import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import models, transforms as T
from tqdm import tqdm


def load_data(path='data/behavioural_data.csv'):
    """ load the behavioural data & add/remove some necessary variables

          Parameters
        ----------
        path: string
          path to behavioural data


        Returns
        ----------
        df: pandas.core.frame.DataFrame
          behavioural dataframe
        consts: dictionary
          dictionary with participant no, trial no, and feature no as keys
    """

    df = pd.read_csv(path)
    const = {}
    const['par'] = df['participant_n'].nunique()
    const['trials'] = df['trial_n'].nunique()
    const['feats'] = len(
        [col for col in df.columns if col.startswith('loadings_left')])

    return df, const


def new_latent_features(df,weight_path,id_path,new_df_path,remake=False):

    """ make new dataframe with the newly extracted loadings """

    if os.path.isfile(new_df_path) and not remake:
        return pd.read_csv(new_df_path)
    
    weights = np.load(weight_path)
    ids = pd.read_csv(id_path)
    ids = ids['id'].values.tolist()
    dims = weights.shape[1]


    #drop original loadings
    df.drop(df.filter(regex='loading').columns, axis=1, inplace=True)
    df.drop(df.filter(regex='Loading').columns, axis=1, inplace=True)

    col_names = [f'loadings_{side}{no}' for side in ['left','right'] for no in range(1,dims+1)]
    new_weights = np.zeros((len(df),len(col_names)))

    for index, row in df.iterrows():
        left_stim = row['ImageLeft'].replace('stimuli/','').replace('.png','')
        right_stim = row['ImageRight'].replace('stimuli/','').replace('.png','')

        left_where = ids.index(left_stim)
        right_where = ids.index(right_stim)

        cur_weights = []
        cur_weights.extend(weights[left_where,:].tolist())
        cur_weights.extend(weights[right_where,:].tolist())
        new_weights[index,:] = cur_weights
    
    df_add = pd.DataFrame(columns=col_names,data=new_weights)
    df = df.join(df_add)
    df.to_csv(new_df_path)

    return df

def get_resnet_features(df,const,path='data/resnet_features.npy',remake=False):

    """ extract features from resnet18"""
  
    if os.path.isfile(path) and not remake:
        return

    transform = T.Compose([T.Resize(224), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    rn18 = models.resnet18(pretrained=True).eval()
    feature_extractor = torch.nn.Sequential(*list(rn18.children())[:-1])


    num_trials = const['trials']
    num_participants = const['par']
    num_features = 512

    # init feature array
    features = torch.zeros(num_participants, num_trials, num_features, 2)

    for i in tqdm(range(num_participants)):
        # get particpant dataframe
        df_participant = df[df['participant_n'] ==f'participant_{str(i + 1).zfill(2)}']

        # list to store image path
        
        leftImages = df_participant['ImageLeft'].values.tolist()
        rightImages = df_participant['ImageRight'].values.tolist()

        # tensor to store images
        batchLeftImages = torch.zeros(len(leftImages), 3, 224, 224)
        batchRightImages = torch.zeros(len(rightImages), 3, 224, 224)

        # load left images to tensors and pass through neural network
        for num_img, image_path in enumerate(leftImages):
            image = Image.open(image_path)
            x = TF.to_tensor(image)
            x = transform(x)
            batchLeftImages[num_img] = x


        features[i, :, :, 0] = feature_extractor(
            batchLeftImages).squeeze().detach().reshape(150,-1)

        # load right images to tensors and pass through neural network
        for num_img, image_path in enumerate(rightImages):
            image = Image.open(image_path)
            x = TF.to_tensor(image)
            x = transform(x)
            batchRightImages[num_img] = x

        features[i, :, :, 1] = feature_extractor(
            batchRightImages).squeeze().detach().reshape(150,-1)
    
    features = features.detach().numpy()
    np.save(path,features)


def run_models(models, features, path="data/humans_and_models.csv",remake=False):
    
    """ run the given models with the given features"""

    if os.path.isfile(path) and not remake:
        return pd.read_csv(path)
    
    correct_features = ['resnet','original','14','82']
    assert set(features).issubset(correct_features), f'features have to be from {correct_features}'
    
    behavioural_df,const = load_data()
    participant_n = list(np.repeat(['participant_' + str(par).zfill(2) for par in range(1,const['par']+1)],const['trials']))
    trial_n = [trial for trial in range(1,const['trials']+1)] * const['par']
    behavioural_df['reward'] = np.where(behavioural_df['rightChoice']==1,behavioural_df['RewardRight'],behavioural_df['RewardLeft'])
   
    agent_df = {
        'participant_n':participant_n,
        'trial_n':trial_n,
        'RewardLeft': behavioural_df['RewardLeft'].tolist(),
        'RewardRight':behavioural_df['RewardRight'].tolist(),
        'PCorrectChoice':behavioural_df['PCorrectChoice'].tolist(),
        'rightChoice':behavioural_df['rightChoice'].tolist(),
        'value_left':0,
        'value_right':0,
        'value_rl_diff':0,
        'reward':behavioural_df['reward'].to_list(),
        'model':'Human',
        'choice_probability':.5,
        'features':'Human'
    }
    agent_df = pd.DataFrame(agent_df)

    model_names = [model.__name__ for model in models]
    all_dfs = []

    y = len(model_names) * const['par'] * len(features)
    x = 0

    for feature in features:

        resnet=True if feature=='resnet' else False
        
        if feature in ['original','resnet']:
            data_path = 'data/behavioural_data.csv'
        else:
            data_path = f'data/behavioural_data_{feature}.csv'
        
        df,const= load_data(data_path)
        agent_dfs = {agent: agent_df.copy() for agent in model_names}
        for key in agent_dfs.keys():
            agent_dfs[key]['model'] = key

        for model, model_name in zip(models, model_names):
            
            values = []
            choice_probabilities = []
            choice_histories = []
            corrects = []
            reward_histories = []
            for par in range(1, const['par']+1):

                # run model
                cur_model = model(df, const, par, resnet=resnet)
                cur_model.fit()
                x += 1
                print(f"{model_name} finished for participant {par}")
                print(f"{x} of {y} models done")


                # save to relevant lists

                values.extend(cur_model.values.T)
                choice_probabilities.extend(cur_model.choice_probability)
                reward_histories.extend(cur_model.reward_history)


            
            agent_dfs[model_name][['value_left','value_right']] = values
            agent_dfs[model_name]['value_rl_diff'] = agent_dfs[model_name]['value_right'] - agent_dfs[model_name]['value_left']
            agent_dfs[model_name]['choice_probability'] = choice_probabilities
            agent_dfs[model_name]['rightChoice'] = df['rightChoice'].values
            agent_dfs[model_name]['PCorrectChoice'] = df['PCorrectChoice'].values
            agent_dfs[model_name]['reward'] = reward_histories
            agent_dfs[model_name]['features'] = feature


        all_dfs.extend(list(agent_dfs.values()))
    
    all_dfs.append(agent_df) # add human behavioural data
    model_df = pd.concat(all_dfs)
    model_df = model_df.reset_index(drop=True)
    model_df.to_csv(path)

    return model_df