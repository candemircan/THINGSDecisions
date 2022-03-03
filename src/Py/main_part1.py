from helpers import *
from learners import *
from model_recovery_and_comparison import recover_models 


if __name__ == "__main__":

    df, const = load_data()

    ## get resnet features

    get_resnet_features(df,const)

    ## get newly extracted features
    
    feature_paths = ['data/spose_embedding_14d_sorted.npy','data/spose_embedding_82d_sorted.npy']
    id_path = 'data/unique_id.csv'
    new_df_paths = ['data/behavioural_data_14.csv','data/behavioural_data_82.csv']

    for feature_path, new_df_path in zip(feature_paths,new_df_paths):
        new_latent_features(df,feature_path,id_path,new_df_path)
    
    ## select and run models
    
    models = [Linear,
              GP,
              SingleCue,
              EqualWeighting,
              ]
    
    features = ['resnet',
                'original',
                '14',
                '82']


    all_df = run_models(models=models, features=features, remake=False)

    ## model recovery analyses

    # agents trained with original features
    og_agents = all_df[all_df['features']=='original']
    recover_models(og_agents, 'results/recovery_original.csv')

    # bin the above dataframe for testing strategy change
    og_agents_b1 = og_agents[og_agents['trial_n'].between(1,50)]
    recover_models(og_agents_b1, 'results/recovery_original_b1.csv')
    og_agents_b2 = og_agents[og_agents['trial_n'].between(51,100)]
    recover_models(og_agents_b2, 'results/recovery_original_b2.csv')
    og_agents_b3 = og_agents[og_agents['trial_n'].between(101,150)]
    recover_models(og_agents_b3, 'results/recovery_original_b3.csv')

    # recover agents trained with different features
    features.remove('original')
    for feature in features:
        feature_df = all_df[all_df['features']==feature]
        recover_models(feature_df, f'results/recovery_{feature}.csv')