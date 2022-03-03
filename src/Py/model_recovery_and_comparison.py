import os
import numpy as np
import pandas as pd
from groupBMC.groupBMC import GroupBMC

def recover_models(df,path,remake=False):

    if os.path.isfile(path) and not remake:
        return pd.read_csv(path)

    agents = list(df['model'].unique())
    recovery_df = pd.DataFrame(index=agents,columns=agents)

    for performing_agent in agents:
        for recovering_agent in agents:
            og_prob = df[df['model'] == performing_agent]['choice_probability'].values
            og_prob = np.where(og_prob>.99,.99,og_prob)
            og_prob = np.where(og_prob<.01,.01,og_prob)
            recovery_prob = df[df['model'] == recovering_agent]['choice_probability'].values
            recovery_prob = np.where(recovery_prob>.99,.99,recovery_prob)
            recovery_prob = np.where(recovery_prob<.01,.01,recovery_prob)
            cross_entropy = -np.sum(og_prob * np.log(recovery_prob) + (1-og_prob) * np.log(1-recovery_prob))
            recovery_df.at[performing_agent,recovering_agent] = cross_entropy
    
    recovery_df.to_csv(path)
    return df

def model_frequency(losses, path,remake=False):

    if os.path.isfile(path) and not remake:
        return pd.read_csv(path)

  
    ids = np.array(losses["p"])
    models = list(losses.columns.values)
    models.remove('p')

    losses = losses.values
    L = np.zeros((len(models), len(np.unique(ids))))

    for i, ID in enumerate(np.unique(ids)):
        idx = np.where(losses[:, -1] == ID)
        for model in range(len(models)):
            l = losses[idx, model]
            L[model, i] = np.sum(l)

    result = GroupBMC(L).get_result()
    df = pd.DataFrame()
    df["mean"] = result.frequency_mean
    df["var"] = result.frequency_var
    df["pxp"] = result.protected_exceedance_probability
    df["xp"] = result.exceedance_probability
    df['model'] = models
    df.to_csv(path, index=False)

    return df

def r_squared(losses,const,path,binned=False,remake=False):

    models = list(losses.columns)
    models.remove('p')
    r2 ={model:0 for model in models}
    trials = 50 if binned else const['trials']
    random_loss = np.log(.5) * trials * const['par']

    for model in models:
        r2[model] = [1- losses[model].sum()/random_loss]
    
    df = pd.DataFrame(r2)
    df.to_csv(path,index=False)
    return df

def nll(losses):

    models = list(losses.columns)
    models.remove('p')
    nll ={model:0 for model in models}

    for model in models:
        nll[model] = [-losses[model].sum()]
    
    df = pd.DataFrame(nll)
    return df