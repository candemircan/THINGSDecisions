import os
import pandas as pd
from helpers import load_data
from learners import *
from model_recovery_and_comparison import model_frequency, r_squared


if __name__ == "__main__":

    # do a model frequency analysis and compute pseudo R² for loo-cv -loglikelihoods of each modelling part

    _, const = load_data()
    
    original_nll = pd.read_csv('results/loo_original.csv')
    model_frequency(original_nll,'results/frequency_original.csv')
    r_squared(original_nll,const,'results/r2_original.csv')

    bins = ['b1','b2','b3']

    for binned in bins:
        binned_loss = pd.read_csv(f'results/loo_original_{binned}.csv')
        model_frequency(binned_loss,f'results/frequency_original_{binned}.csv')
        r_squared(binned_loss,const,binned=True,path=f'results/r2_original_{binned}.csv')

    features = ['original','resnet','14','82']

    for feature in features:
        feature_loss = pd.read_csv(f'results/loo_{feature}.csv')
        model_frequency(feature_loss,f'results/frequency_{feature}.csv')
        r_squared(feature_loss,const,f'results/r2_{feature}.csv')

    # plot results
    os.system("python src/Py/plots.py")
    os.system("python src/Py/supp_plots.py")
