# %%
import string

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
import seaborn as sns

from helpers import load_data


mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams.update({'font.size': 20})
plt.rcParams["figure.figsize"] = (25,25/4)
cmap = ["#fd7f6f", "#7eb0d5", "#8bd3c7", "#bd7ebe", "#ffb55a"]
sns.set_palette(cmap)

mpl.use('pgf')
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})



behavioural_df, const = load_data('../../data/behavioural_data.csv')
s_font_size = 35

# %% [markdown]
# # Figure 1: Design & Behavioural Analyses

# %%
fig1, axs1 = plt.subplots(1, 4)

### 1A ###

task_design = mpimg.imread('../../doc/figures/task_design.png')


axs1[0].spines['bottom'].set_visible(False)
axs1[0].spines['left'].set_visible(False)
axs1[0].imshow(task_design)
axs1[0].get_xaxis().set_ticks([])
axs1[0].get_yaxis().set_ticks([])
axs1[0].annotate('100',
                      xy=(500, 400),
                       weight='bold',ha='center',color='green',fontsize=30)
axs1[0].annotate('2',
                      xy=(1400, 400),
                        weight='bold',ha='center',color='black', fontsize=30)


axs1[0].text(-0.1, 1.2, string.ascii_uppercase[0], transform=axs1[0].transAxes,
                  size=s_font_size, weight='bold')


### 1B ###

behavioural_df['obtained_reward'] = np.where(behavioural_df['responseKey'] == 'left',
                        behavioural_df['RewardLeft'], behavioural_df['RewardRight'])
behavioural_df['regret'] = behavioural_df['MaxReward'] - behavioural_df['obtained_reward']

def cum_mean_regret(df):

    return np.cumsum(df['regret'])/range(1, const['trials']+1)


behavioural_df['cum_mean_regret'] = behavioural_df.groupby('participant_n').apply(cum_mean_regret).values

sns.lineplot(data=behavioural_df, x='trial_n', y='cum_mean_regret', color=cmap[4],
            style='participant_n', ax=axs1[1], alpha=.4)

sns.lineplot(data=behavioural_df, x='trial_n', y='cum_mean_regret', color='k', ax=axs1[1])

axs1[1].set(ylabel=r'Cumulative Mean Regret $\pm$ SE', xlabel='Trial')
axs1[1].text(-0.1, 1.1, string.ascii_uppercase[1], transform=axs1[1].transAxes,
                  size=s_font_size, weight='bold')
axs1[1].legend([], [], frameon=False)
axs1[1].set_ylim(ymin=0)
axs1[1].set_xlim(xmin=0)

### 1C ###

# initialise bin variables
step = 18
behavioural_df['reward_diff_lr'] = behavioural_df['RewardLeft'] - behavioural_df['RewardRight']
min_val = round(min(behavioural_df['reward_diff_lr']))
max_val = round(max(behavioural_df['reward_diff_lr']))
bins = np.arange(min_val, max_val, step) # end element otherwise not included

# total count of responses
total_count = behavioural_df.groupby(['participant_n', pd.cut(
    behavioural_df['reward_diff_lr'], bins)]).agg({'responseKey': 'count'})

# count left vs right responses for each bin
response_count = behavioural_df.groupby(['participant_n', 'responseKey', pd.cut(
    behavioural_df['reward_diff_lr'], bins)]).agg({'responseKey': 'count'})

# calculate probability of left/right choice for each bin
all_counts = response_count.div(total_count)
all_counts = all_counts.add_suffix('_').reset_index().rename(
    columns={'responseKey_': 'PCorrectChoice'})
all_counts['reward_diff_lr'] = all_counts['reward_diff_lr'].astype(str)

sns.pointplot(data=all_counts[all_counts['responseKey'] == 'left'], x='reward_diff_lr',
              y='PCorrectChoice', color='k', ax=axs1[2])
axs1[2].set(
    xlabel='Left - Right Reward Difference ',
    ylabel='p(Left)')

axs1[2].set_xticklabels([str(x) for x in range(-66,61,18)])

axs1[2].text(-0.1, 1.1, string.ascii_uppercase[2], transform=axs1[2].transAxes,
                  size=s_font_size, weight='bold')

### 1D ###

trial_feature_df = pd.read_csv('../../results/feature_trial_to_choice.csv')
trial_feature_df = trial_feature_df[trial_feature_df['effect']=='fixed']
trial_feature_df = trial_feature_df.replace(['RewardDiffRightLeft', 'trial_n', 'RewardDiffRightLeft:trial_n'],
                                            ['Reward Difference', 'Trial', 'Reward Difference x Trial'])
axs1[3].bar(trial_feature_df['term'],trial_feature_df['estimate'],color=cmap[4])
axs1[3].set_ylabel(r'$\hat{\beta}$',rotation=0,labelpad=10)
axs1[3].set_xticklabels(['Reward \nDifference','Trial','Reward \nDifference x Trial'])
axs1[3].set_ylim(ymin=0)


for regressor in trial_feature_df['term'].values:

    axs1[3].plot([regressor,regressor],
    [float(trial_feature_df[trial_feature_df['term']==regressor]['estimate'] + trial_feature_df[trial_feature_df['term']==regressor]['std.error']),
    float(trial_feature_df[trial_feature_df['term']==regressor]['estimate'] - trial_feature_df[trial_feature_df['term']==regressor]['std.error'])],
    color='k')

axs1[3].annotate('***',
                      xy=('Reward Difference', 1.78),
                       weight='bold',ha='center')
axs1[3].annotate('***',
                      xy=('Reward Difference x Trial', .6),
                       weight='bold',ha='center')

axs1[3].annotate(r'*** p $<$ .001',
                    xy=('Reward Difference x Trial',1.55),
                    style='italic',ha='center')


axs1[3].text(-0.1, 1.1, string.ascii_uppercase[3], transform=axs1[3].transAxes,
                  size=s_font_size, weight='bold')

plt.savefig('../../doc/figures/figure1.pgf',bbox_inches='tight')

# %% [markdown]
# # Figure 2: Model-Based Analyses

# %%
fig2, axs2 = plt.subplots(1, 4)

### 2A ###
feature_loglik_df = pd.read_csv('../../results/features_to_choice.csv')
feature_loglik_df['neg_log_lik'] = - feature_loglik_df['loglik']
rand_negloglik = -np.log(.5) * const['trials'] * const['par']
axs2[0].hlines(rand_negloglik,
                     xmin=0, xmax=49, color='r', alpha=.8, linestyles='dashed', linewidth=2)
sns.scatterplot(data=feature_loglik_df, x='feature_no',
                y='neg_log_lik', color='k', ax=axs2[0])

axs2[0].set(ylabel=r'Negative Log Likelihood', xlabel='Feature No')


axs2[0].annotate('metallic',xy=(1.5,feature_loglik_df.iloc[0,2]),xytext=(8, 1950), 
                 arrowprops=dict(arrowstyle="->"), va='center')

axs2[0].annotate('tool-related',xy=(16,feature_loglik_df.iloc[15,2]-5),xytext=(15, 2400), 
                 arrowprops=dict(arrowstyle="->"), va='center')

axs2[0].annotate('construction-related',xy=(39,feature_loglik_df.iloc[38,2]-5),xytext=(16, 2300), 
                 arrowprops=dict(arrowstyle="->"), va='center')

axs2[0].text(-0.1, 1.1, string.ascii_uppercase[0], transform=axs2[0].transAxes,
                   size=s_font_size, weight='bold')

### 2B ###

model_freq_df = pd.read_csv('../../results/frequency_original.csv')
model_freq_df['low'] = model_freq_df['mean'] - model_freq_df['var']
model_freq_df['high'] = model_freq_df['mean'] + model_freq_df['var']

sns.barplot(data=model_freq_df, x='model', y='mean',
            hue='model', ax=axs2[1], dodge=False)
axs2[1].legend([], [], frameon=False)
axs2[1].set(ylabel=r'Model Frequency',xlabel='Agent')

for model in model_freq_df['model'].unique():
    axs2[1].vlines(x=model,ymin=model_freq_df[model_freq_df['model']==model]['low'].values[0],
    ymax=model_freq_df[model_freq_df['model']==model]['high'].values[0],color='k')

axs2[1].set_xticklabels(['Linear','Gaussian \nProcess','Single \nCue','Equal \nWeighting'])
axs2[1].set_xlabel(None)

axs2[1].text(-0.1, 1.1, string.ascii_uppercase[1], transform=axs2[1].transAxes,
                   size=s_font_size, weight='bold')
axs2[1].set_ylim([0,1])


### 2C ###
binned_dfs = {'1-50':[],'51-100':[],'101-150':[]}
for cur_bin,trials in zip(range(1,4),binned_dfs.keys()):
    binned_dfs[trials] = pd.read_csv(f'../../results/frequency_original_b{cur_bin}.csv')
    binned_dfs[trials]['bin'] = trials


binned_main_df = pd.concat(binned_dfs.values())

sns.barplot(data=binned_main_df, x='bin', y='mean', hue='model', ax=axs2[2])

axs2[2].set(ylabel='Model Frequency',xlabel='Agent')
axs2[2].set_xticklabels(['1-50','51-100','101-150'])
axs2[2].set_xlabel('Trial')
axs2[2].legend([], [], frameon=False)

axs2[2].text(-0.1, 1.1, string.ascii_uppercase[2], transform=axs2[2].transAxes,
                   size=s_font_size, weight='bold')

### 2D ###

all_df = pd.read_csv('../../data/humans_and_models.csv')
original_models_df = all_df[all_df['features']=='original']
original_models_df = original_models_df[original_models_df['model'] != 'Human']
original_models_df['choice_history'] = np.where(original_models_df['value_rl_diff']>0,1,0) # argmax
original_models_df['reward'] = np.where(original_models_df['choice_history'] ==
                            0, original_models_df['RewardLeft'], original_models_df['RewardRight'])

original_models_df['MaxReward'] = np.where(original_models_df['RewardLeft'] >= original_models_df['RewardRight'],
                                            original_models_df['RewardLeft'], original_models_df['RewardRight'])
original_models_df['regret'] = original_models_df['MaxReward'] - original_models_df['reward']
original_models_df['cum_mean_regret'] = original_models_df.groupby(['model','participant_n'],sort=False).apply(cum_mean_regret).values

temp_human_df = behavioural_df[['cum_mean_regret','trial_n','participant_n']]
temp_human_df = temp_human_df.assign(model='Human')

temp_og_models_df = original_models_df[['cum_mean_regret','trial_n','participant_n','model']]

og_model_human_df = pd.concat([temp_og_models_df,temp_human_df])

sns.lineplot(data=og_model_human_df, x='trial_n', y='cum_mean_regret', hue='model',
            ax=axs2[3],ci=None, linewidth=3)



axs2[3].set(ylabel=r'Cumulative Mean Regret', xlabel='Trial')
axs2[3].text(-0.1, 1.1, string.ascii_uppercase[3], transform=axs2[3].transAxes,
                   size=s_font_size, weight='bold')
axs2[3].set_ylim(ymin=0)
axs2[3].set_xlim(xmin=1)
axs2[3].legend([], [], frameon=False)

### legend ###

custom_legend_lines = [Line2D([0], [0], color=x, lw=15) for x in cmap]
fig2.legend(custom_legend_lines, ['Linear','Gaussian Process','Single Cue','Equal Weighting','Human'], loc=(.20,.87),ncol=5,frameon=False)



plt.savefig('../../doc/figures/figure2.pgf',bbox_inches='tight')

# %% [markdown]
# # Figure 3: Representational Analyses

# %%
fig3, axs3 = plt.subplots(1, 4)

### 3A ###

r2_og = pd.read_csv('../../results/r2_original.csv')
r2_og['dimensions'] = 'original'

r2_14 = pd.read_csv('../../results/r2_14.csv')
r2_14['dimensions'] = 'low'

r2_82 = pd.read_csv('../../results/r2_82.csv')
r2_82['dimensions'] = 'high'

r2_allsize = pd.concat([r2_og,r2_14,r2_82],ignore_index=True)

r2_allsize = pd.melt(r2_allsize,id_vars='dimensions',value_vars=['Linear','GP','SingleCue','EqualWeighting'])
r2_allsize['agent'] = r2_allsize['variable']
r2_allsize['r2'] = r2_allsize['value']
sns.barplot(data=r2_allsize, x='dimensions', y='value',
             hue='agent', ax=axs3[0],ci=None)
axs3[0].set(ylabel=r'Predictive Accuracy ($R^2$)')
axs3[0].margins(y=0)
axs3[0].set_xlabel('Dimensions')
axs3[0].legend([], [], frameon=False)
axs3[0].set_ylim([0,.3])
axs3[0].text(-0.1, 1.1, string.ascii_uppercase[0], transform=axs3[0].transAxes,
                   size=s_font_size, weight='bold')

### 3B ###

r2_resnet = pd.read_csv('../../results/r2_resnet.csv')
r2_resnet['dimensions'] = 'pixel based'
og_vs_resnet = pd.concat([r2_resnet,r2_og],ignore_index=True)
og_vs_resnet = pd.melt(og_vs_resnet,id_vars='dimensions',value_vars=['Linear','GP','SingleCue','EqualWeighting'])
og_vs_resnet['agent'] = og_vs_resnet['variable']
og_vs_resnet['r2'] = og_vs_resnet['value']
sns.barplot(data=og_vs_resnet, x='dimensions', y='value',
             hue='agent', ax=axs3[1],ci=None,dodge=True)
axs3[1].set(ylabel=r'Predictive Accuracy ($R^2$)')
axs3[1].margins(y=0)
axs3[1].set_xlabel('Dimensions')
axs3[1].set_ylim([0,.3])

axs3[1].legend([], [], frameon=False)
axs3[1].text(-0.1, 1.1, string.ascii_uppercase[1], transform=axs3[1].transAxes,
                   size=s_font_size, weight='bold')

### 3C ###

resnet_models_df = all_df[all_df['features']=='resnet']
resnet_models_df = resnet_models_df[resnet_models_df['model'] != 'Human']
resnet_models_df['choice_history'] = np.where(resnet_models_df['value_rl_diff']>0,1,0) # argmax
resnet_models_df['reward'] = np.where(resnet_models_df['choice_history'] ==
                            0, resnet_models_df['RewardLeft'], resnet_models_df['RewardRight'])

resnet_models_df['MaxReward'] = np.where(resnet_models_df['RewardLeft'] >= resnet_models_df['RewardRight'],
                                            resnet_models_df['RewardLeft'], resnet_models_df['RewardRight'])
resnet_models_df['regret'] = resnet_models_df['MaxReward'] - resnet_models_df['reward']
resnet_models_df['cum_mean_regret'] = resnet_models_df.groupby(['model','participant_n'],sort=False).apply(cum_mean_regret).values


temp_resnet_models_df = resnet_models_df[['cum_mean_regret','trial_n','participant_n','model']]

resnet_model_human_df = pd.concat([temp_resnet_models_df,temp_human_df],ignore_index=True)

sns.lineplot(data=resnet_model_human_df, x='trial_n', y='cum_mean_regret', hue='model',
            ax=axs3[2],ci=None, linewidth=3, legend=False)

axs3[2].set(ylabel=r'Cumulative Mean Regret', xlabel='Trial')
axs3[2].text(-0.1, 1.1, string.ascii_uppercase[2], transform=axs3[2].transAxes,
                   size=s_font_size, weight='bold')
axs3[2].set_ylim(ymin=0)
axs3[2].set_xlim(xmin=1)


### 3D ###

resnet_feature_df = pd.read_csv('../../results/resnet_linear_to_choice.csv')
resnet_feature_df = resnet_feature_df[resnet_feature_df['effect']=='fixed']
resnet_feature_df = resnet_feature_df.replace(['original_rl_diff', 'value_rl_diff'],
                                            ['Latent Dimension \nValue Difference', 'Pixel Based \nValue Difference'])
axs3[3].bar(resnet_feature_df['term'],resnet_feature_df['estimate'],color=cmap[4], width=.5)
axs3[3].set_ylabel(r'$\hat{\beta}$',rotation=0,labelpad=10)


for regressor in resnet_feature_df['term'].values:

    axs3[3].plot([regressor,regressor],
    [float(resnet_feature_df[resnet_feature_df['term']==regressor]['estimate'] + resnet_feature_df[resnet_feature_df['term']==regressor]['std.error']),
    float(resnet_feature_df[resnet_feature_df['term']==regressor]['estimate'] - resnet_feature_df[resnet_feature_df['term']==regressor]['std.error'])],
    color='k')

axs3[3].set_ylim([0,1.9])


axs3[3].annotate('***',
                       xy=('Latent Dimension \nValue Difference', 1.8),
                        weight='bold',ha='center')
axs3[3].annotate('***',
                       xy=('Pixel Based \nValue Difference', .45),
                        weight='bold',ha='center')

axs3[3].annotate(r'*** p$<$.001',
                 xy=('Pixel Based \nValue Difference',1.65),
                    style='italic',ha='center')


axs3[3].text(-0.1, 1.1, string.ascii_uppercase[3], transform=axs3[3].transAxes,
                  size=s_font_size, weight='bold')

### legend ###
custom_legend_lines = [Line2D([0], [0], color=x, lw=15) for x in cmap]
fig3.legend(custom_legend_lines, ['Linear','Gaussian Process','Single Cue','Equal Weighting','Human'], loc=(.20,.87),ncol=5,frameon=False)

plt.savefig('../../doc/figures/figure3.pgf',bbox_inches='tight')
