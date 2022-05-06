# %%
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# # Supplementary Figures: Model Recovery

# %%
original_recovery = pd.io.parsers.read_csv("../../results/recovery_original.csv",index_col=0)
original_recovery = np.log(original_recovery)
sns.heatmap(original_recovery.T,annot=True,cbar_kws={'label': 'negative log likelihood (logged)'})
plt.title('Model Recovery with Original Latent Features')
plt.xlabel('Performing Agent')
plt.ylabel('Recovering Agent')
plt.savefig('../../doc/figures/supp_figure1.png',bbox_inches='tight',dpi=400)

# %%
fig, axes = plt.subplots(1, 3)
fig.suptitle('Model Recovery for Time Binned Data')
for num,binned in enumerate(['b1','b2','b3']):
    binned_df =pd.io.parsers.read_csv(f"../../results/recovery_original_{binned}.csv",index_col=0)
    binned_df = np.log(binned_df)
    sns.heatmap(binned_df.T,ax=axes[num],annot=True,cbar_kws={'label': 'negative log likelihood (logged)'})
    axes[num].set_title(binned)
    
plt.setp(axes[1], xlabel='Performing Agent')
plt.setp(axes[0], ylabel='Recovering Agent')
plt.savefig('../../doc/figures/supp_figure2.png',bbox_inches='tight',dpi=400)

# %%
resnet_recovery = pd.io.parsers.read_csv("../../results/recovery_resnet.csv",index_col=0)
resnet_recovery = np.log(resnet_recovery)
sns.heatmap(resnet_recovery.T,annot=True,cbar_kws={'label': 'negative log likelihood (logged)'})
plt.title('Model Recovery with Resnet Features')
plt.xlabel('Performing Agent')
plt.ylabel('Recovering Agent')
plt.savefig('../../doc/figures/supp_figure3.png',bbox_inches='tight',dpi=400)

# %%
fig, axes = plt.subplots(1, 2)
fig.suptitle('Model Recovery for Different Latent Features')
for num,feats in enumerate(['14','82']):
    binned_df =pd.io.parsers.read_csv(f"../../results/recovery_{feats}.csv",index_col=0)
    binned_df = np.log(binned_df)
    sns.heatmap(binned_df.T,ax=axes[num],annot=True,cbar_kws={'label': 'negative log likelihood (logged)'})
    axes[num].set_title(f'{feats} features')
    
plt.setp(axes[1], xlabel='Performing Agent')
plt.setp(axes[0], ylabel='Recovering Agent')
plt.savefig('../../doc/figures/supp_figure4.png',bbox_inches='tight',dpi=400)