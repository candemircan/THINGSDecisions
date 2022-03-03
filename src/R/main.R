packages <- c('tidyverse')
lapply(packages, library, character.only = TRUE)

source('src/R/mixed_effects.R')

behavioural_df <- read_csv('data/behavioural_data.csv')
models_df <- read_csv('data/humans_and_models.csv')

predict_trial_feature(behavioural_df,'results/feature_trial_to_choice.csv')
predict_with_feature(behavioural_df,'results/features_to_choice.csv')


# LOO-CV with original features
df_original <- models_df %>% filter(features == 'original')
loo_cv(df_original, 'results/loo_original.csv')

# # LOO-CV with binned data
df_original_b1 <- df_original %>% filter(trial_n %in% (1:50))
loo_cv(df_original_b1, 'results/loo_original_b1.csv')
df_original_b2 <- df_original %>% filter(trial_n %in% (51:100))
loo_cv(df_original_b2, 'results/loo_original_b2.csv')
df_original_b3 <- df_original %>% filter(trial_n %in% (101:150))
loo_cv(df_original_b3, 'results/loo_original_b3.csv')

# LOO-CV with other features
features <- c('resnet','14','82')
for (feature in features){
    df_feature <- models_df %>% filter(features==feature)
    path <- sprintf('results/loo_%s.csv',feature)
    loo_cv(df_feature, path)
}

predict_resnet_linear(models_df,'results/resnet_linear_to_choice.csv')
