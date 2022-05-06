packages <- c('tidyverse','lme4','broom.mixed')

lapply(packages, library, character.only = TRUE)


predict_with_feature <-function(df,path,remake=FALSE){

    if (file.exists(path) && remake==FALSE){
        return()
        }

    feat_models <- list()
    for (i in 1:49){ # loop through each feature

        v_name <- sprintf("rl_loading_diff%s",i)
        l <- sprintf("loadings_left%s",i)
        r <- sprintf("loadings_right%s",i)
        df[v_name] <- df[l] - df[r]
        df[v_name] <- scale(df[v_name])
        f <- sprintf('rightChoice ~ -1 + rl_loading_diff%i + (-1+rl_loading_diff%i|participant_n)', i, i)
        m <- glmer(f, family='binomial',data=df)
        feat_models[i] <- logLik(m)
        }

    logliks <- tibble(
        feature_no = 1:49,
        loglik = flatten_chr(feat_models))
        write.table(logliks , file = path,sep=",", row.names=FALSE)
}

predict_trial_feature <- function(df,path,remake=FALSE){

    if (file.exists(path) && remake==FALSE){
        return()
        }

    df$RewardDiffRightLeft <- df$RewardRight - df$RewardLeft
    df$RewardDiffRightLeft <- scale(df$RewardDiffRightLeft)
    df$trial_n <- scale(df$trial_n)

    trial_feature_model <- glmer(rightChoice~-1+RewardDiffRightLeft+
                               RewardDiffRightLeft * trial_n + trial_n +
                               (-1 + RewardDiffRightLeft + trial_n|participant_n),
                             family="binomial",data=df)

    trial_feature_df <- tidy(trial_feature_model)
    write.csv(trial_feature_df,path)
}

predict_resnet_linear <- function(df,path_mixed, path_anova, remake=FALSE){

    if (file.exists(path_mixed) && remake==FALSE){
        return()
        }

    resnet_df <- df %>% filter(features=='resnet')
    original_df <- df %>% filter(features=='original')

    resnet_df$value_rl_diff <- scale(resnet_df$value_rl_diff)[,1]
    resnet_df$original_rl_diff <- scale(original_df$value_rl_diff)[,1]

    # using value estimates of the linear model trained with original features
    null_model <- glmer(
        rightChoice ~-1 + original_rl_diff + (-1 + original_rl_diff | participant_n),
        family="binomial",data=resnet_df)

    # null model + value estimates of linear model using resnet features
    resnet_model <- glmer(
        rightChoice ~-1 + original_rl_diff + value_rl_diff + (-1 + original_rl_diff + value_rl_diff | participant_n),
        family="binomial",data=resnet_df)


    resnet_feature_compare <- tidy(anova(null_model,resnet_model))
    resnet_feature_df <- tidy(resnet_model)
    write.csv(resnet_feature_df,path_mixed)
    write.csv(resnet_feature_compare,path_anova)



}

loo_cv <-function(df,path,remake=FALSE){

  if (file.exists(path) && remake==FALSE){
    return()
  }
  learners <- unique(df$model)
  learners <- learners[learners != "Human"]
  df <- df %>% filter(model %in% learners)
  row_no <- nrow(df) / (length(learners))
  par_no <- length(unique(df$participant_n))
  trial_no <- length(unique(df$trial_n))

  loss <- matrix(nrow=row_no, ncol=length(learners))
  total <- row_no*length(learners)
  progress <- 1
  m_idx <- 1
  for (learner in learners){
    cur_df <- df %>% filter(model==learner)
    cur_df$value_rl_diff <- scale(cur_df$value_rl_diff)
    points <- nrow(cur_df)
    for (i in 1:points){
      training <- cur_df[-c(i), ]
      test <- cur_df %>% slice(i)
      cur_model <- glmer(rightChoice~-1+value_rl_diff+(-1 + value_rl_diff|participant_n), family=binomial, data=training)
      prediction <- predict(cur_model, newdata=test,  type="response")
      outcome <- test$rightChoice
      loglikelihood <- log((prediction * outcome)+((1-prediction) * (1-outcome)))

      loss[i, m_idx] <- loglikelihood
      cat(sprintf('\n %s of %s models fitted',progress,total))
      progress <- progress + 1

    }
    m_idx <- m_idx + 1

  }
  loss <- as.data.frame(loss)
  colnames(loss) <- learners
  loss$p <- rep(1:par_no, times=1, each=trial_no)
  write.csv(loss, path ,row.names = FALSE)


}
