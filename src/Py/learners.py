import numpy as np
import pandas as pd
import GPy
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import scale
from scipy.stats import logistic
from scipy.special import softmax

class Model:

    def __init__(self, df, const, par, resnet):
        self.df = df[df['participant_n'] == 'participant_' + str(par).zfill(2)]
        # [0,] -> left, [1,] -> right
        self.values = np.zeros([2, const['trials']])
        self.length = const['trials']
        self.rl_val_dif = np.zeros(const['trials'])
        self.choice_probability = np.zeros(const['trials']) # probability of choosing the option on the right
        self.choice_history = np.zeros(const['trials'])
        self.reward_history = np.zeros(const['trials'])
        self.correct = np.zeros(const['trials']) 
        self.rewards = self.df[self.df.columns[pd.Series(
            self.df.columns).str.startswith('Reward')]].to_numpy().T
        self.human = self.df['rightChoice'].to_numpy()
   
        # rewards:
        # i -> left(==0) vs right (==1)
        # j -> trial no
        
        # human choices:
        # left(==0) vs right (==1)

        # feature loadings:
        # i -> trial no
        # j -> feature no
        # k -> left (==0) vs right(==1) option

        if resnet:
            self.trials = np.load('data/resnet_features.npy')
            self.trials = self.trials[par-1]

        else:
            left = self.df[self.df.columns[pd.Series(
                self.df.columns).str.startswith('loadings_left')]].to_numpy()
            right = self.df[self.df.columns[pd.Series(
                self.df.columns).str.startswith('loadings_right')]].to_numpy()
            self.trials = np.stack((left, right), axis=-1)


        self.feats = self.trials.shape[1] 

    def get_xy(self):

        X = [self.trials[0, :, 0], self.trials[0, :, 1]]
        y = [self.rewards[0, 0], self.rewards[1, 0]]

        return X, y

    def add_xy(self, X, y, i):

        X.extend([self.trials[i, :, 0], self.trials[i, :, 1]])
        y.extend([self.rewards[0, i], self.rewards[1, i]])

        return X, y

    def model_unique(self, X, y, i):

        pass
    
    def fit(self):

        X, y = self.get_xy()

        for i in range(1, self.length):

            self.model_unique(X, y, i)
            self.rl_val_dif[i] = self.values[1, i] - self.values[0, i]
            X, y = self.add_xy(X, y, i)
        
        self.rl_val_dif = self.values[1,:] - self.values[0,:]
        self.rl_val_dif = self.rl_val_dif.T
        self.decide()
    
    def decide(self):

        """ get choice probabilities from softmax. take probability weighted sum of  options to record reward_history
        note that this does not reflect choice history and self.correct , which are both arrays of 1 for now. They are filled
        with human behavioural data later for regression analyses. """

        
        self.choice_probability = softmax(self.values,axis=0)[1,:]
        self.reward_history = self.rewards[0,:] * (1-self.choice_probability) + self.rewards[1,:] * self.choice_probability

        
        
        


class Linear(Model):

    def __init__(self, df, const, par, resnet):

        Model.__init__(self, df, const, par, resnet)
        self.weights = []
    
    def model_unique(self, X, y, i):

        cur_model = BayesianRidge().fit(np.array(X), np.array(y))
        self.weights.append(cur_model.coef_)
        self.values[:, i] = cur_model.predict(self.trials[i, :, :].T)

class GP(Model):
    
    def __init__(self, df, const, par, resnet):

        Model.__init__(self, df, const, par, resnet)

    def model_unique(self, X, y, i):
        X = np.array(X)
        y = np.array(y)
        kernel = GPy.kern.RBF(input_dim=self.feats,
                              variance=1., lengthscale=1.)
        gp = GPy.models.GPRegression(X, y.reshape(-1, 1), kernel)
        gp.optimize()
        self.values[0, i], _ = gp.predict(
            self.trials[i, :, 0].T.reshape(1, self.feats))
        self.values[1, i], _ = gp.predict(
            self.trials[i, :, 1].T.reshape(1, self.feats))

class SingleCue(Model):

    def __init__(self, df, const, par, resnet):

        Model.__init__(self, df, const, par, resnet)
        self.cumulative_loss = np.zeros(self.feats)
        best = np.amax(self.rewards,axis=0)
        self.best = np.where(best == self.rewards[1,:],1,0)

    
    def model_unique(self, X, y, i):

        X = np.array(X)
        y = np.array(y)

        
        models = [BayesianRidge().fit(X[:,f].reshape(-1,1), y) for f in range(self.feats)]
        
        predictions = np.array([models[f].predict(
            self.trials[i, f, :].reshape(-1,1)) for f in range(self.feats)])

        rl_pred_diff = predictions[:,1] - predictions[:,0]
        rl_pred_diff = scale(rl_pred_diff)
        probs = logistic.cdf(rl_pred_diff)
        

        self.cumulative_loss -= np.log([(probs[j] * self.best[i])+((1-probs[j]) * (1-self.best[i])) for j in range(self.feats)])

        candidates = np.where(self.cumulative_loss == self.cumulative_loss.min())[0]

        cur_model = np.random.choice(candidates)

        self.values[:, i] = predictions[cur_model, :].T

class EqualWeighting(Model):

    def __init__(self, df, const, par, resnet):

        Model.__init__(self, df, const, par, resnet)

    def model_unique(self, X, y, i):

        cur_model = BayesianRidge().fit(
            np.sum(np.array(X), axis=1).reshape(-1, 1), np.array(y))
        self.values[:, i] = cur_model.predict(
            np.sum(self.trials[i, :, :].T, axis=1).reshape(-1, 1))