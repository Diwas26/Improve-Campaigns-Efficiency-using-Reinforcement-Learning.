# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:42:27 2020

@author: Diwas
"""

import pandas as pd
import numpy as np
import sys
import os
import math
from numpy.random import choice

## Defining functions for loading dataset ##

diwas = pd.read_csv("S:/AI_For_Business/Contextual_Bandits_for _Recommendation_Sys/MovieLens_Dataset/ml-25m/ratings.csv" ,
                          engine = 'python' , nrows = 60000)

def read_moviedataset():
    ratings = pd.read_csv("S:/AI_For_Business/Contextual_Bandits_for _Recommendation_Sys/MovieLens_Dataset/ml-25m/ratings.csv" ,
                          engine = 'python' , nrows = 150000)
    
    movies = pd.read_csv("S:/AI_For_Business/Contextual_Bandits_for _Recommendation_Sys/MovieLens_Dataset/ml-25m/movies.csv" , 
                         engine = 'python')
    
    links = pd.read_csv("S:/AI_For_Business/Contextual_Bandits_for _Recommendation_Sys/MovieLens_Dataset/ml-25m/links.csv" , 
                        engine = 'python')
    
    movies = movies.join(movies.genres.str.get_dummies().astype(bool))
    movies.drop('genres', inplace=True, axis=1) ## dropping a column ##
    
    ## Left joining two dataframes ##
    df = ratings.join(movies, on='movieId', how='left', rsuffix='_movie')
    
    return df


def preprocess_movielens_25m(df, min_number_of_reviews=100):
    # remove ratings of movies with < N ratings. too few ratings will cause the recsys to get stuck in offline evaluation
    movies_to_keep = pd.DataFrame(df.movieId.value_counts())\
        .loc[pd.DataFrame(df.movieId.value_counts())['movieId']>=min_number_of_reviews].index
    df = df.loc[df['movieId'].isin(movies_to_keep)]
    # shuffle rows to debias order of user ids
    df = df.sample(frac=1)
    # create a 't' column to represent time steps for the bandit to simulate a live learning scenario
    df['t'] = np.arange(len(df))
    df.index = df['t']
    # rating >= 4.5 stars is a 'like', < 4 stars is a 'dislike'
    df['liked'] = df['rating'].apply(lambda x: 1 if x >= 4.5 else 0)
    return df



def get_ratings_25m(min_number_of_reviews=100):
    df = read_moviedataset()
    df = preprocess_movielens_25m(df, min_number_of_reviews=100)
    return df


###### Setting up testing environment #######
    
# simulation params: slate size, batch size (number of events per training iteration)
slate_size = 5
batch_size = 10

df = get_ratings_25m(min_number_of_reviews=100)


# initialize empty history 
# (the algorithm should be able to see all events and outcomes prior to the current timestep, but no current or future outcomes)
history = pd.DataFrame(data=None, columns=df.columns)
history = history.astype({'movieId': 'int32', 'liked': 'float'})


def replay_score(history, df, t, batch_size, recs):
    # reward if rec matches logged data, ignore otherwise
    actions = df[t:t+batch_size]
    actions = actions.loc[actions['movieId'].isin(recs)]
    actions['scoring_round'] = t
    # add row to history if recs match logging policy
    history = history.append(actions)
    action_liked = actions[['movieId', 'liked']]
    return history, action_liked


# initialize empty list for storing scores from each step
#rewards = []
#
#for t in range(df.shape[0]//batch_size):
#    
#    t = t * batch_size
#    # generate recommendations from a random policy
#    recs = np.random.choice(df.movieId.unique(), size=(slate_size), replace=False)
#    
#    # send recommendations and dataset to a scoring function so the model can learn & adjust its policy in the next iteration
#    
#    history, action_score = replay_score(history, df, t, batch_size, recs)
#    
#    if action_score is not None:
#        action_score = action_score.liked.tolist()
#        rewards.extend(action_score)
        

rewards = []
num_arms = df.movieId.unique().shape[0]
weights = [1.0] * df.movieId.unique().shape[0] # initialize one weight per arm


def distr(weights, gamma=0.0):
    weight_sum = float(sum(weights))
    return tuple((1.0 - gamma) * (w / weight_sum) + (gamma / len(weights)) for w in weights)

def draw(probability_distribution, n_recs=1):
    arm = choice(df.movieId.unique(), size=n_recs,
        p=probability_distribution, replace=False)
    return arm

def update_weights(weights, gamma, movieId_weight_mapping, probability_distribution, actions):
    # iter through actions. up to n updates / rec
    if actions.shape[0] == 0:
        return weights
    for a in range(actions.shape[0]):
        action = actions[a:a+1]
        weight_idx = movieId_weight_mapping[action.movieId.values[0]]
        estimated_reward = 1.0 * action.liked.values[0] / probability_distribution[weight_idx]
        weights[weight_idx] *= math.exp(estimated_reward * gamma / num_arms)
    return weights

def score(history, df, t, batch_size, recs):
    # https://arxiv.org/pdf/1003.5956.pdf
    # replay score. reward if rec matches logged data, ignore otherwise
    #actions = df.copy()[t:t+batch_size]
    actions = df[t:t+batch_size]
    actions = actions.loc[actions['movieId'].isin(recs)]
    actions['scoring_round'] = t
    # add row to history if recs match logging policy
    history = history.append(actions)
    action_liked = actions[['movieId', 'liked']]
    return history, action_liked

def exp3_policy(df, history, t, weights, movieId_weight_mapping, gamma, n_recs, batch_size):
    '''
    Applies EXP3 policy to generate movie recommendations
    Args:
        df: dataframe. Dataset to apply EXP3 policy to
        history: dataframe. events that the offline bandit has access to (not discarded by replay evaluation method)
        t: int. represents the current time step.
        weights: array or list. Weights used by EXP3 algorithm.
        movieId_weight_mapping: dict. Maping between movie IDs and their index in the array of EXP3 weights.
        gamma: float. hyperparameter for algorithm tuning.
        n_recs: int. Number of recommendations to generate in each iteration. 
        batch_size: int. Number of observations to show recommendations to in each iteration.
    '''
    probability_distribution = distr(weights, gamma)
    recs = draw(probability_distribution, n_recs=n_recs)
    history, action_score = score(history, df, t, batch_size, recs)
    weights = update_weights(weights, gamma, movieId_weight_mapping, probability_distribution, action_score)
    action_score = action_score.liked.tolist()
    return history, action_score, weights , recs

gamma = 0.7
n = 5
batch_size = 10
movieId_weight_mapping = dict(map(lambda t: (t[1], t[0]), enumerate(df.movieId.unique())))
max_time = df.shape[0] # total number of ratings to evaluate using the bandit
verbose = True
i = 1
recs_list= []

for t in range(max_time//batch_size): #df.t:
	t = t * batch_size
	if t % 100000 == 0:
		if verbose == 'TRUE':
			print(t)
            
	history, action_score, weights , recs = exp3_policy(df, history, t, weights, movieId_weight_mapping, gamma, n, batch_size)	
	rewards.extend(action_score)

recs_list.append(recs)
    

#history, action_score, weights = exp3_policy(df, history, t, weights, movieId_weight_mapping, gamma, n, batch_size)	
#rewards.extend(action_score)



    

