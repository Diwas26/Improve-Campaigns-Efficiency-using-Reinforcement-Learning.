# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 13:54:39 2020

@author: Diwas
"""

import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import random

### setting the Prameters ###

N = 5000
d = 9

## Creating the simulation ##
conversion_rates = [0.22 , 0.17 , 0.14 , 0.09 , 0.11 , 0.14 , 0.20 , 0.08 , 0.10]

## Initialising the matrix with zeros with dimension of total iteration as rows as no. of strategy as columns ##

## Running simulation ##
x = np.array(np.zeros([N,d]))

### Loop throgh every row, i.e. for every user in the list ###
for i in range(N):
    ## Looping throgh every strategy we have ##
    for j in range(d):
        
        if np.random.rand() <= conversion_rates[j]:
            x[i,j] = 1
            
# "NOTE": The above simulation is done for thomsons sampling. The above simulation is explained as:

#Now the strtegy that we are looking at right now say have a conversion rate of 16%,
#means the customer using thsi startegy has high chance of conversion. 
#Now to simulate this, when we draw a random number from 0-1, it will have
#16% chances of being less that 0.16 and 84% chnace of larger than 0.16, as it came from
#uniform gaussian distribution. Now as the startegy we picked also has 16% chance of 
#converting user, so we give the specific cell, i.e. that user with that specific strtegy
#1, otherwise 0.
            
            
########### THOMSAON SAMPLING ############33

## Implementing thomson sampleing and random strategy for comarasion ##
strategies_selected_rs = []
strategies_selected_ts = []   
total_rewards_rs = 0
total_rewards_ts = 0

## Exclusive to thomson sampling, as this represents total number of "1" accumulated
#over N iteration's and vice-versa total number of "0" accumulated over N iterations ##

number_of_rewards_1 = [0]*d
number_of_rewards_0 = [0]*d

for k in range(N):
    
    ## Deploying Random Strategy ##
    
    strategy_rs = random.randrange(0,d) ## Picking up a strategy randomly ##
    strategies_selected_rs.append(strategy_rs)
    reward_rs = x[k , strategy_rs]
    total_rewards_rs = total_rewards_rs + reward_rs
    
    ## Thomson Sampling ##
    
    # Part1:- Random draw from beta-distribution's for all the 9 or available strategies ##
    max_random = 0
    strategy_ts = 0
    for l in range(0,d):
        
        random_beta = random.betavariate(number_of_rewards_1[l]+1 , 
                                         number_of_rewards_0[l]+1)
        # Part2:- Select the strategy that has the heighest random draws from beta-dist #
        
        if random_beta > max_random:
            max_random = random_beta ## check if the random draw is grater than pre-assigned "max_random". If yes, then replace the "max_random" with random draw ##
            strategy_ts = l ## keep the strategy that gave us the maximum beta ##
        
        # Part3: Finally updating list for strategies where we got "reward-1" and "reward-0" #
        
    reward_ts = x[k , strategy_ts]
    
    # Now, upadte the list, for which strategies we got rewrd-1 or reward-0 #
    if reward_ts == 1:
        number_of_rewards_1[strategy_ts] = number_of_rewards_1[strategy_ts] +1
        
    else:
        number_of_rewards_0[strategy_ts] = number_of_rewards_0[strategy_ts] +1
        
    ## Finally, updating "total_accumulated_rewards" and "strategies_selected" ##
    
    strategies_selected_ts.append(strategy_ts)
    total_rewards_ts = total_rewards_ts + reward_ts
    
## Computing the efficieny of Thomsons Sampling (in terms of incremental revenue the company will make)##
    
abs_return = (total_rewards_ts - total_rewards_rs) * 999 ## for subscriptio to a plan a ustomer has to pay Rs 999/- ##
rel_return = (total_rewards_ts - total_rewards_rs) / total_rewards_rs * 100

print('Absolute Return: Rs {} /-'.format(abs_return))
print('Relative Return: {} %'.format(rel_return))
        
        
## Plotting the strategies selected over 10,000 iteration ##
plt.hist(strategies_selected_ts)
plt.title("Selection of Strategies by Thomson Sampling ")
plt.xlabel("Strategy")
plt.ylabel("No.of times strategy was selected")
plt.show()


         
