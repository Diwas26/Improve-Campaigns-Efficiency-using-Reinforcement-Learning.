# Improve Campaigns Efficiency using Reinforcement-Learning.
Technology has changed the way how we walk, talk, think and even also the way we do business. Even e-commerce, banking are not untouched by this revolution. Using latest and best in class algorithms for business problems have given us some unprecedented results in recent times. Here we introduce a award winning multi armed bandits technique called "Thomson's Sampling" for one of our case study problems, where we need to find the best strategy to use for customer's so that they will subscribe to the subscription plan.

Now for our real life case study, Imagine an Online Retail Business that has million's of customers. These customers are only people buying
some products on the website from time to time, getting them delivered at home. The business is doing good, but the companies senior management has decided to take some action plan to maximize revenue even more. This plan consists of offering to the customers the option to subscribe to a premium plan, which will give them some benefits like reduced prices, special deals, etc. This premium plan is offered at a yearly price of Rs 1000/- , and the goal of this online retail business is of course to get the maximum customers to subscribe to this premium plan (* Idea is inspired by an online training I attended which was artificial intelligence for business).

Also in our case study we will be facing 9 different strategies, and our AI will have no idea of which is the best one, and absolutely no prior information on any of their conversion rates. However we will make the assumption that each of these 9 strategies does have a fixed conversion rate. These strategies were carefully and smartly elaborated by the marketing team, and each of them has the same goal: convert the maximum clients to the premium plan. However, these 9 strategies will be all different.

**Simulation**

In order to simulate this case study, we will assume these strategies have the following conversion rates:
S1 - 0.22 ,  
S2 - 0.17 ,  
S3 - 0.14 ,  
S4 - 0.09 ,  
S5 - 0.11 ,  
S6 - 0.14 ,  
S7 - 0.20 ,  
S8 - 0.08 ,  
S9 - 0.10

However, in a real life situation we would have no idea of what would be these conversion rates. We only know them here for simulation purposes, just so that we can check in the end that our AI manages to figure out the best strategy.

![Simulated_Rewards_Matrix](https://github.com/Diwas26/Improve-Campaigns-Efficiency-using-Reinforcement-Learning./blob/master/1.png)

  

