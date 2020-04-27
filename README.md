# Improve Campaigns Efficiency using Reinforcement-Learning.
Technology has changed the way how we walk, talk, think and even also the way we do business. Even e-commerce, banking are not untouched by this revolution. Using latest and best in class algorithms for business problems have given us some unprecedented results in recent times. Here we introduce a award winning multi armed bandits technique called "Thomson's Sampling" for one of our case study problems, where we need to find the best strategy to use for customer's so that they will subscribe to the subscription plan.

Now for our real life case study, Imagine an Online Retail Business that has million's of customers. These customers are only people buying
some products on the website from time to time, getting them delivered at home. The business is doing good, but the companies senior management has decided to take some action plan to maximize revenue even more. This plan consists of offering to the customers the option to subscribe to a premium plan, which will give them some benefits like reduced prices, special deals, etc. This premium plan is offered at a yearly price of Rs 1000/- , and the goal of this online retail business is of course to get the maximum customers to subscribe to this premium plan (* Idea is inspired by an online training I attended which was artificial intelligence for business).

Also in our case study we will be facing 9 different strategies, and our AI will have no idea of which is the best one, and absolutely no prior information on any of their conversion rates. However we will make the assumption that each of these 9 strategies does have a fixed conversion rate. These strategies were carefully and smartly elaborated by the marketing team, and each of them has the same goal: convert the maximum clients to the premium plan. However, these 9 strategies will be all different.

**Problem Simulation :**

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

**Defining Environmrnt :**

Online Learning is a special branch of Artificial Intelligence, where there is not much need of defining the states and actions. Here, a state would simply be a specific customer onto whom we deploy a strategy, and the action would simply be the strategy selected. Here we are doing online learning. However we do have to define the rewards, since again we will have to make a rewards matrix, where each row corresponds to a user being deployed a strategy, and each column corresponds to one of the 9 strategies. Therefore, since we will actually run this online learning experiment on 10,000 customers, this rewards matrix will have 10,000 rows and 9 columns. Then, each cell will get either a 0 if the customer doesn’t subscribe to the premium plan after being approached by the selected strategy, and a 1 if the customer does subscribe after being approached by the selected strategy. And the values in the cell are exactly, the rewards.

Now a very important thing to understand is that the rewards matrix is only here for the simulation, and in real life we would have no such thing as a rewards matrix. We will just simulate 10,000 customers successively being approached by one of the 9 strategies. Here is below as an example the first few rows of a simulated rewards matrix

![Simulated_Rewards_Matrix](https://github.com/Diwas26/Improve-Campaigns-Efficiency-using-Reinforcement-Learning./blob/master/1.png)


**Algorithm :**

The AI solution that will figure out the best strategy is called **"Thompson Sampling"**. It is by far the best model for that kind of problems. The goal of Thomson Sampling is to select best strategy over many iterations. Below is the pseudo code for how it is achieved:

*For each round n, over some iterations following steps will repeat :*

![Pseudo_Code](https://github.com/Diwas26/Improve-Campaigns-Efficiency-using-Reinforcement-Learning./blob/master/2.png)

**Intuition**

Each strategy has its own beta distribution. Over the rounds, the beta distribution of the strategy with the highest conversion rate will be progressively shifted to the right, and the beta distributions of the strategies with lower conversion rates will be progressively shifted to the left. Therefore, the strategy with the highest conversion rate will be more and more selected. Below is a graph displaying three beta distributions of three strategies, that will help you visualize this:

![Thomsons_Sampling](https://github.com/Diwas26/Improve-Campaigns-Efficiency-using-Reinforcement-Learning./blob/master/Thomsons_Sampling.png)






  

