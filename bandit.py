import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import math
import random
import pandas as pd


    
class bandit:
    def __init__(self, k, eps, iters, prob,strat ):

        # arms
        self.k = k

        #initialize strategy
        self.strategy=strat

        #Initialize list to store policy for action preference strategy
        self.policy_prob=np.zeros(k)

        # exploration prob
        self.eps = eps

        # iterations
        self.iters = iters

        # step count
        self.n = 0

        # step count of each arm
        self.k_n = np.zeros(k)

        # mean reward
        self.mean_reward = 0.0

        #preference
        self.pref = np.zeros(k)

        # total reward
        self.reward = np.zeros(iters)

        # mean reward for each arm
        self.k_reward = np.zeros(k)

        # initializing probability distribution
        self.dis = prob

        # sets probability of reward to normal distribution
        if prob == 'gaussian':
             self.prob = np.random.normal(0,1,k)

        # sets probability of reward to bernoulli distribution
        if prob == 'bernoulli':
            y = bernoulli(0.5)
            self.prob = y.rvs(k)

    def run(self):
        for i in range(self.iters):
           
            #switch statement based on strategy of the agent
            if self.strategy==0:
                self.step()
            elif self.strategy==1:
                self.step_optimistic()
            elif self.strategy==2:
                self.step_UCB()
            elif self.strategy==3:
                self.step_AP()
            self.reward[i] = self.mean_reward


    # Step function for the optimistic initial values strategy
    def step_optimistic(self):
        p=np.random.rand()

        #First step choose a random arm
        if (self.eps == 0 and self.n == 0) or p < self.eps :
            #before we begin training, we set all the arms rewards to 0.2 since we are using an initial optimistic strategy
            self.k_reward=np.full(self.k,0.2)
            action = np.random.choice(self.k)
        else:
            action = np.argmax(self.k_reward)

        #getting rewards
        if self.prob == 'bernoulli':
            y = bernoulli(0.5)
            reward = y.rvs(1)
            
        else:
            reward = np.random.normal(self.prob[action], 1)

        # Update counts
        self.n += 1
        self.k_n[action] += 1

        # Update total
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n

        # Update results

        #if the agent hasnt chosen this lever, then the mean updates without taking into account the optimal initial values
        if self.k_n[action]==1:
            self.k_reward[action] = (reward - self.k_reward[action])/self.k_n[action]
        else:
            self.k_reward[action] = self.k_reward[action] + (reward - self.k_reward[action])/self.k_n[action]
        return

    #this function updates the policy for each action
    def update_policy(self):
        sum=0
        #sums action preference of all actions for the denominator
        for i in self.pref:
            sum+=math.exp(i)

        #fills in list of probabilities of taking an action given a time stamp
        for j in range(0,self.k):
            self.policy_prob[j]=math.exp(self.pref[j])/sum


    # this function updates the H(a) for each action (to be used in the next iteration)
    def update_preference(self,action,current_reward,alpha):
        
        for i in range(self.k):
            if i!=action:
                self.pref[i]=self.pref[i]-alpha*(current_reward-self.mean_reward)*self.policy_prob[i]
            else: 
                self.pref[i]=self.pref[i]+alpha*(current_reward-self.mean_reward)*(1-self.policy_prob[i])
        return


    #Step function for action preferences strategies
    def step_AP(self):
        
        self.update_policy()
        
        # Chooses action based on probabilities calculated
        action = random.choices(list(range(self.k)),self.policy_prob,k=1)[0]

        if self.prob == 'bernoulli':
            y = bernoulli(0.5)
            reward = y.rvs(1)

        else:
            reward = np.random.normal(self.prob[action], 1)

        # Update counts and preference 
        self.n += 1
        self.k_n[action] += 1
        self.update_preference(action,reward, self.eps)

        # Update total
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n

        # Update results
        self.k_reward[action] = self.k_reward[action] + (reward - self.k_reward[action])/self.k_n[action]


    # Step functin for the upper confidence bounds strategy
    def step_UCB(self):
        # p regulates amount of exploration
        p = np.random.rand()
        c=1.5
        
        #First step choose a random arm
        if (self.eps == 0 and self.n == 0) or p < self.eps :
            action = np.random.choice(self.k)
        
        else:
            #Calculate UCB for each action
            UCB=np.zeros(self.k)
            for i in range(self.k):
                UCB[i]= self.k_reward[i]+c*math.sqrt(self.n/self.k_n[i])
            action = np.argmax(UCB)

        #getting rewards based on probability distribution
        if self.prob == 'bernoulli':
            y = bernoulli(0.5)
            reward = y.rvs(1)

        else:
            reward = np.random.normal(self.prob[action], 1)


        # Update counts
        self.n += 1
        self.k_n[action] += 1

        # Update total
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n


        # Update results
        self.k_reward[action] = self.k_reward[action] + (reward - self.k_reward[action])/self.k_n[action]

    # Step function for epsilon-greedy
    def step(self):
        #random number for exploration
        p = np.random.rand()
        
        #First step choose a random arm or if random number below exploration probability
        if (self.eps == 0 and self.n == 0) or p < self.eps :
            action = np.random.choice(self.k)
        
        else:
            action = np.argmax(self.k_reward)

        if self.dis == 'bernoulli':
            y = bernoulli(self.prob[action])
            reward = y.rvs(1)

        else:
            reward = np.random.normal(self.prob[action], 1)

        # Update counts
        self.n += 1
        self.k_n[action] += 1

        # Update total
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n

        # Update results
        self.k_reward[action] = self.k_reward[action] + (reward - self.k_reward[action])/self.k_n[action]

    def optimal_action(self):
    #calculate percentage of pulling the lever with the highest mean rewards for results section
        idx = np.argmax(self.mean_reward)
        optimal_action = self.k_n[idx]/iters
        return optimal_action


if __name__ == '__main__':

    #collect hyper parameters 
    k = int(input("How many arms would you like?: "))
    eps = float(input("Search Probability (epsilon)?: "))
    epochs = int(input("How many epochs? "))
    iters = 1000

    #initialize dictionaries and arrays to store counts
    bands_rewards = {}
    bands_actions = {}
    optimal_action=np.zeros(8)
    for i in range (8):
        bands_rewards[i] = np.zeros(iters)
        bands_actions[i] = np.zeros(k)


    # Dictionary that will contain the rewards and actions of each bandit
    dictionary_bandits={"greedy_g" : [] ,
                        "Optimistic_g": [],
                        "UCB_g":[],
                        "AP_g":[],
                        "greedy_b" : [] ,
                        "Optimistic_b": [],
                        "UCB_b":[],
                        "AP_b":[],
                        }

    # Run experiments
    bandits = {}
    for i in range(epochs):
        print("epoch: ", i)

        # Initialize bandits
        for j in range(4):
                bandits[j] = bandit(k, eps, iters, prob = 'gaussian', strat = j) 
        for j in range (4):
                bandits[j+4] = bandit(k, eps, iters, prob = 'bernoulli', strat = j) 

        for j in range(8):
            bandits[j].run()
            optimal_action[j]+=bandits[j].optimal_action()
         
        
        #Updating the long-term mean rewards
        for j in range(8):
            bands_rewards[j] = bands_rewards[j] + (
                bandits[j].reward - bands_rewards[j]) / (i + 1)
            bands_actions[j] = bands_actions[j] + (
                bandits[j].k_n - bands_actions[j]) / (i + 1)

    #putting all the data into a dictionary for plotting 
    for i,j in zip(dictionary_bandits.keys(),range(8)):
        print(bands_rewards[j])
        dictionary_bandits[i].append(list(bands_rewards[j]))
        dictionary_bandits[i].append(list(bands_actions[j]))
    
    # Calculating and printing the optimal actions percentages
    optimal_action=optimal_action/epochs
    print("PERCENTAGE OF OPTIMAL ACTION : \n")
    print(optimal_action)
    
    #Plotting the mean rewards over training
    plt.figure(figsize=(12,8))
    for i,j in zip(dictionary_bandits.keys(),range(8)):
        plt.plot(dictionary_bandits[i][0])
    plt.legend(dictionary_bandits.keys())
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Average Rewards after " + str(epochs) 
        + " epochs")
    plt.show()

    