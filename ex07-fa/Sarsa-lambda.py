#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

def Q_lambda(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.2, lamb=0.1, num_ep=int(1e4)):
    n_states = 20  
    n_actions = env.action_space.n  
    w = np.random.rand(n_actions,2)  # Number of features in a single state = 2

    # Initialize Q table
    ''' Q = np.random.uniform(low = -1, high = 1, size = (n_states, n_states, n_actions))
    # Deduce the Value function
    V = np.max(Q,axis=2)
    # Initialize eligibility traces:
    e = np.zeros((n_states,n_states))'''
    z=0

    interval = (env.observation_space.high - env.observation_space.low) / n_states
    high = env.observation_space.high
    low  = env.observation_space.low

    total_episode_length = []
    reward_list = []
    avg_reward_list = []

    for ep in tqdm(range(num_ep)):      
        S = env.reset()
        S = [int(s) for s in np.floor((S - low) / interval)]

        z = np.zeros(2)
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)             # Random action
        else:
            action = np.argmax([np.array(S).dot(w[a]) for a in range(n_actions)])


        done = False
        t = 0

        while not done:
            #choose action
            
            S_, R, done, _ = env.step(action)
            S_ = [int(s) for s in np.floor((S_ - low) / interval)] #discretize state
            if np.random.rand() < epsilon:
                next_action = np.random.randint(n_actions)             # Random action
            else:
                next_action = np.argmax([np.array(S).dot(w[a]) for a in range(n_actions)])

            del_v = S  # gradient of the linear function approximator is the state itself
            z = (gamma*lamb)*z + del_v
            delta = R + gamma * np.array(S_).dot(w[next_action]) - np.array(S).dot(w[action])
            w = w + ((alpha*delta) * z)
            S = S_
            action = next_action
            t += 1
            if(R>0):
                print("Reward: ",R, " || Timestep: ",t)

            reward_list.append(R)

            total_episode_length.append(t)
            avg_reward_list.append(np.mean(reward_list))
            reward_list = []
        

        if (ep+1)%1000 == 0:
            V = getValueFunction(w)
            print('Episode {} || Avg Timestep: {} || Avg Reward: {}'.format(ep+1, np.mean(total_episode_length), np.mean(avg_reward_list)))
            total_episode_length = []
            avg_reward_list = []
            plt.imshow(V)
            plt.colorbar()
            plt.show()

    return True

def getValueFunction(w):
    w = np.transpose(w)
    n_states = 20
    V = np.zeros((n_states,n_states),dtype=float)
    for i in range(n_states):
        for j in range(n_states):
            S = np.array((i,j),dtype=int)
            np.reshape(S,(1,-1))
            V[i][j] = np.max(S.dot(w))

    return V

lamb = np.linspace(0,1,10)
env = gym.make('MountainCar-v0')
for l in lamb:
    Q_lambda(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.5, lamb=0.1, num_ep=int(1e4))


# In[ ]:




