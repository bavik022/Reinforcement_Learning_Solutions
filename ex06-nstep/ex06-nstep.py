import gym
import numpy as np
import matplotlib.pyplot as plt
import random

def choose_action(epsilon, state, Q, env):
    if random.random() < epsilon:
        action = random.choice(range(env.action_space.n))
    else:
        action = np.argmax(Q[state])
    return action

def value_iteration(env):           #dynamic programming to calculate ground truth state values
    V_states = np.zeros(env.observation_space.n)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    idx = 0
    while True:
        print("Iteration:" + str(idx))
        idx += 1  # increment iterations
        delta = 0.0  # initialize delta
        for s in range(env.observation_space.n):  # iterate over the number of states
            v = np.copy(V_states)[s]  # store the current value of that state
            max_a = -99999
            for a in range(env.action_space.n):
                probs = np.asarray(
                    env.P[s][a]
                )  # retrieve the probabilities of landing in a particular state and receiving the corresponding reward
                psum = 0.0
                for i in range(len(probs)):
                    p = probs[i][0]
                    s_ = int(probs[i][1])
                    r = probs[i][2]
                    psum = psum + (
                        p * (r + gamma * V_states[s_])
                    )  # calculate the new action-value
                if psum > max_a:
                    max_a = psum
            V_states[s] = max_a  # set the value to the maximum action-value
            delta = max(delta, np.abs(v - V_states[s]))  # set the delta value
        print("delta:" + str(delta))
        if delta < theta:  # check if delta has converged
            print("optimal value function:" + str(V_states))
            break
    return V_states

def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    """ TODO: implement the n-step sarsa algorithm """
    Q = np.random.rand(env.observation_space.n, env.action_space.n) #initialize Q values randomly
    Q[env.observation_space.n - 1, :] = 0
    for i in range(num_ep):
        observations = []
        state = env.reset()
        done = False
        t = 0
        T = np.inf
        states = [state]
        rewards = []
        action = choose_action(epsilon, state, Q, env)
        actions = [action]
        while True:
            if t<T:
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                states.append(state)
                if done:
                    T = t + 1
                else:
                    action = choose_action(epsilon, state, Q, env)
                    actions.append(action)
            tm = t - n + 1
            if tm >= 0:                                                 # update the rewards 
                G = 0
                for j in range(tm + 1, min(tm+n+1, T+1)):
                    G += np.power(gamma, j-tm-1) * rewards[j-1]   
                if tm + n < T:
                    G += np.power(gamma, n) * Q[state][action]  
                Q[states[tm]][actions[tm]] += alpha * (G - Q[states[tm]][actions[tm]])      # update the action-value
            t += 1
            if tm == T - 1:
                break   
    values = np.zeros(env.observation_space.n)
    for j in range(env.observation_space.n):                            # calculate the optimal state values
         values[j] = np.max(Q[j])
    return values         


env=gym.make('FrozenLake-v0', map_name="8x8")
# TODO: run multiple times, evaluate the performance for different n and alpha
values_true = value_iteration(env)
errors = []
alphas = np.linspace(0.1, 1.0, 10)
legends = []
for n in range(1, 11):
    errors.append([])
    for alpha in alphas:
        print("Running nstep sarsa with n = " + str(n) + " and alpha = " + str(alpha))
        values_pred = nstep_sarsa(env, n, alpha)
        errors[n-1].append(np.sqrt(np.mean((values_pred - values_true)**2)))
    plt.plot(alphas, errors[n-1])
    legend = "n = " + str(n)
    legends.append(legend)
plt.legend(legends)
plt.xlabel("alpha")
plt.ylabel("RMS error")
plt.title("Performance of n-step sarsa")
plt.show()
