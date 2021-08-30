import numpy as np
import matplotlib.pyplot as plt
import random


class GaussianBandit:
    def __init__(self):
        self._arm_means = np.random.uniform(0., 1., 10)  # Sample some means
        self.n_arms = len(self._arm_means)
        self.rewards = []
        self.total_played = 0

    def reset(self):
        self.rewards = []
        self.total_played = 0

    def play_arm(self, a):
        reward = np.random.normal(self._arm_means[a], 1.)  # Use sampled mean and covariance of 1.
        self.total_played += 1
        self.rewards.append(reward)
        return reward


def greedy(bandit, timesteps):
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)

    # TODO: init variables (rewards, n_plays, Q) by playing each arm once
    for i in range(bandit.n_arms):                  # run the loop once for each arm    
        rewards[i] += bandit.play_arm(i)            # update the reward received for each arm
        n_plays[i] += 1                             # update the number of times each arm is played by one
        Q[i] = rewards[i]                           # initialize the action-value array Q using the sample average method

    # Main loop
    while bandit.total_played < timesteps:
        # This example shows how to play a random arm:
        # a = random.choice(possible_arms)
        # reward_for_a = bandit.play_arm(a)
        # TODO: instead do greedy action selection
        # TODO: update the variables (rewards, n_plays, Q) for the selected arm

        arm = np.argmax(Q)                          # select the arm with the maximum action-value in the last step
        rewards[arm] += bandit.play_arm(arm)        # add the total reward received from playing that arm
        n_plays[arm] += 1                           # increment the number of times the arm is played
        Q[arm] = rewards[arm]/n_plays[arm]          # update the action-value for the arm played


def epsilon_greedy(bandit, timesteps):
    # TODO: epsilon greedy action selection (you can copy your code for greedy as a starting point)

    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    for i in range(bandit.n_arms):                  # initialize the action-values by playing each arm once
        rewards[i] += bandit.play_arm(i)
        n_plays[i] += 1
        Q[i] = rewards[i]

    epsilon = 0.1                                   # set the epsilon value 

    while bandit.total_played < timesteps:
        # reward_for_a = bandit.play_arm(0)  # Just play arm 0 as placeholder
        if(random.random() < epsilon):              # play a random arm with probability 0.1
            arm = random.choice(possible_arms)      # choose a random arm
            rewards[arm] += bandit.play_arm(arm)    # update the total reward for the arm played        
            n_plays[arm] += 1                       # update the number of times the arm is played
            Q[arm] = rewards[arm]/n_plays[arm]      # update the action-value for the arm played
        else:                                       # choose the greedy approach with probability 0.9
            arm = np.argmax(Q)                      # choose the arm with the maximum action-value till the last step
            rewards[arm] += bandit.play_arm(arm)    # update the total reward for the arm played
            n_plays[arm] += 1                       # update the number of times the arm is played
            Q[arm] = rewards[arm]/n_plays[arm]      # update the action-value for the arm



def main():
    n_episodes = 10000  # TODO: set to 10000 to decrease noise in plot
    n_timesteps = 1000
    rewards_greedy = np.zeros(n_timesteps)
    rewards_egreedy = np.zeros(n_timesteps)

    for i in range(n_episodes):
        if i % 100 == 0:
            print ("current episode: " + str(i))

        b = GaussianBandit()  # initializes a random bandit
        greedy(b, n_timesteps)
        rewards_greedy += b.rewards

        b.reset()  # reset the bandit before running epsilon_greedy
        epsilon_greedy(b, n_timesteps)
        rewards_egreedy += b.rewards

    rewards_greedy /= n_episodes
    rewards_egreedy /= n_episodes
    plt.plot(rewards_greedy, label="greedy")
    print("Total reward of greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_greedy)))
    plt.plot(rewards_egreedy, label="e-greedy")
    print("Total reward of epsilon greedy strategy averaged over " + str(n_episodes) + " episodes: " + str(np.sum(rewards_egreedy)))
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.savefig('bandit_strategies.eps')
    plt.show()


if __name__ == "__main__":
    main()
