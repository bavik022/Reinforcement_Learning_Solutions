import gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
plt.style.use('ggplot')


def run_episode(env):

    states = []
    actions = []
    rewards = []
    state = env.reset()
    obs = env.reset()
    action = 0
    done = False
    while not done:
        #print("Observation:", obs)
        if obs[0] >= 20:
            # print("Stick")
            action = 0
        else:
            # print("Hit")
            action = 1

        obs, reward, done, _ = env.step(action)
        #print("Reward:", reward)
        # print("")
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = obs
    return states, actions, rewards


def first_visit_MC(env, num_episodes):

    # Value table for storing the values of each state
    value_table = defaultdict(float)
    n = defaultdict(int)

    for i in tqdm(range(num_episodes)):

        states, actions, rewards = run_episode(env)
        returns = 0

        # For each step calculate returns as a sum of rewards
        for t in range(len(states) - 1, -1, -1):
            S = states[t]
            R = rewards[t]

            returns += R

            # Check if the episode is visited for the first time =>
            # If Yes: Assign the value of the state as an average of returns

            if S not in states[:t]:
                n[S] += 1
                value_table[S] = (returns + value_table[S] * (n[S] - 1)) / n[S]

    return value_table


def plot_blackjack(V, ax1, ax2):
    player_sum = np.arange(12, 21 + 1)
    dealer_show = np.arange(1, 10 + 1)
    usable_ace = np.array([False, True])
    state_values = np.zeros(
        (len(player_sum), len(dealer_show), len(usable_ace)))

    for i, player in enumerate(player_sum):
        for j, dealer in enumerate(dealer_show):
            for k, ace in enumerate(usable_ace):
                state_values[i, j, k] = V[player, dealer, ace]

    X, Y = np.meshgrid(player_sum, dealer_show)

    ax1.plot_wireframe(X, Y, state_values[:, :, 0])
    ax2.plot_wireframe(X, Y, state_values[:, :, 1])

    for ax in ax1, ax2:
        ax.set_zlim(-1, 1)
        ax.set_ylabel('Player sum')
        ax.set_xlabel('Dealer showing')
        ax.set_zlabel('State Value')
    plt.show()


def main():
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    env = gym.make('Blackjack-v0')
    episodes = 500000
    value = first_visit_MC(env, episodes)

    _, axes = plt.subplots(nrows=2, figsize=(
        6, 9), subplot_kw={'projection': '3d'})
    axes[0].set_title('No Usable ace')
    axes[1].set_title('Usable ace')
    plot_blackjack(value, axes[0], axes[1])


if __name__ == "__main__":
    main()
