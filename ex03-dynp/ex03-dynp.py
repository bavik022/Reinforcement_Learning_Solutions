import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np

# Init environment
#env = gym.make("FrozenLake-v0")
env=gym.make('FrozenLake-v0', map_name="8x8")
# you can set it to deterministic with:
# env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
# random_map = generate_random_map(size=5, p=0.8)
# env = gym.make("FrozenLake-v0", desc=random_map)


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n


def value_iteration():
    V_states = np.zeros(n_states)  # init values as zero
    opt_policy = np.zeros(n_states)
    theta = 1e-8
    gamma = 0.8
    idx = 0
    # TODO: implement the value iteration algorithm and return the policy
    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r
    while True:
        print("Iteration:" + str(idx))
        idx += 1  # increment iterations
        delta = 0.0  # initialize delta
        for s in range(n_states):  # iterate over the number of states
            v = np.copy(V_states)[s]  # store the current value of that state
            max_a = -99999
            for a in range(n_actions):
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
                    opt_policy[s] = a
            V_states[s] = max_a  # set the value to the maximum action-value
            delta = max(delta, np.abs(v - V_states[s]))  # set the delta value
        print("delta:" + str(delta))
        if delta < theta:  # check if delta has converged
            print("optimal value function:" + str(V_states))
            break
    return opt_policy


def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # run the value iteration
    policy = value_iteration()
    print("Computed policy:")
    print(policy)

    # This code can be used to "rollout" a policy in the environment:
    """print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break"""


if __name__ == "__main__":
    main()
