import gym
import numpy as np

# Init environment
# Lets use a smaller 3x3 custom map for faster computations
custom_map3x3 = [
    'SFF',
    'FFF',
    'FHG',
]
env = gym.make("FrozenLake-v0", desc=custom_map3x3)
# TODO: Uncomment the following line to try the default map (4x4):
#env = gym.make("FrozenLake-v0")

# Uncomment the following lines for even larger maps:
# random_map = generate_random_map(size=5, p=0.8)
# env = gym.make("FrozenLake-v0", desc=random_map)

# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n

print("States: ", n_states)
print("Actions: ", n_actions)
# the r vector is zero everywhere except for the goal state (last state)
r = np.zeros(n_states)
r[-1] = 1.

gamma = 0.8


# left = 0; right =2 ; up=1; down=4
steps = [[]]

''' Helper function to generate list of all possible moves (modulo 4 counter) '''


def moduloCount(n, base):

    digits = "0123"  # 4 possible moves

    if n < base:
        x = digits[n]
    else:
        x = moduloCount(n//base, base) + digits[n % base]

    return x


""" This is a helper function that returns the transition probability matrix P for a policy """


def trans_matrix_for_policy(policy):
    transitions = np.zeros((n_states, n_states))
    for s in range(n_states):
        probs = env.P[s][policy[s]]
        for el in probs:
            transitions[s, el[1]] += el[0]
    return transitions


""" This is a helper function that returns terminal states """


def terminals():
    terms = []
    for s in range(n_states):
        # terminal is when we end with probability 1 in terminal:
        if env.P[s][0][0][0] == 1.0 and env.P[s][0][0][3] == True:
            terms.append(s)
    return terms


def value_policy(policy):
    P = trans_matrix_for_policy(policy)
    # TODO: calculate and return v
    # (P, r and gamma already given)
    I = np.eye(n_states)
    v = np.matmul(np.linalg.inv(I - gamma*P), r)
    return v


def bruteforce_policies():
    #terms = terminals()
    optimalpolicies = []
    policy = np.zeros(n_states, dtype=np.int)
    optimalvalue = np.zeros(n_states)
    policyList = []
    valueList = []  # Store values of all policies
    # Total combinations of states = 4^10 (n_actions ^ n_states)
    for i in range(n_actions ** n_states):

        x = moduloCount(i, n_actions)
        L = len(x)
        policy = list('0'*(n_states-L)+x)
        policy = [int(i) for i in policy]
        policyList.append(policy)
        valueList.append(value_policy(policy))
        if(i % 1000 == 0):
            print("State_idx: ", i)
            print(policyList[i - 1])

    optimalvalue = np.amax(valueList, axis=0)
    equalityMatrix = (valueList == optimalvalue)
    idx = 0
    for row in equalityMatrix:
        if(all(row)):  # Check if values of all states are maximum
            optimalpolicies.append(policyList[idx])
        idx += 1

    #print(str(i)+"---  "+policy)
    # in the discrete case a policy is just an array with action = policy[state]

    # TODO: implement code that tries all possible policies, calculate the values using def value_policy.
    # Find the optimal values and the optimal policies to answer the exercise questions.

    print("Optimal value function:")
    print(optimalvalue)
    print("number optimal policies:")
    print(len(optimalpolicies))
    print("optimal policies:")
    print(np.array(optimalpolicies))
    return optimalpolicies


def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # Here a policy is just an array with the action for a state as element
    policy_left = np.zeros(n_states, dtype=np.int)  # 0 for all states
    policy_right = np.ones(n_states, dtype=np.int) * 2  # 2 for all states

    # Value functions:
    # print("Value function for policy_left (always going left):")
    # print(value_policy(policy_left))
    # print("Value function for policy_right (always going right):")
    # print(value_policy(policy_right))

    optimalpolicies = bruteforce_policies()
    # $ optimalpolicies = policy_right
    # This code can be used to "rollout" a policy in the environment:

    # print("rollout policy:")
    # maxiter = 100
    # state = env.reset()
    # for i in range(maxiter):
    #     new_state, reward, done, info = env.step(optimalpolicies[0][state])
    #     env.render()
    #     state = new_state
    #     if done:
    #         print("Finished episode")
    #         break


if __name__ == "__main__":
    main()
