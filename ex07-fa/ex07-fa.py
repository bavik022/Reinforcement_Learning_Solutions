import gym
import numpy as np
import matplotlib.pyplot as plt
import random

position_bounds = [-1.2, 0.6]
velocity_bounds = [-0.07, 0.07]
numIntervals = 20
intervalLengthPosition = (position_bounds[1] - position_bounds[0])/numIntervals
intervalLengthVelocity = (velocity_bounds[1] - velocity_bounds[0])/numIntervals
numActions = 3

def get_active_states(position, velocity):
    positionStateNum = position/intervalLengthPosition
    veclocityStateNum = velocity/intervalLengthVelocity
    activeStates = (positionStateNum, velocityStateNum)

def chooseAction(epsilon, Q, env, state):
    if(random.random() < epsilon):
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

def q_learn(alpha = 0.8, gamma = 0.8, epsilon = 0.1, lmbd = 0.9, env):
    Q = np.random.rand(numIntervals, numActions)
    e = np.zeros(numIntervals)
    state = env.reset()
    action = chooseAction(epsilon, Q, env, state)
    while True:
        next_state, reward, done, _ = env.step(action)
        for s in range(num_states):
            e[s] = gamma * lmbd * e[s]
        activeState = 


def random_episode(env):
    """ This is an example performing random actions on the environment"""
    while True:
        env.render()
        action = env.action_space.sample()
        print("do action: ", action)
        observation, reward, done, info = env.step(action)
        print("observation: ", observation)
        print("reward: ", reward)
        print("")
        if done:
            break


def main():
    env = gym.make('MountainCar-v0')
    env.reset()
    random_episode(env)
    env.close()


if __name__ == "__main__":
    main()
