import gym
import numpy as np
import matplotlib.pyplot as plt

gamma = 0.5
alpha = 0.01


def gradient(state, theta, action):
   probs = np.matmul(state, theta)
   probs = np.exp(probs)
   if action == 0:
       grad = np.zeros([4,2])
       grad[:,0] = np.transpose(state) - probs[0] * np.transpose(state)
       grad[:,1] = -probs[1] * np.transpose(state) 
   else:
       grad = np.zeros([4,2])
       grad[:,0] = -probs[0] * np.transpose(state)
       grad[:,1] = np.transpose(state) - probs[1] * np.transpose(state)
       
   return grad


def generate_discounted_rewards(rewards):
    rewards_sum = 0
    discounted_rewards = np.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        rewards_sum = gamma * rewards_sum
        rewards_sum += rewards[t]
        discounted_rewards[t] = rewards_sum
    return discounted_rewards


def policy(state, theta):
    """ TODO: return probabilities for actions under softmax action selection """
    probs = np.matmul(state, theta)
    probs = np.exp(probs)
    probs /= np.sum(probs)
    return probs # both actions with 0.5 probability => random


def generate_episode(env, theta, display=False):
    """ generates one episode and returns the list of states, the list of rewards and the list of actions of that episode """
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    for t in range(500):
        if display:
            env.render()
        p = policy(state, theta)
        #action = np.random.choice(len(p), p=p)
        action = np.argmax(p)

        state, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break
        states.append(state)

    return states, rewards, actions


def REINFORCE(env):
    theta = np.random.rand(4, 2)  # policy parameters
    episode_lengths = []
    episode_lengths_means = []
    e = 0 
    while e < 100000:
        if e % 1000 == 0:
            states, rewards, actions = generate_episode(env, theta, False)  # display the policy every 300 episodes
        else:
            states, rewards, actions = generate_episode(env, theta, False)

        #print("episode: " + str(e) + " length: " + str(len(states)))
        episode_lengths.append(len(states))
        # TODO: keep track of previous 100 episode lengths and compute mean
        if e % 1000 == 0:
            episode_lengths_means.append(np.mean(episode_lengths))
            print("Mean episode length:" + str(np.mean(episode_lengths)))
            episode_lengths = []
            
        # TODO: implement the reinforce algorithm to improve the policy weights
        discounted_rewards = generate_discounted_rewards(rewards)
        for t in range(len(states) - 1):
            grad = gradient(states[t], theta, actions[t])
            theta += alpha * (gamma ** t) * grad * discounted_rewards[t]
        
        e += 1

        if(episode_lengths_means[-1]<=495 and e>=30000):            
            print("Restarting...")
            env.reset()
            e = 0
            theta = np.random.rand(4, 2)  # policy parameters
            episode_lengths = []
            episode_lengths_means = []
        

        
    plt.plot(range(len(episode_lengths_means)),episode_lengths_means)
    plt.title('Plot of average episode lengths')
    plt.show()

def main():
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    REINFORCE(env)
    env.close()


if __name__ == "__main__":
    main()
