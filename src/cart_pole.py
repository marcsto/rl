"""
  A simple reinforcement learning implementation to solve the CartPole problem.
  It's based on Natural Evolution Strategies found at https://blog.openai.com/evolution-strategies/
  It also makes use of some of the code from http://kvfrans.com/simple-algoritms-for-solving-cartpole/
  
  Author: Marc Stogaitis  
"""
import gym
import numpy as np

EVAL_RUNS = 2

# Setup the environment
env = gym.make('CartPole-v0')
observation_size = env.observation_space.shape[0]
action_size = env.action_space.n
print("Observation Size:", observation_size)
print("Action size:", action_size)

def computeReward(agent):
    """ Computes the reward signal for a specified agent. """
    all_run_reward_sum = 0
    # Take the average of multiple runs to smooth out results
    for _ in range(EVAL_RUNS):
        observation = env.reset()
        step_count = 0
        reward_sum = 0
        while True:
            env.render()
            action = agent.take_action(observation)
            observation, reward, done, _ = env.step(action)
            reward_sum += reward
            step_count += 1
            if done:
                all_run_reward_sum += reward_sum
                break
    return all_run_reward_sum / EVAL_RUNS

""" An agent takes an action based on observations """
class Agent:
    def take_action(self, observation):
        raise NotImplementedError()

""" Randomly selects an action. Useful for testing """
class RandomAgent(Agent):
    def take_action(self, observation):
        return env.action_space.sample()
    
""" Chooses between 2 actions based on a set of weights """
class BinaryActionAgent(Agent):
    def __init__(self, weights):
        self.weights = weights

    def take_action(self, observation):
        return 0 if np.matmul(self.weights, observation) < 0 else 1
    
""" 
    Runs the Natural Evolution optimization strategy.
    Algorithm description: 
    1. Create a population of weight vectors by making random modification to an initial weight vector.
    2. Compute the reward for each weight vector
    3. Move the main weight vector towards the weight vectors that had the most success.
    4. Repeat the process above.
    
    https://blog.openai.com/evolution-strategies/  
"""
class NaturalEvolutionOptimizer:
    def optimize(self):
        iterations = 10
        npop = 20  # population size  
        sigma = 0.1  # noise standard deviation  
        alpha = 0.1  # learning rate
          
        w = np.random.randn(observation_size)  # initial guess  
        for i in range(iterations):
            print("Starting iteration ", i)
            N = np.random.randn(npop, observation_size)
            R = np.zeros(npop)
            for j in range(npop):
                w_try = w + sigma * N[j]
                agent = BinaryActionAgent(w_try)
                R[j] = computeReward(agent)  # f(w_try)
                print(R[j], w_try, j, i)
            
            stdDev = np.std(R)
            if (stdDev == 0):
                break
            A = (R - np.mean(R)) / np.std(R)
            w = w + alpha / (npop * sigma) * np.dot(N.T, A)
        
        print(w)
        return BinaryActionAgent(w)
    
        
    
if __name__ == '__main__':
    best_agent = NaturalEvolutionOptimizer().optimize()
    final_reward = computeReward(best_agent)
    print("Final reward = ", final_reward)
    


