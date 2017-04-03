"""
  A simple reinforcement learning implementation to solve the several OpenAi gym problem.
  It's based on Natural Evolution Strategies found at https://blog.openai.com/evolution-strategies/
  It also makes use of some of the code from http://kvfrans.com/simple-algoritms-for-solving-cartpole/
  
  Author: Marc Stogaitis  
"""
import gym
import numpy as np
from gym.spaces.box import Box
from gym import wrappers
import os
    
EVAL_RUNS = 1
print(os.environ['PATH'])
# Setup the environment
PROBLEM_NAME_CAR_POLE = 'CartPole-v0'
PROBLEM_NAME_MOUNTAIN_CAR = 'MountainCar-v0'
#'Acrobot-v1' 'Pendulum-v0'
#'BipedalWalker-v2'
problem_name = 'BipedalWalker-v2'
 
env = gym.make(problem_name)
env = wrappers.Monitor(env, 'E:/tmp/BipedalWalker-v2-06')
observation_size = env.observation_space.shape[0]

action_is_discrete = False
if isinstance(env.action_space, Box):
    action_size = env.action_space.shape[0]
    print(env.action_space.high)
    print(env.action_space.low)
else:
    print("Not Box")
    action_size = env.action_space
    action_is_discrete = True

print("Observation Size:", observation_size)
print("Action size:", action_size)

def computeReward(agent, render=False):
    """ Computes the reward signal for a specified agent. """
    all_run_reward_sum = 0
    all_frame_sum = 0
    # Take the average of multiple runs to smooth out results
    actions = []
    won = False
    for _ in range(EVAL_RUNS):
        observation = env.reset()
        step_count = 0
        reward_sum = 0
        still_frames = 0
        while True:
            if (render):
                env.render()
            action = agent.take_action(observation)
            actions.append(action)
            observation, reward, done, _ = env.step(action)
            #won = won or observation[0] >= 0.5
            reward_sum += reward
            step_count += 1
            
            if False and abs(observation[2]) <= 0.000001:
                still_frames += 1
                if (still_frames > 100):
                    break
            else:
                still_frames = 0
            all_frame_sum += 1
            if done:
                all_run_reward_sum += reward_sum
                break
    avg_reward = all_run_reward_sum / EVAL_RUNS
    avg_frame = all_frame_sum / EVAL_RUNS
    won = False
    return avg_reward, actions, avg_frame, won

def summarize_actions(actions):
    summarized = []
    last_action = -1
    count = 0
    for action in actions:
        if (last_action != action):
            if (last_action != -1):
                summarized.append((last_action, count))
            count = 0
            last_action = action
        count += 1
    summarized.append((last_action, count))
    return summarized
        

""" An agent takes an action based on observations """
class Agent:
    def take_action(self, observation):
        raise NotImplementedError()

""" Chooses between actions based on a set of weights """
class ActionAgent(Agent):
    def __init__(self, weights):
        self.weights = weights

    def take_action(self, observation):
        result = np.zeros(action_size)
        for i in range(action_size):
            start_index = i * observation_size
            end_index = start_index + observation_size
            result[i] = np.matmul(observation, self.weights[start_index: end_index])
            if (not action_is_discrete):
                pass
                # Clamp the result to the bounds of the action for this problem
                #result[i] = result[i] 
        
        if (action_is_discrete):
            selected_action = np.argmax(result)
        else:
            selected_action = result
        return selected_action
    
    
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
        npop = 30  # population size  
        initial_sigma = 0.2
        sigma = initial_sigma  # noise standard deviation  
        alpha = 0.1  # learning rate
        
        w = np.random.randn(observation_size * action_size)  # initial guess  
        i = 0
        last_all_equal = False
        while True:
            if (last_all_equal):
                sigma = sigma + 0.1
                print('New sigma is', sigma)
            elif (sigma > initial_sigma):
                sigma = sigma - 0.1
                print('New sigma is', sigma) 
            N = np.random.randn(npop, observation_size * action_size)
            R = np.zeros(npop)
            F = np.zeros(npop)
            for j in range(npop):
                w_try = w + sigma * N[j]
                agent = ActionAgent(w_try)
                
                R[j], actions, F[j], won = computeReward(agent, False)
                #print(R[j], w_try, j, i, summarize_actions(actions), won)
                #if (won):
                #    _, _, _ = computeReward(agent, render=True)
            
            if i % 25 == 0:
                best_vector_index = np.argmax(R)
                reward, _, frames, _ = computeReward(ActionAgent(w + sigma * N[best_vector_index]), True)
                print(reward, frames, i)
            
            
            mean = np.mean(R)
            #if (mean <= -10000):
                # Use frame count as the target function instead of the reward.
            #    R = F
            #    mean = np.mean(R)
            stdDev = np.std(R)
            
            if (stdDev == 0):
                print("All equal")
                if (won):
                    break;
                # Set a new random starting position
                if (i < 1000):
                    print("Resetting weights to random point")
                    w = np.random.randn(observation_size * action_size)
                #last_all_equal = True
            else:
                last_all_equal = False
                A = (R - mean) / np.std(R)
                w = w + alpha / (npop * sigma) * np.dot(N.T, A)
                if (i % 10 == 0):
                    print("Mean", mean, won, i, repr(w))
                else:
                    print("Mean", mean, won, i)
            i += 1
        print(w)
        return ActionAgent(w)
    
        
    
if __name__ == '__main__':
    best_agent = NaturalEvolutionOptimizer().optimize()

    #w = [0.35431518  ,-7.80998257   ,0.08284797  ,17.61420183   ,3.44219892, 0.96508483] # CartPole-v1
    #w = [-2.18829    ,-22.43855965  ,-1.37641763  ,-5.1617201   ,-2.37288426, 26.50576672] # MountainCar
    #w = [0.3191604 , -2.7798147,-1.64798003,2.70150274,0.41343009, -4.47046953,1.89583941,1.69446406, -3.55064147, -0.41513307, -1.37656644, -0.96194778,1.47853167,0.83565499 ,-1.2619065,-4.63712679, -4.22746252,1.58055732] #Acrobot-v1
    #best_agent = ActionAgent(w)
    final_reward, actions, won = computeReward(best_agent, True)
    print("Final reward = ", final_reward, summarize_actions(actions))
    


