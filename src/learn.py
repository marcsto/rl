"""
  A simple reinforcement learning implementation to solve the several OpenAi gym problem.
  It's based on Natural Evolution Strategies found at https://blog.openai.com/evolution-strategies/
  It also makes use of some of the code from http://kvfrans.com/simple-algoritms-for-solving-cartpole/
  
  Author: Marc Stogaitis  
"""

import numpy as np
import task

import os
import random
import sys
from natural_evolution_optimizer import NaturalEvolutionOptimizer
from task import computeReward
from model import LinearModel
import weight
from gym.scoreboard import scoring



print(os.environ['PATH'])
# Setup the environment


        

def run_evolvolve(initial_weight = None):
    best_initial_reward = -10000
    if initial_weight == None:
        for initial_attempt in range(30):
            initial_weights = weight.random_weights()  # initial guess
            print("Optimizing new initial weights", initial_attempt)
            optimized_weights, reward = NaturalEvolutionOptimizer().optimize(initial_weights, 1) # 50
            if (reward > best_initial_reward):
                print("New best initial weight", reward, optimized_weights)
                best_initial_reward = reward
                best_initial_weights = optimized_weights
    else:
        best_initial_weights = initial_weight
    
    print("Starting optimization from best initial candidate", best_initial_reward, best_initial_weights)
    NaturalEvolutionOptimizer().optimize(best_initial_weights, sys.maxsize)

def run_manual():
    w = []
    final_reward, actions, _ = computeReward(LinearModel(w), True)
    print("Final reward = ", final_reward, task.summarize_actions(actions))
    
if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    #run_evolvolve(get_initial_weight())    
    run_evolvolve(weight.random_weights())
    #print(scoring.score_from_local(hyper_param.OUTPUT_DIR))
    #run_manual()
    #w = [0.35431518  ,-7.80998257   ,0.08284797  ,17.61420183   ,3.44219892, 0.96508483] # CartPole-v1
    #w = [-2.18829    ,-22.43855965  ,-1.37641763  ,-5.1617201   ,-2.37288426, 26.50576672] # MountainCar
    #w = [0.3191604 , -2.7798147,-1.64798003,2.70150274,0.41343009, -4.47046953,1.89583941,1.69446406, -3.55064147, -0.41513307, -1.37656644, -0.96194778,1.47853167,0.83565499 ,-1.2619065,-4.63712679, -4.22746252,1.58055732] #Acrobot-v1
    #action = [1, 0.2, 0, 0]
    #best_agent = StaticAgent(action)
    #final_reward, actions, won = computeReward(best_agent, True)
    #print("Final reward = ", final_reward, summarize_actions(actions))
    
    


