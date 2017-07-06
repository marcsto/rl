import task
import numpy as np
from model import LinearModel
from hyper_param import npop, sigma, alpha

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
    def optimize(self, initial_weight, steps):
        
        w = initial_weight  
        last_all_equal = False
        for i in range(steps):
            #if (last_all_equal):
            #    sigma = sigma + 0.1
            #    print('New sigma is', sigma)
            #elif (sigma > initial_sigma):
            #    sigma = sigma - 0.1
            #    print('New sigma is', sigma) 
            N = np.random.randn(npop, LinearModel.weight_count)
            #N[0] = np.zeros((observation_size + 1) * action_size) # Always try the initial vector
            R = np.zeros(npop)
            F = np.zeros(npop)
            for j in range(npop):
                w_try = w + sigma * N[j]
                agent = LinearModel(w_try)
                
                R[j], actions, F[j], won = task.computeReward(agent, False)
                #print(R[j], w_try, j, i, summarize_actions(actions), won)
                #if (won):
                #    _, _, _ = computeReward(agent, render=True)
            
            best_vector_index = np.argmax(R)
            if i % 100 == 0:
                reward, actions, frames, _ = task.computeReward(LinearModel(w + sigma * N[best_vector_index]), True)
                print(reward, frames, i, task.summarize_actions(actions))
                print("break")
            
            
            mean = np.mean(R)
            #if (mean <= -10000):
                # Use frame count as the target function instead of the reward.
            #    R = F
            #    mean = np.mean(R)
            stdDev = np.std(R)
                
            if (stdDev == 0):
                print("All equal")
                if (won):
                    task.close()
                    break;
                # Set a new random starting position
                #w = np.random.randn(LinearModel.weight_count)
                #last_all_equal = True
            else:
                last_all_equal = False
                A = (R - mean) / np.std(R)
                w = w + alpha / (npop * sigma) * np.dot(N.T, A)
                if ((i + 1) % 25 == 0):
                    print("Mean", mean, won, i, repr(w))
                else:
                    dist = np.linalg.norm(w-initial_weight)
                    print("Mean", mean, "Best", R[best_vector_index], won, dist, i)
        print(w)
        return w + sigma * N[best_vector_index], R[best_vector_index]