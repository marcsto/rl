import task
import numpy as np

""" An agent takes an action based on observations """
class Model:
    def take_action(self, observation):
        raise NotImplementedError()

""" Chooses between actions based on a set of weights
    Uses a simple linear model, y = wx + b
    There's 1 equation per action.
    If it's a discrete problem, the action with the largest value is returned.
    If it's a continuous problem, the result of each equation is used as the action vector.
"""
class LinearModel(Model):
    weight_count = (task.observation_size + 1) * task.action_size
     
    def __init__(self, weights):
        self.weights = weights
    
    def take_action(self, observation):
        result = np.zeros(task.action_size)
        for i in range(task.action_size):
            start_index = i * (task.observation_size + 1)
            end_index = start_index + task.observation_size
            result[i] = np.matmul(observation, self.weights[start_index: end_index]) + self.weights[end_index]
            if (not task.action_is_discrete):
                pass
                # Clamp the result to the bounds of the action for this problem
                #result[i] = result[i] 
        
        if (task.action_is_discrete):
            selected_action = np.argmax(result)
        else:
            selected_action = result
        return selected_action

""" Simple model that returns a hard-coded action. Used for testing """
class StaticModel(Model):
    def __init__(self, action):
        self.action = action
    
    def take_action(self, observation):
        return self.action