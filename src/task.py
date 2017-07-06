import gym
from gym import wrappers
from gym.spaces.box import Box
import hyper_param

PROBLEM_NAME_CAR_POLE = 'CartPole-v0'

PROBLEM_NAME_MOUNTAIN_CAR = 'MountainCar-v0'
#'Acrobot-v1' 'Pendulum-v0'
#'BipedalWalker-v2'


problem_name = PROBLEM_NAME_CAR_POLE #PROBLEM_NAME_MOUNTAIN_CAR #'BipedalWalker-v2'
WINNING_REWARD = 195.0
 
env = gym.make(problem_name)
env = wrappers.Monitor(env, hyper_param.OUTPUT_DIR, video_callable=False, force=True)
observation_size = env.observation_space.shape[0]

action_is_discrete = False
if isinstance(env.action_space, Box):
    action_size = env.action_space.shape[0]
    print(env.action_space.high)
    print(env.action_space.low)
else:
    print("Not Box")
    action_size = env.action_space.n
    action_is_discrete = True
    
print("Observation Size:", observation_size)
print("Action size:", action_size)

EVAL_RUNS = 1
def computeReward(agent, render=False, debug=False):
    """ Computes the reward signal for a specified agent. """
    all_run_reward_sum = 0
    all_frame_sum = 0
    # Take the average of multiple runs to smooth out results
    actions = []
    won = False
    for i in range(EVAL_RUNS):
        observation = env.reset()
        reward_sum = 0
        still_frames = 0
        while True:
            if (render and i == 0):
                env.render()
            action = agent.take_action(observation)
            if (debug): print(action)
            actions.append(action)
            observation, reward, done, _ = env.step(action)
            #won = won or observation[0] >= 0.5
            reward_sum += reward
            
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
    
    won = avg_reward >= WINNING_REWARD
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

def close():
    env.close()