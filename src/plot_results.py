from gym.wrappers import monitoring
import hyper_param
import matplotlib.pyplot as plt

results = monitoring.load_results(hyper_param.OUTPUT_DIR)
print(results)


plt.plot(results['episode_rewards'])
plt.ylabel('some numbers')
plt.show()