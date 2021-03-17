import numpy as np
from gridworld import GridWorld
import random as r
 
env = GridWorld()

DISCOUNT = 0.995
# optimal policy, obtained for example from value iteration...
policy_matrix = np.array([[1, 1, 1, -1],
						  [0, np.nan, 0, -1],
						  [0, 3, 3, 3]])	

utility_matrix = np.zeros((3,4),dtype=np.float32)
running_mean_matrix = np.full((3,4),1e-10)

def get_return(steps_list):
	utility = 0
	for t,step in enumerate(steps_list):
		reward = step[1]
		utility += reward*np.power(DISCOUNT,t)

	return utility

# MC prediction
# 1.In every episode, run the agent and collect states and rewards
# 2.At the end of an episode, update the utility matrix
NUM_EPISODES = 20_000
for _ in range(NUM_EPISODES):
	episode_list = list()
	state = env.reset() #env.reset(exploring_starts=True)
	done = False
	while not done:
		action = policy_matrix[state[0],state[1]]
		state,reward,done = env.step(action)
		episode_list.append([state,reward])

	checkup_matrix = np.zeros((3,4))

	for i,step in enumerate(episode_list):
		state = step[0]
		if checkup_matrix[state[0],state[1]] == 0:
			utility = get_return(episode_list[i:])
			utility_matrix[state[0],state[1]] += utility
			running_mean_matrix[state[0],state[1]] += 1

print('Utility matrix:')
print(utility_matrix/running_mean_matrix)

