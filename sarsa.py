import numpy as np
from gridworld import GridWorld
import random as r

# SARSA with optional eligibility traces
DISCOUNT = 0.999
EPISODES = 250_000
EPS = 0.1 # eps-greedy policy
LAMBDA = 0 
LR = 1e-3
START_EPS_DECAYING = 1
END_EPS_DECAYING = EPISODES // 2
decaying_eps_value = EPS/(END_EPS_DECAYING - START_EPS_DECAYING)

SHOW_EVERY = 2000

policy_matrix = np.random.randint(0, 4, (3,4))
policy_matrix[0,3] = policy_matrix[1,3] = policy_matrix[1,1] = -1
q_matrix = np.zeros((3,4,4))

env = GridWorld()

def update_q(state,next_state,action,next_action,reward):
	q = q_matrix[state[0],state[1],action]
	next_q = q_matrix[next_state[0],next_state[1],next_action]
	q_matrix[state[0],state[1],action] += LR*(reward + DISCOUNT*next_q - q)

def update_policy(state):
	policy_matrix[state[0],state[1]] = np.argmax(q_matrix[state[0],state[1],:])

for episode in range(EPISODES):
	state = env.reset(exploring_starts=True)
	done = False
	is_starting = True
	while not done:
		u = r.random()
		if is_starting:
			u = 0
			is_starting = False
		action = policy_matrix[state[0],state[1]] if u>EPS else r.randint(0,3)
		next_state,reward,done = env.step(action)
		next_action = policy_matrix[next_state[0],next_state[1]]
		update_q(state,next_state,action,next_action,reward)
		update_policy(state)
		state = next_state

	if END_EPS_DECAYING >= episode >= START_EPS_DECAYING:
		EPS -= decaying_eps_value


	if episode%SHOW_EVERY == 0:
		print('episode: {}'.format(episode))
		print(policy_matrix)
		print('\n')