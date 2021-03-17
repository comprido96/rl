import gym
import numpy as np
from course1_gym import get_discrete_state, update_q, decay_eps
env = gym.make('MountainCar-v0')

NUM_WINDOWS = (20,20)
WINDOW_SIZE = (env.observation_space.high - env.observation_space.low) / NUM_WINDOWS 

DISCOUNT = 0.95
NUM_EPISODES = 25_000
EPS = 0.4
LR = 1e-1
SHOW_EVERY = 2_000
START_EPS_DECAYING = 1
END_EPS_DECAYING = NUM_EPISODES // 2
decaying_eps_value = EPS/(END_EPS_DECAYING - START_EPS_DECAYING)

q_values = np.random.uniform(low = -2.0, high = 0.0, size=(NUM_WINDOWS + (env.action_space.n,) )) # shape: 20x20x3

exploratory_policy = np.load('policy.npy')

for episode in range(NUM_EPISODES):
	discrete_state = get_discrete_state(env.reset())
	done = False
	render = False
	if episode%SHOW_EVERY==0:
		print(episode)
		render = True
	while not done:
		action = exploratory_policy[discrete_state]
		next_state,reward,done,_ = env.step(action)
		if render:
			env.render()

		next_discrete_state = get_discrete_state(next_state)
		update_q(discrete_state,action,next_discrete_state,reward)
		discrete_state = next_discrete_state
	
	if END_EPS_DECAYING >= episode >= START_EPS_DECAYING:
		EPS -= decaying_eps_value

	if next_state[0] >= env.goal_position:
		print('We made it on episode {}'.format(episode))

env.close()
