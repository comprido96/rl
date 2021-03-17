import gym
import numpy as np

env = gym.make('MountainCar-v0')
print(env.observation_space.low) # [-1.2 -0.07]
print(env.observation_space.high) # [0.6 0.07]

NUM_WINDOWS = (20,20)
WINDOW_SIZE = (env.observation_space.high - env.observation_space.low) / NUM_WINDOWS 

print(WINDOW_SIZE)

DISCOUNT = 0.95
NUM_EPISODES = 15_000
EPS = 0.3
LR = 1e-1
SHOW_EVERY = 2_000
START_EPS_DECAYING = 1
END_EPS_DECAYING = NUM_EPISODES // 2
decaying_eps_value = EPS/(END_EPS_DECAYING - START_EPS_DECAYING)

def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) // WINDOW_SIZE
	return tuple(discrete_state.astype(np.int))

def update_q(state,action,next_state,reward):
	current_q = q_values[state[0],state[1],action]
	next_q = np.max(q_values[next_state[0],next_state[1],:])
	q_values[state[0],state[1],action] = current_q + LR*(reward + DISCOUNT*next_q - current_q)

def decay_eps():
	EPS -= decaying_eps_value

'''
state = env.reset()
print('state: ')
print(state)
discrete_state = get_discrete_state(state)
print('discrete_state: ')
print(discrete_state)


rewards = []
for _ in range(500):
	done = False
	state = env.reset()
	while not done:
		state,reward,done,_ = env.step(env.action_space.sample())
		rewards.append(reward)

np_rewards = np.array(rewards)
print('Min reward: ')
print(np.min(np_rewards))
print('Max reward: ')
print(np.max(np_rewards))
print('Mean reward: ')
print(np.mean(np_rewards))
'''



# since reward is always -1 except for terminal states, we can initialize our q_values this way
q_values = np.random.uniform(low = -2.0, high = 0.0, size=(NUM_WINDOWS + (env.action_space.n,) )) # shape: 20x20x3

for episode in range(NUM_EPISODES):
	discrete_state = get_discrete_state(env.reset())
	done = False
	render = False
	if episode%SHOW_EVERY==0:
		print(episode)
		#render = True
	while not done:
		u = np.random.uniform()	
		action = np.argmax(q_values[discrete_state]) if u > EPS else np.random.randint(0,env.action_space.n) 
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

policy_matrix = np.argmax(q_values,axis=2)
print(policy_matrix.shape)
np.save('policy',policy_matrix)