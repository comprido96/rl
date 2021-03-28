import gym
import numpy as np
from matplotlib import pyplot as plt

env = gym.make('MountainCar-v0')
# For reproducibility:
RANDOM_SEED = 23

env.seed(RANDOM_SEED)
env.action_space.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

#print(env.observation_space.low) 
#print(env.observation_space.high) 

def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / window_size
	return tuple(discrete_state.astype(np.int))

def update(algo,action):
	if algo=='sarsa':
		return sarsa_update(action)
	elif algo=='qlearning':
		return qlearning_update(action)

	else:
		print('Wrong algorithm name!')
		env.close()
		sys.exit()

def sarsa_update(action):
	u = np.random.random()
	next_action = np.argmax(q_values[new_discrete_state]) if u > EPSILON else env.action_space.sample()
	current_q = q_values[discrete_state+(action, )] 
	future_q = q_values[new_discrete_state+(next_action, )]
	new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT*future_q)
	return new_q


def qlearning_update(action):
	current_q = q_values[discrete_state+(action, )] 
	max_future_q = np.max(q_values[new_discrete_state])
	new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT*max_future_q)
	return new_q


NUM_WINDOWS = [30,30]
DISCRETE_SPACE_SIZE = NUM_WINDOWS#[NUM_WINDOWS]*len(env.observation_space.low)
window_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_SPACE_SIZE 

q_values = np.random.uniform(low=-2.0, high=0.0, size=(DISCRETE_SPACE_SIZE + [env.action_space.n]))

LEARNING_RATE = 7e-2
DISCOUNT = 0.995
EPISODES = 30_000
EPSILON = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
decaying_eps_value = EPSILON/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
SHOW_EVERY = 3000
STATS_EVERY = 100

ALGO='sarsa'

# training

# For stats
ep_returns = []
aggr_ep_returns = {'ep': [], 'avg': [], 'max': [], 'min': []}
successes = np.zeros(EPISODES)
max_return = -110
for episode in range(EPISODES):
	ep_return = 0
	if episode % SHOW_EVERY == 0:
		render = True
		print(episode)
	else:
		render = False

	discrete_state = tuple(get_discrete_state(env.reset()))
	done = False
	while not done:
		# eps-greedy policy:
		u = np.random.random()
		action = np.argmax(q_values[discrete_state]) if u > EPSILON else env.action_space.sample() 
		
		new_state,reward,done,_ = env.step(action)
		new_discrete_state = get_discrete_state(new_state)
		if render:
			env.render()
		if not done:
			new_q = update(ALGO,action)
			q_values[discrete_state+(action, )] = new_q
		# check for episode succesful termination:
		elif new_state[0] >= env.goal_position:
			successes[episode] = 1
			print('We made it on episode {}'.format(episode))

		ep_return += reward
		discrete_state = new_discrete_state

	# decay epsilon
	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		EPSILON -= decaying_eps_value

	ep_returns.append(ep_return)		
	if (episode+1)%STATS_EVERY==0:
		average_return = sum(ep_returns[-STATS_EVERY:])/STATS_EVERY
		aggr_ep_returns['ep'].append(episode)
		aggr_ep_returns['avg'].append(average_return)
		aggr_ep_returns['max'].append(max(ep_returns[-STATS_EVERY:]))
		aggr_ep_returns['min'].append(min(ep_returns[-STATS_EVERY:]))
		print(f'Episode: {episode:>5d}, average return {average_return:>4.1f}, current epsilon: {EPSILON:>1.2f}')

	if ep_return > max_return:
		max_return = ep_return
		np.save('qmatrix',q_values)

# Final statistic:
START_FROM = (EPISODES//3)*2
print('Percentage of succesful episodes is {}%'.format(100*np.sum(successes[START_FROM:])/len(successes[START_FROM:])))


q = np.load('qmatrix.npy')
# test solver:
TRIALS = 100 
tot_return = 0
for trial in range(TRIALS):
	done = False
	discrete_state = tuple(get_discrete_state(env.reset()))
	while not done:
		action = np.argmax(q[discrete_state])
		new_state,reward,done,_ = env.step(action)
		new_discrete_state = get_discrete_state(new_state)
		tot_return += reward
		discrete_state = new_discrete_state

avg_return = tot_return/TRIALS

print('Env solved!') if avg_return >= -110 else print('Env not solved...')
print('avg return is {}'.format(avg_return))
env.close()

plt.plot(aggr_ep_returns['ep'], aggr_ep_returns['avg'], label="average returns")
plt.plot(aggr_ep_returns['ep'], aggr_ep_returns['max'], label="max returns")
plt.plot(aggr_ep_returns['ep'], aggr_ep_returns['min'], label="min returns")
plt.legend(loc=4)
plt.show()