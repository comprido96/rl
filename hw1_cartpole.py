import gym
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def get_discrete_state(state):
	discrete_state = (state - OBS_LOW) / window_size
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

# For reproducibility:
RANDOM_SEED = 23

env = gym.make('CartPole-v1')
env.seed(RANDOM_SEED)
env.action_space.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

'''
print(env.observation_space.low) # [-4.8, -Inf, -0.419, -Inf]
print(env.observation_space.high) # [4.8, Inf, 0.419, Inf]

cart_velocity = []
pole_angular_velocity = []

for i in range(15_000):
	state = env.reset()
	done = False
	while not done:
		action = env.action_space.sample()
		next_state,reward,done,_ = env.step(action)
		#print(next_state)
		cart_velocity.append(next_state[1])
		pole_angular_velocity.append(next_state[3])

np_cart_v = np.array(cart_velocity)
np_pole_ang_v = np.array(pole_angular_velocity)

print('min cart vel: {}'.format(np.min(np_cart_v)))
print('mean cart vel: {}'.format(np.mean(np_cart_v)))
print('max cart vel: {}'.format(np.max(np_cart_v)))

print('min pole vel: {}'.format(np.min(np_pole_ang_v)))
print('mean pole vel: {}'.format(np.mean(np_pole_ang_v)))
print('max pole vel: {}'.format(np.max(np_pole_ang_v)))

sns.displot(np_cart_v)
sns.displot(np_pole_ang_v)

plt.show()
'''


MIN_CART_POSITION = -4.9
MAX_CART_POSITION = 4.9

MIN_CART_VELOCITY = -4.0
MAX_CART_VELOCITY = 4.0

MIN_POLE_ANGLE = -0.419
MAX_POLE_ANGLE = 0.419

MIN_POLE_ANG_VELOCITY = -4.0
MAX_POLE_ANG_VELOCITY = 4.0

OBS_LOW = np.array((MIN_CART_POSITION,MIN_CART_VELOCITY,MIN_POLE_ANGLE,MIN_POLE_ANG_VELOCITY))
OBS_HIGH = np.array((MAX_CART_POSITION, MAX_CART_VELOCITY, MAX_POLE_ANGLE, MAX_POLE_ANG_VELOCITY))

NUM_WINDOWS = 20
DISCRETE_SPACE_SIZE = [NUM_WINDOWS]*len(env.observation_space.low)
window_size = (OBS_HIGH - OBS_LOW) / DISCRETE_SPACE_SIZE 


ALGO='qlearning'

LEARNING_RATE = 1e-1
DISCOUNT = 0.995
EPISODES = 20_000
EPSILON = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
decaying_eps_value = EPSILON/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
SHOW_EVERY = 2000
STATS_EVERY = 100

q_values = np.zeros(shape=(DISCRETE_SPACE_SIZE + [env.action_space.n]))

# training

# For stats
ep_returns = []
aggr_ep_returns = {'ep': [], 'avg': [], 'max': [], 'min': []}
successes = np.zeros(EPISODES)

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
			pass#env.render()
		if not done:
			new_q = update(ALGO,action)
			q_values[discrete_state+(action, )] = new_q

		ep_return += reward
		discrete_state = new_discrete_state

	# decay epsilon
	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		EPSILON -= decaying_eps_value

	# check for episode succesful termination:
	'''
	if ep_return > 199:
		successes[episode] = 1
		print('We made it on episode {}'.format(episode))
	'''

	ep_returns.append(ep_return)		
	if (episode+1)%STATS_EVERY==0:
		average_return = np.sum(ep_returns[-STATS_EVERY:])/STATS_EVERY
		aggr_ep_returns['ep'].append(episode)
		aggr_ep_returns['avg'].append(average_return)
		aggr_ep_returns['max'].append(np.max(ep_returns[-STATS_EVERY:]))
		aggr_ep_returns['min'].append(np.min(ep_returns[-STATS_EVERY:]))
		print(f'Episode: {episode:>5d}, average return {average_return:>4.1f}, current epsilon: {EPSILON:>1.2f}')



# Final statistic:
START_FROM = (EPISODES//3)*2
print('Percentage of succesful episodes is {}%'.format(100*np.sum(successes[START_FROM:])/len(successes[START_FROM:])))


# test solver:
TRIALS = 100 
tot_return = 0
for trial in range(TRIALS):
	done = False
	discrete_state = tuple(get_discrete_state(env.reset()))
	while not done:
		action = np.argmax(q_values[discrete_state])
		new_state,_,done,_ = env.step(action)
		new_discrete_state = get_discrete_state(new_state)
		tot_return += 1
		discrete_state = new_discrete_state

avg_return = tot_return/TRIALS
print('Env solved!') if avg_return >= 490 else print('Env not solved...')
print('avg return is {}'.format(avg_return))
env.close()

plt.plot(aggr_ep_returns['ep'], aggr_ep_returns['avg'], label="average returns")
plt.plot(aggr_ep_returns['ep'], aggr_ep_returns['max'], label="max returns")
plt.plot(aggr_ep_returns['ep'], aggr_ep_returns['min'], label="min returns")
plt.legend(loc=4)
plt.show()
