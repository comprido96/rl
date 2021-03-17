import numpy as np

########
# BUILDING ENVIRONMENT
########
# define reward matrix
reward_matrix = np.full((3,4),-0.04,dtype=np.float32)
reward_matrix[0,3] = 1
reward_matrix[1,3] = -1
reward_matrix[1,1] = 0

transition_matrix = np.zeros((3,4,4,3,4),dtype=np.float32)

# given the tuple (s,a) current_state-action, action is performed correctly with prob. 0.8.
# with prob. 0.1 it goes left, with prob. 0.1 it goes right. If agent bounces on the wall it stays in the current_state
# Actions are 0:UP, 1:RIGHT, 2:DOWN, 3:LEFT
actions = {0:'UP',1:'RIGHT',2:'DOWN',3:'LEFT'}
PROB_CORRECT = 0.8
PROB_LEFT = 0.1
PROB_RIGHT = 0.1	
DISCOUNT = 0.999
EPS = 1e-3

# this function return the state that the agent moves to if action succesfully executed
def correct_state(current_state,action):
	next_state = None
	if action=='UP':
		next_state = [current_state[0]-1,current_state[1]]
	if action=='RIGHT':
		next_state = [current_state[0],current_state[1]+1]
	if action=='DOWN':
		next_state = [current_state[0]+1,current_state[1]]
	if action=='LEFT':
		next_state = [current_state[0],current_state[1]-1]
	
	# checking for wall bounces:
	if next_state[0]<0:
		next_state[0]=0
	if next_state[0]>2:
		next_state[0]=2
	if next_state[1]<0:
		next_state[1]=0
	if next_state[1]>3:
		next_state[1]=3

	return next_state

# this function returns the state the agent moves to after a perturbation towards its right 
def right_state(current_state,action):
	next_state = None
	if action=='UP':
		next_state = correct_state(current_state,'RIGHT')
	if action=='RIGHT':
		next_state = correct_state(current_state,'DOWN')
	if action=='DOWN':
		next_state = correct_state(current_state,'LEFT')
	if action=='LEFT':
		next_state = correct_state(current_state,'UP')

	return next_state

# this function returns the state the agent moves to after a perturbation towards its left
def left_state(current_state,action):
	next_state = None
	if action=='UP':
		next_state = correct_state(current_state,'LEFT')
	if action=='RIGHT':
		next_state = correct_state(current_state,'UP')
	if action=='DOWN':
		next_state = correct_state(current_state,'RIGHT')
	if action=='LEFT':
		next_state = correct_state(current_state,'DOWN')

	return next_state

# filling up transition matrix
for row in range(3):
	for column in range(4):
		for a in range(4):
			action = actions[a]
			correct_s = correct_state([row,column],action)
			right_s = right_state([row,column],action)
			left_s = left_state([row,column],action)
			transition_matrix[row,column,a,correct_s[0],correct_s[1]] += PROB_CORRECT
			transition_matrix[row,column,a,right_s[0],right_s[1]] += PROB_RIGHT
			transition_matrix[row,column,a,left_s[0],left_s[1]] += PROB_LEFT


# zeroing probabilities of reaching obstacle and those of moving from terminal states and obstacle
# zeroing probs of moving to obstacle
transition_matrix[:,:,:,1,1] = 0

# correcting for bouncing against obstacle
transition_matrix[0,1,1,0,1] += PROB_LEFT
transition_matrix[0,1,2,0,1] += PROB_CORRECT
transition_matrix[0,1,3,0,1] += PROB_RIGHT

transition_matrix[1,2,0,1,2] += PROB_LEFT
transition_matrix[1,2,2,1,2] += PROB_RIGHT
transition_matrix[1,2,3,1,2] += PROB_CORRECT

transition_matrix[2,1,0,2,1] += PROB_CORRECT
transition_matrix[2,1,1,2,1] += PROB_LEFT
transition_matrix[2,1,3,2,1] += PROB_RIGHT

transition_matrix[1,0,0,1,0] += PROB_RIGHT
transition_matrix[1,0,1,1,0] += PROB_CORRECT
transition_matrix[1,0,2,1,0] += PROB_LEFT

# zeroing probs of moving from obstacle
transition_matrix[1,1,:,:,:] = 0

# zeroing probs of moving from charge station
transition_matrix[0,3,:,:,:] = 0

# zeroing probs of moving from stairs
transition_matrix[1,3,:,:,:] = 0


#define policy matrix
policy_matrix = np.full((3,4), np.nan)


def return_state_utility(state):
	utilities = np.zeros(4)
	for action in range(4):
		u = 0
		for row in range(3):
			for column in range(4):
				u += transition_matrix[state[0],state[1],action,row,column]*value_matrix2[row,column]
		utilities[action] = u

	return np.max(utilities)



########
# VALUE ITERATION
########
value_matrix = np.zeros((3,4))
value_matrix2 = np.zeros((3,4))
#for _ in range(30):
while True:
	value_matrix = value_matrix2.copy()
	for row in range(3):
		for column in range(4):
			reward = reward_matrix[row,column]
			utility = return_state_utility([row,column])
				

			value_matrix2[row,column] = reward + DISCOUNT*utility

	if np.max(np.abs(value_matrix2[row,column]-value_matrix[row,column])) < EPS/DISCOUNT*(1-DISCOUNT):	
		break	
			
# Now we can retrieve the optimal policy
for row in range(3):
	for column in range(4):
		utilities = np.zeros(4)
		for action in range(4):
			u = np.sum(np.multiply(transition_matrix[row,column,action,:,:],value_matrix2))
			utilities[action] = u

		policy_matrix[row,column] = np.argmax(utilities)

# adjusting for terminal states and obstacle:
policy_matrix[1,1] = np.nan
policy_matrix[0,3] = -1
policy_matrix[1,3] = -1


print('\nValue Matrix:')
print(value_matrix2)
print('\nPolicy Matrix:')
print(policy_matrix)

