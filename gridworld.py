import numpy as np
import random as r

# 4x3 grid environment for cleaning robot examples
class GridWorld():
	def __init__(self,default=True):
		self.S = np.zeros((3,4)) # state matrix
		self.R = np.zeros((3,4),dtype=np.float32) # reward matrix
		self.T = np.zeros((3,4,4,3,4),dtype=np.float32) # transition matrix((state),action,(next_state))
		self.actions = {0:'UP',1:'RIGHT',2:'DOWN',3:'LEFT'}
		self.PROB_CORRECT = 0.8
		self.PROB_LEFT = 0.1
		self.PROB_RIGHT = 0.1	
		if default:
			self.set_states()
			self.set_rewards()
			self.set_transitions()

		self.done=False
		self.action_dict = {0:'UP',1:'RIGHT',2:'DOWN',3:'LEFT'}
		self.current_state = [2,0]
		self.render_dict = {0:'-',-1:'#',1:'*'}

	def set_states(self,state_matrix=None):
		if state_matrix is None:
			self.S[1,1] = -1
			self.S[0,3] = 1
			self.S[1,3] = 1

		else:	
			self.S = state_matrix


	def set_rewards(self,reward_matrix=None):
		if reward_matrix is None:
			reward_matrix = np.full((3,4),-0.04,dtype=np.float32)
			reward_matrix[0,3] = 1
			reward_matrix[1,3] = -1
			self.R = reward_matrix
		else:
			self.R = reward_matrix


	# this function return the state that the agent moves to if action succesfully executed
	def correct_state(self,current_state,action):
		next_state = None
		if action=='UP':
			next_state = [current_state[0]-1,current_state[1]]
		if action=='RIGHT':
			next_state = [current_state[0],current_state[1]+1]
		if action=='DOWN':
			next_state = [current_state[0]+1,current_state[1]]
		if action=='LEFT':
			next_state = [current_state[0],current_state[1]-1]
		
		# checking for wall bounces
		if next_state[0]<0:
			next_state[0]=0
		if next_state[0]>2:
			next_state[0]=2
		if next_state[1]<0:
			next_state[1]=0
		if next_state[1]>3:
			next_state[1]=3

		# checking for obstacle bounces
		if next_state==[1,1]:
			if action=='UP':
				next_state[0] += 1
			if action=='RIGHT':
				next_state[1] -= 1
			if action=='DOWN':
				next_state[0] -= 1 
			if action=='LEFT':
				next_state[1] += 1
		return next_state


	# this function returns the state the agent moves to after a perturbation towards its right 
	def right_state(self,current_state,action):
		next_state = None
		if action=='UP':
			next_state = self.correct_state(current_state,'RIGHT')
		if action=='RIGHT':
			next_state = self.correct_state(current_state,'DOWN')
		if action=='DOWN':
			next_state = self.correct_state(current_state,'LEFT')
		if action=='LEFT':
			next_state = self.correct_state(current_state,'UP')

		return next_state

	# this function returns the state the agent moves to after a perturbation towards its left
	def left_state(self,current_state,action):
		next_state = None
		if action=='UP':
			next_state = self.correct_state(current_state,'LEFT')
		if action=='RIGHT':
			next_state = self.correct_state(current_state,'UP')
		if action=='DOWN':
			next_state = self.correct_state(current_state,'RIGHT')
		if action=='LEFT':
			next_state = self.correct_state(current_state,'DOWN')

		return next_state


	def set_transitions(self,transition_matrix=None):
		if transition_matrix is None:
			transition_matrix = np.zeros((3,4,4,3,4),dtype=np.float32)
			# filling up transition matrix
			for row in range(3):
				for column in range(4):
					for a in range(4):
						action = self.actions[a]
						correct_s = self.correct_state([row,column],action)
						right_s = self.right_state([row,column],action)
						left_s = self.left_state([row,column],action)
						transition_matrix[row,column,a,correct_s[0],correct_s[1]] += self.PROB_CORRECT
						transition_matrix[row,column,a,right_s[0],right_s[1]] += self.PROB_RIGHT
						transition_matrix[row,column,a,left_s[0],left_s[1]] += self.PROB_LEFT


			# zeroing probabilities of reaching obstacle and those of moving from terminal states and obstacle
			# zeroing probs of moving to obstacle
			transition_matrix[:,:,:,1,1] = 0

			# correcting for bouncing against obstacle
			transition_matrix[0,1,1,0,1] += self.PROB_LEFT
			transition_matrix[0,1,2,0,1] += self.PROB_CORRECT
			transition_matrix[0,1,3,0,1] += self.PROB_RIGHT

			transition_matrix[1,2,0,1,2] += self.PROB_LEFT
			transition_matrix[1,2,2,1,2] += self.PROB_RIGHT
			transition_matrix[1,2,3,1,2] += self.PROB_CORRECT

			transition_matrix[2,1,0,2,1] += self.PROB_CORRECT
			transition_matrix[2,1,1,2,1] += self.PROB_LEFT
			transition_matrix[2,1,3,2,1] += self.PROB_RIGHT

			transition_matrix[1,0,0,1,0] += self.PROB_RIGHT
			transition_matrix[1,0,1,1,0] += self.PROB_CORRECT
			transition_matrix[1,0,2,1,0] += self.PROB_LEFT

			# zeroing probs of moving from obstacle
			transition_matrix[1,1,:,:,:] = 0

			# zeroing probs of moving from charge station
			transition_matrix[0,3,:,:,:] = 0

			# zeroing probs of moving from stairs
			transition_matrix[1,3,:,:,:] = 0
			self.T = transition_matrix
		else:
			self.T = transition_matrix


	def reset(self,exploring_starts=False):
		self.done=False
		if exploring_starts:
			self.current_state = [1,1]
			while self.current_state==[1,1] or self.current_state==[0,3] or self.current_state==[1,3]: # cause we don't want to start here!
				row = r.randint(0,self.S.shape[0]-1)
				col = r.randint(0,self.S.shape[1]-1)
				self.current_state = [row,col]
		else:
			self.current_state = [2,0]

		return self.current_state


	def render(self):
		for row in range(self.S.shape[0]):
			render_string = ''
			for col in range(self.S.shape[1]):
				if self.current_state == [row,col]:
					render_string += 'o '
				else:
					render_string += self.render_dict[self.S[row,col]]+' '

			print(render_string)

		print('\n')


	def step(self,action):
		action = self.action_dict[action]
		u = r.random()
		if u<0.8:
			next_state = self.correct_state(self.current_state,action)
		else:
			u = r.randint(0,1)
			if u==0:
				next_state = self.right_state(self.current_state,action)
			else:
				next_state = self.left_state(self.current_state,action)

		reward = self.R[next_state[0],next_state[1]]
		if next_state==[0,3] or next_state==[1,3]:
			self.done=True
		self.current_state=next_state

		return next_state,reward,self.done


	def get_T(self):
		return self.T


	def get_R(self):
		return self.R