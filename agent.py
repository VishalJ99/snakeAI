"""
implement Dq learning training loop
figure out at what stage in the training loop is exp replay populated
"""

from collections import deque 
from game import SnakeGame
import numpy as np
import random
from model import Linear_QNet, QTrainer
import torch
from utils import plot
# hyper params taken from ...
MAXMEM = 10000
BATCHSIZE = 1000
LR = 0.0001
EPISODES = 10000
import matplotlib.pyplot as plt
from IPython import display
model = 0
class Agent():
	def __init__(self):
		self.replay_stack = deque(maxlen = MAXMEM)
		self.model = Linear_QNet(11,256,3)
		self.game = SnakeGame()
		self.eps = 0.001 # learning rate
		self.gamma = 0.9 # discount rate
		self.trainer = QTrainer(self.model, self.eps, self.gamma)

		self.exploration_rate = 1
		self.min_exploration_rate = 0.02
		self.max_exploration_rate = 1
		self.exploration_decay_rate = 0.03
	
	def train(self):
		total_frames = 0
		plot_scores = []
		plot_mean_scores = []
		total_score = 0
		record = 0



		for episode in range(EPISODES):
			# reset game state 
			done = False
			self.game.reset()
			
   
			if episode>0: 
				total_frames += frame
				print('current episode:',episode,' total_frames:',total_frames, ' exploration rate:', self.exploration_rate, ' best score:',record,' avg score:',total_score/episode)
    
   			# every 1000 frames copy weights from policy to target models
			if (total_frames % 1000)<200:
				print('updating target network') 
				self.trainer.target.load_state_dict(self.trainer.policy.state_dict())
				
			frame = 0
			while done != True:
    			# get current state
				curr_state = self.game.get_state() 
				
				# choose action
				action = [0,0,0]
				exploration_threshold = random.uniform(0,1)
				if exploration_threshold > self.exploration_rate:
					# using model
					curr_state_tensor = torch.tensor(curr_state, dtype=torch.float)
					action_idx = torch.argmax(self.trainer.policy(torch.unsqueeze(curr_state_tensor, 0))).item()
				else:
					# randomly 
					action_idx = random.randint(0,2)
				
				action[action_idx] = 1 
    
				# populate replay stack with experience
				done, score, reward = self.game.play_step(action)
				new_state = self.game.get_state()
				experience_tuple = (curr_state, action, reward, new_state, done)
				self.replay_stack.appendleft(experience_tuple)
				self.trainer.train_step(curr_state, action, reward, new_state, done)
				frame+=1
    
			if len(self.replay_stack) > BATCHSIZE:
				# randomly sample from replay stack
				batch = random.sample(self.replay_stack,BATCHSIZE)
				curr_states, actions, rewards, new_states, dones = zip(*batch)
				self.trainer.train_step(curr_states, actions, rewards, new_states, dones)

			# plot data
			plot_scores.append(score)
			total_score += score
			mean_score = total_score / (episode + 1)
			plot_mean_scores.append(mean_score)
			plot(plot_scores, plot_mean_scores)
				
			# decay exploration rate
			self.exploration_rate = self.min_exploration_rate + (self.max_exploration_rate - self.min_exploration_rate)*np.exp(-(self.exploration_decay_rate*episode))

			if score>record:
				record = score 
				self.trainer.policy.save()





plt.ion()
agent = Agent()
agent.train()

# game = SnakeGame()
# results_tuple = game.play_step([0,0,1])
# print(results_tuple)


