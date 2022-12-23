from collections import deque 
from game import SnakeGame
import numpy as np
import random
from model import Linear_QNet, QTrainer
import torch
from utils import plot
import matplotlib.pyplot as plt

random.seed(42)
MAXMEM = 10000
BATCHSIZE = 1000
EPISODES = 10000

class Agent():
	def __init__(self):
		self.replay_stack = deque(maxlen = MAXMEM)
		self.model = Linear_QNet(11,256,3)
		self.game = SnakeGame()
		self.eps = 0.001 # learning rate
		self.gamma = 0.9 # discount rate
		self.trainer = QTrainer(self.model, self.eps, self.gamma)

		self.exploration_rate = 1
		self.exploration_decay_rate = 0.03
		self.min_exploration_rate = 0.02
		self.max_exploration_rate = 1
	
	def train(self):
		total_frames = 0
		total_score = 0
		record = 0
		plot_scores = []
		plot_mean_scores = []
  
		for episode in range(EPISODES):
			# reset game state 
			done = False
			self.game.reset()

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
					# exploiting
					curr_state_tensor = torch.tensor(curr_state, dtype=torch.float)
					action_idx = torch.argmax(self.trainer.policy(torch.unsqueeze(curr_state_tensor, 0))).item()
				else:
					# exploring
					action_idx = random.randint(0,2)
				
				action[action_idx] = 1 
    
				# populate replay stack with experience
				done, score, reward = self.game.play_step(action)
				new_state = self.game.get_state()
				experience_tuple = (curr_state, action, reward, new_state, done)
				self.replay_stack.appendleft(experience_tuple)
				
				# train policy model with current step
				self.trainer.train_step(curr_state, action, reward, new_state, done)
				
				frame+=1
    
			if len(self.replay_stack) > BATCHSIZE:
				# randomly sample from replay stack and train
				batch = random.sample(self.replay_stack,BATCHSIZE)
				curr_states, actions, rewards, new_states, dones = zip(*batch)
				self.trainer.train_step(curr_states, actions, rewards, new_states, dones)
			
   			# print training statistics 
			total_frames += frame
			print('current episode:',episode,' total_frames:',total_frames, ' exploration rate:', self.exploration_rate, ' best score:',record,' avg score:',total_score/(episode+1))
    
			# plot data
			plot_scores.append(score)
			total_score += score
			mean_score = total_score / (episode + 1)
			plot_mean_scores.append(mean_score)
			plot(plot_scores, plot_mean_scores)
				
			# decay exploration rate
			self.exploration_rate = self.min_exploration_rate + (self.max_exploration_rate - self.min_exploration_rate)*np.exp(-(self.exploration_decay_rate*episode))
			
			# save policy model
			if score>record:
				record = score 
				self.trainer.policy.save()

if __name__ == '__main__':
	plt.ion()
	agent = Agent()
	agent.train()



