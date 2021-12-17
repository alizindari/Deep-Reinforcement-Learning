import torch
from torch._C import device, wait
from torch.functional import Tensor
import torch.nn as nn        
import torch.optim as optim  
import matplotlib.pyplot as plt
import numpy as np
import IPython
import collections
import pdb
# from gymWrappers import *

IMAGESIZE = [3, 30, 30]
MEAN_REWARD_BOUND = 19.0           

gamma = 0.99                   
BATCH_SIZE = 32 
REPLAY_SIZE = 10000            
LEARING_RATE = 1e-4 *1    
SYNC_TARGET_FRAMES = 1000      
REPLAY_START_SIZE = 10000      

EPS_INITIAL=1.0
EPS_DECAY=0.99
EPS_MIN=0.02


class ExperienceReplay:
	def __init__(self, capacity):
		self.buffer = collections.deque(maxlen=capacity)
		self.len = 0
		self.cap = capacity
		
		self.index = 0
		self.states = torch.empty([capacity, *IMAGESIZE])
		self.actions = torch.empty([capacity,], dtype= torch.int64)
		self.rewards = torch.empty([capacity,])
		self.dones = torch.empty([capacity,], dtype=torch.bool)
		self.statePs = torch.empty([capacity, *IMAGESIZE])
		
	def __len__(self):
		return self.len

	def append(self, experience):
		state, action, reward, done, stateP = experience
		self.states[self.index] = torch.tensor(state)
		self.actions[self.index] = action
		self.rewards[self.index] = reward
		self.dones[self.index] = done
		self.statePs[self.index] = torch.tensor(stateP)

		self.index = (self.index + 1) % self.cap
		self.len = min(self.len + 1,self.cap)
		
	def sample(self, batch_size):
		indices = torch.randint(0,self.len,[batch_size,])
		return [
			self.states[indices],
			self.actions[indices],
			self.rewards[indices],
			self.dones[indices],
			self.statePs[indices]
		]


class DQN(nn.Module):
	def __init__(self, input_shape, n_actions):
		super(DQN, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1),
		    nn.ReLU(),
		    nn.Conv2d(16, 32, kernel_size=3, stride=1),
		    nn.ReLU(),
		    nn.Conv2d(32, 16, kernel_size=3, stride=1),
		    nn.ReLU()
        )

		conv_out_size = self._get_conv_out(input_shape)
		self.fc = nn.Sequential(
			nn.Linear(conv_out_size, 64),
		    nn.ReLU(),
		    nn.Linear(64, n_actions)
		)

	def _get_conv_out(self, shape):
		o = self.conv(torch.zeros(1, *shape))
		return int(np.prod(o.size()))

	def forward(self, x):
		conv_out = self.conv(x).view(x.size()[0], -1)
		return self.fc(conv_out)

class Agent():
	def __init__(self,action_space) -> None:
		self.device = torch.device("cpu")
		self.epsilon = EPS_INITIAL
		self.action_space = action_space
		self.interaction_counter = 0
		self.batchSize = BATCH_SIZE

		self.mainNet = DQN(IMAGESIZE,3).to(self.device)
		self.targetNet = DQN(IMAGESIZE,3).to(self.device)
		self.memory = ExperienceReplay(REPLAY_SIZE)

		self.temp_history = collections.deque(maxlen=300)
		self.training_history = [[],[]]
		self.optimizer = optim.Adam(self.mainNet.parameters(), lr=LEARING_RATE)
		self.totalReward = 0

	def dicide(self,state):
		if np.random.random() < self.epsilon:
			action = np.random.randint(0,3)
			
		else:
			state = torch.tensor(state).unsqueeze(0).to(self.device)
			q_vals = self.mainNet(state)
			_, action = torch.max(q_vals, dim=1)
			action = int(action.item())

		return action

	def giveFeedBack(self,xp):
		self.memory.append(xp)
		self.interaction_counter += 1	
		self.epsilon = max(self.epsilon*EPS_DECAY, EPS_MIN)
		self.totalReward += xp[2]

		if len(self.memory) >= REPLAY_START_SIZE and self.interaction_counter% int(self.batchSize/32) == 0:

			batch = self.memory.sample(self.batchSize)
			s, a, r, done, sP = [tensor.to(self.device) for tensor in batch]
# 			pdb.set_trace()

			vsP = self.targetNet(sP).max(1)[0].detach()
			vsP[done] = 0.0
			update_vs = vsP * gamma + r

			vs = self.mainNet(s).gather(1, a.unsqueeze(-1)).squeeze(-1)
			
			loss = nn.MSELoss()(vs, update_vs)
			self.temp_history.append(float(loss.detach()))
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			
		if self.interaction_counter % SYNC_TARGET_FRAMES*int(self.batchSize/32) == 0 and len(self.memory) >= REPLAY_START_SIZE:
			self.targetNet.load_state_dict(self.mainNet.state_dict())
			self.training_history[0].append(np.mean(self.temp_history))
			self.training_history[1].append(np.var(self.temp_history))

	def reset(self):
		self.totalReward = 0

	def loadKnowlage(self,path):
		self.mainNet.load_state_dict(torch.load(path))
		self.targetNet.load_state_dict(torch.load(path))

	def saveKnowlage(self,path):
		torch.save(self.mainNet.state_dict(),path)

# if __name__ == '__main__':
# 	env = make_env("PongNoFrameskip-v4")
# 	jimi = Agent(env.action_space)
# 	state = env.reset()

# 	while True:
# 		action = jimi.dicide(state)
# 		newState, reward, done, _ = env.step(action)
# 		jimi.giveFeedBack([state,action,reward,done,newState])
# 		state = newState.copy()

# 		if done:

# 			print(jimi.totalReward, jimi.interaction_counter)
# 			state = env.reset()
# 			jimi.reset()

