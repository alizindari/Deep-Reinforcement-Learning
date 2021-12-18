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
from Environment import *
from experience_replay import *
from brain import *
from Hyperparameters import *

class Agent():
    def __init__(self):
        self.param = Hyperparameters()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epsilon = self.param.EPS_INITIAL
        self.action_space = self.param.ACTION_SPACE
        self.interaction_counter = 0
        self.batchSize = self.param.BATCH_SIZE

        self.mainNet = DQN(self.param.IMAGESIZE,self.param.CHANNEL_NUM).to(self.device)
        self.targetNet = DQN(self.param.IMAGESIZE,self.param.CHANNEL_NUM).to(self.device)
        self.memory = ExperienceReplay(REPLAY_SIZE)

        self.temp_history = collections.deque(maxlen=300)
        self.training_history = [[],[]]
        self.optimizer = optim.Adam(self.mainNet.parameters(), lr=self.param.LEARING_RATE)
        self.totalReward = 0

    def find_action(self,state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0,self.param.ACTION_SPACE)
            
        else:
            state = torch.tensor(state).unsqueeze(0).to(self.device)
            q_vals = self.mainNet(state)
            _, action = torch.max(q_vals, dim=1)
            action = int(action.item())

        return action

    def giveFeedBack(self,experience):
        self.memory.add_to_memory(experience)
        self.interaction_counter += 1
        self.epsilon = max(self.epsilon*self.param.EPS_DECAY, self.param.EPS_MIN)
        self.totalReward += experience[2]

        if self.memory.current_len >= self.param.REPLAY_START_SIZE and self.interaction_counter% int(self.batchSize/32) == 0:

            random_samples = self.memory.random_sample()
            current_states,next_states,rewards,actions,dones = [torch.tensor(element,dtype=torch.float32).to(self.device) for element in random_samples]
            ##################################
            max_next_q_values = self.targetNet(next_states).max(1)[0].detach()
            dones = torch.tensor(dones,dtype=torch.bool)
            actions = torch.tensor(actions,dtype=torch.int64)
#             print(actions.dtype)
            max_next_q_values[dones] = 0.0
            
            new_q_values = max_next_q_values * self.param.GAMMA + rewards
            current_q_values = self.mainNet(current_states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            ##################################
            loss = nn.MSELoss()(current_q_values, new_q_values)
#             self.temp_history.append(float(loss.detach()))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        if self.interaction_counter % self.param.SYNC_TARGET_FRAMES*int(self.batchSize/32) == 0 and self.memory.current_len >= self.param.REPLAY_START_SIZE:
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

