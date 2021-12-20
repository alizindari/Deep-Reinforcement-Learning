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
import cv2

class Agent():
    def __init__(self):
        self.param = Hyperparameters()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epsilon = self.param.EPS_INITIAL
        self.action_space = self.param.ACTION_SPACE
        self.interaction_counter = 0
        self.batchSize = self.param.BATCH_SIZE
        self.success_counter = 0
        self.success_list = []

        self.mainNet = DQN().to(self.device)
        self.targetNet = DQN().to(self.device)
        self.memory = ExperienceReplay(self.param.REPLAY_SIZE)

        self.temp_history = collections.deque(maxlen=300)
        self.training_history = [[],[]]
        self.optimizer = optim.Adam(self.mainNet.parameters(), lr=self.param.LEARING_RATE)
        self.totalReward = 0

    def preprocessing(self,current_state,game):
        if game == 'pong':
            current_state = current_state[35:195] 
            current_state = current_state[::2, ::2, :]
            current_state = current_state[:, :, 0]
            current_state[current_state == 144] = 0
            current_state[current_state == 109] = 0
            current_state[current_state != 0] = 1
            return current_state 
        elif game == 'breakout':
            current_state = current_state[32:193,8:150,0]
            current_state = cv2.resize(current_state,(self.param.IMAGESIZE[0], self.param.IMAGESIZE[1]))/255
            return current_state

    def find_action(self,state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0,self.param.ACTION_SPACE)
            
        else:
            state = torch.tensor(state).unsqueeze(0).to(self.device)
            q_vals = self.mainNet(state)
            _, action = torch.max(q_vals, dim=1)
            action = int(action.item())

        return action

    
    def fill_memory_with_success(self,env):
        for i in range(self.param.ITER_SUCCESS_FILL):

            stacked_frames = deque(maxlen=self.param.CHANNEL_NUM)
            current_frame = env.reset()
            current_frame = self.preprocessing(current_frame,'pong') 

            for IT in range(self.param.CHANNEL_NUM):
                stacked_frames.append(current_frame) 
            
            current_state = torch.from_numpy(np.stack(stacked_frames,axis=0)).float()
            done = False

            while True:

                action = np.random.randint(0,self.param.ACTION_SPACE)
                new_frame, reward, done,_= env.step(action)
                new_frame = self.preprocessing(new_frame,'pong')
                stacked_frames.append(new_frame)
                new_state = torch.from_numpy(np.stack(stacked_frames,axis=0)).float()
                if reward == 1:
                    self.success_list.append([current_state.clone(),action,reward,done,new_state])
                    print(f'iteration: {i}, last_reward: {reward}')
                current_state = new_state.clone()
            
                if done:
                    break

    def update_weights(self,experience):
        self.memory.add_to_memory(experience)
        self.interaction_counter += 1
        if experience[2] == 1:
            self.success_counter += 1

        if self.success_counter < self.param.THRESH_START_DECAY:
            self.epsilon = self.param.FIXED_EPSILON
        else:
            self.epsilon = max(self.epsilon*self.param.EPS_DECAY, self.param.EPS_MIN)

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

