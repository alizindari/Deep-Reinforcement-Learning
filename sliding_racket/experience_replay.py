import numpy as np
import IPython
import collections
from Environment import *
import torch
IMAGESIZE = 30
MEAN_REWARD_BOUND = 19.0           
CHANNEL_NUM = 3
ACTION_SPACE = 3
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
    def __init__(self,buffer_size):
        self.buffer_size = buffer_size
        self.current_states = torch.empty([buffer_size,CHANNEL_NUM,IMAGESIZE,IMAGESIZE])
        self.next_states    = torch.empty([buffer_size,CHANNEL_NUM,IMAGESIZE,IMAGESIZE])
        self.actions        = torch.empty([buffer_size])
        self.rewards        = torch.empty([buffer_size])
        self.dones          = torch.empty([buffer_size])
        self.index          = 0
        self.current_len    = 0
        
    def add_to_memory(self,experience):
        current_state,action,reward,done,next_state = experience
        
        self.current_states[self.index] = current_state
        self.next_states[self.index]    = next_state
        self.rewards[self.index]       = reward
        self.actions[self.index]       = action
        self.dones[self.index]         = done
        
        self.index = (self.index + 1) % self.buffer_size
        self.current_len = min(self.current_len + 1,self.buffer_size)
        
    def random_sample(self):
        idx = torch.randint(0,self.current_len,[BATCH_SIZE])
        samples = [
            self.current_states[idx],
            self.next_states[idx],
            self.rewards[idx],
            self.actions[idx],
            self.dones[idx]   
        ]
        
        return samples