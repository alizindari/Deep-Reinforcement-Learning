import numpy as np
import IPython
import collections
from Environment import *
import torch
from Hyperparameters import *


class ExperienceReplay:
    def __init__(self,buffer_size):
        self.param = Hyperparameters()
        self.buffer_size = buffer_size
        self.current_states = torch.empty([buffer_size,self.param.CHANNEL_NUM,self.param.IMAGESIZE[0],self.param.IMAGESIZE[1]])
        self.next_states    = torch.empty([buffer_size,self.param.CHANNEL_NUM,self.param.IMAGESIZE[0],self.param.IMAGESIZE[1]])
        self.actions        = torch.empty([buffer_size])
        self.rewards        = torch.empty([buffer_size])
        self.dones          = torch.empty([buffer_size])
        self.index          = 0
        self.current_len    = 0
        
    def add_to_memory(self,experience):
        current_state,action,reward,done,next_state = experience
        
        self.current_states[self.index] = current_state
        self.next_states[self.index]    = next_state
        self.rewards[self.index]        = reward
        self.actions[self.index]        = action
        self.dones[self.index]          = done
        
        self.index = (self.index + 1) % self.buffer_size
        self.current_len = min(self.current_len + 1,self.buffer_size)
        
    def random_sample(self):
        idx = torch.randint(0,self.current_len,[self.param.BATCH_SIZE])
        samples = [
            self.current_states[idx],
            self.next_states[idx],
            self.rewards[idx],
            self.actions[idx],
            self.dones[idx]   
        ]
        
        return samples