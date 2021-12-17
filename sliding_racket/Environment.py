
import numpy as np
from collections import deque
from experience_replay import *
from brain import *
from agent import *

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


class Environment:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.buff = deque(maxlen= 3)

        
    def reset(self):
        self.map = np.zeros([self.x,self.y])
        self.x_pos = 0
        self.y_pos = np.random.randint(0,self.y)
        self.map[self.x_pos,self.y_pos] = 1
        self.bar_init = np.random.randint(0,self.y-7)
        self.bar = [self.bar_init,self.bar_init+5]
        self.map[self.x-1,self.bar[0]:self.bar[1]] = 1

        self.buff.append(self.map)
        self.buff.append(self.map)
        self.buff.append(self.map)

        return torch.tensor(np.stack(self.buff,axis=0),dtype= torch.float32).to('cpu')
    
    def step(self,action):
        
        self.x_pos += 1
        done = False
        reward = 0
        if self.x_pos == self.x-1 and self.map[self.x_pos,self.y_pos] == 1:
            reward = 5
            done = True
        if self.x_pos == self.x-1 and self.map[self.x_pos,self.y_pos] == 0:
            reward = -1
            done = True
            
        if action == 0:
            if self.bar[0] == 0:
                pass
            else:
                self.bar[0] -= 1
                self.bar[1] -= 1
                
        if action == 1:
            if self.bar[1] == self.y:
                pass
            else:
                self.bar[1] += 1
                self.bar[0] += 1
                
        if action == 2:
            pass
                
        self.map = np.zeros([self.x,self.y])
        self.map[self.x_pos,self.y_pos] = 1
        self.map[self.x-1,self.bar[0]:self.bar[1]] = 1
        
        self.buff.append(self.map)
        return torch.tensor(np.stack(self.buff,axis=0),dtype= torch.float32).to('cpu'),reward,done
        
        
        

        