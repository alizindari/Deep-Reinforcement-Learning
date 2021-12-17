from os import stat
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
import torch.optim as optim 
from torch import nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import gym
import gym.spaces
import torch
import torch.nn as nn        # Pytorch neural network package
import torch.optim as optim  # Pytorch optimization package
import matplotlib.pyplot as plt
import IPython
import time
import numpy as np
import collections
from ai import *
# from gymWrappers import *
# from IPython.display import clear_output





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
        
        
        
device = torch.device("cpu")
DEFAULT_ENV_NAME = "PongNoFrameskip-v4" 
rewards = collections.deque(maxlen=20)



env = Environment(30,30)
jimi = Agent(None)
jimi.loadKnowlage('best.dat')
jimi.epsilon = 0

state = env.reset()

for itr in range(10):
    for eps in range(50):
        image = state[0].cpu().detach().numpy().copy()
        image = cv2.resize(image,(1000,1000))
        cv2.imshow('game',image)
        cv2.waitKey(10)

        action = jimi.dicide(state)
        newState, reward, done= env.step(action)
        state = newState.clone()

        if done:
            state = env.reset()
            jimi.reset()
            break

            

