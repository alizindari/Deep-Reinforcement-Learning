
import torch
from torch._C import device, wait
from torch.functional import Tensor
import torch.nn as nn        
import torch.optim as optim  
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

class DQN(nn.Module):
    def __init__(self, input_shape, ACTION_SPACE):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = 9216
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_SPACE)
        )


    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)