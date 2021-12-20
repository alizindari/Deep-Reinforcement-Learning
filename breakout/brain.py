
import torch
from torch._C import device, wait
from torch.functional import Tensor
import torch.nn as nn        
import torch.optim as optim  
from Hyperparameters import *
import numpy as np


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.param = Hyperparameters()
        self.conv = nn.Sequential(
            nn.Conv2d(self.param.CHANNEL_NUM, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self.get_conv_output_shape([1,self.param.CHANNEL_NUM,self.param.IMAGESIZE[0],self.param.IMAGESIZE[1]])
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.param.ACTION_SPACE)
        )

    def get_conv_output_shape(self,inp_shape):
        inp = torch.zeros(inp_shape)
        out = self.conv(inp)
        return np.prod(out.shape)


    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)