import torch
import numpy as np
import torch.nn as nn
import os

class simpletest(nn.Module):
    def __init__(self):
        super(simpletest, self).__init__()
        self.net = []
        self.net += [nn.Linear(3, 16),
                nn.Linear(16,32),
                nn.Linear(32,64),
                nn.Linear(64,32),
                nn.Linear(32,16),
                nn.Linear(16,8),
                nn.Linear(8,1)]

    def forward(self, x):
        x = self.net
        return x




