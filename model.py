#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """Simple FNN for MNIST"""

    def __init__(self):
        nn.Module.__init__(self)

        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)

        return output
