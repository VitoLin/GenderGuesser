import torch
import torch.nn as nn
from torch.nn.functional import relu, sigmoid

class MLPClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.input = nn.Linear(512, 1000)
        self.hidden1 = nn.Linear(1000, 1000)
        self.hidden2 = nn.Linear(1000, 1000)
        self.out = nn.Linear(1000, 1)
    
    def forward(self, x):
        
        x = relu(self.input(x))
        x = relu(self.hidden1(x))
        x = relu(self.hidden2(x))
        x = sigmoid(self.out(x)) # to convert to 0, 1
        return x