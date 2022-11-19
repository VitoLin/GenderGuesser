import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.nn.functional import relu, sigmoid
import pytorch_lightning


class MLPClassifier(nn.Module):
    '''
    Input: 512
    ReLU
    Hidden1: 1000
    ReLU
    Hidden2: 1000
    ReLU
    Out: 1
    '''
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

class MLPClassifierWrapper(pytorch_lightning.LightningModule):
    
    
    def __init__(self, trained_model):
        self.trained_model = trained_model
        self.model = MLPClassifier()
        self.loss_func = nn.BCELoss()
    
    
    def training_step(self, batch, batch_idx):
        # training loop
        x, y = batch
        yhat = self.model(x)
        loss = self.loss_func(y, yhat)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation loop
        x, y = batch
        yhat = self.model(x)
        loss = self.loss_func(y, yhat)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x,y = batch
        yhat = self.model(x)
        loss = self.loss_func(y, yhat)
        self.log("test_loss", loss)
        return loss


if __name__ == '__main__':
    
    
    
    MLPClassifierWrapper()