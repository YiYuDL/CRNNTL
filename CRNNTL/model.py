# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 09:02:33 2021

@author: yuyi6
"""

import torch
import torch.nn as nn

input_size = 60
hidden_size = 60
num_layers = 1      
sequence_length = 64
num_classes = 600
dropout1 = 0.2
dropout2 = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'  

class CRNN_cla(nn.Module):
    def __init__(self):
        super(CRNN_cla, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=15, kernel_size=5, stride = 2, padding = 2)
            , nn.ReLU()
            , nn.BatchNorm1d(15)
            , nn.Conv1d(in_channels=15, out_channels=30, kernel_size=2, stride = 2)
            , nn.ReLU()
            , nn.BatchNorm1d(30)
            , nn.Conv1d(in_channels=30, out_channels=60, kernel_size=5, stride = 2, padding = 2)
            , nn.BatchNorm1d(60)
        )
          
        self.gc = nn.Sequential(
            nn.Flatten(start_dim=1)
            , nn.Linear(in_features=64*60, out_features=40)
            , nn.ReLU()
            , nn.Dropout(p = dropout1)
            , nn.Linear(in_features=40, out_features=2)
            , nn.Softmax(dim = 1)
        )
    
    def forward(self, x):
        x = self.conv(x)
        out = self.gc(x)
        return out

CRNN_cla = CRNN_cla()
    
class CRNN_reg(nn.Module):
    def __init__(self):
        super(CRNN_reg, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=15, kernel_size=5, stride = 2, padding = 2)
            , nn.ReLU()
            , nn.BatchNorm1d(15)
            , nn.Conv1d(in_channels=15, out_channels=30, kernel_size=2, stride = 2)
            , nn.ReLU()
            , nn.BatchNorm1d(30)
            , nn.Conv1d(in_channels=30, out_channels=60, kernel_size=5, stride = 2, padding = 2)
            , nn.BatchNorm1d(60)
        )
        
        self.gc = nn.Sequential(
            nn.Flatten(start_dim=1)
            , nn.Linear(in_features=64*60, out_features=40)
            , nn.ReLU()
            , nn.Dropout(p = dropout1)
            , nn.Linear(in_features=40, out_features=1)
        )
    
    def forward(self, x):
        x = self.conv(x)
        out = self.gc(x)
        return out

CRNN_reg = CRNN_reg()
 
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first = True,bidirectional = True)
        self.fc = nn.Linear(2*hidden_size*sequence_length, num_classes)
        
    def forward(self,t):
        h0 = torch.zeros(self.num_layers*2, t.size(0), self.hidden_size).to(device)
        out, _ =self.gru(t, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out
    
class PER(nn.Module):
    def __int__(self):
        super(PER, self).__init__()
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        return x
    
CRNN_cla_gc = nn.Sequential(
    nn.Flatten(start_dim=1)
    , nn.Linear(in_features=64*60, out_features=40)
    , nn.ReLU()
    , nn.Dropout(p = dropout1)
    , nn.Linear(in_features=40, out_features=2)
    , nn.Softmax(dim = 1)
    )

CRNN_reg_gc = nn.Sequential(
    nn.Flatten(start_dim=1)
    , nn.Linear(in_features=64*60, out_features=40)
    , nn.ReLU()
    , nn.Dropout(p = dropout1)
    , nn.Linear(in_features=40, out_features=1)
    )
