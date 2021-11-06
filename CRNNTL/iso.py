# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 10:56:38 2021

@author: yuyi6
"""

from sklearn.base import clone
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
import numpy as np
from preprocessing import data_pre
from skorch import NeuralNetRegressor
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from skorch.callbacks import EpochScoring,EarlyStopping
from CRNNTL.model import *
import torch
from sklearn.svm import SVR
import torch.nn as nn
import pandas as pd

def iso_CNN_CRNN_SVM(file):
    file_name = 'datasets/' + file + '.csv'
    iso_df = pd.read_csv(file_name, index_col=False)
    Xi, DFi = data_pre(iso_df)
    
    yy = DFi['label']
    
    y = yy.to_numpy()
    
    y = y.reshape(-1, 1)
    y = y.astype('float32')
    
    X = Xi.astype('float32')
    X = X.reshape(-1,1,512)
    
    groups = DFi['groups'].to_numpy()
    groups = groups.astype('int64')
    logo = LeaveOneGroupOut()
    logo.get_n_splits(X, y, groups=groups)
    
    result = []
    result1 = []
    result2 = []
    saved_models = []
    nv = 0
   
    for train_index, test_index in logo.split(X, y, groups):
        
        print ("1DCNN")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'  
        torch.manual_seed(2)
    
        Ir = 0.0001
        max_epochs=20  
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        valid_ds = Dataset(X_test, y_test)
        
        net = NeuralNetRegressor(
            CRNN_reg,
            batch_size=20,
            train_split=predefined_split(valid_ds),
            optimizer = torch.optim.Adam,
            max_epochs= max_epochs,  
            lr=Ir,
            callbacks = [EpochScoring('r2',lower_is_better = False),
                         EarlyStopping(patience = 3, threshold = 0.001)],
            device=device,
        )
        
        net = clone(net)
        net.fit(X_train, y_train)
     
        rs = net.score(X_test, y_test)
        result.append(rs)
        saved_models.append(net)
        
        print('1DCNN_reg', rs)
        print('start CRNN1')
        
        CRNN_reg.gc = nn.Sequential(
            PER(),
            RNN(input_size, hidden_size, num_layers, num_classes),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(in_features=num_classes, out_features=40),
            nn.ReLU(),
            nn.Dropout(p = 0.0),
            nn.Linear(in_features=40, out_features=1),)
        
        for param in CRNN_reg.conv.parameters():
            param.requires_grad = False
            
        net = NeuralNetRegressor(
            module=CRNN_reg,
            train_split=predefined_split(valid_ds),
            batch_size=20,
            max_epochs= max_epochs,
            optimizer = torch.optim.Adam,
            lr= 0.0001,
            callbacks = [EpochScoring('r2',lower_is_better = False),
                          EarlyStopping(patience = 1, threshold = 0.001)],
            device=device,
        )
        
        net.fit(X_train, y_train)

        y_pred = net.predict(X_test)
        rs1 = net.score(X_test, y_test)
        
        result1.append(rs1)
        print('CRNN1 ',rs1)
        
        print('start CRNN2')
        
        for param in CRNN_reg.conv.parameters():
            param.requires_grad = True
              
        max_epochs=30
        net = NeuralNetRegressor(
            module=CRNN_reg,
            train_split=predefined_split(valid_ds),
            batch_size=20,
            max_epochs= max_epochs,
            optimizer = torch.optim.Adam,
            lr= 0.00001,
            callbacks = [EpochScoring('r2',lower_is_better = False),
                           EarlyStopping(patience = 1, threshold = 0.001)],
            device=device
            )
       
        net.fit(X_train, y_train)

        y_pred = net.predict(X_test)
        rs2 = net.score(X_test, y_test)
        result2.append(rs2)
        print('CRNN2 ',rs2)

        CRNN_reg.gc = CRNN_reg_gc
        nv = nv + 1
        
    X = Xi.reshape(-1,1,512)
    X = X.astype('float64')
    X = X.reshape(-1,512)
    
    y =  DFi.label.to_numpy(dtype = 'float32')
    
    
    y = (y-y.mean())/y.std()
    y = y.astype('float64')
    y = y.reshape(-1,)
    
    groups = DFi.groups.to_numpy(dtype = 'int64')
    
    clf = SVR(C=5)
        
    result3 = cross_val_score(clf,
                              X,
                              y,
                              groups,
                              cv=LeaveOneGroupOut(),   
                              scoring = 'r2',
                              n_jobs=5)
    result1 = np.mean(result)
    result2 = np.mean(result2)
    result3 = np.mean(result3)
    
    print('CNN',"%0.2f"%(result1))
    
    print('CRNN',"%0.2f"%(result2))

    print('SVM', "%0.2f"%(result3))
     
