# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 10:56:38 2021

@author: yuyi6
"""

from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np
from skorch import NeuralNetRegressor, NeuralNetClassifier
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from sklearn.model_selection._validation import _fit_and_score
from skorch.callbacks import EpochScoring,EarlyStopping
from sklearn.metrics._scorer import check_scoring
from CRNNTL.model import *
import torch
import torch.nn as nn

def CNN_CRNN_train(X,DF):
    
    yy = DF['label']
    DF_len = len(DF)
    
    if DF_len >= 300:
        batch = 128
    elif 100 < DF_len < 300:
        batch = 64
    else:
        batch = 32

   
    if yy.isin([0,1]).all() == False:
# start cross validation for regression  
        task = 'r2'
        print('regression begins')
        y = yy.to_numpy()
        
        y = y.reshape(-1, 1)
        y = y.astype('float32')
        
        X = X.astype('float32')
        X = X.reshape(-1,1,512)
        
        groups = DF['groups'].to_numpy()
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
            max_epochs=50  
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            valid_ds = Dataset(X_test, y_test)
            
            net = NeuralNetRegressor(
                CRNN_reg,
                batch_size=batch,
                train_split=predefined_split(valid_ds),
                optimizer = torch.optim.Adam,
                max_epochs= max_epochs,  
                lr=Ir,
                callbacks = [EpochScoring('r2',lower_is_better = False),
                             EarlyStopping(patience = 1, threshold = 0.0005)],
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
                nn.Dropout(p = dropout1),
                nn.Linear(in_features=num_classes, out_features=40),
                nn.ReLU(),
                nn.Dropout(p = dropout2),
                nn.Linear(in_features=40, out_features=1),)
            
            for param in CRNN_reg.conv.parameters():
                param.requires_grad = False
                
            net = NeuralNetRegressor(
                module=CRNN_reg,
                train_split=predefined_split(valid_ds),
                batch_size=batch,
                max_epochs= max_epochs,
                optimizer = torch.optim.Adam,
                lr= 0.0005,
                callbacks = [EpochScoring('r2',lower_is_better = False),
                              EarlyStopping(patience = 2, threshold = 0.003)],
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
                  
            # max_epochs=50
            net = NeuralNetRegressor(
                module=CRNN_reg,
                train_split=predefined_split(valid_ds),
                batch_size=batch,
                max_epochs= max_epochs,
                optimizer = torch.optim.Adam,
                lr= 0.00003,
                callbacks = [EpochScoring('r2',lower_is_better = False),
                               EarlyStopping(patience = 2, threshold = 0.0005)],
                device=device
                )
           
            net.fit(X_train, y_train)
    
            y_pred = net.predict(X_test)
            rs2 = net.score(X_test, y_test)
            result2.append(rs2)
            print('CRNN2 ',rs2)

            CRNN_reg.gc = CRNN_reg_gc
            nv = nv + 1
            
    else:
# start cross validation for regression 
        task = 'AUC-ROC'
        y = yy.to_numpy()
        y = y.astype('int64')
        
        X = X.astype('float32')
        X = X.reshape(-1,1,512)
        
        groups = DF['groups'].to_numpy()
        groups = groups.astype('int64')
        logo = LeaveOneGroupOut()
        logo.get_n_splits(X, y, groups=groups)
        
        logo.get_n_splits(X, y, groups=groups)
        
        result = []
        result1 = []
        result2 = []
        nv = 0
        
        for train_index, test_index in logo.split(X, y, groups):
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'  
            torch.manual_seed(2)
        
            Ir = 0.0001
            max_epochs=30
            
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            valid_ds = Dataset(X_test, y_test)
            
            net = NeuralNetClassifier(
                CRNN_cla,
                batch_size=batch,
                train_split=predefined_split(valid_ds),
                optimizer = torch.optim.Adam,
                max_epochs= max_epochs,
                lr=Ir,
                callbacks = [EpochScoring('roc_auc',lower_is_better = False),
                             EarlyStopping(patience = 1, threshold = 0.001)],    
                device=device,
            )
            net = clone(net)
            scorers = check_scoring(net, 'roc_auc')
            y_pred = _fit_and_score(net, X, y, scorer = scorers, return_parameters=True,
                                    train = train_index, test = test_index,
                                    verbose = 0,
                                    return_estimator = True,
                                    parameters = None,
                                    fit_params = None)
    
            result.append(y_pred["test_scores"])
            print('1DCNN_cla', y_pred["test_scores"])
            
            print('start CRNN1')
            CRNN_cla.gc = nn.Sequential(
                PER(),
                RNN(input_size, hidden_size, num_layers, num_classes),
                nn.ReLU(),
                nn.Dropout(p = dropout1),
                nn.Linear(in_features=num_classes, out_features=40),
                nn.ReLU(),
                nn.Dropout(p = dropout2),
                nn.Linear(in_features=40, out_features=2),
                nn.Softmax(dim = 1),)  
            
            
            for param in CRNN_cla.conv.parameters():
                param.requires_grad = False
                
            net = NeuralNetClassifier(
                CRNN_cla,
                batch_size=batch,
                train_split=predefined_split(valid_ds),
                optimizer = torch.optim.Adam,
                max_epochs= max_epochs,
                lr=0.0003,
                callbacks = [EpochScoring('roc_auc',lower_is_better = False),
                              EarlyStopping(patience = 2, threshold = 0.003)],    
                device=device,
            )
            
            y_pred = _fit_and_score(net, X, y, scorer = scorers, return_parameters=True,
                                    train = train_index, test = test_index,
                                    verbose = 0,
                                    return_estimator = True,
                                    parameters = None,
                                    fit_params = None)
            
            result1.append(y_pred["test_scores"])
            print('CRNN1', y_pred["test_scores"])
            
            print('start CRNN2')
            for param in CRNN_cla.conv.parameters():
                param.requires_grad = True
            
            net = NeuralNetClassifier(
                CRNN_cla,
                batch_size=batch,
                train_split=predefined_split(valid_ds),
                optimizer = torch.optim.Adam,
                max_epochs= max_epochs,
                lr=0.00003,
                callbacks = [EpochScoring('roc_auc',lower_is_better = False),
                             EarlyStopping(patience = 2, threshold = 0.0005)],    
                device=device,
            )
            
            y_pred = _fit_and_score(net, X, y, scorer = scorers, return_parameters=True,
                                    train = train_index, test = test_index,
                                    verbose = 0,
                                    return_estimator = True,
                                    parameters = None,
                                    fit_params = None)
            
            result2.append(y_pred["test_scores"])
            CRNN_cla.gc = CRNN_cla_gc
            print('CRNN2', y_pred["test_scores"])
            
    print('CNN',task,"%0.2f"%(np.mean(result)))
    
    print('CRNN',task,"%0.2f"%(np.mean(result2)))
    result = np.mean(result)
    result2 = np.mean(result2)
    return result, result2
     
