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
from sklearn.model_selection._validation import _fit_and_score,_score
from skorch.callbacks import EpochScoring,EarlyStopping
from sklearn.metrics._scorer import check_scoring
from CRNNTL.model import *
import torch
import torch.nn as nn




def augCRNN_train(X,DF,augX,augDF):
    
    yy = DF['label']
    
    batch = len(DF)
    if DF_len >= 300:
        batch = 128
    elif 100 < DF_len < 300:
        batch = 64
    else:
        batch = 32
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    if yy.isin([0,1]).all() == False:
        task = 'r2'
        y = yy.to_numpy()
        y = y.reshape(-1, 1)
        y = y.astype('float32')
        
        X = X.astype('float32')
        X = X.reshape(-1,1,512)
        
        groups = DF['groups'].to_numpy()
        groups = groups.astype('int64')
        
        testX = []
        testY = []
        
        logo = LeaveOneGroupOut()
        logo.get_n_splits(X, y, groups=groups)
        
        for train_index, test_index in logo.split(X, y, groups):
            
            a=X[test_index]
            b=y[test_index]
            testX.append(a)
            testY.append(b)
        
        augy = augDF['label'].to_numpy()
        augy = augy.reshape(-1, 1)
        augy = augy.astype('float32')
        
        augX = augX.reshape(-1,1,512)
        augX = augX.astype('float32')
        
        auggroups = augDF['groups'].to_numpy()
        auggroups = auggroups.astype('int64')
        
        logo = LeaveOneGroupOut()
        logo.get_n_splits(augX, augy, auggroups)
        
        result = []
        result1 = []
        result2 = []
        saved_models = []
        nv = 0
        
          
        
        for train_index, test_index in logo.split(augX, augy, auggroups):
            
            print ("aug1DCNN")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'  
            torch.manual_seed(2)
        
            Ir = 0.0001
            max_epochs=30  
            
            X_train = augX[train_index]
            X_test = testX[nv]
            y_train = augy[train_index]
            y_test = testY[nv]
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
            
            # name = '1DCNN' + ' ' + file + str(nv) + '.pkl'
            # net.save_params(f_params= name)
            # nv = nv + 1
             
            # y_pred = net.predict(X_test)
            rs = net.score(X_test, y_test)
            result.append(rs)
            saved_models.append(net)
            
            print('aug1DCNN_reg', rs)
            
            print('start augCRNN1')
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'  
            # CNN_reg.load_state_dict(torch.load(name))
            CRNN_reg.gc = nn.Sequential(
                PER(),
                RNN(input_size, hidden_size, num_layers, num_classes),
                nn.ReLU(),
                nn.Dropout(p = 0.2),
                nn.Linear(in_features=num_classes, out_features=40),
                nn.ReLU(),
                nn.Dropout(p = 0),
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
                              EarlyStopping(patience = 1, threshold = 0.003)],
                device=device,
            )
            
            net.fit(X_train, y_train)
    
            y_pred = net.predict(X_test)
            rs1 = net.score(X_test, y_test)
            # result1 = []
            result1.append(rs1)
            print('augCRNN1 ',rs1)
            
            print('start augCRNN2')
            
            for param in CRNN_reg.conv.parameters():
                param.requires_grad = True
                  
            max_epochs=20
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
            print('augCRNN2 ',rs2)
            # net = clone(net)
            CRNN_reg.gc = CRNN_reg_gc
            nv = nv + 1
        
    else:
        task = 'AUC-ROC'
        y = yy.to_numpy()
        y = y.astype('int64')
        X = X.reshape(-1,1,512)
        X = X.astype('float32')
        
        groups = DF['groups'].to_numpy()
        groups = groups.astype('int64')
        
        testX = []
        testY = []
        
        logo = LeaveOneGroupOut()
        logo.get_n_splits(X, y, groups=groups)
        
        for train_index, test_index in logo.split(X, y, groups):
            
            a=X[test_index]
            b=y[test_index]
            testX.append(a)
            testY.append(b)
        
        augy = augDF['label'].to_numpy()
        augy = augy.astype('int64')
        
        augX = augX.reshape(-1,1,512)
        augX = augX.astype('float32')
        
        auggroups = augDF['groups'].to_numpy()
        auggroups = auggroups.astype('int64')
        
        logo = LeaveOneGroupOut()
        logo.get_n_splits(augX, augy, auggroups)
        
        result = []
        result1 = []
        result2 = []
        nv = 0
        
        for train_index, test_index in logo.split(augX, augy, auggroups):
            
            print ("aug1DCNN")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'  
            torch.manual_seed(2)
        
            Ir = 0.0001
            max_epochs=30  
            
            X_train = augX[train_index]
            X_test = testX[nv]
            y_train = augy[train_index]
            y_test = testY[nv]
            valid_ds = Dataset(X_test, y_test)
            
            net = NeuralNetClassifier(
                CRNN_cla,
                batch_size=batch,
                train_split=predefined_split(valid_ds),
                optimizer = torch.optim.Adam,
                max_epochs= max_epochs,
                lr=Ir,
                callbacks = [EpochScoring('roc_auc',lower_is_better = False),
                             EarlyStopping(patience = 2, threshold = 0.0005)],    
                device=device,
            )
            net = clone(net)
            scorers = check_scoring(net, 'roc_auc')
            y_pred = _fit_and_score(net, augX, augy, scorer = scorers, return_parameters=True,
                                    train = train_index, test = test_index,
                                    verbose = 0,
                                    return_estimator = True,
                                    parameters = None,fit_params = None)
            
            CNN_auc = _score(net, X_test, y_test, scorers, error_score="raise")
            result.append(CNN_auc)
            print('aug1DCNN_cla', CNN_auc)
            
            print('start augCRNN1')
            CRNN_cla.gc = nn.Sequential(
                PER(),
                RNN(input_size, hidden_size, num_layers, num_classes),
                nn.ReLU(),
                nn.Dropout(p = 0.2),
                nn.Linear(in_features=num_classes, out_features=40),
                nn.ReLU(),
                nn.Dropout(p = 0),
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
            
            y_pred = _fit_and_score(net, augX, augy, scorer = scorers, return_parameters=True,
                                    train = train_index, test = test_index,
                                    verbose = 0,
                                    return_estimator = True,
                                    parameters = None,
                                    fit_params = None)
            
            # result1.append(y_pred[0])
            # print('augCRNN1', y_pred[0])
            
            print('start augCRNN2')
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
                             EarlyStopping(patience = 2, threshold = 0.001)],    
                device=device,
            )
            
            y_pred = _fit_and_score(net, augX, augy, scorer = scorers, return_parameters=True,
                                    train = train_index, test = test_index,
                                    verbose = 0,
                                    return_estimator = True,
                                    parameters = None,
                                    fit_params = None)
            
            CRNN_auc = _score(net, X_test, y_test, scorers, error_score="raise")
            result2.append(CRNN_auc)
            CRNN_cla.gc = CRNN_cla_gc
            print('augCRNN2', CRNN_auc)
            nv = nv + 1
        
    print('CRNN',task,"%0.2f"%(np.mean(result2)))
     
