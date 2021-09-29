# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 13:58:17 2021

@author: yuyi6
"""
 
from preprocessing import data_pre
from train import CNN_CRNN_train
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np
from skorch import NeuralNetRegressor, NeuralNetClassifier
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from sklearn.model_selection._validation import _fit_and_score
from skorch.callbacks import EpochScoring,EarlyStopping
from sklearn.metrics._scorer import check_scoring
from model import *
import torch
import torch.nn as nn

def TL(source, target):

# source training
    Xs, DFs = data_pre(source)
    yy = DFs['label']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    if yy.isin([0,1]).all() == False:
        y = yy.to_numpy()
        
        y = y.reshape(-1, 1)
        y = y.astype('float32')
        
        X = Xs.astype('float32')
        X = X.reshape(-1,1,512)
        
        groups = DFs['groups'].to_numpy()
        groups = groups.astype('int64')
        logo = LeaveOneGroupOut()
        logo.get_n_splits(X, y, groups=groups)
        nv = 0
        result1 = []
        saved_models = []
        
        for train_index, test_index in logo.split(X, y, groups):
            
            print ("1DCNN")
            
            torch.manual_seed(2)
        
            Ir = 0.0001
            max_epochs=30  
            
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            valid_ds = Dataset(X_test, y_test)
            nets = NeuralNetRegressor(
                CRNN_reg,
                batch_size=128,
                train_split=predefined_split(valid_ds),
                optimizer = torch.optim.Adam,
                max_epochs= max_epochs,  
                lr=Ir,
                callbacks = [EpochScoring('r2',lower_is_better = False),
                              EarlyStopping(patience = 1, threshold = 0.0005)],
                device=device,
            )
            
            nets = clone(nets)
            
            nets.fit(X_train, y_train)
            
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
                
            nets = NeuralNetRegressor(
                module=CRNN_reg,
                train_split=predefined_split(valid_ds),
                batch_size=128,
                max_epochs= max_epochs,
                optimizer = torch.optim.Adam,
                lr= 0.0003,
                callbacks = [EpochScoring('r2',lower_is_better = False),
                              EarlyStopping(patience = 2, threshold = 0.003)],
                device=device,
            )
            
            nets.fit(X_train, y_train)
    
            
            for param in CRNN_reg.conv.parameters():
                param.requires_grad = True
                  
        
            nets = NeuralNetRegressor(
                module=CRNN_reg,
                train_split=predefined_split(valid_ds),
                batch_size=128,
                max_epochs= 50,
                optimizer = torch.optim.Adam,
                lr= 0.00003,
                callbacks = [EpochScoring('r2',lower_is_better = False),
                                EarlyStopping(patience = 2, threshold = 0.0005)],
                device=device
                )
           
            nets.fit(X_train, y_train)

            name = 'specific' + ' ' + str(nv) + '.pkl'
            nets.save_params(f_params= name)
            nv = nv + 1
             
            rs = nets.score(X_test, y_test)
            result1.append(rs)
            saved_models.append(nets)
            CRNN_reg.gc = CRNN_reg_gc
            
            print('1DCNN_reg', rs)
            
        max_s = max(result1)
        max_i = result1.index(max_s)
        TL_model = saved_models[max_i]
        name1 = 'specific' + ' ' + str(max_i) + '.pkl'
    
    else:
        y = yy.to_numpy()
        
        y = y.reshape(-1, 1)
        y = y.astype('int64')
        
        X = X.astype('float32')
        X = X.reshape(-1,1,512)
        
        groups = DFs['groups'].to_numpy()
        groups = groups.astype('int64')
        logo = LeaveOneGroupOut()
        logo.get_n_splits(X, y, groups=groups)
        
        logo.get_n_splits(X, y, groups=groups)
        
        result1 = []
        saved_models = []
        
        torch.manual_seed(2)
        for train_index, test_index in logo.split(X, y, groups):
            
            Ir = 0.0001
            max_epochs=30
            
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            valid_ds = Dataset(X_test, y_test)
            
            net = NeuralNetClassifier(
                CRNN_cla,
                batch_size=128,
                train_split=predefined_split(valid_ds),
                optimizer = torch.optim.Adam,
                max_epochs= max_epochs,
                lr=Ir,
                callbacks = [EpochScoring('roc_auc',lower_is_better = False),
                              EarlyStopping(patience = 1, threshold = 0.0005)],    
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
            
            name = 'specific' + ' ' + str(nv) + '.pkl'
            net.save_params(f_params= name)
            nv = nv + 1
            result1.append(y_pred[0])
            
        max_s = max(result1)

        max_i = result1.index(max_s)
        name1 = 'specific' + ' ' + str(max_i) + '.pkl'
        TL_model = saved_models[max_i]
        
    
#learning from scratch
    Xt, DFt = data_pre(target)
    result, result2 = CNN_CRNN_train(Xt,DFt)
    print('learning from scratch:', result2)
    
#transfer learning
    print('start TL')
    yy = DFt['label']
    if yy.isin([0,1]).all() == False:
        y = yy.to_numpy()
        
        y = y.reshape(-1, 1)
        y = y.astype('float32')
        
        X = Xt.astype('float32')
        X = X.reshape(-1,1,512)
        
        groups = DFt['groups'].to_numpy()
        groups = groups.astype('int64')
        logo = LeaveOneGroupOut()
        logo.get_n_splits(X, y, groups=groups)
       
        result3 = []
        
        for train_index, test_index in logo.split(X, y, groups):
            
            print ("1DCNN")
            
            torch.manual_seed(2)
        
            Ir = 0.0001
            max_epochs=30  
            
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            valid_ds = Dataset(X_test, y_test)
            
            CRNN_reg.gc = nn.Sequential(
                PER(),
                RNN(input_size, hidden_size, num_layers, num_classes),
                nn.ReLU(),
                nn.Dropout(p = 0.2),
                nn.Linear(in_features=num_classes, out_features=40),
                nn.ReLU(),
                nn.Dropout(p = 0.0),
                nn.Linear(in_features=40, out_features=1),)
            
            CRNN_reg.load_state_dict(torch.load(name1))
            
            for param in CRNN_reg.conv.parameters():
                param.requires_grad = False
                
            net = NeuralNetRegressor(
                module=CRNN_reg,
                train_split=predefined_split(valid_ds),
                batch_size=128,
                max_epochs= max_epochs,
                optimizer = torch.optim.Adam,
                lr= 0.0001,
                callbacks = [EpochScoring('r2',lower_is_better = False),
                              EarlyStopping(patience = 2, threshold = 0.001)],
                device=device,
            )
            
            net.fit(X_train, y_train)
            
            for param in CRNN_reg.conv.parameters():
                param.requires_grad = True
                  
            max_epochs=20
            net = NeuralNetRegressor(
                module=CRNN_reg,
                train_split=predefined_split(valid_ds),
                batch_size=128,
                max_epochs= 30,
                optimizer = torch.optim.Adam,
                lr= 0.0001,
                callbacks = [EpochScoring('r2',lower_is_better = False),
                               EarlyStopping(patience = 2, threshold = 0.001)],
                device=device
                )
           
            net.fit(X_train, y_train)
            
            rs = net.score(X_test, y_test)
            result3.append(rs)
            
            print('tl:', rs)
        
    else:
        y = yy.to_numpy()
        
        y = y.reshape(-1, 1)
        y = y.astype('int64')
        
        X = X.astype('float32')
        X = X.reshape(-1,1,512)
        
        groups = DFs['groups'].to_numpy()
        groups = groups.astype('int64')
        logo = LeaveOneGroupOut()
        logo.get_n_splits(X, y, groups=groups)
        
        logo.get_n_splits(X, y, groups=groups)
        
        result3 = []
        
        torch.manual_seed(2)
        for train_index, test_index in logo.split(X, y, groups):
            
            print ("1DCNN")
            
            torch.manual_seed(2)
        
            Ir = 0.0001
            max_epochs=30  
            
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            valid_ds = Dataset(X_test, y_test)
            
            CRNN_cla.gc = nn.Sequential(
                PER(),
                RNN(input_size, hidden_size, num_layers, num_classes),
                nn.ReLU(),
                nn.Dropout(p = 0.2),
                nn.Linear(in_features=num_classes, out_features=40),
                nn.ReLU(),
                nn.Dropout(p = 0.0),
                nn.Linear(in_features=40, out_features=1),)
            
            CRNN_cla.load_state_dict(torch.load(name1))
            
            for param in CRNN_cla.conv.parameters():
                param.requires_grad = False
                
            net = NeuralNetClassifier(
                module=CRNN_cla,
                train_split=predefined_split(valid_ds),
                batch_size=128,
                max_epochs= max_epochs,
                optimizer = torch.optim.Adam,
                lr= 0.0001,
                callbacks = [EpochScoring('roc_auc',lower_is_better = False),
                              EarlyStopping(patience = 2, threshold = 0.001)],
                device=device,
            )
            
            y_pred = _fit_and_score(net, X, y, scorer = scorers, return_parameters=True,
                                    train = train_index, test = test_index,
                                    verbose = 0,
                                    return_estimator = True,
                                    parameters = None,
                                    fit_params = None)
            
            for param in CRNN_cla.conv.parameters():
                param.requires_grad = True
                  
            max_epochs=20
            net = NeuralNetClassifier(
                module=CRNN_cla,
                train_split=predefined_split(valid_ds),
                batch_size=128,
                max_epochs= 30,
                optimizer = torch.optim.Adam,
                lr= 0.0001,
                callbacks = [EpochScoring('roc_auc',lower_is_better = False),
                               EarlyStopping(patience = 2, threshold = 0.001)],
                device=device
                )
           
            y_pred = _fit_and_score(net, X, y, scorer = scorers, return_parameters=True,
                                    train = train_index, test = test_index,
                                    verbose = 0,
                                    return_estimator = True,
                                    parameters = None,
                                    fit_params = None)
            
            
            result3.append(y_pred[0])
            
            print('tl:', rs)
            

        print('haha')
    result3 = np.mean(result3)    
    print('Learning from scratch',"%0.2f"%(result))
    print('Transfer learning from',source,"%0.2f"%(result3))
   
            
        
    
    
    
    
        
    
                    
                    

    