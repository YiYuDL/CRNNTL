# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 21:41:10 2021

@author: yuyi6
"""

import numpy as np
from preprocessing import data_pre
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut    

def SVM(file):
    

    Xi, DFi = data_pre(file)
    yy = DFi['label']
    
    if yy.isin([0,1]).all() == False:
        
        print('regression task')
        
        X = Xi.reshape(-1,1,512)
        X = X.astype('float64')
        X = X.reshape(-1,512)
        
        y =  DFi.label.to_numpy(dtype = 'float32')
        
        
        y = (y-y.mean())/y.std()
        y = y.astype('float64')
        y = y.reshape(-1,)
        
        groups = DFi.groups.to_numpy(dtype = 'int64')
        
        clf = SVR(C=5)
            
        result = cross_val_score(clf,
                                  X,
                                  y,
                                  groups,
                                  cv=LeaveOneGroupOut(),   
                                  scoring = 'r2',
                                  n_jobs=5)
        
        result = np.mean(result)
        print('baseline r2',"%0.2f"%(result))

    else:
        print('classification task')
        X = Xi.reshape(-1,1,512)
        X = X.astype('float64')
        X = X.reshape(-1,512)
        y =  DFi.label.to_numpy(dtype = 'int64')
        groups = DFi.groups.to_numpy(dtype = 'int64')
        
        clf = SVC(C=5)
            
        result = cross_val_score(clf,
                                  X,
                                  y,
                                  groups,
                                  cv=LeaveOneGroupOut(),   
                                  scoring = 'roc_auc',
                                  n_jobs=5)
        result = np.mean(result)
        print('baseline AUC-ROC',"%0.2f"%(result))
 