# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:28:44 2021

@author: yuyi6
"""

import pandas as pd
import numpy as np
import collections
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from cddd.inference import InferenceModel
from sklearn.model_selection import StratifiedKFold, KFold

def data_pre(file):
# input csv
    file = 'datasets/' + file + '.csv'
    df = pd.read_csv(file,encoding='gb18030')
                   
    print('number of molecules before cleanning',len(df))
       
    df1 = df['smiles'].astype('str')
    
# transfer the smile into consistent format
    def clean(a):
        try:
            mol = Chem.MolFromSmiles(a)
            Chem.RemoveStereochemistry(mol)
            Chem.RemoveHs(mol)
            remover = SaltRemover()
            mol1 = remover.StripMol(mol)
            aa = Chem.MolToSmiles(mol1)
            return aa
        except:
            return ''
        
    df['smilessd'] = df1.map(clean)
    df = df.dropna()
   
    print('number of molecules after cleaning',len(df))
         
    e = np.where(df["smiles"] == df["smilessd"], True, False)
    print('number of molecules smiles changed', collections.Counter(e))  
        
    df4 = df['smilessd']
    
    smiles_list = df4.tolist()
    
    inference_model = InferenceModel()
    smiles_embedding = inference_model.seq_to_emb(smiles_list)
           
    dataX = smiles_embedding     
    dataX = (dataX - dataX.mean()) / dataX.std()
    X = dataX.reshape(-1,1,512)
    X = X.astype('float32')
    
    yy = df['label']
    y = yy.to_numpy()
# add group 
    try:
        groups = df['groups'].to_numpy()
        print('groups in the file')
        if df['label'].isin([0,1]).all() == False:
            y = y.astype('float32')
            if np.any(y<=0) == False:
                y = np.log10(y)
                y = (y - y.mean()) / y.std()
            else:
                y = (y - y.mean()) / y.std()
            
        else:
            y = y.astype('int64')
        DF = pd.DataFrame(columns = ['label', 'smiles', 'groups'])
        
        DF['label'] = pd.Series(y)
        DF['groups'] = pd.Series(groups)
        DF['smiles'] = df4
        
    except:
        print('creating groups')
        if df['label'].isin([0,1]).all() == False:
        
            y = y.astype('float32')
            if np.any(y<=0) == False:
                y = np.log10(y)
                y = (y - y.mean()) / y.std()
            else:
                y = (y - y.mean()) / y.std()
            n_splits = 5
            SKF = KFold(n_splits = n_splits, shuffle = True, random_state=66)
            
            List1 = []
            
            for train_index, val_index in SKF.split(X):
                
                combined = np.vstack((y[val_index],val_index))
                list0 = combined.tolist()
                List1.append(list0)
            
            DF = pd.DataFrame(columns = ['label', 'ind', 'groups'])
            for n in range(5):
                dff = pd.DataFrame(columns = ['label', 'ind', 'groups'])
                dff = pd.DataFrame(np.asarray(List1[n]).T, columns = ['label', 'ind'])
                dff['groups'] = pd.DataFrame([n]*np.array(List1[n]).shape[1])
                DF = pd.concat([DF,dff],axis = 0, ignore_index=True)
            
            DF['smiles'] = df4
            DF=DF.sort_values(by=['ind']) 
          
        else:
           
            y = y.astype('int64')
            
            n_splits = 5
            SKF = StratifiedKFold(n_splits=n_splits, shuffle = True,random_state=66)
            
            List1 = []
            
            for train_index, val_index in SKF.split(X,y):
               
                combined = np.vstack((y[val_index],val_index))
                list2 = combined.tolist()
                List1.append(list2)
                
            print(combined.shape)
            print(len(List1))
            
            DF = pd.DataFrame(columns = ['label', 'ind', 'groups'])
           
            for n in range(5):
                dff = pd.DataFrame(columns = ['label', 'ind', 'groups'])
                dff = pd.DataFrame(np.asarray(List1[n]).T, columns = ['label', 'ind'])
                dff['groups'] = pd.DataFrame([n]*np.array(List1[n]).shape[1])
                DF = pd.concat([DF,dff],axis = 0, ignore_index=True)
            DF['smiles'] = df4
            DF=DF.sort_values(by=['ind']) 
            
    return X, DF    
    
   
