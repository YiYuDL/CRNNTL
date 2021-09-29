# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 23:59:43 2021

@author: yuyi6
"""
import pandas as pd
from rdkit import Chem
import numpy as np
from cddd.inference import InferenceModel

def aug(X,DF):

    le = len(X)
    DF = DF 
    groups = DF['groups'].to_numpy()
    smiles = DF['smiles']
    y = DF['label'].to_numpy()
    
    aa = []
    a2 = []
    a3 = []
    
    for e in range(le):
        aa.append(smiles[e])
        
        aa2 = []
        aa2.append(y[e])
        aa2 = aa2*10
        a2 = a2 + aa2
        
        aa3 = []
        aa3.append(groups[e])
        aa3 = aa3*10
        a3 = a3 + aa3
        
        for i in range(9):
            m = Chem.MolFromSmiles(smiles[e])
            ans = list(range(m.GetNumAtoms()))
            np.random.shuffle(ans)
            nm = Chem.RenumberAtoms(m,ans)
            ab = Chem.MolToSmiles(nm, canonical=False, isomericSmiles=True)
            aa.append(ab)
    
    augy = pd.Series(a2)
    auggroups= pd.Series(a3)
    augsmiles = pd.Series(aa)
    
    augDF = pd.DataFrame(columns = ['label', 'groups','smiles'])
    augDF['groups'] = auggroups
    augDF['label'] = augy
    augDF['smiles'] = augsmiles
    augDF = augDF.sample(frac=1)
    aa2 = augDF['smiles'].tolist()
    
    inference_model = InferenceModel()
    augX = inference_model.seq_to_emb(aa2)
    augX = (augX - augX.mean()) / augX.std()

    return augX, augDF

