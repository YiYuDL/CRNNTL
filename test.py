# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 18:45:54 2021

@author: yuyi6
"""
from CRNNTL.preprocessing import data_pre
from CRNNTL.train import CNN_CRNN_train
from CRNNTL.baseline import SVM

def main():
    X_fgfr1, DF_fgfr1 = data_pre('fgfr1')
    CNN_result,CRNN_result = CNN_CRNN_train(X_fgfr1, DF_fgfr1)
    SVM_result = SVM('fgfr1')
    print('CNN','fgfr1',"%0.2f"%(CNN_result))
    print('CRNN','fgfr1',"%0.2f"%(CRNN_result))
    print('baseline SVM',"%0.2f"%(SVM_result))
    
if __name__ == "__main__":
   
    main()