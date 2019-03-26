# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:31:35 2015

@author: gferry
"""

import pandas as pd
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import scale


# entrée => dataFrame + liste champs à vectoriser + seuil
# sortie => DataFrame vectorisé
# !!! NaN




def vectorize(DataFrame, cols, thres):
    mat = pd.DataFrame(DataFrame)
    nrows = len(mat)
    newmat = pd.DataFrame(dtype=int)    
    
    for field in cols:
        m = np.array((mat[field].value_counts()/nrows).reset_index())        
        m = np.array(filter(lambda row: row[1]>thres, m))        

        for e in m:
            newmat[field + '|' + str(e[0])] = mat[field].apply(lambda row: 1 if(row==e[0]) else 0)
        
        if float(mat[field].isnull().sum())/nrows>thres:
            newmat[field + '|NaN'] = mat[field].isnull().astype(int)
        
    print newmat.sum()     
    return newmat



def kpca_vector(DataFrame,cols,gamma,n_comp=3,thres=0.001):
    mat = pd.DataFrame(DataFrame)
    mat = mat[cols]
    vector = vectorize(mat,cols,thres)
    mat = pd.concat([mat,vector],axis=1)
    mat.drop(cols,axis=1,inplace=True)
        
    kern = scale(np.array(mat,dtype=np.float))
    kpca = RBFSampler(n_components=n_comp, gamma=gamma)
    kern = kpca.fit_transform(kern)
    mat.drop(mat.columns,axis=1,inplace=True)
    cols = ['kpca'+ str(i) for i in range(n_comp)]
    for c in cols:
        mat[c]=np.zeros(len(mat))
    
    mat[cols]=kern
    return mat
    
