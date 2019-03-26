# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:39:01 2015

@author: dataquanty
"""

import numpy as np
import pandas as pd
from vectorizer import vectorize
import scipy
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from operator import itemgetter
import matplotlib.pyplot as plt

mat = pd.read_csv('data.csv',parse_dates=['quote_date'])

mat.drop('tube_assembly_id',inplace=True,axis=1)
mat['year']=mat['quote_date'].apply(lambda row: row.year)
mat['month']=mat['quote_date'].apply(lambda row: row.month)
mat.drop('quote_date',inplace=True,axis=1)



cols = ['material_id','supplier']
vectorSpec = vectorize(mat,cols,0.001)
mat = pd.concat([mat,vectorSpec],axis=1)
mat.drop(cols,axis=1,inplace=True)

cols = ['component_type_id','weight_tot','component_id','n_components']
mat[cols]=np.nan_to_num(np.array(mat[cols]))

colRemoved = ['cost','id']
cols = mat.drop(colRemoved,axis=1).columns

for c in cols:
    try:
        mat[c]=pd.cut(mat[c],bins=32,labels=False)
    except:
        print c
        
for c in cols:
    try:
        mat[c]=mat[c].astype(np.int8)
    except:
        print c

cols = ['end_a','end_x']
for c in cols:
    mat[c]=mat[c].apply(lambda row: hash(row))


X = mat[np.isnan(mat['id'])].drop(colRemoved,axis=1)
Y = mat[np.isnan(mat['id'])]['cost']
Y = np.log1p(Y)
Xtest = mat[~np.isnan(mat['id'])].drop(colRemoved,axis=1)


def report(grid_scores, n_top=5):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


paramDist = {'n_estimators': [50],
#             'criterion': ['entropy'],
             'max_features':['auto'],
             'max_depth': scipy.stats.expon(scale=10),
             'min_samples_split':[2],
             'min_samples_leaf':scipy.stats.expon(scale=1)}

             
paramDist = {'n_estimators': scipy.stats.randint(20,50),
             'learning_rate': [0.1],
             'max_features':['auto'],
             'max_depth': scipy.stats.expon(scale=7),
#             'min_samples_split':scipy.stats.expon(scale=2),
             'min_samples_leaf':[1]}



Rforest = RandomForestRegressor()
GBM = GradientBoostingRegressor()
grid_search = RandomizedSearchCV(Rforest,cv=3,param_distributions=paramDist,n_iter=40,n_jobs=4,scoring='mean_squared_error')
grid_search = RandomizedSearchCV(GBM,cv=5,param_distributions=paramDist,n_iter=12,n_jobs=4,scoring='mean_squared_error')


grid_search.fit(X, Y)

scoresGrid = grid_search.grid_scores_
print grid_search.best_score_
print grid_search.best_estimator_
report(grid_search.grid_scores_)


cols = np.array(mat.drop(colRemoved,axis=1).columns)
importance = grid_search.best_estimator_.feature_importances_
featImport = pd.concat((pd.DataFrame(cols),pd.DataFrame(importance)),axis=1)
featImport.columns=['f','v']
featImport.sort('v',ascending=False,inplace=True)
featImport.set_index('f',inplace=True)
featImport.plot(kind='bar')
plt.subplots_adjust(bottom = 0.3)
plt.show()


