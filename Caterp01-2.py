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
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error



mat = pd.read_csv('data.csv',parse_dates=['quote_date'])

mat.drop('tube_assembly_id',inplace=True,axis=1)
mat['year']=mat['quote_date'].apply(lambda row: row.year)
mat['month']=mat['quote_date'].apply(lambda row: row.month)
mat.drop('quote_date',inplace=True,axis=1)
mat['quantity_tr']=np.sqrt(np.log(mat['quantity']))



cols = ['material_id','supplier']
vectorSpec = vectorize(mat,cols,0.001)
mat = pd.concat([mat,vectorSpec],axis=1)
mat.drop(cols,axis=1,inplace=True)

colRemoved = ['cost','id']
cols = mat.drop(colRemoved,axis=1).columns
mat[cols]=mat[cols].fillna(0)
#mat[cols]=np.nan_to_num(np.array(mat[cols]))

"""
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
"""
cols = ['end_a','end_x']
for c in cols:
    mat[c]=mat[c].apply(lambda row: hash(row))


X = mat[np.isnan(mat['id'])].drop(colRemoved,axis=1)
Y = mat[np.isnan(mat['id'])]['cost']
Y = np.log1p(Y)
Xtest = mat[~np.isnan(mat['id'])].drop(colRemoved,axis=1)




X, Y = shuffle(X,Y)


offset = int(X.shape[0] * 0.8)
X_train, y_train = X[:offset], Y[:offset]
X_test, y_test = X[offset:], Y[offset:]


n_est = 200

#test_score_roc = np.zeros((n_est,), dtype=np.float64)
test_score = np.zeros((n_est,), dtype=np.float64)
#train_score = np.zeros((n_est,), dtype=np.float64)


params = {'n_estimators': n_est, 
          'max_depth': 12, 
          'min_samples_split': 1,
          'learning_rate': 0.05, 
          'subsample':0.9,
#          'max_features':'sqrt'
          }


clf = GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)

print np.sqrt(mean_squared_error(y_test,clf.predict(X_test)))

"""
train_score[i:(i+1)*10] = (mean_squared_error(y_train,clf.predict(X_train)))
test_score_roc[i:(i+1)*10] = (mean_squared_error(y_test, clf.predict(X_test)))
print("iteration ",i*10)
    
np.sqrt(mean_squared_error(Y,clf.predict(X)))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Area under ROC curve')
plt.plot(np.arange(n_est) + 1, train_score, 'b-',
         label='Training Set ROC')
plt.plot(np.arange(n_est) + 1, test_score_roc, 'r-',
         label='Test Set ROC')
plt.legend(loc='lower right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
"""

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')




cols = np.array(mat.drop(colRemoved,axis=1).columns)
importance = clf.feature_importances_
featImport = pd.concat((pd.DataFrame(cols),pd.DataFrame(importance)),axis=1)
featImport.columns=['f','v']
featImport.sort('v',ascending=False,inplace=True)
featImport.set_index('f',inplace=True)
featImport.plot(kind='bar')
plt.subplots_adjust(bottom = 0.3)
plt.show()

