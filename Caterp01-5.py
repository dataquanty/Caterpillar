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
from sklearn.svm import SVR
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


mat = pd.read_csv('data.csv',parse_dates=['quote_date'])

#mat.drop('tube_assembly_id',inplace=True,axis=1)
mat['year']=mat['quote_date'].apply(lambda row: row.year)
mat['month']=mat['quote_date'].apply(lambda row: row.month)
mat.drop('quote_date',inplace=True,axis=1)
mat['quantity_tr']=np.sqrt(np.log(mat['quantity']))



cols = ['supplier']
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
        mat[c]=pd.cut(mat[c],bins=128,labels=False)
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
Y = mat[np.isnan(mat['id'])][['tube_assembly_id','cost']]
Y['cost'] = np.log1p(Y['cost'])
Xtest = mat[~np.isnan(mat['id'])].drop(colRemoved,axis=1)

#X.drop('tube_assembly_id',inplace=True,axis=1)
#Xtest.drop('tube_assembly_id',inplace=True,axis=1)
#Y.drop('tube_assembly_id',inplace=True,axis=1)


X, Y = shuffle(X,Y)
lstTubes = X['tube_assembly_id'].drop_duplicates()
lstTubesTrain = pd.DataFrame(lstTubes[:int(lstTubes.shape[0]*0.85)])
lstTubesTrain['train']=1

X = X.merge(lstTubesTrain,how='left',left_on='tube_assembly_id',right_on='tube_assembly_id')
Y = Y.merge(lstTubesTrain,how='left',left_on='tube_assembly_id',right_on='tube_assembly_id')

X.drop('tube_assembly_id',inplace=True,axis=1)
Y.drop('tube_assembly_id',inplace=True,axis=1)
Xtest.drop('tube_assembly_id',inplace=True,axis=1)

scaler = StandardScaler()
X_train,y_train = scaler.fit_transform(X[X['train']==1].drop('train',axis=1)) , np.array(Y[Y['train']==1].drop('train',axis=1)).ravel()
X_test, y_test = scaler.fit_transform(np.array(X[X['train']!=1].drop('train',axis=1))) , np.array(Y[Y['train']!=1].drop('train',axis=1))

"""
offset = int(X.shape[0] * 0.8)
X_train, y_train = X[:offset], Y[:offset]
X_test, y_test = X[offset:], Y[offset:]
"""

clf = LinearRegression()
clf.fit(X_train,y_train)
clf.coef_

for i in range(X_train.shape[1]):
    X_train[i]=X_train[i]*clf.coef_[i]
    X_test[i]=X_test[i]*clf.coef_[i]





def report(grid_scores, n_top=5):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

paramDist = {'kernel': ['rbf'],
             'C':[40,80,120,200]}


clf = SVR(kernel='rbf',C=40,epsilon=0.8,gamma=0.005)
grid_search = GridSearchCV(clf,param_grid=paramDist,n_jobs=4,scoring='mean_squared_error')

grid_search.fit(X_train, y_train)


scoresGrid = grid_search.grid_scores_
print grid_search.best_score_
print grid_search.best_estimator_
report(grid_search.grid_scores_)


clf.fit(X_train,y_train)
print np.sqrt(mean_squared_error(y_test,grid_search.best_estimator_.predict(X_test)))
print np.sqrt(mean_squared_error(y_test,clf.predict(X_test)))


test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
    #test_score[i] = clf.loss_(y_test, y_pred)
    test_score[i] = np.sqrt(mean_squared_error(y_test, y_pred))

plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')




cols = np.array(X_train.columns)
importance = clf.feature_importances_
featImport = pd.concat((pd.DataFrame(cols),pd.DataFrame(importance)),axis=1)
featImport.columns=['f','v']
featImport.sort('v',ascending=False,inplace=True)
featImport.set_index('f',inplace=True)
featImport.plot(kind='bar')
plt.subplots_adjust(bottom = 0.3)
plt.show()


GBM = GradientBoostingRegressor(n_estimators=500, learning_rate= 0.01, max_depth=8, subsample=0.9,min_samples_split=1)
Rforest = RandomForestRegressor(n_estimators=200, max_depth=30)

Bagging = BaggingRegressor(base_estimator=Rforest,n_estimators=4,random_state=42,n_jobs=2,bootstrap=True,max_samples=0.8)
Bagging = BaggingRegressor(base_estimator=GBM,n_estimators=10,random_state=23,n_jobs=4,bootstrap=True,max_samples=0.8)
Bagging.fit(X,Y)



Ytest = np.expm1(Bagging.predict(Xtest))

Ytest = np.expm1(clf.predict(Xtest))


pred = np.vstack([np.array(mat[~np.isnan(mat['id'])]['id'],dtype=np.int16),Ytest]).T
pred = pd.DataFrame(pred)
pred.columns = ['id','cost']
pred['id']=pred['id'].astype(np.int16)
pred.to_csv('pred03-GBM.csv',index=False)
