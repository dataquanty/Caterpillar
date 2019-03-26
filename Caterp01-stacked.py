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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import time, datetime


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


X, Y = shuffle(X,Y)
lstTubes = X['tube_assembly_id'].drop_duplicates()
lstTubesTrain = pd.DataFrame(lstTubes[:int(lstTubes.shape[0]*0.85)])
lstTubesTrain['train']=1

X = X.merge(lstTubesTrain,how='left',left_on='tube_assembly_id',right_on='tube_assembly_id')
Y = Y.merge(lstTubesTrain,how='left',left_on='tube_assembly_id',right_on='tube_assembly_id')

X.drop('tube_assembly_id',inplace=True,axis=1)
Y.drop('tube_assembly_id',inplace=True,axis=1)
Xtest.drop('tube_assembly_id',inplace=True,axis=1)

X_train,y_train = X[X['train']==1].drop('train',axis=1) , np.array(Y[Y['train']==1].drop('train',axis=1)).ravel()
X_test, y_test = np.array(X[X['train']!=1].drop('train',axis=1)) , np.array(Y[Y['train']!=1].drop('train',axis=1)).ravel()

"""
offset = int(X.shape[0] * 0.8)
X_train, y_train = X[:offset], Y[:offset]
X_test, y_test = X[offset:], Y[offset:]
"""



params = {'n_estimators': 500, 
          'max_depth': 8, 
          'min_samples_split': 1,
          'learning_rate': 0.01, 
          'subsample':0.9,
#          'max_features':'sqrt'
          }

clf = GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)
print np.sqrt(mean_squared_error(y_test,clf.predict(X_test)))
Hold_out = clf.predict(X_test)
Ypredict = clf.predict(Xtest)



params = {'n_estimators': 200,
#             'criterion': ['entropy'],
             'max_features':'auto',
             'max_depth': 30,
             'min_samples_split':2,
             'min_samples_leaf':1}


clf = RandomForestRegressor(**params)
clf.fit(X_train, y_train)
print np.sqrt(mean_squared_error(y_test,clf.predict(X_test)))

Hold_out = np.vstack((Hold_out,clf.predict(X_test))).T
Ypredict = np.vstack((Ypredict,clf.predict(Xtest))).T

scaler = StandardScaler()
X_test,X_train,Xtest = scaler.fit_transform(X_test), scaler.fit_transform(X_train) , scaler.fit_transform(Xtest)


clf = SVR(kernel='rbf',C=0.8,epsilon=0.8,gamma=0.005)
clf.fit(X_train, y_train)
print np.sqrt(mean_squared_error(y_test,clf.predict(X_test)))

Hold_out = np.hstack((Hold_out,clf.predict(X_test).reshape(len(X_test),1)))
Ypredict = np.hstack((Ypredict,clf.predict(Xtest).reshape(len(Xtest),1)))


np.savetxt('Hold_out_0.csv',Hold_out)
np.savetxt('HO_Ypredict_0.csv',Ypredict)
np.savetxt('HO_y_test_0.csv',y_test)

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
             
Rforest = RandomForestRegressor()
grid_search = RandomizedSearchCV(Rforest,cv=3,param_distributions=paramDist,n_iter=100,n_jobs=4,scoring='mean_squared_error')


grid_search.fit(Hold_out, y_test)

scoresGrid = grid_search.grid_scores_
print grid_search.best_score_
print grid_search.best_estimator_
report(grid_search.grid_scores_)         

finalpred = np.expm1(grid_search.predict(Ypredict))


pred = np.vstack([np.array(mat[~np.isnan(mat['id'])]['id'],dtype=np.int16),finalpred]).T
pred = pd.DataFrame(pred)
pred.columns = ['id','cost']
pred['id']=pred['id'].astype(np.int16)
ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S') 
pred.to_csv('pred06-stack'+ ts +'.csv',index=False)


"""
p1 = pd.read_csv('pred04-stack.csv')
p2 = pd.read_csv('pred05-stack.csv')

pred = pd.concat([p1['id'],(p1['cost']+p2['cost'])/2],axis=1)
pred.to_csv('pred05-stack05-04.csv',index=False)
"""

"""
clf = grid_search.best_estimator_
cols = np.array(X_train.columns)
cols = np.array([1,2,3])
importance = clf.feature_importances_
featImport = pd.concat((pd.DataFrame(cols),pd.DataFrame(importance)),axis=1)
featImport.columns=['f','v']
featImport.sort('v',ascending=False,inplace=True)
featImport.set_index('f',inplace=True)
featImport.plot(kind='bar')
plt.subplots_adjust(bottom = 0.3)
plt.show()
"""
