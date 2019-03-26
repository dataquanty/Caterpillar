from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD
from keras.optimizers import Adagrad
from keras.regularizers import l2

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error
import matplotlib.pyplot as plt
from vectorizer import vectorize
from sklearn.utils import shuffle



'''
    This demonstrates how to reach a score of 0.4890 (local validation)
    on the Kaggle Otto challenge, with a deep net using Keras.
    Compatible Python 2.7-3.4 
    Recommended to run on GPU: 
        Command: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python kaggle_otto_nn.py
        On EC2 g2.2xlarge instance: 19s/epoch. 6-7 minutes total training time.
    Best validation score at epoch 21: 0.4881 
    Try it at home:
        - with/without BatchNormalization (BatchNormalization helps!)
        - with ReLU or with PReLU (PReLU helps!)
        - with smaller layers, largers layers
        - with more layers, less layers
        - with different optimizers (SGD+momentum+decay is probably better than Adam!)
    Get the data from Kaggle: https://www.kaggle.com/c/otto-group-product-classification-challenge/data
'''

#np.random.seed(1337) # for reproducibility

def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X) # https://youtu.be/uyUXoap67N8
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def make_submission(y_prob, ids, encoder, fname):
    with open(fname, 'w') as f:
        f.write('id,')
        f.write(','.join([str(i) for i in encoder.classes_]))
        f.write('\n')
        for i, probs in zip(ids, y_prob):
            probas = ','.join([i] + [str(p) for p in probs.tolist()])
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(fname))

def grow1(df,n_times):
    a = df[df[:,1]==1].copy()
    for i in range(n_times):
        df = np.vstack((df,a))
    
    return df

def growWeights(df,weightsCol):
    m = df[df[weightsCol]>1].copy()
    m = m.reset_index()
    for i in range(len(m)):
        n_times=int(m.iloc[i][weightsCol])
        a = m.iloc[i]
        for i in range(n_times-1):
            df = df.append(a)
    
    return df



print("Loading data...")


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







dims= mat.drop(colRemoved,axis=1).shape[1]-1

print("Building model...")


NN = 64
dropout = 0

def build_model():
    """Build model"""
    
    
    model = Sequential()
    
    #model.add(Dense(dims, 30, W_regularizer = l1(.01)))
    
    model.add(Dense(dims, 64, init='he_normal'))
    model.add(Activation('relu'))
    #model.add(PReLU((32,)))
    model.add(Dropout(dropout))
    """
    model.add(Dense(512, NN, init='glorot_uniform'))
    model.add(Activation('tanh'))
    model.add(PReLU((NN,)))
    model.add(BatchNormalization((NN,)))
    model.add(Dropout(dropout))
    """
    model.add(Dense(64, 64, init='he_normal'))
    model.add(Activation('relu'))
    #model.add(PReLU((32,)))
    model.add(Dropout(dropout))
    
    #model.add(Dense(5, 5, W_regularizer = l2(.5)))
    #model.add(Dense(20, 20, W_regularizer = l2(.01)))
    
    model.add(Dense(64, 1, init='he_normal'))
    
    model.compile(loss='mse', optimizer=sgd)

    
    return model



#sgd = SGD(lr=100*1e-3, decay=1e-5, momentum=0.9, nesterov=True)
sgd = SGD(lr=1*1e-5, decay=1e-12, momentum=0.9, nesterov=True)
adagrad = Adagrad(lr=0.1, epsilon=1e-6)



print ("load weights")
#model.load_weights('weightsNN')

nbTries = 1
rocHist = np.zeros(nbTries)

for i in range(nbTries):
    print("Transfo data...")
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
    
    scaler = StandardScaler()
    #X = scaler.fit_transform(np.nan_to_num(X))
    #Xtest = scaler.fit_transform(np.nan_to_num(Xtest))
    
    X_train,y_train = scaler.fit_transform(X[X['train']==1].drop('train',axis=1)) , np.array(Y[Y['train']==1].drop('train',axis=1))
    X_test, y_test = scaler.fit_transform(np.array(X[X['train']!=1].drop('train',axis=1))) , np.array(Y[Y['train']!=1].drop('train',axis=1))
    
    
    
    model=build_model()
    print ("Training ... ")
    hist = model.fit(X_train, y_train, nb_epoch=500, batch_size=256,
                     validation_data=(X_test, y_test),show_accuracy=False,verbose=2,shuffle=True)
    
    pred = model.predict(X_test,verbose=2)
    rocHist[i] = np.sqrt(mean_squared_error(y_test,pred))
    print (rocHist[i], "ROC")
    plt.figure()
    plt.subplot(1,1,1)
    plt.title('Loss')
    plt.plot(hist['loss'],'b-',label='Training Loss')
    plt.plot(hist['val_loss'], 'r-',label = 'Validation Loss')
    plt.ylim( (0, 2) )
    plt.legend(loc='upper right')
    plt.show()

print ( np.average(rocHist), "Average ROC ")

"""
print("Training final model ...")
X,Y = shuffle(X,Y)
model = build_model()
hist = model.fit(X, Y, nb_epoch=11, batch_size=64,verbose=2)

pred = model.predict(X,verbose=2)
print("Final ROC", np.sqrt(mean_squared_error(Y,pred)))

plt.figure()
plt.subplot(1,2,2)
plt.title('Loss')
plt.plot(hist['loss'],'b-',label='Training Loss')
plt.ylim( (0, 0.5) )
plt.legend(loc='upper right')
plt.show()
"""

print("Generating submission...")


#proba = model.predict_proba(XTest,verbose=2)
#res = pd.DataFrame(mat['Id'][trainLen:].astype(int))
#res['WnvPresent']=proba[:,1]
#res.to_csv('sub26-NN.csv',index=False,float_format="%.12f")


#pd.DataFrame(proba[:,1]).to_csv('submission.csv',index=False)
#print (np.sum(proba[:,1]>0.5))
#make_submission(proba, ids, encoder, fname='submission.csv')







"""
sgd = SGD(lr=1e-2, decay=1e-3, momentum=0.8, nesterov=True)
model.fit(X, y, nb_epoch=100, batch_size=128, validation_split=0.2,show_accuracy=True,verbose=2,shuffle=True)
"""

"""
NN = 650
sgd = SGD(lr=1e-2, decay=1e-7, momentum=0.9, nesterov=True)
model.fit(X, y, nb_epoch=200, batch_size=1010, validation_split=0.2,show_accuracy=True,verbose=2,shuffle=True)
"""



