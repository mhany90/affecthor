import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python import SKCompat
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer
from numpy import mean
import os
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold




def read_and_shuffle(f):
    df = pd.read_csv(f, header=None, skiprows=0)
    df = shuffle(df)
    #x = df.iloc[0:]
    #y = df[0]
    #return x.iloc[:-200], y.iloc[:-200], x.iloc[-200:], y.iloc[-200:]
    return df

def read_and_shuffle_cv(f):
    df = pd.read_csv(f, header=None, skiprows=0)
    df = shuffle(df)
    x = df.iloc[:,1:]
    y = df[0]
    return x, y


def pearson(y_pred, y_true):
    p = -1* pearsonr(y_pred, y_true)[0]
    p = convert_to_tensor(
    p,
    dtype=None,
    name=None,
    preferred_dtype=None)
    return p


def l1_norm(prediction, target, inputs):
    return tf.reduce_sum(tf.abs(prediction - target), name='l1')


#model
network = input_data(shape=[None, 4139])
network = fully_connected(network, 3000 , activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, 1500, activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, 750, activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, 350, activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, 150, activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, 75, activation='relu')
network = tflearn.fully_connected(network, 1, activation='sigmoid')
network = tflearn.regression(network, optimizer='adam', learning_rate=0.001,
                         loss='mean_square', metric='R2' )




#DIR = '/data/s3094723/extra_features/'
#files = [DIR+f for f in os.listdir(DIR) if "sadness" in f]
files = ['/data/s3094723/extra_features/EI-reg-En-joy-train.txt.sent.lex']
#test_file = '/home/s3094723/SEMEVAL/affecthor/data/EI-reg/En/dev/'	



for f in files:
    print(f)
    data = read_and_shuffle(f)
    #CORRELATIONS
    correlations_avg = []
    correlations_dnn = []
    correlations_gbt = []
    #SPLIT INTO KFOLDS
    kf = KFold(n_splits=10)
    for train, test in kf.split(data):
        print("index test: " ,test)
        print("index train:", train)
        #TEST AND TRAIN INDICIES FOR FOLD
        train_data = data.iloc[train,:]
        test_data = data.iloc[test,:]
        #TAKE OUT FIRST COL (LABELS)
        train_y = train_data.iloc[:,0]
        test_y = test_data.iloc[:,0]
        train_data = train_data.drop(train_data.columns[0], axis=1)
        test_data = test_data.drop(test_data.columns[0], axis=1)
        #CHECK SHAPE
        print("shape: ", test_data.shape, train_data.shape)
        #GBT TAKES A ROW, DNN A COL
        train_y_gbt = train_y.values   
        train_y = np.reshape(train_y.values,(-1,1))
        #TRAIN MODELS
        #DNN
        model =  tflearn.DNN(network, tensorboard_verbose=0)
        model.fit(train_data.values, train_y,  show_metric = True, batch_size=10)
        #GBT
        regr = GradientBoostingRegressor(max_depth=3,n_estimators=450, learning_rate = 0.05,subsample=0.9, max_leaf_nodes=37000)
        regr.fit(train_data.values, train_y_gbt)
        #PREDICT
        predictions_gbt = regr.predict(test_data.values)
        predictions_dnn = model.predict(test_data.values)
        predictions_dnn = [item for sublist in predictions_dnn for item in sublist]
        #avg
        predictions = np.mean([predictions_gbt, predictions_dnn], axis = 0)
        #PREDICTIONS AND LABELS
        for p, t in zip(predictions, test_y.values):
            print('Test predictions: {}, Truth: {}'.format(p, t))

        #CORRELATIONS OF EACH MODEL AND AVG. CORR.  
        corr_avg = pearsonr(predictions, test_y.values)
        print("Corr. Avg: ", corr_avg[0])
        correlations_avg.append(corr_avg[0])
        #DNN CORR.
        corr_dnn = pearsonr(predictions_dnn, test_y.values)
        print("Corr. DNN: ", corr_dnn[0])
        correlations_dnn.append(corr_dnn[0])
        #GBT CORR.
        corr_gbt = pearsonr(predictions_gbt, test_y.values)
        print("Corr. GBT: ", corr_gbt[0])
       	correlations_gbt.append(corr_dnn[0])

print("Mean Corr. of Avgeraged models: ", np.mean(correlations_avg))
print("Mean Corr. of DNN: ", np.mean(correlations_dnn))
print("Mean Corr. of GBT: ", np.mean(correlations_gbt))

