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
    #df = shuffle(df)
    x = df.iloc[0:]
    #y = df[0]
    #return x.iloc[:-200], y.iloc[:-200], x.iloc[-200:], y.iloc[-200:]
    return x

def read_and_shuffle_cv(f):
    df = pd.read_csv(f, header=None, skiprows=1)
    df = shuffle(df)
    x = df.iloc[:,1:]
    y = df[0]
    return x, y


#def kfold(x):
#    kf_total = KFold(len(x), n_folds=10, shuffle=True, random_state=4)
#    return kf_total

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
network = fully_connected(network, 300, activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, 150, activation='relu')
network = dropout(network, 0.8)
network = fully_connected(network, 50, activation='relu')
network = tflearn.fully_connected(network, 1, activation='sigmoid')
network = tflearn.regression(network, optimizer='adam', learning_rate=0.001,
                         loss='mean_square', metric='R2' )




#DIR = '/data/s3094723/extra_features/'


#files = [DIR+f for f in os.listdir(DIR) if "sadness" in f]


files = ['/data/s3094723/extra_features/EI-reg-En-joy-train.txt.sent.lex']
#test_file = '/home/s3094723/SEMEVAL/affecthor/data/EI-reg/En/dev/'	



#model = tflearn.DNN(network, tensorboard_verbose=0)
#model.fit(train_x, train_y, validation_set=(test_x, test_y), show_metric=True,
 #         batch_size=8)


for f in files:
    print(f)
#    train_x, train_y, test_x, test_y = read_and_shuffle(f)
    data = read_and_shuffle(f)
    act_data = data.values
    correlations = []
    kf = KFold(n_splits=8, shuffle = True)
    for train, test in kf.split(data):
        train_data = data.iloc[train[0]:]
        test_data = data.iloc[:test[-1]]
        train_y = train_data.iloc[:,0]
        test_y = test_data.iloc[:,0]
        del train_data[0]
        del test_data[0]
        print(len(test_data), len(train_data))   
        train_y = np.reshape(train_y.values,(-1,1))
        model =  tflearn.DNN(network, tensorboard_verbose=0)
        model.fit(train_data.values, train_y,  show_metric = True, batch_size=10)
        predictions = model.predict(test_data.values)
        for p, t in zip(predictions, test_y.values):
            print('Test predictions: {}, Truth: {}'.format(p[0], t))

        predictions = [item for sublist in predictions for item in sublist]

        corr = pearsonr(predictions, test_y.values)
        print(corr[0])
        correlations.append(corr[0])

print("Mean Corr.:", np.mean(correlations))
