from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python import SKCompat
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from mord import LogisticAT, LAD, LogisticIT
from scipy.stats import pearsonr, spearmanr
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


feature_size = 4096

#model
network = input_data(shape=[None, feature_size])
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
network = tflearn.fully_connected(network, 4, activation='softmax')
network = tflearn.regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy' )




#DIR = '/data/s3094723/extra_features/'
#files = [DIR+f for f in os.listdir(DIR) if "sadness" in f]
#files = ['/data/s3094723/extra_features/EI-reg-En-sadness-train.txt.sent.lex']
#test_file = '/home/s3094723/SEMEVAL/affecthor/data/EI-reg/En/dev/'	
files = ['/home/s3094723/SEMEVAL/affecthor/data/EI-oc/En/train/EI-oc-En-fear-train.tok.sent']
#files = ['/home/s3094723/SEMEVAL/affecthor/data/EI-reg/Ar/train/features/EI-reg-Ar-anger-train.combined.tweets.csv']

for f in files:
    print(f)
    data = read_and_shuffle(f)
    #CORRELATIONS
    correlations_oc3 = []
    correlations_oc2 = []
    correlations_oc1 = []
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
        #oc TAKES A ROW, DNN A COL
        train_y_oc = train_y
        # Transform string labels to one-hot encodings
        encoder = LabelBinarizer()
        train_y = encoder.fit_transform(train_y.values) # Use encoder.classes_ to find mapping of one-hot indices to string labels
        #test_y = encoder.fit_transform(test_y.values)
        print(train_y)
        # Get mapping from labels to classes
        [print('{} is Column: {}'.format(item, num)) for num,item in enumerate(encoder.classes_)]
        train_data = train_data.drop(train_data.columns[0], axis=1)
        test_data = test_data.drop(test_data.columns[0], axis=1)
        #CHECK SHAPE
        print("shape: ", test_data.shape, train_data.shape)
        #train_y = np.reshape(train_y.values,(-1,4))
        #TRAIN MODELS
        #DNN
       # model =  tflearn.DNN(network, tensorboard_verbose=0)
        #model.fit(train_data.values, train_y,  show_metric = True, batch_size=10)
        #oc
        oc1 =  LogisticAT()
        oc2 =  LogisticIT( alpha = 0.1)
        oc3 =  LAD()
        #oc = GradientBoostingClassifier(max_depth=3,n_estimators=350, learning_rate = 0.05,subsample=0.9, max_leaf_nodes=30000)
        oc1.fit(train_data.values, train_y_oc)
        oc2.fit(train_data.values, train_y_oc)
        oc3.fit(train_data.values, train_y_oc)
        #PREDICT
        predictions_oc1 = oc1.predict(test_data.values)
        predictions_oc2 = oc2.predict(test_data.values)
        predictions_oc3 = oc3.predict(test_data.values)

        #predictions_dnn = model.predict(test_data.values)
        #predictions_dnn = [item for sublist in predictions_dnn for item in sublist]
        #avg
        #predictions = np.mean([predictions_oc, predictions_dnn], axis = 0)
        #PREDICTIONS AND LABELS
        #for p, t in zip(predictions, test_y.values):
        #    print('Test predictions: {}, Truth: {}'.format(p, t))

        #CORRELATIONS OF EACH MODEL AND AVG. CORR.  
        #corr_avg = pearsonr(predictions, test_y.values)
        #print("Corr. Avg: ", corr_avg[0])
        #correlations_avg.append(corr_avg[0])
        #DNN CORR.
        #predictions_dnn = np.array(predictions_dnn)
        #predictions_dnn = np.reshape(predictions_dnn, (-1, 4))
        #print(predictions_dnn)
        #print(len(predictions_dnn))
        #predictions_dnn = np.argmax(predictions_dnn, axis = 1)
        #corr_dnn = spearmanr(predictions_dnn, test_y.values)
        #print("Corr. DNN: ", corr_dnn[0])
        #correlations_dnn.append(corr_dnn[0])
        #oc CORR.
        corr_oc1 = spearmanr(predictions_oc1, test_y.values)
        corr_oc2 = spearmanr(predictions_oc2, test_y.values)
        corr_oc3 = spearmanr(predictions_oc3, test_y.values)

        print("Corr.  AT: ", corr_oc1[0])
       	correlations_oc1.append(corr_oc1[0])
        print("Corr.  IT: ", corr_oc2[0])
        correlations_oc2.append(corr_oc2[0])
        print("Corr. Multi: ", corr_oc3[0])
        correlations_oc3.append(corr_oc3[0])

#print("Mean Corr. of Avgeraged models: ", np.mean(correlations_avg))
#print("Mean Corr. of DNN: ", np.mean(correlations_dnn))
print("Mean Corr. of AT: ", np.mean(correlations_oc1))
print("Mean Corr. of IT: ", np.mean(correlations_oc2))
print("Mean Corr. of Multi: ", np.mean(correlations_oc3))

