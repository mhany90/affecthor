from att import AttentionWithContext
from att2 import Attention
from collections import OrderedDict
from collections import defaultdict
from keras.initializers import lecun_normal
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, AveragePooling1D, LSTM, Conv1D, MaxPooling1D, Flatten, Embedding, GRU, TimeDistributed, Reshape,  Bidirectional, Permute, merge, concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from scipy.stats import pearsonr
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
import random
from keras.initializers import TruncatedNormal, glorot_uniform, glorot_normal
import csv
import codecs


def read_shuffle(f):
    data = pd.read_csv(f, header=None)
    return data


def read_embeds(efile, word_index):    
    embedding_index = {}
    f = open(efile, "r")
    for line in f:
        values = line.split("\t")
        embedding_dim = len(values)-1
        word = values[-1].strip()
        coefs = np.asarray(values[:-1], dtype='float32')
        embedding_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, embedding_dim

def pearson(true, pred):
    return pearsonr(true, pred)[0]

init = 'TruncatedNormal'

def make_dnn(input_shape1, input_shape2):
    input1 = Input(shape=input_shape1)
    input2 = Input(shape=input_shape2)
    input  = concatenate([input1, input2])

    x = Dense(5000, activation="relu", kernel_initializer = init )(input)
    #x2 = Dense(3000, activation="relu", kernel_initializer = init )(input2)

    #x  = concatenate([x1, x2])
    x = Dropout(0.2)(x)
    x = Dense(1500, activation="relu", kernel_initializer = init)(x)
    x = Dropout(0.2)(x)
    x = Dense(750, activation="relu", kernel_initializer = init)(x)
    #x = Dropout(0.2)(x)
    x = Dense(350, activation="relu", kernel_initializer = init)(x)
  #  x = Dropout(0.2)(x)

    #separate 
    x1 = Dense(150, activation="relu", kernel_initializer = init)(x)
    x2 = Dense(150, activation="relu", kernel_initializer = init)(x)

    x1 = Dense(70, activation="relu", kernel_initializer = init)(x1)
    x2 = Dense(70, activation="relu", kernel_initializer = init)(x2)

    x1 = Dense(35, activation="relu", kernel_initializer = init)(x1)
    x2 = Dense(35, activation="relu", kernel_initializer = init)(x2)

    preds1 = Dense(1, activation="sigmoid", kernel_initializer = init)(x1)

    preds2 = Dense(1, activation="sigmoid", kernel_initializer = init)(x2)

    return input1, input2, x, preds1, preds2


def cross_validate(feat_file1, feat_file2,  dev_feat_file, test_feat_file,  n_folds, CV):
    print(feat_file1, feat_file2)
    data1 = read_shuffle(feat_file1)
    data2 = read_shuffle(feat_file2)


    kf = KFold(n_splits=n_folds)
    corrs_dnn1, corrs_dnn2 = [], []

    #sent and lex train data
    #combine dev and train sent.lex
    #data = data.append(dev_data, ignore_index=True)
    #char and seq train data (remove test data)

    #CV on the dev and train
    for train, test in kf.split(data1):
        print("index test: " ,test[0],":",test[-1])
        train_data1 = data1.iloc[train,:]
        test_data1 = data1.iloc[test,:]
        train_data2 = data2.iloc[train,:]
        test_data2 = data2.iloc[test,:]
        #cv train
        feat_train_x1 = train_data1.drop(train_data1.columns[0], axis=1).values
        train_y1 = train_data1.iloc[:,0].values
        feat_train_x2 = train_data2.drop(train_data2.columns[0], axis=1).values
        train_y2 = train_data2.iloc[:,0].values
        #cv test
        feat_test_x1 = test_data1.drop(test_data1.columns[0], axis=1).values
        test_y1 = test_data1.iloc[:,0].values
        feat_test_x2 = test_data2.drop(test_data2.columns[0], axis=1).values
        test_y2 = test_data2.iloc[:,0].values
        #optim.
        adam = optimizers.Adam(lr = 0.001)
        #dense
        dense_input1, dense_input2, dnn, dnn_preds1, dnn_preds2  = make_dnn((feat_train_x1.shape[1],), (feat_train_x2.shape[1],))
        model_dnn = Model(inputs=[dense_input1, dense_input2], outputs=[dnn_preds1, dnn_preds2])
        model_dnn.compile(loss='mse',
                      optimizer=adam)

        if CV:
            #fit all
            model_dnn.fit([feat_train_x1,feat_train_x2], [train_y1, train_y2], epochs=16, batch_size=10)
            #pred all
            preds_dnn1, preds_dnn2  = model_dnn.predict([feat_test_x1, feat_test_x2])
            corr_dnn1 = pearson(test_y1, preds_dnn1.flatten())
            corr_dnn2 = pearson(test_y2, preds_dnn2.flatten())

            print("Pearson correlation for fold DNN1: ", corr_dnn1)
            print("Pearson correlation for fold DNN2: ", corr_dnn2)

            corrs_dnn1.append(corr_dnn1)
            corrs_dnn2.append(corr_dnn2)


    if CV:
        #mean_dnn1 = float(np.mean(corrs_dnn1))
        #print("Pearson correlation for fold dnn1:", corr_dnn1)
        #print("Pearson correlation for fold dnn2:", corr_dnn2)

        print("Average Pearson correlation DNN1:", np.mean(corrs_dnn1))
        print("Average Pearson correlation DNN2:", np.mean(corrs_dnn2))

EFILE = "/data/s3094723/embeddings/en/w2v.twitter.edinburgh10M.400d.csv"
feat_file1 = "/data/s3094723/extra_features/EI-reg/V-reg-En-valence-train.tok.sent.lex"
feat_file2 = "/data/s3094723/extra_features/EI-reg/EI-reg-En-anger-train.tok.sent.lex.short"

dev_feat_file = '/data/s3094723/extra_features/EI-reg/V-reg-En-valence-dev.tok.sent.lex'
test_feat_file = '/data/s3094723/extra_features/EI-reg/V-reg-En-valence-test.tok.sent.lex'

CV = True

cross_validate(feat_file1, feat_file2, dev_feat_file, test_feat_file, 5, CV)

