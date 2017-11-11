from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import re
from random import randint
import tensorflow as tf


strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def read_files(train,test):
    train = pd.read_table(train, header=None)
    test = pd.read_table(test, header=None)
    train_x, train_y, test_x, test_y = train[1], train[3], test[1], test[3]
    return train_x, train_y, test_x, test_y

def get_embeddings(efile_name):
    efile = open(efile_name, 'r')
    first_400 = [x for x in range(0,400)]
    embeddings = np.loadtxt(efile, dtype=float, usecols = first_400, skiprows= 1, delimiter = '\t')
    efile.close()
    efile = open(efile_name, 'r')
    vocab = np.loadtxt(efile,  usecols = 400, dtype = np.unicode_, skiprows = 1, delimiter = '\t')
    return embeddings, vocab

def vectorize(train_x, test_x, embeddings, vocab, maxSeqLength):
    tweets_vec_train = np.zeros(shape=(len(train_x), maxSeqLength), dtype=np.int32)
    tweets_vec_test = np.zeros(shape=(len(test_x), maxSeqLength), dtype=np.int32)
    for no, tweet in enumerate(train_x):
        wordIndex = 0
        cleanedLine = cleanSentences(tweet)
        split = cleanedLine.split()
        for word in split:
           try:
#               print(word, np.where(vocab==word)[0][0])
               if word in vocab:
                   tweets_vec_train[no][wordIndex] = np.where(vocab==word)[0][0]
           except ValueError:
               tweets_vec_train[no][wordIndex] = np.where(vocab=='mva-unk')[0][0] #Vector for unkown words
           wordIndex = wordIndex + 1
           if wordIndex >= maxSeqLength:
               break
    for no, tweet in enumerate(test_x):
        wordIndex = 0
        cleanedLine = cleanSentences(tweet)
        split = cleanedLine.split()
        for word in split:
           try:
               tweets_vec_test[no][wordIndex] = np.where(vocab==word)[0]
           except ValueError:
               tweets_vec_test[no][wordIndex] = np.where(vocab=='mva-unk')[0] #Vector for unkown words
           wordIndex = wordIndex + 1
           if wordIndex >= maxSeqLength:
               break
    print(tweets_vec_train[1][1])
    return tweets_vec_train, tweets_vec_test


train_file = "EI-reg-En-anger-train.txt"
test_file = "EI-reg-En-anger-dev.txt"
efile_name = "/data/s3094723/embeddings/en/w2v.twitter.edinburgh10M.400d.csv"

train_x, train_y, test_x, test_y = read_files(train_file, test_file)


print(train_y.values)
train_y = np.reshape(train_y.values,(-1,1))
test_y = np.reshape(test_y.values,(-1,1))
#train_y = tf.constant(train_y.values)
#print(train_y)
# Data preprocessing
embeddings, vocab = get_embeddings(efile_name)
print(embeddings.shape)
train_x, test_x = vectorize(train_x, test_x, embeddings, vocab, 50)
train_x = pad_sequences(train_x, maxlen=50, value=0.)
test_x = pad_sequences(test_x, maxlen=50, value=0.)

# Network building
net = tflearn.input_data([None, 50])
net = tflearn.embedding(net, input_dim=len(vocab), output_dim=400, trainable=False, name="EmbeddingLayer")
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 1, activation='sigmoid')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='mean_square')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
model.set_weights(embeddingWeights, embeddings)
model.fit(train_x, train_y, validation_set=(test_x, test_y), show_metric=True,
          batch_size=1)
