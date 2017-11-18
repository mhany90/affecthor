import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
from collections import defaultdict
import re
from random import randint


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

def get_avg_sent_len(train_x):
    lens = 0
    for x in train_x:
        lens = lens + len(x.split())
    return int(lens/len(train_x))

def vectorize(train_x, test_x, embeddings, vocab, maxSeqLength):
    tweets_vec_train = np.zeros(shape=(len(train_x), maxSeqLength), dtype='int32')
    tweets_vec_test = np.zeros(shape=(len(test_x), maxSeqLength), dtype='int32')
    for no, tweet in enumerate(train_x):
        wordIndex = 0
        cleanedLine = cleanSentences(tweet)
        split = cleanedLine.split()
        for word in split:
            print(np.where(vocab==word)[0][0])
        for word in split:
            vocabIndex = embeddings[np.where(vocab==word)[0][0]]
            if len(vocabIndex) > 0:
                tweets_vec_train[no][wordIndex] = embeddings[np.where(vocab=='mva-unk')[0][0]] #Vector for unkown words
            else:
                tweets_vec_train[no][wordIndex] = embeddings[np.where(vocab==word)[0][0]]
            wordIndex = wordIndex + 1
            if wordIndex >= maxSeqLength:
                break
    for no, tweet in enumerate(test_x):
        wordIndex = 0
        cleanedLine = cleanSentences(tweet)
        split = cleanedLine.split()
        for word in split:
            vocabIndex = embeddings[np.where(vocab==word)[0][0]]
            if len(vocabIndex) > 0:
                tweets_vec_test[no][wordIndex] = embeddings[np.where(vocab=='mva-unk')[0][0]] #Vector for unkown words
            else:
                tweets_vec_test[no][wordIndex] = embeddings[np.where(vocab==word)[0][0]]
            wordIndex = wordIndex + 1
            if wordIndex >= maxSeqLength:
                break
    return tweets_vec_train, tweets_vec_test


def to_tensor(tweets_vec, labels):
    tweets_vec = tf.constant(tweets_vec, dtype=tf.float32)  # X is a np.array
    labels = tf.constant(train_y, dtype=tf.string)  # y is a np.array
    return tweets_vec, labels


def generate_batches(tweets_vec_train, train_y, batch_size, num_epochs=None):
    x,y = to_tensor(tweets_vec_train, train_y)
    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_size
    feature_batch, label_batch = tf.train.shuffle_batch([x, y], 
                                                        batch_size=batch_size, 
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    #print(label_batch)

    return feature_batch, label_batch

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

train_file = "../../data/EI-reg-English-Train/EI-reg-en_anger_train.txt"
test_file = "../../data/dev/EI-reg-En-anger-dev.txt"
train_feature_file= "EI-reg-En-anger-train.vectors.without.random.train.csv"
test_feature_file= "EI-reg-En-anger-train.vectors.without.random.test.csv"
efile_name = "/data/s3094723/embeddings/en/w2v.twitter.edinburgh10M.400d.csv"

train_x, train_y, test_x, test_y = read_files(train_file, test_file)
maxSentLength = get_avg_sent_len(train_x)
embeddings, vocab = get_embeddings(efile_name)

train_x, test_x = vectorize(train_x, test_x, embeddings, vocab, maxSentLength)
print(train_x)
