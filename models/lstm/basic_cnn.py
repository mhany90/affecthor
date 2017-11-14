from __future__ import division, print_function, absolute_import
import numpy as np
import pandas as pd
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import re
from random import randint
import tensorflow as tf
from scipy.stats.stats import pearsonr
import codecs
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.core import input_data, dropout, fully_connected

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
    first_400 = [x for x in range(0,400)]
    embeddings  = pd.read_csv(efile_name, usecols=first_400, dtype=float, sep = '\t', header=None, skiprows=1)
    vocab  = pd.read_csv(efile_name, usecols=[400], dtype=np.unicode_, sep = '\t', header=None, skiprows=1)
    return embeddings, vocab

def vectorize(train_x, test_x, embeddings, vocab, maxSeqLength, unknown_token):
    tweets_vec_train = np.zeros(shape=(len(train_x), maxSeqLength), dtype=np.int32)
    tweets_vec_test = np.zeros(shape=(len(test_x), maxSeqLength), dtype=np.int32)
    count = 0
    for no, tweet in enumerate(train_x):
        wordIndex = 0
        cleanedLine = cleanSentences(tweet)
        split = cleanedLine.split()
        for word in split:
           try:
               if word in vocab:
                   tweets_vec_train[no][wordIndex] = np.where(vocab==word)[0]
           except ValueError:
               tweets_vec_train[no][wordIndex] = np.where(vocab==unknown_token)[0] #Vector for unkown words
               print('UNK')
               count = count + 1
           wordIndex = wordIndex + 1
           if wordIndex >= maxSeqLength:
               break
    for no, tweet in enumerate(test_x):
        wordIndex = 0
        cleanedLine = cleanSentences(tweet)
        split = cleanedLine.split()
        for word in split:
           try:
               tweets_vec_test[no][wordIndex] =  np.where(vocab==word)[0]
           except ValueError:
               tweets_vec_test[no][wordIndex] = np.where(vocab==unknown_token)[0]  #Vector for unkown words
           wordIndex = wordIndex + 1
           if wordIndex >= maxSeqLength:
               break
    print(count)
    return tweets_vec_train, tweets_vec_test


def l1_norm(prediction, target, inputs):
    return tf.reduce_sum(tf.abs(prediction - target), name='l1')

unknown_token = 'mva-unk'
train_file = "EI-reg-En-anger-train.txt"
test_file = "EI-reg-En-anger-dev.txt"
efile_name = "/data/s3094723/embeddings/en/w2v.twitter.edinburgh10M.400d.csv"
#efile_name = "/data/s3094723/embeddings/en/400M/w2v.400M.reformated.csv"

train_x, train_y, test_x, test_y = read_files(train_file, test_file)


print(train_y.values)
train_y = np.reshape(train_y.values,(-1,1))
test_y = np.reshape(test_y.values,(-1,1))
#train_y = tf.constant(train_y.values)
print(test_y)
# Data preprocessing
embeddings, vocab = get_embeddings(efile_name)
print(embeddings.shape)
vocab_size, embeddings_dim = embeddings.shape
print(vocab_size, embeddings_dim)
train_x, test_x = vectorize(train_x, test_x, embeddings, vocab, 25, unknown_token)

embeddings = embeddings.as_matrix()
train_x = pad_sequences(train_x, maxlen=25, value=0.)
test_x = pad_sequences(test_x, maxlen=25, value=0.)

batch_size = 8

# net building
net = tflearn.input_data([None, 25])
net = tflearn.embedding(net, input_dim=len(vocab), output_dim=400, trainable=False, name="EmbeddingLayer")
net = tflearn.layers.normalization.batch_normalization(net)
#print('shape: ', net.shape)
#net = tflearn.reshape(net, [None, 25, 400])
print('shape: ', net.shape)
#net = conv_1d(net, 400, 25, padding='valid', activation='relu', regularizer="L2")
branch1 = conv_1d(net, 400, 1, padding='valid', activation='relu', regularizer="L2") #128
branch2 = conv_1d(net, 400, 2, padding='valid', activation='relu', regularizer="L2") #128
branch3 = conv_1d(net, 400, 3, padding='valid', activation='relu', regularizer="L2") #128 [batch_size, new steps1, nb_filters]. padding:"VALID",only ever drops the right-most columns
branch4 = conv_1d(net, 400, 4, padding='valid', activation='relu', regularizer="L2") #128 [batch_size, new steps2, nb_filters]
branch5 = conv_1d(net, 400, 5, padding='valid', activation='relu', regularizer="L2") #128 [batch_size, new steps3, nb_filters]
branch6 = conv_1d(net, 400, 6, padding='valid', activation='relu', regularizer="L2") #128 [batch_size, new steps3, nb_filters] #ADD
branch7 = conv_1d(net, 400, 7, padding='valid', activation='relu', regularizer="L2") #128 [batch_size, new steps3, nb_filters] #ADD
branch8 = conv_1d(net, 400, 7, padding='valid', activation='relu', regularizer="L2") #128 [batch_size, new steps3, nb_filters] #ADD
branch9 = conv_1d(net, 400, 8, padding='valid', activation='relu', regularizer="L2") #128 [batch_size, new steps3, nb_filters] #ADD
branch10 = conv_1d(net,400, 9, padding='valid', activation='relu', regularizer="L2") #128 [batch_size, new steps3, nb_filters] #ADD
net = merge([branch1, branch2, branch3,branch4,branch5,branch6, branch7, branch8,branch9,branch10], mode='concat', axis=1) # merge a list of `Tensor` into a single one.===>[batch_size, new steps1+new step2+new step3, nb_filters]
print('shape: ', net.shape)
net = tf.expand_dims(net, 1)
print('shape: ', net.shape)
net = global_max_pool(net)
net = dropout(net, 0.7)
net = tflearn.fully_connected(net, 200, activation='relu')
net = dropout(net, 0.7)
#net = tflearn.lstm(net, 300, dropout=0.8)
net = tflearn.fully_connected(net, 1, activation='sigmoid')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='mean_square', metric='R2' )
# Training
model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='tflearn_logs/')
embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
embedding_placeholder = tf.placeholder(tf.float32, [vocab_size,embeddings_dim])
embedding_init = embeddingWeights.assign(embedding_placeholder)
#model.set_weights(embeddingWeights, embeddings)
sess = tf.Session()
sess.run(embedding_init, feed_dict={embedding_placeholder: embeddings})
model.fit(train_x, train_y, validation_set=(test_x, test_y), show_metric=True,
          batch_size=8)

score = model.evaluate(test_x, test_y)
print('Test accuarcy: %0.4f%%' % (score[0] * 100))
print(score)
scores = model.predict(test_x)


print('Test scores:', scores)
print('Corr:')
print('labels:', test_y)

scores = [item for sublist in scores for item in sublist]
test_y = [item for sublist in test_y for item in sublist]

corr = pearsonr(scores, test_y)
print(corr[0])

