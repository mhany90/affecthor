import numpy as np
import pandas as pd
import keras
from collections import defaultdict
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Bidirectional
from keras.layers import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing import text, sequence
from keras.utils import np_utils
from keras.preprocessing import sequence
from scipy.stats import pearsonr

def read_files(train,test):
    train = pd.read_table(train, header=None)
    test = pd.read_table(test, header=None)

    train_x, train_y, test_x, test_y = train[1], train[3], test[1], test[3]

    return train_x, train_y, test_x, test_y

def read_features(train, test):
    train = pd.read_csv(train, header=None)
    test = pd.read_csv(test, header=None)

    return train.iloc[:,2:], test.iloc[:,2:]

def get_avg_sent_len(train_x):
    lens = 0
    for x in train_x:
        lens = lens + len(x.split())

    return int(lens/len(train_x))

def integerize(train_x, test_x):
    lex = defaultdict()
    padding = get_avg_sent_len(train_x)
    ints_train = np.zeros(shape = (len(train_x), padding))
    ints_test = np.zeros(shape = (len(test_x), padding))

    for n, tweet in enumerate(train_x):
        ints = np.zeros((padding), dtype = "int32")
        tweet = tweet.split()
        for word in tweet:
            if (tweet.index(word) + 1) >= padding:
                break
            if lex.get(word.lower()) == None:
               lex[word.lower()] = len(lex.keys())+1
            ints[tweet.index(word)] = lex[word.lower()]
        ints_train[n] = ints

    for n, tweet in enumerate(test_x):
        ints = np.zeros((padding), dtype="int32")
        tweet = tweet.split()
        for word in tweet:
            if (tweet.index(word) + 1) >= padding:
                break
            try:
                ints[tweet.index(word)] = lex[word.lower()]
            except KeyError:
                ints[tweet.index(word)] = len(set(lex.keys()))+1
        ints_test[n] = ints

    return ints_train, ints_test, lex

def pearson(true, pred):
    return pearsonr(true, pred)

train_file = "../../data/EI-reg-English-Train/EI-reg-en_anger_train.txt"
test_file = "../../data/dev/EI-reg-En-anger-dev.txt"
train_feature_file= "EI-reg-En-anger-train.vectors.without.random.train.csv"
test_feature_file= "EI-reg-En-anger-train.vectors.without.random.test.csv"

train_x, train_y, test_x, test_y = read_files(train_file, test_file)
train_x, test_x, lex = integerize(train_x, test_x)
train_features, test_features = read_features(train_feature_file, test_feature_file)

print(train_x.shape)
print(train_x)


model = Sequential()
#model.add(Embedding(input_dim=len(word2idx), output_dim=embedding_vector_length, input_length=max_sent_length, mask_zero=True))
print("=================================================================")
print("training basic LSTM...")
print("=================================================================")
#model.add(Embedding(input_dim=len(lex.keys()), output_dim=100, input_length=train_x.shape[1], mask_zero=True))
model.add(LSTM(128, input_shape=(train_x.shape[1])))#, 
                        #mask_zero=True)))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=["accuracy"])

model.fit(train_x, train_y, batch_size=16, epochs=100)
print(model.summary())
#pred = model.predict_classes(test_x)
#print(pred)
#print(pearsonr(test_y, pred))
#score = model.evaluate(test_x, test_y, batch_size=16)

#print(score)
