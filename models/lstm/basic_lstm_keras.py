import numpy as np
import pandas as pd
import re
import keras
from collections import defaultdict
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Bidirectional
from keras.layers import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing import text, sequence
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.stats import pearsonr

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def read_files(train,test):
    train = pd.read_table(train, header=None)
    test = pd.read_table(test, header=None)

    train_x, train_y, test_x, test_y = train[1], train[3], test[1], test[3]

    return train_x, train_y.values, test_x, test_y.values

def read_features(train, test):
    train = pd.read_csv(train, header=None)
    test = pd.read_csv(test, header=None)

    return train.iloc[:,2:], test.iloc[:,2:]

def get_avg_sent_len(train_x):
    lens = 0
    for x in train_x:
        lens = lens + len(x.split())

    return int(lens/len(train_x))

def get_embeddings(efile_name):
    first_400 = [x for x in range(0,400)]
    embeddings  = pd.read_csv(efile_name, usecols=first_400, dtype=float, sep = '\t', header=None, skiprows=1)
    vocab  = pd.read_csv(efile_name, usecols=[400], dtype=np.unicode_, sep = '\t', header=None, skiprows=1)
    
    return embeddings, vocab

def vectorize(train_x, test_x, embeddings, vocab, maxSeqLength, unknown_token):
    tweets_vec_train = np.zeros(shape=(len(train_x), maxSeqLength, 400), dtype=object)
    tweets_vec_test = np.zeros(shape=(len(test_x), maxSeqLength, 400), dtype=object)
    for no, tweet in enumerate(train_x):
        cleanedLine = cleanSentences(tweet)
        split = cleanedLine.split()
        for i in range(maxSeqLength):
            if i >= len(split):
                tweets_vec_train[no][i] = np.zeros(shape=400, dtype=np.float64)
            else:
                try:
                    tweets_vec_train[no][i] = embeddings.iloc[[np.where(vocab==split[i])[0][0]]].values
                except IndexError:
                    tweets_vec_train[no][i] = embeddings.iloc[[np.where(vocab==unknown_token)[0][0]]].values #Vector for unkown words

    for no, tweet in enumerate(test_x):
        cleanedLine = cleanSentences(tweet)
        split = cleanedLine.split()
        for i in range(maxSeqLength):
            if i >= len(split):
                tweets_vec_test[no][i] = np.zeros(shape=400, dtype=np.float64)
            else:                
                try:
                    tweets_vec_test[no][i] = embeddings.iloc[[np.where(vocab==split[i])[0][0]]].values
                except IndexError:
                    tweets_vec_test[no][i] = embeddings.iloc[[np.where(vocab==unknown_token)[0][0]]].values #Vector for unkown words

    return tweets_vec_train, tweets_vec_test

def pearson(true, pred):
    print(true.shape, pred.shape)
    return pearsonr(true, pred)

train_file = "../../data/EI-reg-English-Train/EI-reg-en_anger_train.txt"
test_file = "../../data/dev/EI-reg-En-anger-dev.txt"
train_feature_file= "EI-reg-En-anger-train.vectors.without.random.train.csv"
test_feature_file= "EI-reg-En-anger-train.vectors.without.random.test.csv"
efile_name = "/data/s3094723/embeddings/en/w2v.twitter.edinburgh10M.400d.csv"
unknown_token = 'mva-unk'

train_x, train_y, test_x, test_y = read_files(train_file, test_file)
maxlen = get_avg_sent_len(train_x)
embeddings, vocab = get_embeddings(efile_name)
train_x, test_x = vectorize(train_x, test_x, embeddings, vocab, maxlen, unknown_token)

model = Sequential()
print("=================================================================")
print("training basic LSTM...")
print("=================================================================")

model.add(LSTM(256, input_shape=(train_x.shape[1], train_x.shape[2])))

model.add(Dropout(0.5))
#model.add(Dense(50, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='mean_squared_error',
              optimizer='adam')

# estimator = KerasRegressor(build_fn=model, nb_epoch=100, batch_size=5, verbose=0)
# estimator.fit(train_x, train_y)
# scores = estimator.predict(test_x)
# print(scores)



model.fit(train_x, train_y, batch_size=8, epochs=40)
print(model.summary())
score = model.evaluate(test_x, test_y, batch_size=8)
pred = model.predict(test_x)
print("\nSCORE")
print(pred)
print(pearson(test_y, pred.flatten()))
