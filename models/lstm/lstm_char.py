import pandas as pd
import numpy as np
from collections import defaultdict
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Flatten, Bidirectional, TimeDistributed
from keras.models import Model
from keras import optimizers
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

def get_max_word_len(tweets):
    
    max_len = 0
    for tweet in tweets:
        tweet = tweet.split(" ")
        tweet_max = len(max(tweet, key=len))
        if tweet_max > max_len:
            max_len = tweet_max
            
    return max_len

def proc_chars(tweets, max_seq):
    c2i = defaultdict(lambda: len(c2i))
    max_word = get_max_word_len(tweets)

    chars = np.zeros((tweets.shape[0], max_seq, max_word))
    print(chars.shape)

    for i, tweet in enumerate(tweets):
        tweet = tweet.split(" ")
        for j, word in enumerate(tweet):
            for k, char in enumerate(word):
                chars[i][j][k] = c2i[char]
    
    return chars, c2i, max_word

def read_and_proc(f):
    data = pd.read_table(f, header=None)
    data = shuffle(data)
    tweets = data[1].values

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)

    word_index = tokenizer.word_index
    max_seq = len(max(sequences, key=len))

    padded = pad_sequences(sequences, maxlen=max_seq)
    chars, char_index, max_word = proc_chars(tweets, max_seq)

    labels = data[3].values

    return padded, chars, labels, word_index, char_index, max_seq, max_word

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
    true = Flatten()(true)
    pred = Flatten()(pred)
    return pearsonr(true,pred)[0]

def cross_validate(f, efile, n_folds):
    print(f)
    seqs, chars, labels, word_index, char_index, MAX_SEQ, MAX_WORD = read_and_proc(f)
    embeds, EMBEDDING_DIM = read_embeds(efile, word_index)
    kf = KFold(n_splits=n_folds)
    corrs = []
    for train, test in kf.split(seqs):
        print("index test: " ,test[0],":",test[-1])
        seqs_train_x = seqs[train]
        chars_train_x = chars[train]
        train_y = labels[train]
        seqs_test_x = seqs[test]
        chars_test_x = chars[test]
        test_y = labels[test]
        
        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights = [embeds],
                                    input_length=MAX_SEQ,
                                    trainable = False)

        char_embedding_layer = TimeDistributed(Embedding(len(char_index) + 1,
                                                         output_dim = 32,
                                                         input_length = MAX_WORD))

        sequence_input = Input(shape=(MAX_SEQ,), dtype='int32')
        char_input = Input(shape=(MAX_SEQ, MAX_WORD,), dtype='int32')
        
        embedded_sequences = embedding_layer(sequence_input)
        x = LSTM(256, activation="relu")(embedded_sequences)

        embedded_chars = char_embedding_layer(char_input)
        y = TimeDistributed(Flatten())(embedded_chars)
        y = LSTM(50, activation="relu")(y)
        
        merged = keras.layers.concatenate([x, y], axis=1)

        #x = Conv1D(250, 5, activation='relu')(embedded_sequences)
        #x = MaxPooling1D()(x)
        #x = Flatten()(x)
        #x = Dense(125, activation='relu')(x)
        #x = Dense(50, activation='relu')(merged)
        preds = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[sequence_input, char_input], outputs=preds)
        adam = optimizers.Adam(lr = 0.001)
        model.compile(loss="mse",
                      optimizer=adam)
        
        model.fit([seqs_train_x, chars_train_x], train_y, epochs=10, batch_size=10)

        preds = model.predict([seqs_test_x, chars_test_x]).flatten()
        corr = pearsonr(test_y, preds)[0]
        print("Pearson correlation for fold:", corr)
        corrs.append(corr)
    
    print("Average Pearson correlation:", np.mean(corrs))
    return corrs

cross_validate("../../data/EI-reg/En/train/EI-reg-En-anger-train.tok", 
               "/data/s3273512/en/w2v/w2v.twitter.edinburgh10M.400d.csv", 5)
