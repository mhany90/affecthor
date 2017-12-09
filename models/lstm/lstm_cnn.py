import pandas as pd
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Flatten, Bidirectional, TimeDistributed
from keras.models import Model
from keras import optimizers
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

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
    labels = data[3].values

    return padded, labels, word_index, max_seq

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

def cross_validate(f, efile, n_folds):
    print(f)
    data, labels, word_index, MAX_SEQ = read_and_proc(f)
    embeds, EMBEDDING_DIM = read_embeds(efile, word_index)
    kf = KFold(n_splits=10)
    corrs = []
    for train, test in kf.split(data):
        print("index test: " ,test[0],":",test[-1])
        train_x = data[train]
        train_y = labels[train]
        test_x = data[test]
        test_y = labels[test]
        
        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights = [embeds],
                                    input_length=MAX_SEQ,
                                    trainable = False)

        sequence_input = Input(shape=(MAX_SEQ,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        #x = LSTM(300, activation="relu")(embedded_sequences)
        x = Conv1D(250, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D()(x)
        x = Flatten()(x)
        x = Dense(125, activation='relu')(x)
        x = Dense(50, activation='relu')(x)
        preds = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=sequence_input, outputs=preds)
        adam = optimizers.Adam(lr = 0.001)
        model.compile(loss='mse',
                      optimizer=adam)

        
        model.fit(train_x, train_y, epochs=10, batch_size=10)

        preds = model.predict(test_x).flatten()
        corr = pearsonr(test_y, preds)[0]
        print("Pearson correlation for fold:", corr)
        corrs.append(corr)
    
    print("Average Pearson correlation:", np.mean(corrs))
    return corrs

cross_validate("../../data/EI-reg/En/train/EI-reg-En-anger-train.tok", "/data/s3273512/en/w2v.twitter.edinburgh10M.400d.csv", 10)
