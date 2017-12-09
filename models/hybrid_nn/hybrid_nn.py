import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from scipy.stats import pearsonr
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import random

def read_shuffle(f):
    data = pd.read_csv(f, header=None)
    #data = shuffle(data)
    return data

def read_seqs(f):
    data = pd.read_table(f, header=None)
    tweets = data[1].values

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)

    word_index = tokenizer.word_index
    max_seq = len(max(sequences, key=len))

    data = pad_sequences(sequences, maxlen=max_seq)
    #labels = data[3].values

    return data, word_index, max_seq

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

def make_dnn(input_shape):
    input = Input(shape=input_shape)
    x = Dense(3000, activation="relu")(input)
    x = Dropout(0.5)(x)
    x = Dense(1500, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(750, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(300, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(150, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation="relu")(x)
    #preds = Dense(1, activation="sigmoid")(x)

    #return Model(inputs=inputs, outputs=preds)

    return input, x

def make_cnn_lstm(word_index, max_seq):
    embeds, embed_dim = read_embeds(EFILE, word_index)

    embedding_layer = Embedding(len(word_index) + 1,
                            embed_dim,
                            weights = [embeds],
                            input_length=max_seq,
                            trainable = False)

    sequence_input = Input(shape=(max_seq,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = LSTM(300, activation="relu", return_sequences=True)(embedded_sequences)
    x = LSTM(150, activation="relu")(x)
    #x = Conv1D(256, 5, activation='relu')(embedded_sequences)
    #x = MaxPooling1D()(x)
    #x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    #x = Dense(50, activation='relu')(x)
    #preds = Dense(1, activation='sigmoid')(x)

    return sequence_input, x

def concat_models(dense_input, dense_layers, seq_input, seq_layers):
    x = keras.layers.concatenate([dense_layers, seq_layers])
    x = Dense(50, activation="relu")(x)
    preds = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[dense_input, seq_input], outputs=[preds])
    adam = optimizers.Adam(lr = 0.001)
    model.compile(optimizer=adam,
                  loss="mse")

    return model

def cross_validate(feat_file, tok_file, n_folds):
    print(feat_file)
    data = read_shuffle(feat_file)
    seq_data, word_index, max_seq = read_seqs(tok_file)
    data, seq_data = shuffle(data, seq_data)
    kf = KFold(n_splits=10)
    corrs = []
    for train, test in kf.split(data):
        print("index test: " ,test[0],":",test[-1])
        train_data = data.iloc[train,:]
        test_data = data.iloc[test,:]
        feat_train_x = train_data.drop(train_data.columns[0], axis=1).values
        seq_train_x = seq_data[train]
        train_y = train_data.iloc[:,0].values
        feat_test_x = test_data.drop(test_data.columns[0], axis=1).values
        seq_test_x = seq_data[test]
        test_y = test_data.iloc[:,0].values
        
        dense_input, dnn = make_dnn((feat_train_x.shape[1],))
        seq_input, cnn_lstm = make_cnn_lstm(word_index, max_seq)

        model = concat_models(dense_input, dnn, seq_input, cnn_lstm)
        
        model.fit([feat_train_x, seq_train_x],
                  [train_y],
                  epochs=10,
                  batch_size=10)
        
        preds = model.predict([feat_test_x, seq_test_x]).flatten()
        corr = pearson(test_y, preds)
        print("Pearson correlation for fold:", corr)
        corrs.append(corr)

    return corrs

EFILE = "/data/s3273512/en/w2v.twitter.edinburgh10M.400d.csv"
feat_file = "/data/s3094723/extra_features/EI-reg-En-anger-train.txt.sent.lex"
tok_file = "../../data/EI-reg/En/train/EI-reg-En-anger-train.tok"

corrs = cross_validate(feat_file, tok_file, 10)
print("Average Pearson correlation:", np.mean(corrs))
