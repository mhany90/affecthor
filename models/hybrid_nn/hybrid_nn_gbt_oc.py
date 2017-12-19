import random
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, AveragePooling1D, LSTM, Conv1D, MaxPooling1D, Flatten, Embedding, GRU
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras.initializers import TruncatedNormal, glorot_uniform, glorot_normal
from scipy.stats import pearsonr
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

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



init = 'TruncatedNormal'


def make_dnn(input_shape):
    input = Input(shape=input_shape)
    x = Dense(3000, activation="relu", kernel_initializer = init )(input)
    x = Dropout(0.2)(x)
    x = Dense(1500, activation="relu", kernel_initializer = init)(x)
    x = Dropout(0.2)(x)
    x = Dense(750, activation="relu", kernel_initializer = init)(x)
   # x = Dropout(0.2)(x)
    x = Dense(350, activation="relu", kernel_initializer = init)(x)
  #  x = Dropout(0.2)(x)
    x = Dense(150, activation="relu", kernel_initializer = init)(x)
 #   x = Dropout(0.2)(x)
    x = Dense(70, activation="relu", kernel_initializer = init)(x)
    x = Dense(35, activation="relu", kernel_initializer = init)(x)
    preds = Dense(1, activation="sigmoid", kernel_initializer = init)(x)

    #return Model(inputs=inputs, outputs=preds)

    return input, x, preds

def make_cnn_lstm(word_index, max_seq):
    embeds, embed_dim = read_embeds(EFILE, word_index)

    embedding_layer = Embedding(len(word_index) + 1,
                            embed_dim,
                            weights = [embeds],
                            input_length=max_seq,
                            trainable = False)

    sequence_input = Input(shape=(max_seq,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = LSTM(256, activation="relu", return_sequences=True, kernel_initializer = init)(embedded_sequences)
    #x = LSTM(150, activation="relu")(x)
    x = Conv1D(150, 5, activation='relu', kernel_initializer = init)(x)
    x = AveragePooling1D()(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu', kernel_initializer = init)(x)
    #x = Dropout(0.1)(x)
    #x = Dense(50, activation='relu', kernel_initializer = init)(x)
    preds = Dense(1, activation='sigmoid', kernel_initializer = init)(x)

    return sequence_input, x, preds

def concat_models(dense_input, dense_layers, seq_input, seq_layers):
    x = keras.layers.concatenate([dense_layers, seq_layers])
    x = Dense(50, activation="relu")(x)
    preds = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[dense_input, seq_input], outputs=[preds])
    adam = optimizers.Adam(lr = 0.001)
    model.compile(optimizer=adam,
                  loss="mse")

    return model

def cross_validate(feat_file, tok_file, ord_file, n_folds):
    print(feat_file)
    data = read_shuffle(feat_file)
    seq_data, word_index, max_seq = read_seqs(tok_file)
    ord_data = pd.read_table(ord_file, header=None)[3]
    data, seq_data, ord_data = shuffle(data, seq_data, ord_data)
    print(data.shape, seq_data.shape, ord_data.shape)
    kf = KFold(n_folds)
    preds_dnn, preds_cnn_lstm, preds_gbt = [], [], [] 
    ord_labels = []
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
        ord_y = ord_data.iloc[test].values
        
        dense_input, dnn, dnn_preds = make_dnn((feat_train_x.shape[1],))
        seq_input, cnn_lstm, cnn_lstm_preds= make_cnn_lstm(word_index, max_seq)
        adam = optimizers.Adam(lr = 0.001)

        model_cnn_lstm = Model(inputs=seq_input, outputs=cnn_lstm_preds)
        model_cnn_lstm.compile(loss='mse',
                      optimizer=adam)

        model_dnn = Model(inputs=dense_input, outputs=dnn_preds)        
        model_dnn.compile(loss='mse',
                      optimizer=adam)
        regr = GradientBoostingRegressor(max_depth=3,n_estimators=450, learning_rate = 0.05,subsample=0.9, max_leaf_nodes=37000)

        regr.fit(feat_train_x, train_y)
        model_cnn_lstm.fit(seq_train_x, train_y, epochs=10, batch_size=8)
        model_dnn.fit(feat_train_x, train_y, epochs=10, batch_size=8)

        preds_dnn.extend(model_dnn.predict(feat_test_x).flatten())
        preds_cnn_lstm.extend(model_cnn_lstm.predict(seq_test_x).flatten())
        preds_gbt.extend(regr.predict(feat_test_x))

        ord_labels.extend(ord_y.tolist())

    preds = np.mean([preds_cnn_lstm, preds_dnn, preds_gbt], axis = 0)
    return preds.reshape(-1,1), np.asarray(preds_dnn).reshape(-1,1), np.asarray(preds_cnn_lstm).reshape(-1,1), np.asarray(preds_gbt).reshape(-1,1), np.asarray(ord_labels).reshape(-1,1)

EFILE = "/data/s3094723/embeddings/en/w2v.twitter.edinburgh10M.400d.csv"
feat_file = "/data/s3094723/extra_features/EI-reg-En-fear-train.txt.sent.lex"
tok_file = "/home/s3094723/SEMEVAL/affecthor/data/EI-reg/En/train/EI-reg-En-fear-train.tok"
ord_file = "/home/s3094723/SEMEVAL/affecthor/data/EI-oc/En/train/EI-oc-En-fear-train.tok"

preds, preds_dnn, preds_cnn_lstm, preds_gbt, ord_labels  = cross_validate(feat_file, tok_file, ord_file, 5)
concat = np.concatenate((preds_dnn, preds_cnn_lstm, preds_gbt), axis=1)
kf = KFold(n_splits=5)
for train, test in kf.split(preds):
    lg = LogisticRegression()
    lg_concat = LogisticRegression()
    lg.fit(preds[train], ord_labels[train])
    lg_concat.fit(concat[train], ord_labels[train])
    print("AVG", lg.score(preds[test], ord_labels[test]))
    print("CONCAT", lg_concat.score(concat[test], ord_labels[test]))

    
