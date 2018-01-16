import random
#from mord import LogisticAT, LAD, LogisticIT
import numpy as np
import pandas as pd
import keras
from collections import defaultdict
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, AveragePooling1D, LSTM, Conv1D, MaxPooling1D, Flatten, Embedding, GRU, Bidirectional, TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras.initializers import TruncatedNormal, glorot_uniform, glorot_normal
from scipy.stats import pearsonr, spearmanr
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

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

    chars = np.zeros((tweets.shape[0], max_seq , max_word))
    print(chars.shape)

    for i, tweet in enumerate(tweets):
       # print(len(tweet))
        tweet = tweet.split(" ")
        #print(len(tweet))
        for j, word in enumerate(tweet):
            #print(word)
            for k, char in enumerate(word):
                #print(k)
                chars[i][j][k] = c2i[char]

    return chars, c2i, max_word

def read_shuffle(f):
    data = pd.read_csv(f, header=None)
    #data = shuffle(data)
    return data

def read_seqs_cv(f):
    data = pd.read_table(f, header=None)
    tweets = data[1].values

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)

    word_index = tokenizer.word_index
    max_seq = len(max(sequences, key=len))

    data = pad_sequences(sequences, maxlen=max_seq)
    chars, char_index, max_word = proc_chars(tweets, max_seq)
    #labels = data[3].values

    return data, word_index, max_seq, chars, char_index, max_word

def read_seqs_train(train, test):
    train_raw = pd.read_table(train, header=None)
    test_raw = pd.read_table(test, header=None)
    train = train_raw[1].values
    test = test_raw[1].values
    train_index = train.shape[0]
    concat = np.concatenate((train,test))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train)
    seqs_train = tokenizer.texts_to_sequences(train)
    seqs_test = tokenizer.texts_to_sequences(test)

    word_index = tokenizer.word_index
    max_seq = len(max(seqs_train, key=len))

    seqs_train = pad_sequences(seqs_train, maxlen=max_seq)
    seqs_test = pad_sequences(seqs_test, maxlen=max_seq)

    chars, char_index, max_word = proc_chars(concat, max_seq)
    chars_train = chars[:train_index]
    chars_test = chars[train_index:]

    return seqs_train, seqs_test, word_index, max_seq, chars_train, chars_test, char_index, max_word

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
    #x = Dropout(0.2)(x)
    x = Dense(350, activation="relu", kernel_initializer = init)(x)
    #x = Dropout(0.2)(x)
    x = Dense(150, activation="relu", kernel_initializer = init)(x)
    #x = Dropout(0.2)(x)
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
    x = Bidirectional(LSTM(256, activation="relu", return_sequences=True, kernel_initializer = init))(embedded_sequences)
    #x = LSTM(150, activation="relu")(x)
    x = Conv1D(150, 5, activation='relu')(x)
    x = MaxPooling1D()(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

    return sequence_input, x, preds

def make_char_lstm(word_index, char_index, MAX_SEQ, MAX_WORD):
    embeds, EMBEDDING_DIM = read_embeds(EFILE, word_index)

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
    x = Bidirectional(LSTM(256, activation="relu"))(embedded_sequences)

    embedded_chars = char_embedding_layer(char_input)
    y = TimeDistributed(Flatten())(embedded_chars)
    y = Bidirectional(LSTM(50, activation="relu"))(y)

    merged = keras.layers.concatenate([x, y], axis=1)
    preds = Dense(1, activation='sigmoid')(merged)

    return sequence_input, char_input, merged, preds

def concat_models(dense_input, dense_layers, seq_input, seq_layers):
    x = keras.layers.concatenate([dense_layers, seq_layers])
    x = Dense(50, activation="relu")(x)
    preds = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[dense_input, seq_input], outputs=[preds])
    adam = optimizers.Adam(lr = 0.001)
    model.compile(optimizer=adam,
                  loss="mse")

    return model

def train(feat_file_train, feat_file_test, tok_file_train, tok_file_test):
    feats_train = read_shuffle(feat_file_train)
    feats_test = read_shuffle(feat_file_test)
    seqs_train_x, seqs_test_x, word_index, max_seq, chars_train_x, chars_test_x, char_index, max_word = read_seqs_train(tok_file_train, tok_file_test)
    feats_train_x = feats_train.drop(feats_train.columns[0], axis=1).values
    train_y = feats_train.iloc[:,0].values
    feats_test_x = feats_test.drop(feats_test.columns[0], axis=1).values
    # test_y = feats_test.iloc[:,0].values
        
    dense_input, dnn, dnn_preds = make_dnn((feats_train_x.shape[1],))
    seq_input, char_input, char_lstm, char_lstm_preds = make_char_lstm(word_index, char_index, max_seq, max_word)
    # seq_input, cnn_lstm, cnn_lstm_preds = make_cnn_lstm(word_index, max_seq)
    adam = optimizers.Adam(lr = 0.001)

    # model_cnn_lstm = Model(inputs=seq_input, outputs=cnn_lstm_preds)
    # model_cnn_lstm.compile(loss='mse',
    #                     optimizer=adam)

    model_dnn = Model(inputs=dense_input, outputs=dnn_preds)        
    model_dnn.compile(loss='mse',
                      optimizer=adam)

    model_char_lstm = Model(inputs=[seq_input, char_input], outputs=char_lstm_preds)
    model_char_lstm.compile(loss='mse',
                            optimizer=adam)

    regr = GradientBoostingRegressor(max_depth=3,n_estimators=450, learning_rate = 0.05,subsample=0.9, max_leaf_nodes=37000)

    regr.fit(feats_train_x, train_y)
    # model_cnn_lstm.fit(seqs_train_x, train_y, epochs=10, batch_size=8)
    model_dnn.fit(feats_train_x, train_y, epochs=10, batch_size=8)
    model_char_lstm.fit([seqs_train_x, chars_train_x], train_y, epochs=10, batch_size=8)

    preds_dnn = model_dnn.predict(feats_test_x).flatten()
    # preds_cnn_lstm = model_cnn_lstm.predict(seqs_test_x).flatten()
    preds_char_lstm = model_char_lstm.predict([seqs_test_x, chars_test_x]).flatten()
    preds_gbt = regr.predict(feats_test_x)

    # corr_dnn = pearson(test_y, preds_dnn)
    # corr_char_lstm = pearson(test_y, preds_char_lstm)
    # corr_cnn_lstm = pearson(test_y, model_cnn_lstm.predict(seqs_test_x).flatten())
    # corr_gbt = pearson(test_y, regr.predict(feats_test_x))

    # print("Pearson correlation for DNN:", corr_dnn)
    # print("Pearson correlation for CNN_LSTM:", corr_cnn_lstm)
    # print("Pearson correlation for CHAR_LSTM:", corr_char_lstm)
    # print("Pearson correlation for GBT:", corr_gbt)

    preds = np.mean([preds_char_lstm, preds_dnn, preds_gbt], axis = 0)
    return preds.reshape(-1,1), np.asarray(preds_dnn).reshape(-1,1), np.asarray(preds_char_lstm).reshape(-1,1), np.asarray(preds_gbt).reshape(-1,1)

def cross_validate(feat_file_train, feat_file_dev, tok_file, ord_file, n_folds):
    print(feat_file)
    data = read_shuffle(feat_file_train)
    dev = read_shuffle(feat_file_dev)
    data = data.append(dev, ignore_index=True)
    seq_data, word_index, max_seq, char_data, char_index, max_word = read_seqs_cv(tok_file)
    ord_data = pd.read_table(ord_file, header=None)[3]
    data, seq_data, char_data, ord_data = shuffle(data, seq_data, char_data, ord_data)
    print(data.shape, seq_data.shape, char_data.shape, ord_data.shape)
    kf = KFold(n_folds)
    preds_dnn, preds_char_lstm, preds_gbt = [], [], [] 
    ord_labels = []
    for train, test in kf.split(data):
        print("index test: " ,test[0],":",test[-1])
        train_data = data.iloc[train,:]
        test_data = data.iloc[test,:]
        feat_train_x = train_data.drop(train_data.columns[0], axis=1).values
        seq_train_x = seq_data[train]
        char_train_x = char_data[train]
        train_y = train_data.iloc[:,0].values
        feat_test_x = test_data.drop(test_data.columns[0], axis=1).values
        seq_test_x = seq_data[test]
        char_test_x = char_data[test]
        test_y = test_data.iloc[:,0].values
        ord_y = ord_data.iloc[test].values
        
        dense_input, dnn, dnn_preds = make_dnn((feat_train_x.shape[1],))
        seq_input, char_input, char_lstm, char_lstm_preds = make_char_lstm(word_index, char_index, max_seq, max_word)
        # seq_input, cnn_lstm, cnn_lstm_preds= make_cnn_lstm(word_index, max_seq)
        adam = optimizers.Adam(lr = 0.001)

        # model_cnn_lstm = Model(inputs=seq_input, outputs=cnn_lstm_preds)
        # model_cnn_lstm.compile(loss='mse',
        #               optimizer=adam)

        model_dnn = Model(inputs=dense_input, outputs=dnn_preds)        
        model_dnn.compile(loss='mse',
                      optimizer=adam)

        model_char_lstm = Model(inputs=[seq_input, char_input], outputs=char_lstm_preds)
        model_char_lstm.compile(loss='mse',
                      optimizer=adam)
        
        regr = GradientBoostingRegressor(max_depth=3,n_estimators=450, learning_rate = 0.05,subsample=0.9, max_leaf_nodes=37000)

        regr.fit(feat_train_x, train_y)
        #model_cnn_lstm.fit(seq_train_x, train_y, epochs=10, batch_size=8)
        model_dnn.fit(feat_train_x, train_y, epochs=10, batch_size=8)
        model_char_lstm.fit([seq_train_x, char_train_x], train_y, epochs=10, batch_size=8)

        preds_dnn.extend(model_dnn.predict(feat_test_x).flatten())
        preds_char_lstm.extend(model_char_lstm.predict([seq_test_x, char_test_x]).flatten())
        preds_gbt.extend(regr.predict(feat_test_x))

        corr_dnn = pearson(test_y, model_dnn.predict(feat_test_x).flatten())
        corr_char_lstm = pearson(test_y, model_char_lstm.predict([seq_test_x, char_test_x]).flatten())
        corr_gbt = pearson(test_y, regr.predict(feat_test_x))

        print("Pearson correlation for fold DNN:", corr_dnn)
        print("Pearson correlation for fold CHAR_LSTM:", corr_char_lstm)
        print("Pearson correlation for fold GBT:", corr_gbt)

        ord_labels.extend(ord_y.tolist())

    preds = np.mean([preds_char_lstm, preds_dnn, preds_gbt], axis = 0)
    return preds.reshape(-1,1), np.asarray(preds_dnn).reshape(-1,1), np.asarray(preds_char_lstm).reshape(-1,1), np.asarray(preds_gbt).reshape(-1,1), np.asarray(ord_labels).reshape(-1,1)

EFILE = "/data/s3094723/embeddings/en/w2v.twitter.edinburgh10M.400d.csv"
feat_file_train = "/data/s3094723/extra_features/EI-reg/EI-reg-En-anger-train.tok.sent.lex"
feat_file_dev = "/data/s3094723/extra_features/EI-reg/EI-reg-En-anger-dev.tok.sent.lex"
feat_file_test = "/data/s3094723/extra_features/EI-reg/EI-reg-En-anger-dev.tok.sent.lex"
tok_file_train = "/home/s3094723/SEMEVAL/affecthor/data/EI-reg/En/train/EI-reg-En-anger-train.tok"
tok_file_dev = "/home/s3094723/SEMEVAL/affecthor/data/EI-reg/En/dev/EI-reg-En-anger-dev.tok"
tok_file_test = "/home/s3094723/SEMEVAL/affecthor/data/EI-reg/En/dev/EI-reg-En-anger-dev.tok"
tok_file_traindev = "/home/s3094723/SEMEVAL/affecthor/data/EI-reg/En/traindev/EI-reg-En-anger-traindev.tok"

feat_file = "/data/s3094723/extra_features/EI-reg/EI-reg-En-anger-train.tok.sent.lex"
tok_file = "/home/s3094723/SEMEVAL/affecthor/data/EI-reg/En/train/EI-reg-En-anger-train.tok"
ord_file = "/home/s3094723/SEMEVAL/affecthor/data/EI-oc/En/traindev/EI-oc-En-anger-traindev.tok"

#preds, preds_dnn, preds_cnn_lstm, preds_gbt = train(feat_file_train, feat_file_test, tok_file_train, tok_file_test)
preds, preds_dnn, preds_char_lstm, preds_gbt, ord_labels  = cross_validate(feat_file_train, feat_file_dev, tok_file_traindev, ord_file, 5)
# concat = np.concatenate((preds_dnn, preds_gbt, preds_cnn_lstm), axis=1)

# avg_oc_it = []
# avg_lgc =[]
# avg_oc1 = []
# avg_oc2 = []

# kf = KFold(n_splits=5)
# for train, test in kf.split(preds):
#     lg_concat = LogisticRegression()
#     oc_it = LogisticIT(alpha = 1)
#     oc2 =  LogisticAT(alpha = 1.5)
#     oc1 =  LogisticAT(alpha = 2.5)
#     #fit
#     lg_concat.fit(concat[train], ord_labels[train])      
#     oc2.fit(concat[train], ord_labels[train])
#     oc1.fit(concat[train], ord_labels[train])
#     oc_it.fit(concat[train], ord_labels[train])
#     #predict
#     predictions_lg_concat = lg_concat.predict(concat[test])
#     predictions_oc2 = oc2.predict(concat[test])	
#     predictions_oc1 = oc1.predict(concat[test])  
#     predictions_oc_it = oc_it.predict(concat[test])

		
#     print("CONCAT_lg: ", spearmanr(predictions_lg_concat, ord_labels[test])[0])
#     print("OC2: ", spearmanr(predictions_oc2, ord_labels[test])[0])
#     print("OC1: ", spearmanr(predictions_oc1, ord_labels[test])[0])
#     print("OC IT: ", spearmanr(predictions_oc_it, ord_labels[test])[0])

    


		
#     avg_lgc.append(spearmanr(predictions_lg_concat, ord_labels[test])[0])
#     avg_oc2.append(spearmanr(predictions_oc2, ord_labels[test])[0])
#     avg_oc1.append(spearmanr(predictions_oc1, ord_labels[test])[0])
#     avg_oc_it.append(spearmanr(predictions_oc_it, ord_labels[test])[0])
    
# print("AVG CONCAT lg: ", np.mean(avg_lgc))
# print("AVG OC2 AT: ", np.mean(avg_oc2))  
# print("AVG OC1 AT: ", np.mean(avg_oc1))
# print("AVG OC IT: ", np.mean(avg_oc_it))

