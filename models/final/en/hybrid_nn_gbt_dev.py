from sklearn.linear_model import LinearRegression
from keras.engine.topology import Layer
from att import AttentionWithContext
from att2 import Attention
from collections import defaultdict
from keras.initializers import lecun_normal
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, AveragePooling1D, LSTM, Conv1D, MaxPooling1D, Flatten, Embedding, GRU,  TimeDistributed, Reshape, Bidirectional, Permute, merge, GaussianNoise
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from scipy.stats import pearsonr
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
import random
from keras.initializers import TruncatedNormal, glorot_uniform, glorot_normal
import csv
import codecs


#Char stuff

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

def read_and_proc(f, fdev):
    data = pd.read_table(f, header=None)
    #data = shuffle(data)
    dev_data = pd.read_table(fdev, header=None)
    train_index = data.shape[0]
    data = data.append(dev_data, ignore_index=True)

    tweets = data[1].values

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)
    word_index = tokenizer.word_index
    max_seq = len(max(sequences, key=len)) + 12

    padded = pad_sequences(sequences, maxlen=max_seq)
    chars, char_index, max_word = proc_chars(tweets, max_seq)

    labels = data[3].values

    return padded, chars, word_index, char_index, max_seq, max_word, train_index


def char_model(word_index, char_index,  MAX_SEQ, MAX_WORD):
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
#end char stuff


def read_shuffle(f):
    data = pd.read_csv(f, header=None)
    #data = shuffle(data)
    return data

def read_seqs(f, fdev):
    data = pd.read_table(f, header=None)
    dev_data = pd.read_table(fdev, header=None) 
    train_index = data.shape[0]
    test_index = data.shape[0] + dev_data.shape[0]
    print(train_index, test_index)
    data = data.append(dev_data, ignore_index=True)

    tweets = data[1].values
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)

    word_index = tokenizer.word_index
    max_seq = len(max(sequences, key=len))
    data = pad_sequences(sequences, maxlen=max_seq)

    return data, word_index, max_seq, train_index, test_index


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
    input = Input(shape=(input_shape))
   # attention_probs = Dense(input_shape[0], activation='softmax', name='attention_vec')(input)
    #attention_mul = merge([input, attention_probs], output_shape=input_shape[0], name='attention_mul', mode='mul')

    x = Dense(3000, activation="relu", kernel_initializer = init)(input)
    x = Dropout(0.2)(x)
    x = Dense(1500, activation="relu", kernel_initializer = init)(x)
    x = Dropout(0.2)(x)
    x = Dense(750, activation="relu", kernel_initializer = init)(x)
    #x = Dropout(0.2)(x)
    x = Dense(350, activation="relu", kernel_initializer = init)(x)
    #x = Dropout(0.2)(x)
    x = Dense(150, activation="relu", kernel_initializer = init)(x)
    x = Dense(70, activation="relu", kernel_initializer = init)(x)
    x = Dense(35, activation="relu", kernel_initializer = init)(x)
    preds = Dense(1, activation="sigmoid", kernel_initializer = init)(x)

    return input, x, preds

def make_cnn(word_index, max_seq):
    embeds, embed_dim = read_embeds(EFILE, word_index)

    embedding_layer = Embedding(len(word_index) + 1,
                            embed_dim,
                            weights = [embeds],
                            input_length=max_seq,
                            trainable = False)

    sequence_input = Input(shape=(max_seq,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequences = GaussianNoise(0.01)(embedded_sequences)

    x = Conv1D(250, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D()(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu')(x)
#    x = Dropout(0.3)(x)
    x = Dense(50, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

    return sequence_input, x, preds


def make_cnn_LSTM3(word_index , max_seq):
    embeds, embed_dim = read_embeds(EFILE, word_index)

    embedding_layer = Embedding(len(word_index) + 1,
                            embed_dim,
                            weights = [embeds],
                            input_length=max_seq,
                            trainable = False)

    sequence_input = Input(shape=(max_seq,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    x = Reshape((-1, max_seq, embed_dim)) (embedded_sequences)
    x = TimeDistributed(Conv1D(50, 3, activation='relu', padding='valid', kernel_initializer = lecun_normal(seed=None), input_shape=(1,max_seq, embed_dim)))(x)
    x = TimeDistributed(MaxPooling1D())(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(300, activation="relu", kernel_initializer = lecun_normal(seed=None))(x)
    x = Dropout(0.3)(x)
    x = Dense(150, activation='relu', kernel_initializer = lecun_normal(seed=None))(x)
    x = Dense(75, activation='relu', kernel_initializer = lecun_normal(seed=None))(x)

    preds = Dense(1, activation='sigmoid')(x)

    return sequence_input, x, preds



def make_cnn_LSTM4(word_index, max_seq):
    embeds, embed_dim = read_embeds(EFILE, word_index)

    embedding_layer = Embedding(len(word_index) + 1,
                            embed_dim,
                            weights = [embeds],
                            input_length=max_seq,
                            trainable = False)

    sequence_input = Input(shape=(max_seq,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Bidirectional(LSTM(256, activation="relu", return_sequences=True))(embedded_sequences)
 #   x = Bidirectional(LSTM(256, activation="relu", return_sequences=True))(x)
   
    x = AttentionWithContext()(x)
    x = Dropout(0.2)(x)
    x = Dense(150, activation='relu')(x)
    x = Dense(75, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

    return sequence_input, x, preds


def cross_validate(feat_file, tok_file, dev_file, dev_tok_file):
    train_feats = read_shuffle(feat_file[0])
    dev_feats = read_shuffle(feat_file[1])
    data = pd.concat([train_feats, dev_feats])
    dev_data = read_shuffle(dev_file)
    seq_data, char_data, word_index, char_index, max_seq, max_word, train_index = read_and_proc(tok_file, dev_tok_file)
    #seq_data, word_index, max_seq, train_index, test_index = read_seqs(tok_file, dev_tok_file)

    #sent and lex train data
    feat_train_x = data.drop(data.columns[0], axis=1).values
    #char and seq train data
    print("Seq then char data: ", seq_data.shape, char_data.shape)
    seq_train_x = seq_data[:train_index,]
    char_train_x = char_data[:train_index,]
    print("train seq them char: ", seq_train_x.shape, char_train_x.shape)
    #train labels
    train_y = data.iloc[:,0].values
    #sent and lex test data
    feat_test_x = dev_data.drop(dev_data.columns[0], axis=1).values
    print(feat_test_x.shape)
    #char and seq test data
    seq_test_x = seq_data[train_index:,]
    char_test_x = char_data[train_index:,]
   # print(seq_test_x.shape)
    #test labels
    test_y = dev_data.iloc[:,0].values
    
    #buildmodels    
    dense_input, dnn, dnn_preds = make_dnn((feat_train_x.shape[1],))
    seq_input, char_input, cnn_LSTM, cnn_LSTM_preds = char_model(word_index, char_index,  max_seq, max_word)
    cnn_input, cnn, cnn_preds = make_cnn_LSTM4(word_index,max_seq)

    adam = optimizers.Adam(lr = 0.001)

    model_cnn_LSTM = Model(inputs=[seq_input, char_input], outputs=cnn_LSTM_preds)
    model_cnn_LSTM.compile(loss='mse',
                      optimizer=adam)

    model_cnn = Model(inputs=cnn_input, outputs=cnn_preds)
    model_cnn.compile(loss='mse',
                      optimizer=adam)

    model_dnn = Model(inputs=dense_input, outputs=dnn_preds)        
    model_dnn.compile(loss='mse',
                      optimizer=adam)
    regr = GradientBoostingRegressor(max_depth=3,n_estimators=45, learning_rate = 0.05,subsample=0.9, max_leaf_nodes=37000)
     
    #Fit
    print("Fitting \n")
    model_cnn.fit(seq_train_x, train_y, epochs=2, batch_size=8)
    preds_cnn = model_cnn.predict(seq_test_x).flatten()
    corr_cnn = pearson(test_y, preds_cnn)
    print("Pearson correlation for dev CNN: ", corr_cnn)

    model_dnn.fit(feat_train_x, train_y, epochs=2, batch_size=8)
    preds_dnn = model_dnn.predict(feat_test_x).flatten()
    corr_dnn = pearson(test_y, preds_dnn)
    print("Pearson correlation for dev DNN:", corr_dnn)
 
    model_cnn_LSTM.fit([seq_train_x, char_train_x], train_y, epochs=2, batch_size=4)
    regr.fit(feat_train_x, train_y)
    model_dnn.fit(feat_train_x, train_y, epochs=2, batch_size=8)
    #Predict
    preds_dnn = model_dnn.predict(feat_test_x).flatten()
    preds_cnn_LSTM = model_cnn_LSTM.predict([seq_test_x, char_test_x]).flatten()
    preds_cnn = model_cnn.predict(seq_test_x).flatten()
    #for d, y in zip(preds_cnn_LSTM, test_y):
    #    print(d, y)
    preds_gbt = regr.predict(feat_test_x)
    preds = np.mean([preds_cnn_LSTM, preds_dnn, preds_gbt, preds_cnn], axis = 0)
    corr_dnn = pearson(test_y, preds_dnn)
    corr_cnn_LSTM = pearson(test_y, preds_cnn_LSTM)
    corr_cnn = pearson(test_y, preds_cnn)
    corr_gbt = pearson(test_y, preds_gbt)
    corr = pearson(test_y, preds)

    print("Pearson correlation for dev AVG:", corr)
    print("Pearson correlation for dev DNN:", corr_dnn)
    print("Pearson correlation for dev CNN_LSTM:", corr_cnn_LSTM)
    print("Pearson correlation for dev GBT:", corr_gbt)
    print("Pearson correlation for dev CNN: ", corr_cnn)

    #write scores to file 
    format = codecs.open(format_file, 'r', "utf-8")
    out = codecs.open(out_file, 'w', "utf-8")
    reader = csv.reader(format, delimiter='\t')
    writer = csv.writer(out, delimiter='\t')
    #write header and skip line
    writer.writerow(next(reader))
    #the rest
    for row, pred in zip(reader, preds):
        row = row[:-1]
        row.append(pred)
        writer.writerow(row)

    out.close()
    format.close()

    
EFILE = "/data/s3094723/embeddings/en/w2v.twitter.edinburgh10M.400d.csv"
feat_file_train = "/data/s3094723/extra_features/EI-reg/EI-reg-En-joy-train.tok.sent.lex"
feat_file_dev = "/data/s3094723/extra_features/EI-reg/EI-reg-En-joy-dev.tok.sent.lex"
tok_file = "../../../data/EI-reg/En/traindev/EI-reg-En-joy-traindev.tok"

test_feat_file = '/data/s3094723/extra_features/EI-reg/EI-reg-En-joy-test.tok.sent.lex'
test_tok_file = "/home/s3094723/SEMEVAL/affecthor/data/EI-reg/En/test/EI-reg-En-joy-test.tok"
format_file = "/home/s3094723/SEMEVAL/affecthor/data/EI-reg/En/test/EI-reg-En-joy-test.txt"
out_file = format_file + '.scores'

cross_validate([feat_file_train, feat_file_dev], tok_file, test_feat_file, test_tok_file)
