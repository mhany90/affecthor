from att import AttentionWithContext
from att2 import Attention
from collections import OrderedDict
from collections import defaultdict
from keras.initializers import lecun_normal
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, AveragePooling1D, LSTM, Conv1D, MaxPooling1D, Flatten, Embedding, GRU, TimeDistributed, Reshape,  Bidirectional, Permute, merge, GaussianNoise
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

def read_and_proc(f, fdev, ftest):
    data = pd.read_table(f, header=None)
    dev_data = pd.read_table(fdev, header=None)
    test_data =  pd.read_table(ftest, header=None)

    #combine dev and train
    data = data.append(dev_data, ignore_index=True)
    train_index = data.shape[0] #get train index
    test_index = data.shape[0] + test_data.shape[0]

    #add test data to mix
    data = data.append(test_data, ignore_index=True)
    print(train_index, test_index)

    tweets = data[1].values

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)
    word_index = tokenizer.word_index
    max_seq = len(max(sequences, key=len)) + 11

    padded = pad_sequences(sequences, maxlen=max_seq)
    chars, char_index, max_word = proc_chars(tweets, max_seq)

    labels = data[3].values

    return padded, chars, word_index, char_index, max_seq, max_word, train_index


def make_char_lstm(word_index, char_index,  MAX_SEQ, MAX_WORD):
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
   # x = Dense(35, activation="relu", kernel_initializer = init)(merged) 
    preds = Dense(1, activation='sigmoid')(merged)

    return sequence_input, char_input, merged, preds
#end char stuff

def read_shuffle(f):
    data = pd.read_csv(f, header=None, skiprows = 1)
    return data

#dev data and train data combined, test data seperate
def read_seqs(f, fdev, ftest):
    data = pd.read_table(f, header=None)
    dev_data = pd.read_table(fdev, header=None)
    test_data =  pd.read_table(ftest, header=None)
 
    #combine dev and train
    data = data.append(dev_data, ignore_index=True)   
    train_index = data.shape[0] #get train index
    test_index = data.shape[0] + test_data.shape[0] 
  
    #add test data to mix
    data = data.append(test_data, ignore_index=True)
    print(train_index, test_index)

    tweets = data[1].values
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)

    word_index = tokenizer.word_index
    max_seq = len(max(sequences, key=len))
    data = pad_sequences(sequences, maxlen=max_seq)

    return data, word_index, max_seq, train_index

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
    x = Dense(300, activation="relu", kernel_initializer = init )(input)
    x = Dropout(0.2)(x)
    x = Dense(150, activation="relu", kernel_initializer = init)(x)
    x = Dropout(0.3)(x)
    #x = Dense(75, activation="relu", kernel_initializer = init)(x)
    # x = Dropout(0.2)(x)
    # x = Dense(35, activation="relu", kernel_initializer = init)(x)
    #  x = Dropout(0.2)(x)
    #x = Dense(150, activation="relu", kernel_initializer = init)(x)
    #   x = Dropout(0.2)(x)
    #    x = Dense(10, activation="relu", kernel_initializer = init)(x)
    preds = Dense(1, activation="sigmoid", kernel_initializer = init)(x)

    #return Model(inputs=inputs, outputs=preds)

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
    #x = Dense(50, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

    return sequence_input, x, preds

def make_lstm(word_index, max_seq):
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
    x = Dropout(0.1)(x)
    x = Dense(150, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

    return sequence_input, x, preds


def make_lstm2(word_index, max_seq):
    embeds, embed_dim = read_embeds(EFILE, word_index)

    embedding_layer = Embedding(len(word_index) + 1,
                            embed_dim,
                            weights = [embeds],
                            input_length=max_seq,
                            trainable = False)

    sequence_input = Input(shape=(max_seq,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Bidirectional(LSTM(256, activation="relu", return_sequences=True))(embedded_sequences)
    #x = LSTM(256, activation="relu", return_sequences=True, kernel_initializer = init)(embedded_sequences)
    #x = Bidirectional(LSTM(100, activation="relu"))(x)
    # x = Conv1D(100, 5, activation='relu', kernel_initializer = init)(embedded_sequences)
    # x = MaxPooling1D()(x)
    x = Flatten()(x)
    # x = Dense(128, activation='relu')(x)
    #x = Dropout(0.2)(x)
    x = Dense(50, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

    return sequence_input, x, preds


def make_cnn_lstm(word_index , max_seq):
    embeds, embed_dim = read_embeds(EFILE, word_index)

    embedding_layer = Embedding(len(word_index) + 1,
                            embed_dim,
                            weights = [embeds],
                            input_length=max_seq,
                            trainable = False)

    sequence_input = Input(shape=(max_seq,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    x = Reshape((-1, max_seq, embed_dim)) (embedded_sequences)
    x = TimeDistributed(Conv1D(100, 3, activation='relu', padding='valid', kernel_initializer = lecun_normal(seed=None), input_shape=(1,max_seq, embed_dim)))(x)
    x = TimeDistributed(MaxPooling1D())(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(300, activation="relu", kernel_initializer = lecun_normal(seed=None))(x)
    x = Dropout(0.2)(x)
    x = Dense(150, activation='relu', kernel_initializer = lecun_normal(seed=None))(x)
    x = Dense(75, activation='relu', kernel_initializer = lecun_normal(seed=None))(x)

    preds = Dense(1, activation='sigmoid')(x)

    return sequence_input, x, preds


def cross_validate(feat_file, feat_file_gbt, dev_feat_file,dev_feat_gbt_file, test_feat_file,test_feat_gbt_file, tok_file, dev_tok_file, test_tok_file,  n_folds, CV):
    print(feat_file)
    data = read_shuffle(feat_file)
    dev_data = read_shuffle(dev_feat_file)
    final_test_data = read_shuffle(test_feat_file)
    data_gbt = read_shuffle(feat_file_gbt)
    dev_gbt = read_shuffle(dev_feat_gbt_file)
    final_test_data_gbt= read_shuffle(test_feat_gbt_file)


    seq_data, char_data, word_index, char_index, max_seq, max_word, train_index = read_and_proc(tok_file, dev_tok_file, test_tok_file)
    #seq_data, word_index, max_seq = read_seqs(tok_file)
    #data, seq_data = shuffle(data, seq_data)

    kf = KFold(n_splits=n_folds)
    corrs = []
    corrs_lstm, corrs_char_lstm, corrs_lstm2 = [], [], []

    #sent and lex train data
    #combine dev and train sent.lex
    data = data.append(dev_data, ignore_index=True)
    data_gbt =  data_gbt.append(dev_gbt, ignore_index=True) 
    #char and seq train data (remove test data)
    print("Seq then char data: ", seq_data.shape, char_data.shape)
    seq_train = seq_data[:train_index,]
    char_train = char_data[:train_index,]
    print("train seq them char: ", seq_train.shape, char_train.shape)

    #CV on the dev and train
    for train, test in kf.split(data):
        print("index test: " ,test[0],":",test[-1])
        train_data = data.iloc[train,:]
        test_data = data.iloc[test,:]
        train_data_gbt = data_gbt.iloc[train,:]
        test_data_gbt = data_gbt.iloc[test,:]
        #cv train
        cols = [0,1]
        feat_train_x = train_data.drop(train_data.columns[cols], axis=1).values
        feat_train_x_gbt = train_data_gbt.drop(train_data_gbt.columns[cols], axis=1).values
        seq_train_x = seq_train[train]
        char_train_x = char_train[train]
        train_y = train_data.iloc[:,1].values
        #cv test
        cols = [0,1]
        feat_test_x = test_data.drop(test_data.columns[cols], axis=1).values
        feat_test_x_gbt = test_data_gbt.drop(test_data_gbt.columns[cols], axis=1).values
        seq_test_x = seq_train[test]
       	char_test_x = char_train[test]
        test_y = test_data.iloc[:,1].values
        #optim.
        adam = optimizers.Adam(lr = 0.001)

        #lstm
        seq_input, lstm, lstm_preds= make_lstm(word_index, max_seq)
        model_lstm = Model(inputs=seq_input, outputs=lstm_preds)
        model_lstm.compile(loss='mse',
                      optimizer=adam)

        #lstm2
        seq_input2, lstm2, lstm_preds2= make_lstm2(word_index, max_seq)
        model_lstm2 = Model(inputs=seq_input2, outputs=lstm_preds2)
        model_lstm2.compile(loss='mse',
                      optimizer=adam)

        #char_lstm (seq input gets overwritten)
        seq_input, char_input, char_lstm, char_lstm_preds = make_char_lstm(word_index, char_index,  max_seq, max_word)
        model_char_lstm = Model(inputs=[seq_input, char_input], outputs=char_lstm_preds)
        model_char_lstm.compile(loss='mse',
                      optimizer=adam)

        #cnn
        #cnn_input, cnn, cnn_preds = make_cnn(word_index,max_seq)
        #model_cnn = Model(inputs=cnn_input, outputs=cnn_preds)
        #model_cnn.compile(loss='mse',
        #              optimizer=adam)

        #cnn lstm
        #seq_input, cnn_lstm, cnn_lstm_preds= make_cnn_lstm(word_index, max_seq)
        #model_cnn_lstm = Model(inputs=seq_input, outputs=cnn_lstm_preds)
        #model_cnn_lstm.compile(loss='mse',
        #              optimizer=adam)

        #regr
        #regr = GradientBoostingRegressor(max_depth=3,n_estimators=450, learning_rate = 0.05,subsample=0.9, max_leaf_nodes=37000)

        if CV:
            #fit all
            #regr.fit(feat_train_x_gbt, train_y)
            #model_cnn_lstm.fit(seq_train_x, train_y, epochs=6, batch_size=4, verbose = 1)
            #model_dnn.fit(feat_train_x, train_y, epochs=8, batch_size=8, verbose = 1)
            model_char_lstm.fit([seq_train_x, char_train_x], train_y, epochs=6, batch_size=4, verbose = 1)
            model_lstm.fit(seq_train_x, train_y, epochs=6, batch_size=8, verbose = 1)
            model_lstm2.fit(seq_train_x, train_y, epochs=6, batch_size=8, verbose = 1)

            #pred all
            #preds_dnn = model_dnn.predict(feat_test_x).flatten()
            #preds_cnn_lstm = model_cnn_lstm.predict(seq_test_x).flatten()
            # preds_gbt = regr.predict(feat_test_x_gbt)
            preds_char_lstm = model_char_lstm.predict([seq_test_x, char_test_x]).flatten()
            preds_lstm = model_lstm.predict(seq_test_x).flatten()
            preds_lstm2 = model_lstm2.predict(seq_test_x).flatten()

            #corr_dnn = pearson(test_y, preds_dnn)
            #corr_cnn_lstm = pearson(test_y, preds_cnn_lstm)
            corr_char_lstm = pearson(test_y, preds_char_lstm)
            # corr_gbt = pearson(test_y, preds_gbt)
            corr_lstm = pearson(test_y, preds_lstm)
            corr_lstm2 = pearson(test_y, preds_lstm2)

            #print("Pearson correlation for fold DNN: ", corr_dnn)
            #print("Pearson correlation for fold CNN_LSTM: ", corr_cnn_lstm)
            #print("Pearson correlation for fold GBT: ", corr_gbt)
            print("Pearson correlation for fold lstm: ", corr_lstm)
            print("Pearson correlation for fold CHAR_LSTM:", corr_char_lstm)
            print("Pearson correlation for fold lstm 2: ", corr_lstm2)

            #corrs_dnn.append(corr_dnn)
            #corrs_cnn_lstm.append(corr_cnn_lstm)
            #corrs_gbt.append(corr_gbt)
            corrs_char_lstm.append(corr_char_lstm)
            corrs_lstm.append(corr_lstm)
            corrs_lstm2.append(corr_lstm2)

            preds = np.mean([ preds_lstm2 , preds_char_lstm, preds_lstm ], axis = 0)
            corr = pearson(test_y, preds)
            corrs.append(corr)
            print("Pearson correlation for fold:", corr)

    if CV:
        #mean_dnn = float(np.mean(corrs_dnn))
        #mean_cnn_lstm = float(np.mean(corrs_cnn_lstm))
        #mean_gbt = float(np.mean(corrs_gbt))
        mean_lstm = float(np.mean(corrs_lstm))
        mean_lstm2 = float(np.mean(corrs_lstm2))
        mean_char_lstm =  float(np.mean(corrs_char_lstm))

        #rank systems
        systems = {'preds_lstm':mean_lstm, 'preds_char_lstm':mean_char_lstm, 'preds_lstm2':mean_lstm2 }
        print(systems)
        sys_ranking = [k for k in sorted(systems, key=systems.get, reverse = True)]
        top_systems  = sys_ranking[:3]

        #print("Average Pearson correlation AVG:", np.mean(corrs))
        #print("Average Pearson correlation DNN:", np.mean(corrs_dnn))
        #print("Average Pearson correlation CNN_LSTM:", np.mean(corrs_cnn_lstm))
        #print("Average Pearson correlation GBT:", np.mean(corrs_gbt))
        print("Average Pearson correlation LSTM:", np.mean(corrs_lstm))
        print("Average Pearson correlation CHAR_LSTN:", np.mean(corrs_char_lstm))
        print("Average Pearson correlation LSTM2:", np.mean(corrs_lstm2))

        print("Best systems: ", top_systems)
    else:
        #top_systems = ['preds_lstm', 'preds_dnn', 'preds_gbt' ]
        top_systems = ['preds_lstm', 'preds_char_lstm', 'preds_lstm2']
        print("Best systems (no CV): ", top_systems)

    #final test
    #fit all again on full train_dev set
    #prepare data
    #final train
    feat_train_x = data.drop(data.columns[cols], axis=1).values
    feat_train_x_gbt = data_gbt.drop(data_gbt.columns[cols], axis=1).values
    train_y = data.iloc[:,1].values
    #final test
    feat_test_x = final_test_data.drop(final_test_data.columns[cols], axis=1).values
    feat_test_x_gbt = final_test_data_gbt.drop(final_test_data_gbt.columns[cols], axis=1).values
    final_seq_test_x = seq_data[train_index:,]
    final_char_test_x = char_data[train_index:,]

    #run x times and use average
    run_times = 3
    all_char_lstm, all_lstm , all_lstm2 = [],[],[]

    for x in range(1, run_times):
        print("run: ", x)
        #fit all again
        #regr.fit(feat_train_x_gbt, train_y)
        #model_cnn_lstm.fit(seq_train, train_y, epochs=8, batch_size=4, verbose = 1)
        #model_dnn.fit(feat_train_x, train_y, epochs=8, batch_size=8, verbose = 1)
        model_char_lstm.fit([seq_train, char_train], train_y, epochs=6, batch_size=4, verbose = 1)
        model_lstm.fit(seq_train, train_y, epochs=6, batch_size=6, verbose = 1)
        model_lstm2.fit(seq_train, train_y, epochs=6, batch_size=6, verbose = 1)

        #pred all
        #preds_dnn = model_dnn.predict(feat_test_x).flatten()
        #preds_cnn_lstm = model_cnn_lstm.predict(final_seq_test_x).flatten()
        #preds_gbt = regr.predict(feat_test_x_gbt)
        preds_char_lstm = model_char_lstm.predict([final_seq_test_x, final_char_test_x]).flatten()
        preds_lstm = model_lstm.predict(final_seq_test_x).flatten()
        preds_lstm2 = model_lstm2.predict(final_seq_test_x).flatten()

        #add to list
        #all_dnn.append(preds_dnn)
        #all_cnn_lstm.append(preds_cnn_lstm)
        #all_gbt.append(preds_gbt)
        all_char_lstm.append(preds_char_lstm)
        all_lstm.append(preds_lstm)
        all_lstm2.append(preds_lstm2)

    #get avg
    #preds_dnn = np.mean(all_dnn,  axis = 0)
    #preds_cnn_lstm = np.mean(all_cnn_lstm,  axis = 0)
    #preds_gbt = np.mean(all_gbt,  axis = 0)
    preds_char_lstm = np.mean(all_char_lstm, axis = 0)
    preds_lstm = np.mean(all_lstm, axis = 0)
    preds_lstm2 = np.mean(all_lstm2, axis = 0)

    #use best systems from CV
    all = defaultdict(list)
    #all = {'preds_dnn':preds_dnn, 'preds_cnn_lstm':preds_cnn_lstm, 'preds_gbt':preds_gbt, 'preds_lstm':preds_lstm, 'preds_char_lstm':preds_char_lstm}
    all = {'preds_lstm':preds_lstm, 'preds_char_lstm':preds_char_lstm, 'preds_lstm2':preds_lstm2}

    best = [value for key, value in all.items() if key in top_systems]
    print(best)
    preds = np.mean(best, axis = 0)
    #preds = np.mean([preds_dnn, preds_gbt,  preds_cnn_lstm, preds_cnn,  preds_char_lstm], axis = 0)

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


EFILE = "/data/s3094723/embeddings/es/es.tweets_two.reformated.csv"
feat_file = "/home/s3094723/SEMEVAL/affecthor/data/EI-reg/Es/train/features/EI-reg-Es-fear-train.emb.tweets_two.csv"

feat_file_gbt = "/home/s3094723/SEMEVAL/affecthor/data/EI-reg/Es/train/features/EI-reg-Es-fear-train.combined.tweets_two.csv"

tok_file = "/home/s3094723/SEMEVAL/affecthor/data/EI-reg/Es/train/EI-reg-Es-fear-train.tok"

dev_feat_file = '/home/s3094723/SEMEVAL/affecthor/data/EI-reg/Es/dev/features/EI-reg-Es-fear-dev.emb.tweets_two.csv'

dev_feat_gbt_file = '/home/s3094723/SEMEVAL/affecthor/data/EI-reg/Es/dev/features/EI-reg-Es-fear-dev.combined.tweets_two.csv'

dev_tok_file = "/home/s3094723/SEMEVAL/affecthor/data/EI-reg/Es/dev/EI-reg-Es-fear-dev.tok"
test_feat_file = '/home/s3094723/SEMEVAL/affecthor/data/EI-reg/Es/test/features/EI-reg-Es-fear-test.emb.tweets_two.csv'

test_feat_gbt_file = '/home/s3094723/SEMEVAL/affecthor/data/EI-reg/Es/test/features/EI-reg-Es-fear-test.combined.tweets_two.csv'

test_tok_file = "/home/s3094723/SEMEVAL/affecthor/data/EI-reg/Es/test/EI-reg-Es-fear-test.tok"
format_file = "/home/s3094723/SEMEVAL/affecthor/data/EI-reg/Es/test/EI-reg-Es-fear-test.txt"
out_file = '/home/s3479307/testing/scores/EI-reg-Es-fear-test.txt.scores'

CV = False

cross_validate(feat_file,feat_file_gbt,  dev_feat_file, dev_feat_gbt_file, test_feat_file,test_feat_gbt_file, tok_file, dev_tok_file, test_tok_file, 5, CV)
