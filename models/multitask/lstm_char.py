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
    max_seq = len(max(sequences, key=len)) + 12

    padded = pad_sequences(sequences, maxlen=max_seq)
    chars, char_index, max_word = proc_chars(tweets, max_seq)

    labels = data[3].values

    return padded, chars, labels, word_index, char_index, max_seq, max_word

def read_and_proc_multiple(files, efile):
    out = []
    for f in files:
        print("Processing {}...".format(f))
        p, ch, ls, w_i, c_i, m_s, m_w = read_and_proc(f)
        procd = {"padded": p,
                 "chars": ch,
                 "labels": ls,
                 "word_index": w_i,
                 "char_index": c_i,
                 "max_seq": m_s,
                 "max_word": m_w,
                 "embeds": None}
        print("Reading embeddings...")
        e, _ = read_embeds(efile, procd["word_index"])
        procd["embeds"] = e
        out.append(procd)

    return out

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
    return pearsonr(true, pred)[0]

# Annoying helper for CV
def get_min(pfs):
    min_samples = pfs[0]["labels"].shape[0]
    index = 0
    for i, a in enumerate(pfs[0:]):
        if a["labels"].shape[0] < min_samples:
            min_samples = a["labels"].shape[0]
            index = i
    return index

def make_char_lstm(word_index, char_index,  MAX_SEQ, MAX_WORD):
    embeds, EMBEDDING_DIM = read_embeds(EFILE, word_index)

    embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights = [embeds],
                                    input_length=MAX_SEQ,
                                    trainable = False)

    char_embedding_layer = TimeDistributed(Embedding(len(char_index) + 1,
                                                     output_dim=32,
                                                     input_length=MAX_WORD))

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


def make_dnn(input_shape):
    input = Input(shape=input_shape)
    x = Dense(3000, activation="relu", kernel_initializer=init)(input)
    x = Dropout(0.2)(x)
    x = Dense(1500, activation="relu", kernel_initializer=init)(x)
    x = Dropout(0.2)(x)
    x = Dense(750, activation="relu", kernel_initializer=init)(x)
    # x = Dropout(0.2)(x)
    x = Dense(350, activation="relu", kernel_initializer=init)(x)
    # x = Dropout(0.2)(x)
    x = Dense(150, activation="relu", kernel_initializer=init)(x)
    # x = Dropout(0.2)(x)
    x = Dense(70, activation="relu", kernel_initializer=init)(x)
    x = Dense(35, activation="relu", kernel_initializer=init)(x)
    preds = Dense(1, activation="sigmoid", kernel_initializer=init)(x)

    return input, x, preds


def make_cnn(word_index, max_seq):
    embeds, embed_dim = read_embeds(EFILE, word_index)

    embedding_layer = Embedding(len(word_index) + 1,
                                embed_dim,
                                weights=[embeds],
                                input_length=max_seq,
                                trainable=False)

    sequence_input = Input(shape=(max_seq,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    embedded_sequences = GaussianNoise(0.01)(embedded_sequences)

    x = Conv1D(250, 3, activation='relu')(embedded_sequences)
    x = MaxPooling1D()(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu')(x)
    # x = Dropout(0.3)(x)
    # x = Dense(50, activation='relu')(x)
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
    x = Bidirectional(LSTM(256,
                           activation="relu",
                           return_sequences=True))(embedded_sequences)
    # x = Bidirectional(LSTM(256, activation="relu", return_sequences=True))(x)
    x = AttentionWithContext()(x)
    x = Dropout(0.2)(x)
    x = Dense(150, activation='relu')(x)
    x = Dense(75, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

    return sequence_input, x, preds


def split_train_test(train, test, pfs):
    out = []
    for pf in pfs:
        split = {"train":
                 {"seqs": pf["padded"][train],
                  "chars": pf["chars"][train],
                  "labels": pf["labels"][train]},
                 "test":
                 {"seqs": pf["padded"][test],
                  "chars": pf["chars"][test],
                  "labels": pf["labels"][test]},
                 "word_index": pf["word_index"],
                 "char_index": pf["char_index"],
                 "max_seq": pf["max_seq"],
                 "max_word": pf["max_word"],
                 "embeds": pf["embeds"]}
        out.append(split)

    return out

def format_for_fitting(pfs, split="train"):
    ins = []
    labels = []
    for pf in pfs:
        ins.append(pf[split]["padded"])
        ins.append(pf[split]["chars"])
        labels.append(pf[split]["labels"])
    return ins, labels

def make_model(pfs):
    inputs = []
    layers = []
    preds = []
    for pf in pfs:
        embedding_layer = Embedding(len(pf["word_index"]) + 1,
                                    EMBEDDING_DIM,
                                    weights=pf["embeds"],
                                    input_length=pf["max_seq"],
                                    trainable = False)

        char_embedding_layer = TimeDistributed(Embedding(len(pf["char_index"]) + 1,
                                                         output_dim = 32,
                                                         input_length = pf["max_word"]))

        sequence_input = Input(shape=(pf["max_seq"],), dtype='int32')
        char_input = Input(shape=(pf["max_seq"], pf["max_word"],),
                           dtype='int32')
        inputs.append(sequence_input)
        inputs.append(char_input)
        embedded_sequences = embedding_layer(sequence_input)
        x = LSTM(200,
                 activation="relu",
                 return_sequences=True)(embedded_sequences)
        # x = Conv1D(150, 5, activation='relu')(x)
        # x = MaxPooling1D()(x)
        # x = Flatten()(x)

        embedded_chars = char_embedding_layer(char_input)
        y = TimeDistributed(Flatten())(embedded_chars)
        y = LSTM(50, activation="relu", return_sequences=True)(y)

        merged = keras.layers.concatenate([x, y], axis=1)
        layers.append(merged)

    merged = keras.layers.concatenate(merged, axis=1)
    x = Dense(100, activation="relu", kernel_initializer =init)(merged)
    for n in range(len(pfs)):
        pred = Dense(1, activation="sigmoid", kernel_initializer=init)(x)
        preds.append(pred)

    return inputs, preds

def cross_validate(files, efile, n_folds):
    pfs = read_and_proc_multiple(files, efile)
    kf = KFold(n_splits=n_folds)
    corrs = []
    min_samples = get_min(pfs)
    for train, test in kf.split(pfs[min_samples]["labels"]):
        print("index test: ",test[0],":",test[-1])
        pfs = split_train_test(train, test, pfs)
        inputs, preds = make_model(pfs)
        # seqs_train_x = seqs[train]
        # chars_train_x = chars[train]
        # train_y = labels[train]
        # seqs_test_x = seqs[test]
        # chars_test_x = chars[test]
        # test_y = labels[test]
        # embedding_layer = Embedding(len(word_index) + 1,
        #                             EMBEDDING_DIM,
        #                             weights = [embeds],
        #                             input_length=MAX_SEQ,
        #                             trainable = False)

        # char_embedding_layer = TimeDistributed(Embedding(len(char_index) + 1,
        #                                                  output_dim = 32,
        #                                                  input_length = MAX_WORD))

        # sequence_input = Input(shape=(MAX_SEQ,), dtype='int32')
        # char_input = Input(shape=(MAX_SEQ, MAX_WORD,), dtype='int32')        
        # embedded_sequences = embedding_layer(sequence_input)
        # x = LSTM(200, activation="relu", return_sequences=True)(embedded_sequences)
        # # x = Conv1D(150, 5, activation='relu')(x)
        # # x = MaxPooling1D()(x)
        # # x = Flatten()(x)

        # embedded_chars = char_embedding_layer(char_input)
        # y = TimeDistributed(Flatten())(embedded_chars)
        # y = LSTM(50, activation="relu", return_sequences=True)(y)
        # merged = keras.layers.concatenate([x, y], axis=2)
        # x = Conv1D(250, 5, activation='relu')(merged)
        # x = MaxPooling1D()(x)
        # x = Flatten()(x)
        # #x = Dense(125, activation='relu')(x)
        # x = Dense(50, activation='relu')(x)
        # preds = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=preds)
        adam = optimizers.Adam(lr=0.001)
        model.compile(loss="mse",
                      optimizer=adam)
        x_train, y_train = format_for_fitting(pfs, split="train")
        x_test, y_test = format_for_fitting(pfs, split="test")
        model.fit(x_train, y_train, epochs=10, batch_size=10)
        preds = model.predict(x_test)
        print(preds)
        print(preds.shape)
    #     corr = pearsonr(test_y, preds)[0]
    #     print("Pearson correlation for fold:", corr)
    #     corrs.append(corr)    
    # print("Average Pearson correlation:", np.mean(corrs))
    # return corrs

EMBEDDING_DIM = 300
EFILE = "/data/s3094723/embeddings/en/w2v.twitter.edinburgh10M.400d.csv"
init = 'TruncatedNormal'
cross_validate(["../../data/EI-reg/En/traindev/EI-reg-En-anger-traindev.tok",
                # "../../data/EI-reg/En/traindev/EI-reg-En-fear-traindev.tok",
                "../../data/EI-reg/En/traindev/EI-reg-En-joy-traindev.tok",
                "../../data/EI-reg/En/traindev/EI-reg-En-sadness-traindev.tok"],
               EFILE, 5)
