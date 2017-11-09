import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
from collections import defaultdict
import re
from random import randint


strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def read_files(train,test):
    train = pd.read_table(train, header=None)
    test = pd.read_table(test, header=None)
    train_x, train_y, test_x, test_y = train[1], train[3], test[1], test[3]
    return train_x, train_y, test_x, test_y


def get_embeddings(efile_name):
    efile = open(efile_name, 'r')
    first_400 = [x for x in range(0,400)]
    embeddings = np.loadtxt(efile, dtype=float, usecols = first_400, skiprows= 1, delimiter = '\t')
    efile.close()
    efile = open(efile_name, 'r')
    vocab = np.loadtxt(efile,  usecols = 400, dtype = np.unicode_, skiprows = 1, delimiter = '\t')
    return embeddings, vocab

def get_avg_sent_len(train_x):
    lens = 0
    for x in train_x:
        lens = lens + len(x.split())
    return int(lens/len(train_x))


def vectorize(train_x, test_x, embeddings, vocab, maxSeqLength):
    tweets_vec_train = np.zeros(shape=(len(train_x), maxSeqLength), dtype='int32')
    tweets_vec_test = np.zeros(shape=(len(test_x), maxSeqLength), dtype='int32')
    for no, tweet in enumerate(train_x):
        wordIndex = 0
        cleanedLine = cleanSentences(tweet)
        split = cleanedLine.split()
        for word in split:
           try:
               tweets_vec_train[no][wordIndex] = embeddings[np.where(vocab==word)]
           except ValueError:
               tweets_vec_train[no][wordIndex] = embeddings[np.where(vocab=='<unk>')] #Vector for unkown words
           wordIndex = wordIndex + 1
           if wordIndex >= maxSeqLength:
               break
    for no, tweet in enumerate(test_x):
        wordIndex = 0
        cleanedLine = cleanSentences(tweet)
        split = cleanedLine.split()
        for word in split:
           try:
               tweets_vec_test[no][wordIndex] = embeddings[np.where(vocab==word)]
           except ValueError:
               tweets_vec_test[no][wordIndex] = embeddings[np.where(vocab=='<unk>')] #Vector for unkown words
           wordIndex = wordIndex + 1
           if wordIndex >= maxSeqLength:
               break
    return tweets_vec_train, tweets_vec_test


def to_tensor(tweets_vec, labels):
    tweets_vec = tf.constant(tweets_vec, dtype=tf.float32)  # X is a np.array
    labels = tf.constant(train_y, dtype=tf.string)  # y is a np.array
    return tweets_vec, labels


def generate_batches(tweets_vec_train, train_y, batch_size, num_epochs=None):
    x,y = to_tensor(tweets_vec_train, train_y)
    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_size
    feature_batch, label_batch = tf.train.shuffle_batch([x, y], 
                                                        batch_size=batch_size, 
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    #print(label_batch)

    return feature_batch, label_batch

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

train_file = "../../data/EI-reg-English-Train/EI-reg-en_anger_train.txt"
test_file = "../../data/dev/EI-reg-En-anger-dev.txt"
train_feature_file= "EI-reg-En-anger-train.vectors.without.random.train.csv"
test_feature_file= "EI-reg-En-anger-train.vectors.without.random.test.csv"
efile_name = "/data/s3094723/embeddings/w2v.twitter.edinburgh10M.400d.csv"

train_x, train_y, test_x, test_y = read_files(train_file, test_file)
train_x, train_y, lex = integerize(train_x, test_x)
train_features, test_features = read_features(train_feature_file, test_feature_file)


learning_rate = 0.001
training_epochs = 100
batch_size = 128
display_step = 20
embedding_dim = 400

maxSeqLenght = get_avg_sent_len(train_x)
padding = train_x.shape[0] #??
timesteps = 1 #??
num_hidden = 128 
num_classes = 1 
total_samples = train_x.shape[0] #??


# #load embeddings
# embeddings, vocab, vocab_size = get_embeddings(efile_name)
#
# W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
#                    trainable=False, name="W")
# V = tf.Variable(tf.constant('', shape=[vocab_size]),
#                    trainable=False, name="V", dtype =tf.string )
# embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
# vocab_placeholder = tf.placeholder(tf.string, [vocab_size])
# embedding_init = W.assign(embedding_placeholder)
# vocab_init = V.assign(vocab_placeholder)
# #sess.run([embedding_init, vocab_init], feed_dict={embedding_placeholder: embeddings, vocab_placeholder: vocab})


X = tf.placeholder("float", [None, padding, timesteps])
Y = tf.placeholder("float", [None, num_classes])

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

output = RNN(X, weights, biases)
prediction = tf.nn.softmax(output)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.losses.mean_squared_error(Y, output))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon = 1e-09)
train_op = optimizer.minimize(loss_op)

#Evaluate model
pearson = tf.contrib.metrics.streaming_pearson_correlation(output, Y, name="pearson")

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    #sess.run(tf.local_variables_initializer())
    features, labels = generate_batches(train_x, train_y, batch_size)
    #coord = tf.train.Coordinator()
    #threads = tf.train.start_queue_runners(coord=coord)    
    for epoch in range(training_epochs):
        avg_cost = 0.
        print(total_samples)
        total_batch = int(total_samples/batch_size)
        print(total_batch)
        # Loop over all batches
        for x in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            feature_batch, label_batch = sess.run([features, labels])
            print(feature_batch.shape)
            batch_y = np.reshape(label_batch, (-1, 1))
            batch_x =  [x for x in feature_batch]
            batch_x = np.reshape(batch_x, (batch_size, timesteps, padding))
            #batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            _, c, p = sess.run([train_op, loss_op, pearson], feed_dict={X: batch_x,
                                                             Y: batch_y, keep_prob : 1.0})
        # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")
    #coord.request_stop()
    #coord.join(threads)
