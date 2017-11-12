import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
from collections import defaultdict
import re
from random import randint
from scipy.stats.stats import pearsonr

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def read_files(train,test):
    train = pd.read_table(train, header=None)
    test = pd.read_table(test, header=None)
    train_x, train_y, test_x, test_y = train[1], train[3], test[1], test[3]
    #print(train_x)
    return train_x, train_y, test_x, test_y


def get_embeddings(efile_name):
    efile = open(efile_name, 'r')
    first_400 = [x for x in range(0,400)]
    embeddings = np.loadtxt(efile, dtype=np.float32, usecols = first_400, skiprows= 1, delimiter = '\t')
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
    tweets_vec_train = np.zeros(shape=(len(train_x), maxSeqLength), dtype=np.int32)
    tweets_vec_test = np.zeros(shape=(len(test_x), maxSeqLength), dtype=np.int32)
    for no, tweet in enumerate(train_x):
        wordIndex = 0
        cleanedLine = cleanSentences(tweet)
        split = cleanedLine.split()
        for word in split:
           try:
#               print(word, np.where(vocab==word)[0][0])
               if word in vocab:
                   tweets_vec_train[no][wordIndex] = np.where(vocab==word)[0][0]
           except ValueError:
               tweets_vec_train[no][wordIndex] = np.where(vocab=='mva-unk')[0][0] #Vector for unkown words
           wordIndex = wordIndex + 1
           if wordIndex >= maxSeqLength:
               break
    for no, tweet in enumerate(test_x):
        wordIndex = 0
        cleanedLine = cleanSentences(tweet)
        split = cleanedLine.split()
        for word in split:
           try:
               if word in vocab:
                   tweets_vec_test[no][wordIndex] = np.where(vocab==word)[0][0]             
           except ValueError:
               tweets_vec_test[no][wordIndex] = np.where(vocab=='mva-unk')[0][0] #Vector for unkown words
           wordIndex = wordIndex + 1
           if wordIndex >= maxSeqLength:
               break
    #print(tweets_vec_train[1][1])
    return tweets_vec_train, tweets_vec_test


def to_tensor(tweets_vec, labels): 
    tweets_vec = tf.constant(tweets_vec)  # X is a np.array
    labels = tf.constant(labels.values)  # y is a np.array
    return tweets_vec, labels


def generate_batches(tweets_vec_train, train_y, batch_size, num_epochs):
    x,y = to_tensor(tweets_vec_train, train_y)
    x_1,y_1 = tf.train.slice_input_producer([x,y])    
    print(y_1)
    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_size
    feature_batch, label_batch = tf.train.shuffle_batch([x_1, y_1], 
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

train_file = "EI-reg-En-anger-train.txt"
test_file = "EI-reg-En-anger-dev.txt"
efile_name = "/data/s3094723/embeddings/en/w2v.twitter.edinburgh10M.400d.csv"
train_x, train_y, test_x, test_y = read_files(train_file, test_file)
test_y = test_y.values
print(test_y.shape)
learning_rate = 0.001
training_epochs = 40
batch_size = 8
display_step = 1
embedding_dim = 400

maxSeqLength = get_avg_sent_len(train_x)
timesteps = maxSeqLength
num_hidden = 256 
num_classes = 1 
total_samples = train_x.shape[0]

embeddings, vocab = get_embeddings(efile_name)
train_x, test_x = vectorize(train_x, test_x, embeddings, vocab, maxSeqLength)

X = tf.placeholder(tf.int32, [batch_size, maxSeqLength])
Y = tf.placeholder(tf.float32, [batch_size, num_classes])

data = tf.Variable(tf.zeros([batch_size, maxSeqLength, embedding_dim], dtype=tf.float32),dtype=tf.float32)
data = tf.nn.embedding_lookup(embeddings,X)


weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

output = RNN(data, weights, biases)
prediction = tf.nn.sigmoid(output)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.losses.mean_squared_error(Y, prediction))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon = 1e-09)
train_op = optimizer.minimize(loss_op)

#Evaluate model
pearson = tf.contrib.metrics.streaming_pearson_correlation(output, Y, name="pearson")

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    features, labels = generate_batches(train_x, train_y, batch_size, num_epochs=training_epochs)
    print(labels)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)    
    for epoch in range(training_epochs):
        avg_cost = 0.
        print(total_samples)
        total_batch = int(total_samples/batch_size)
        print(total_batch)
        # Loop over all batches
        for x in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            feature_batch, label_batch = sess.run([features, labels])
#            print(feature_batch.shape)
            batch_y = np.reshape(label_batch, (-1, 1))
            batch_x =  [x for x in feature_batch]
            batch_x = np.reshape(batch_x, (batch_size, timesteps))
            #batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            _, c, p = sess.run([train_op, loss_op, pearson], feed_dict={X: batch_x,
                                                             Y: batch_y})
            print(p[0])
        # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")
    coord.request_stop()
    coord.join(threads)
    all_scores = []     
    no_of_sub_arrays = len(test_y)/batch_size
    test_x_sub = np.split(test_x, no_of_sub_arrays)
    test_y_sub = np.split(test_y, no_of_sub_arrays)
    for i,j in zip(test_x_sub,test_y_sub):
        x = tf.convert_to_tensor(i)   
        y = tf.convert_to_tensor(j)
        test_x_1, test_y_1 = sess.run([x, y])
      #  print(test_y_1)
        test_y_1 = np.reshape(test_y_1, (batch_size, 1))
        test_x_1 = np.reshape(test_x_1, (batch_size, 16))
        score = sess.run(prediction, feed_dict = {X: test_x_1})
     #   print(score, test_y_1)
        score = score.tolist()
        all_scores.extend(score)

    all_scores = [item for sublist in all_scores for item in sublist]
    print(all_scores, test_y) 
    corr = pearsonr(all_scores, test_y)
    print(corr[0])
