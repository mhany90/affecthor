import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
from collections import defaultdict

def read_files(train,test):
    train = pd.read_table(train, header=None)
    test = pd.read_table(test, header=None)

    train_x, train_y, test_x, test_y = train[1], train[3], test[1], test[3]

    return train_x, train_y, test_x, test_y

def read_features(train, test):
    train = pd.read_csv(train, header=None)
    test = pd.read_csv(test, header=None)

    return train.iloc[:,2:], test.iloc[:,2:]

def get_avg_sent_len(train_x):
    lens = 0
    for x in train_x:
        lens = lens + len(x.split())

    return int(lens/len(train_x))

def integerize(train_x, test_x):
    lex = defaultdict()
    padding = get_avg_sent_len(train_x)
    ints_train = np.zeros(shape = (len(train_x), padding))
    ints_test = np.zeros(shape = (len(test_x), padding))

    for n, tweet in enumerate(train_x):
        ints = np.zeros((padding), dtype = "int32")
        tweet = tweet.split()
        for word in tweet:
            if (tweet.index(word) + 1) > padding:
                break
            if lex.get(word.lower()) == None:
               lex[word.lower()] = len(lex.keys())
            ints[tweet.index(word)] = lex[word.lower()]
        ints_train[n] = ints

    for n, tweet in enumerate(test_x):
        ints = np.zeros((padding), dtype="int32")
        tweet = tweet.split()
        for word in tweet:
            if (tweet.index(word) + 1) > padding:
                break
            try:
                ints[tweet.index(word)] = lex[word.lower()]
            except KeyError:
                ints[tweet.index(word)] = len(set(lex.keys()))
        ints_test[n] = ints

    return ints_train, ints_test, lex

def generate_batches(x, y, batch_size, num_epochs=None):
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

train_x, train_y, test_x, test_y = read_files(train_file, test_file)
train_x, train_y, lex = integerize(train_x, test_x)
train_features, test_features = read_features(train_feature_file, test_feature_file)

learning_rate = 0.001
training_epochs = 100
batch_size = 128
display_step = 200

padding = train_x.shape[0]
#timesteps = 43
timesteps = 1
num_hidden = 128 
num_classes = 1 
total_samples = train_x.shape[0]

X = tf.placeholder("float", [None, padding, timesteps])
Y = tf.placeholder("float", [None, num_classes])

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.losses.mean_squared_error(Y, logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon = 1e-09)
train_op = optimizer.minimize(loss_op)

#Evaluate model
pearson = tf.contrib.metrics.streaming_pearson_correlation(logits, Y, name="pearson")

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

lwith tf.Session() as sess:

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
