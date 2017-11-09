# WORK IN PROGRESS











from __future__ import print_function
import numpy as np
import csv
import os
from scipy.stats.stats import pearsonr
import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
filename_queue = tf.train.string_input_producer(["EI-reg-En-anger-train.vectors.without.random.train.csv"])


def read_my_file_format(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=0)
    _, csv_row = reader.read(filename_queue)
    record_defaults =  [['a'] for cl in range((45))]
    parsed_line = tf.decode_csv(csv_row,record_defaults = record_defaults )
    label = parsed_line[1] #the intensity
    
    # first element is the label
    del parsed_line[1] # Delete first element
    del parsed_line[0] #the emotion
    feature = parsed_line
#    print(label)
    return feature, label

def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, 
                                                    num_epochs=num_epochs, 
                                                    shuffle=True)
    feature, label = read_my_file_format(filename_queue)
    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_size
    feature_batch, label_batch = tf.train.shuffle_batch([feature, label], 
                                                        batch_size=batch_size, 
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    print(label_batch)
    return feature_batch, label_batch


input_data = "EI-reg-En-anger-train.vectors.without.random.train.csv"

total_lines = len(open("EI-reg-En-anger-train.vectors.without.random.train.csv", 'r').readlines()) - 1

# Training Parameters
learning_rate = 0.001
training_steps = 100
batch_size = 3
display_step = 1

# Network Parameters
num_input = 43 # MNIST data input (img shape: 28*28)
timesteps = 43 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 1 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Store layers weight & bias
# weights = {
#     'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1])),
#     'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
#     'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
#     'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
#     'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_classes]))
# }
# biases = {
#     'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
#     'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
#     'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
#     'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
#     'out': tf.Variable(tf.truncated_normal([n_classes]))
# }

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
# def multilayer_perceptron(x):
#     # Hidden fully connected layer with 256 neurons
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     layer_1 = tf.nn.relu(layer_1)
#     drop_out = tf.nn.dropout(layer_1, keep_prob) 
#     # Hidden fully connected layer with 256 neurons
#     layer_2 = tf.add(tf.matmul(drop_out, weights['h2']), biases['b2'])
#     layer_2 = tf.nn.relu(layer_2)
#     # Hidden fully connected layer with 256 neurons
#     layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
#     layer_3 = tf.nn.relu(layer_3)
#     # Output fully connected layer with a neuron for each class
#     layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
#     layer_4 = tf.nn.relu(layer_4)
#     out_layer = tf.nn.sigmoid(tf.matmul(layer_4, weights['out']) + biases['out'])
#  #   out_layer = tf.matmul(layer_4, weights['out']) + biases['out']   
#     return out_layer

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

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.losses.mean_squared_error(Y, logits))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon = 1e-09)
train_op = optimizer.minimize(loss_op)

pearson = tf.contrib.metrics.streaming_pearson_correlation(logits, Y, name="pearson")
accuracy = tf.reduce_mean(pearson)

init = tf.global_variables_initializer()

"""
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    #test_len = 128
    #test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    #test_label = mnist.test.labels[:test_len]
    #print("Testing Accuracy:", \
    #    sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

# Construct model
#intensity = multilayer_perceptron(X)

# Define loss and optimizer
#pearson correlation as loss function
#length = batch_size

#apply regularization (l2)
#Beta = 0.01
#regularizer = tf.nn.l2_loss(weights['h1']) +   tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['h3']) + tf.nn.l2_loss(weights['h4'])

#MSE loss


#loss_op = tf.reduce_mean(loss_op + Beta * regularizer)

#Switch to absol. diff. as loss?
#loss_op = tf.reduce_mean(tf.losses.absolute_difference(Y, intensity))

#used to report correlation 
#pearson = tf.contrib.metrics.streaming_pearson_correlation(intensity, Y, name="pearson")


#pearson corr. as loss?
# multiply by -1 to maximize correlation i.e. minimize negative correlation 
#original_loss = -1 * length * tf.reduce_sum(tf.multiply(intensity, Y)) - (tf.reduce_sum(intensity) * tf.reduce_sum(Y))
#divisor = tf.sqrt(
#            (length * tf.reduce_sum(tf.square(intensity)) - tf.square(tf.reduce_sum(intensity)))) *\
#            tf.sqrt(
#            length * tf.reduce_sum(tf.square(Y)) - tf.square(tf.reduce_sum(Y)))

#correlation = tf.truediv(original_loss, divisor)


#Init optimizer
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon = 1e-09)
#train_op = optimizer.minimize(loss_op)
"""

# Initializing the variables
#init = tf.global_variables_initializer()
#init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:
    # start populating filename queue
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    features, labels = input_pipeline([input_data], batch_size)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
      
    # Training cycle
    for epoch in range(training_steps):
        avg_cost = 0.
        print(total_lines)
        total_batch = int(total_lines/batch_size)
        print(total_batch)
        # Loop over all batches
        for x in range(total_batch) :
            # Run optimization op (backprop) and cost op (to get loss value)
            feature_batch, label_batch = sess.run([features, labels])
            batch_y = np.reshape(label_batch, (-1, 1))
            batch_x =  [x for x in feature_batch]
            batch_x = np.reshape(batch_x, (-1, 43))
#            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
 #                                                            Y: batch_y, keep_prob : 0.85})

            _, c, p = sess.run([train_op, loss_op, pearson], feed_dict={X: batch_x,
                                                             Y: batch_y, keep_prob : 1.0})
            
            #q = sess.run (intensity, feed_dict= {X: batch_x, keep_prob: 0.85}) #for debugging
#            print(p[0])  
        # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")
    coord.request_stop()
    coord.join(threads) 


    #Load and fix up test data
    range = [i for i in range(2,45)]
    df=pd.read_csv('EI-reg-En-anger-train.vectors.without.random.test.csv',usecols=range, header=None)
    d = df.values
    l = pd.read_csv('EI-reg-En-anger-train.vectors.without.random.test.csv',usecols = [1] ,header=None)
    labels = l.values
    data = np.float32(d)
    labels = np.array(l,'float')


    x = tf.convert_to_tensor(data, dtype=tf.float32)   
    y = tf.convert_to_tensor(labels, dtype=tf.float32)
    test_x, test_y = sess.run([x, y])
    #print(test_y)
    test_y = np.reshape(test_y, (-1, 1))
    test_x = np.reshape(test_x, (-1, 43))
   # print(test_x)
  



    
    # Test model
    #run model on test data
    scores = sess.run(intensity, feed_dict = {X: test_x,  keep_prob:1})
    for x,y in zip(scores, test_y):
         print(x, y)
    corr = pearsonr(scores, test_y)
    print(corr[0])