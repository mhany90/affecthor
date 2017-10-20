
from __future__ import print_function
import numpy as np
import csv
import os
import tensorflow as tf
import pandas as pd
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
filename_queue = tf.train.string_input_producer(["v100.csv"])

batch_size = 100
def read_my_file_format(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = reader.read(filename_queue)
    record_defaults =  [['a'] for cl in range((44))]
    parsed_line = tf.decode_csv(csv_row,record_defaults = record_defaults )
    label = parsed_line[0] # first element is the label
    del parsed_line[0] # Delete first element
    feature = parsed_line
    return feature, label

def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, 
                                                    num_epochs=num_epochs, 
                                                    shuffle=True)
    feature, label = read_my_file_format(filename_queue)
    min_after_dequeue = 500
    capacity = min_after_dequeue + 3 * batch_size
    feature_batch, label_batch = tf.train.shuffle_batch([feature, label], 
                                                        batch_size=batch_size, 
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
    return feature_batch, label_batch


input_data = 'v100.csv'

total_lines = len(open('v100.csv', 'r').readlines()) - 1
 
# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 50 # 1st layer number of neurons
n_hidden_2 = 50 # 2nd layer number of neurons
n_hidden_3 = 50
n_input = 43 # MNIST data input (img shape: 28*28)
n_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("int32", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Hidden fully connected layer with 256 neurons
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # start populating filename queue
    sess.run(init)
    #call input pipeline and start queue runners (to load training bata in batches)
    features, labels = input_pipeline([input_data], batch_size)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
  
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0. 
        print(total_lines) 
        total_batch = int(total_lines/batch_size) #total number of training examples / batch size
        print(total_batch)
        # Loop over all batches
        for x in range(total_batch) :
            # Run optimization op (backprop) and cost op (to get loss value)
            feature_batch, label_batch = sess.run([features, labels])  #get a new batch            
            batch_y = sess.run(tf.one_hot(label_batch, 2, 1.0, 0.0))   #convert labels of 1(positive) and 0(negative) to one-hot-tensors
            batch_y = np.reshape(batch_y, (-1, 2))       #-1 because the batch size can be anything
            batch_x =  [x for x in feature_batch]        #probably okay to just say batch_x = feature_batch
            batch_x = np.reshape(batch_x, (-1, 43))      #43 is the number of features, -1 for batch size
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,       #Run training operation, and loss operation, returns loss
                                                              Y: batch_y})
          # Compute average loss
            avg_cost += c / total_batch            #average loss over all batches till now 
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")
    coord.request_stop()
    coord.join(threads) 


    #Load and fix up test data
    range = [i for i in range(1,44)]
    df=pd.read_csv('vtest2.csv',usecols=range, header=None)
    d = df.values
    l = pd.read_csv('vtest2.csv',usecols = [0] ,header=None)
    labels = l.values
    data = np.float32(d)
    labels = np.array(l,'str')


    x = tf.convert_to_tensor(data, dtype=tf.float32)   
    y = tf.convert_to_tensor(labels, dtype=tf.string)
    test_x, test_y = sess.run([x, y])
   # print(test_y)
    test_y = sess.run(tf.one_hot(test_y, 2, 1.0, 0.0))
   # print(test_y)
    test_y = np.reshape(test_y, (-1, 2))
    test_x = np.reshape(test_x, (-1, 43))
   # print(test_x)
  



    
    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(test_y, 1))
#    print(sess.run(correct_prediction))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: test_x, Y: test_y}))
