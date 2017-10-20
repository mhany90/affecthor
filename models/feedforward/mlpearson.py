
from __future__ import print_function
import numpy as np
import csv
import os
from scipy.stats.stats import pearsonr
import tensorflow as tf
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
 
# Parameters
learning_rate = 0.001
training_epochs = 100
display_step = 1
batch_size = 3
# Network Parameters
n_hidden_1 = 20 # 1st layer number of neurons
n_hidden_2 = 15 # 2nd layer number of neurons
n_hidden_3 = 10
n_hidden_4 = 5
n_input = 43 
n_classes = 1 

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    drop_out = tf.nn.dropout(layer_1, keep_prob) 
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(drop_out, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Hidden fully connected layer with 256 neurons
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Output fully connected layer with a neuron for each class
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    out_layer = tf.nn.sigmoid(tf.matmul(layer_4, weights['out']) + biases['out'])
 #   out_layer = tf.matmul(layer_4, weights['out']) + biases['out']   
    return out_layer

# Construct model
intensity = multilayer_perceptron(X)

# Define loss and optimizer
#pearson correlation as loss function
length = batch_size

#apply regularization (l2)
Beta = 0.01
regularizer = tf.nn.l2_loss(weights['h1']) +   tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['h3']) + tf.nn.l2_loss(weights['h4'])

#MSE loss
loss_op = tf.reduce_mean(tf.losses.mean_squared_error(Y, intensity))
loss_op = tf.reduce_mean(loss_op + Beta * regularizer)

#Switch to absol. diff. as loss?
#loss_op = tf.reduce_mean(tf.losses.absolute_difference(Y, intensity))

#used to report correlation 
pearson = tf.contrib.metrics.streaming_pearson_correlation(intensity, Y, name="pearson")


#pearson corr. as loss?
# multiply by -1 to maximize correlation i.e. minimize negative correlation 
#original_loss = -1 * length * tf.reduce_sum(tf.multiply(intensity, Y)) - (tf.reduce_sum(intensity) * tf.reduce_sum(Y))
#divisor = tf.sqrt(
#            (length * tf.reduce_sum(tf.square(intensity)) - tf.square(tf.reduce_sum(intensity)))) *\
#            tf.sqrt(
#            length * tf.reduce_sum(tf.square(Y)) - tf.square(tf.reduce_sum(Y)))

#correlation = tf.truediv(original_loss, divisor)


#Init optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon = 1e-09)
train_op = optimizer.minimize(loss_op)


# Initializing the variables
init = tf.global_variables_initializer()
#init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:
    # start populating filename queue
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    features, labels = input_pipeline([input_data], batch_size)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
      
    # Training cycle
    for epoch in range(training_epochs):
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

