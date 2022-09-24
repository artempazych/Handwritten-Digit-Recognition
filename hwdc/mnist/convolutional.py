# Import the necessary libraries to work with neural networks
import os
import model
import tensorflow as tf

# Download the MNIST data set
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Describe the graph model that will be responsible for the convolutional neural network
with tf.variable_scope("convolutional"):
    x = tf.placeholder(tf.float32, [None, 784])
    keep_prob = tf.placeholder(tf.float32)
    y, variables = model.convolutional(x, keep_prob)

# Train the network
y_ = tf.placeholder(tf.float32, [None, 10])
# Describe the Crossentropy cost function
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# Use AdamOprimizer as a kind of gradient descent
# learning rate - 1e-4
# minimize the cost function described above
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# Enter the variable responsible for the correct prediction
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# Enter the variable that corresponds to the accuracy of the training
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        # mini-batch size - 50
        batch = data.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # Display the accuracy of the training in the appropriate step
            print("step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))

    # Save the received parameters (weights, biases) in the corresponding file
    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'data', 'convolutional.ckpt'),
        write_meta_graph=False, write_state=False)
    print("Saved:", path)
