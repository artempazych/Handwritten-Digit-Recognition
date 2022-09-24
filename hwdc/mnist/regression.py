# Import the necessary libraries to work with neural networks
import os
import model
import tensorflow as tf

# Download the MNIST data set
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Describe the graph model that is responsible for the neural network of regression
with tf.variable_scope("regression"):
    x = tf.placeholder(tf.float32, [None, 784])
    y, variables = model.regression(x)

# Train the network
y_ = tf.placeholder("float", [None, 10])
# Describe the Crossentropy cost function
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# Use GradientDescentOptimizer (градіентний спуск)
# learining rate - 0.01
# minimize the cost function described above
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# Enter the variable responsible for the correct prediction
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# Enter the variable that corresponds to the accuracy of the training
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        # mini-batch size - 100
        batch_xs, batch_ys = data.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # Display the accuracy of the training 
    print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels}))

    # Save the received parameters (weights, biases) in the corresponding file
    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'),
        write_meta_graph=False, write_state=False)
    print("Saved:", path)
