# Import a library for working with neural networks
import tensorflow as tf


# Create 'regression' function, which will have variables for the regression neural network
# The function receives the parameter х - input signals of the neural network
# Variable 'W' - matrix of weights 
# Variable 'b' - matrix of biases   
# Variable 'y' - matrix of weighted sums + bias 
# Returns the variables listed above   
def regression(x):
    W = tf.Variable(tf.zeros([784, 10]), name="W")
    b = tf.Variable(tf.zeros([10]), name="b")
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    return y, [W, b]


# Create 'convolutional' function, which will have variables for the multi-layer convolutional neural network  
# The function receives the parameter х - input signals of the neural network, keep_prob - dropout rate
def convolutional(x, keep_prob):

    # Create 'conv2d' function
    # Returns parameters of convolutional layer
    # (x - matrix of input signals,
    # W - matrix of weights,
    # strides - step of convolutional kernel)
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # Create'max_pool_2x2' function    
    # Returns parameters of pooling layer
    # (x - matrix of input signals,
    # ksize - pooling field 2х2,
    # strides - step of pooling field)
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Create 'weight_variable' function, initializes weights for network
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # Create 'bias_variable' function, initializes biases for network
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # Firsh convolutional layer
    # Change tensor size
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # Set the size of the kernel of the convolution (5x5), and the number of feature maps (32)
    W_conv1 = weight_variable([5, 5, 1, 32])
    # Set the size of weights matrix
    b_conv1 = bias_variable([32])
    # Use the activation function ReLU to weighted sum + bias
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # Seal feature maps using a pooling layer
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer
    # Set the size of the kernel of the convolution (5x5)
    # the number of input feature maps (32),
    # the number of output feature maps (64)
    W_conv2 = weight_variable([5, 5, 32, 64])
    # Set the size of weights matrix
    b_conv2 = bias_variable([64])
    # Use the activation function ReLU to weighted sum + bias
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # Seal feature maps using a pooling layer
    h_pool2 = max_pool_2x2(h_conv2)

    # Move to fully-connected neural network
    # Set the corresponding variables and their parameters
    # Use the activation function ReLU to weighted sum + bias
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout layer to avoid overfitting
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Output layer, appropriate variables and their parameters
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return y, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]
