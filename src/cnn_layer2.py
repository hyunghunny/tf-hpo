#!/usr/bin/python

"""Module CNN layer 2 hyperparameter optimization evaluation.
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

Revised for hyperparameter optimization evaluation.
Author: Hyunghun Cho

Set the appropriate arguments like follows:

    python cnn_layer2.py {neurons in layer 1} {neurons in layer 2} {neurons in fully connected layer}
    
    Parameters:
    {neurons in layer 1} : mendatory
    {neurons in layer 2} : optional, if it is skipped, a predetermined number of neurons in each layer 2 and fully connected layers will be used repeatly
    {neurons in fully connected layer} : optional, if it is skipped, a predetermined number of neurons in fully connected layers will be used repeatly
"""
import sys
import getopt
import tensorflow as tf
import time
import logging
import logging.handlers

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
neurons = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
logger = None

# MNIST importer
def get_mnist():
    # Import MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    return mnist

# logging untility    
def create_logger():
    # Create logger instance
    logger = logging.getLogger('tflogger')

    # Create logger formatter
    #fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
    file_formatter = logging.Formatter('%(asctime)s, %(message)s')
    # Create handles to redirect the log to each stream and file
    fileHandler = logging.FileHandler('../log/mnist-cnn-2.csv')
    #streamHandler = logging.StreamHandler()

    # Set the formatter for each handlers
    fileHandler.setFormatter(file_formatter)
    #streamHandler.setFormatter(file_formatter)

    # Attach the handlers to logger instance
    logger.addHandler(fileHandler)
    #logger.addHandler(streamHandler)

    logger.setLevel(logging.DEBUG)
    #print "logger created"
    return logger

def delete_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    #print "logger deleted"
        
# Create 2 layers conv model with a specific neurons
def conv_net_2(x, conv_1_output, conv_2_output, fully_output, dropout):
    
    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, a specific layer 1 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, conv_1_output])),
        # 5x5 conv, the layer 1 inputs, a specific layer 2 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, conv_1_output, conv_2_output])),
        # fully connected, 7*7*64 inputs, a specific outputs
        'wd1': tf.Variable(tf.random_normal([7*7*conv_2_output, fully_output])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([fully_output, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([conv_1_output])),
        'bc2': tf.Variable(tf.random_normal([conv_2_output])),
        'bd1': tf.Variable(tf.random_normal([fully_output])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out    


# Design scalable and iterative training function
def train_mnist_nn(tag, mnist, model_func, **params):
    '''training cnn with  various hyperparameters
    
    keyword arguments:
    tag -- the brief name of neural network
    dataset -- dataset for 
    model_func -- the model function to be used
    params -- vararg for neurons at each hidden layers   
    '''
    # declare global variable use
    global logger
        
    # tensorboard configuration
    tb_logs_path = "./logs/" + tag 
    
    # Parameters
    learning_rate = 0.001
    training_iters = 200000
    batch_size = 128
    display_step = 10  
    dropout = 0.75 # Dropout, probability to keep units
    
    # Structure parameters
    conv_1_output = params["conv_1_output"]
    conv_2_output = params["conv_2_output"]
    fully_output = params["fully_output"]     

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # dropout (keep probability)
    keep_prob = tf.placeholder(tf.float32) 
    
    # Construct model
    pred = model_func(x, conv_1_output, conv_2_output, fully_output, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # create a summary for our cost and accuracy
    #tf.scalar_summary("cost", cost)
    #tf.scalar_summary("accuracy", accuracy)

    # merge all summaries into a single "operation" which we can execute in a session 
    #summary_op = tf.merge_all_summaries()

    
    # Initializing the variables
    init = tf.initialize_all_variables()
    logger.info("Tag Name,L1,L2,L3,Iteration,Minibatch Loss,Training Accuracy,Elapsed Time,Testing Accuracy" )

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1

        # create log writer object
        #writer = tf.train.SummaryWriter(tb_logs_path, graph=tf.get_default_graph())

        timestamp = time.time() # measure start timestamp for each display steps
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)            
            #_, summary = sess.run([optimizer, summary_op], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            _ = sess.run([optimizer], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                perf = tag +"," + str(step*batch_size) + "," + \
                      "{:.6f}".format(loss) + "," + \
                      "{:.5f}".format(acc) + "," + "{0:.3g}".format(time.time() - timestamp)
                logger.debug(perf)
            step += 1

            # write log
            #writer.add_summary(summary, step * batch_size)

            timestamp = time.time() # reset timestamp
            
        # Calculate accuracy for 256 mnist test images
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                          y: mnist.test.labels[:256],
                                          keep_prob: 1.})
        logger.info(tag +",,,,," + str(test_acc))
        print tag + " : " + str(test_acc)
        sess.close()
        
    return

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def train(dataset, layer1_out, layer2_out, fully) :

    tag = "mnist_cnn_2," + str(layer1_out) + "," + str(layer2_out) + "," + str(fully)
    #print "start training by " + tag
    
    train_mnist_nn(tag, dataset, conv_net_2, conv_1_output=layer1_out, conv_2_output=layer2_out, fully_output=fully)
    return


def main():
    # declare global variable use
    global logger    
    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    except getopt.error, msg:
        print msg
        print "for help use --help"
        sys.exit(2)
    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)
   
    # create global logger
    if logger is None:
        logger = create_logger()

    # get MNIST dataset
    mnist = get_mnist();

    try:
        # process arguments
        if len(args) == 3: 
            train(mnist, int(args[0]), int(args[1]), int(args[2])) 
        elif len(args) == 2:
            for i in neurons:
                train(mnist, int(args[0]), int(args[1]), i)
        elif len(args) == 1:
            print "TODO:future works"
        else:
            print __doc__
    except:
        e = sys.exc_info()[0]
        print e
        delete_logger(logger) # catch exception to remove logger

    delete_logger(logger)
    sys.exit(0)
                      
if __name__ == "__main__":
    main()


