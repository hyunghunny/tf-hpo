# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import time
import traceback

import numpy
from six.moves import urllib
from six.moves import xrange    # pylint: disable=redefined-builtin

import mnist_data as mnist

import tensorflow as tf
#from config import Config
#from util import CSVLogger

import HPOlib.benchmark_util as benchmark_util
import HPOlib.benchmark_functions as benchmark_functions


# DEFINE FLAGS
def define_flags():
    tf.app.flags.DEFINE_boolean("log_only", True, "True if logging test accuracy only.")    
    
    tf.app.flags.DEFINE_integer("filter_size", 5, "Set filter size. default is 5")
    tf.app.flags.DEFINE_integer("conv1_depth", 32, "Set first conv layer depth. default is 32.")
    tf.app.flags.DEFINE_integer("conv2_depth", 64, "Set second conv layer depth. default is 64.")
    tf.app.flags.DEFINE_integer("fc_depth", 512, "Set fully connected layer depth. default is 512.")
        
FLAGS = tf.app.flags.FLAGS

# DEFINE CONSTANTS
IMAGE_SIZE = mnist.IMAGE_SIZE
NUM_CHANNELS = mnist.NUM_CHANNELS
PIXEL_DEPTH = mnist.PIXEL_DEPTH
NUM_LABELS = mnist.NUM_LABELS

VAR_INIT_VALUE = 0.1 # hyperparameter for variable initial value
DROPOUT_RATE = 0.5
SEED = 66478    # Set to None for random seed.

VALIDATION_SIZE = 5000    # Size of the validation set.
BATCH_SIZE = 64
NUM_EPOCHS = 3
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100    # Number of steps between evaluations.

TRAIN_DEVICE_ID = "gpu:1"
EVAL_DEVICE_ID ="gpu:2"
LOG_PATH = "test.log"

    
# initialize tensorflow variables which are required to learning
def init_vars(filter_size, conv1_depth, conv2_depth, fc_depth):
    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
            tf.truncated_normal([filter_size, filter_size, NUM_CHANNELS, conv1_depth],
                                                    stddev=VAR_INIT_VALUE,
                                                    seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([conv1_depth]))
    
    conv2_weights = tf.Variable(
            tf.truncated_normal([filter_size, filter_size, conv1_depth, conv2_depth],
                                                    stddev=VAR_INIT_VALUE,
                                                    seed=SEED))
    
    conv2_biases = tf.Variable(tf.constant(VAR_INIT_VALUE, shape=[conv2_depth]))
    
    fc1_weights = tf.Variable(    # fully connected
            tf.truncated_normal(
                    [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * conv2_depth, fc_depth],
                    stddev=VAR_INIT_VALUE,
                    seed=SEED))
    
    fc1_biases = tf.Variable(tf.constant(VAR_INIT_VALUE, shape=[fc_depth]))
    
    fc2_weights = tf.Variable(tf.truncated_normal([fc_depth, NUM_LABELS],
                                                    stddev=VAR_INIT_VALUE,
                                                    seed=SEED))
    fc2_biases = tf.Variable(tf.constant(VAR_INIT_VALUE, shape=[NUM_LABELS]))    

    return {
        "conv1_weights" : conv1_weights,
        "conv1_biases" : conv1_biases,
        "conv2_weights" : conv2_weights,
        "conv2_biases" : conv2_biases,
        "fc1_weights" : fc1_weights,
        "fc1_biases" : fc1_biases,
        "fc2_weights" : fc2_weights,
        "fc2_biases" : fc2_biases
        }
            

# We will replicate the model structure for the training subgraph, as well
# as the evaluation subgraphs, while sharing the trainable parameters.
def model(vars, data, train=False):
    """The Model definition."""
    
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        vars["conv1_weights"],
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, vars["conv1_biases"]))
    
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    
    conv = tf.nn.conv2d(pool,
                        vars["conv2_weights"],
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    
    relu = tf.nn.relu(tf.nn.bias_add(conv, vars["conv2_biases"]))
    
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, vars["fc1_weights"]) + vars["fc1_biases"])
    
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
        hidden = tf.nn.dropout(hidden, DROPOUT_RATE, seed=SEED)
    
    return tf.matmul(hidden, vars["fc2_weights"]) + vars["fc2_biases"]


    
def train_mnist(dataset, flags):    

    train_data = dataset["train_data"]
    train_labels = dataset["train_labels"]
    validation_data = dataset["validation_data"]
    validation_labels = dataset["validation_labels"]
    test_data = dataset["test_data"]
    test_labels = dataset["test_labels"]
        
    num_epochs = dataset["num_epochs"]
    '''
    if flags.log_only:
        print("Logging test accuracy at " + LOG_PATH)
        logger = CSVLogger(LOG_PATH)
        logger.create(3, 3) # create log with 3 layers and mninbatch, validation, test accuracy        
    '''
    # Small utility function to evaluate a dataset by feeding batches of data to
    # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.
    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
        for begin in xrange(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                   eval_prediction,
                   feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        
        return predictions    
    
        
    train_size = train_labels.shape[0]
    
    with tf.device(TRAIN_DEVICE_ID):    
        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.
        train_data_node = tf.placeholder(
            tf.float32,
            shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
        eval_data = tf.placeholder(
            tf.float32,
            shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))


        # Initialize required variables
        vars = init_vars(flags.filter_size, flags.conv1_depth, flags.conv2_depth, flags.fc_depth)
    
        # Training computation: logits + cross-entropy loss.
        logits = model(vars, train_data_node, True)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, train_labels_node))

        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(vars["fc1_weights"]) + tf.nn.l2_loss(vars["fc1_biases"]) +
                                    tf.nn.l2_loss(vars["fc2_weights"]) + tf.nn.l2_loss(vars["fc2_biases"]))
        # Add the regularization term to the loss.
        loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
        
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,                                # Base learning rate.
        batch * BATCH_SIZE,    # Current index into the dataset.
        train_size,                    # Decay step.
        0.95,                                # Decay rate.
        staircase=True)
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

    with tf.device(TRAIN_DEVICE_ID): 
        # Predictions for the current training minibatch.
        train_prediction = tf.nn.softmax(logits)

    with tf.device(EVAL_DEVICE_ID): 
        # Predictions for the test and validation, which we'll compute less often.
        eval_prediction = tf.nn.softmax(model(vars, eval_data))

    # Create a local session to run the training.
    start_time = time.time()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    
    with tf.Session(config = config) as sess:
        # Run all the initializers to prepare the trainable parameters.
        init = tf.initialize_all_variables()
        sess.run(init)
        print_out('Initialized!')
        
        # Loop through training steps.
        print("epoch num: " + str(num_epochs) + ", total steps: " + str(int(num_epochs * train_size) // BATCH_SIZE))
        for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
            '''
            if flags.log_only:
                logger.setTimer()
                logger.setLayers(flags.conv1_depth, flags.conv2_depth, flags.fc_depth)
            '''
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {train_data_node: batch_data,
                                     train_labels_node: batch_labels}
            
            
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = sess.run(
                [optimizer, loss, learning_rate, train_prediction],
                feed_dict=feed_dict)
            '''
            if step % EVAL_FREQUENCY == 0:
                with tf.device(EVAL_DEVICE_ID):
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print_out('Step %d (epoch %.2f), %.1f ms' %
                        (step, float(step) * BATCH_SIZE / train_size,
                         1000 * elapsed_time / EVAL_FREQUENCY))
                    print_out('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                    minibatch_err = error_rate(predictions, batch_labels)
                    print_out('Minibatch error: %.1f%%' % minibatch_err)
                    validation_err = error_rate(eval_in_batches(validation_data, sess), validation_labels)
                    print_out('Validation error: %.1f%%' % validation_err)
                    sys.stdout.flush()
                
                    if flags.log_only:
                        # log test accuracy 
                        test_accuracy = 100.0 - error_rate(eval_in_batches(test_data, sess), test_labels)
                        tag = str(flags.filter_size) + "_" + str(flags.conv1_depth) + \
                            "_" + str(flags.conv2_depth) + "_" + str(flags.fc_depth)
                        logger.measure(tag, step, (100.0 - minibatch_err), (100.0 - validation_err), test_accuracy)
              
           '''
        with tf.device(EVAL_DEVICE_ID):
            test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
            print('Test error: %.1f%%' % test_error)

        sess.close()
        
        return test_error    

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
            100.0 *
            numpy.sum(numpy.argmax(predictions, 1) == labels) /
            predictions.shape[0])

def print_out(*args):
    if FLAGS.log_only is False:
        print(args)

def main(params, **kwargs):    # pylint: disable=unused-argument
    
    print('Params: '+ str(params))

    # Get the data.
    train_data_filename = mnist.maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = mnist.maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = mnist.maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = mnist.maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = mnist.extract_data(train_data_filename, mnist.NUM_TRAIN_DATA)
    train_labels = mnist.extract_labels(train_labels_filename, mnist.NUM_TRAIN_DATA)
    test_data = mnist.extract_data(test_data_filename, mnist.NUM_TEST_DATA)
    test_labels = mnist.extract_labels(test_labels_filename, mnist.NUM_TEST_DATA)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS
         
    dataset = {
        "train_data" : train_data,
        "train_labels" : train_labels,
        "validation_data" : validation_data,
        "validation_labels" : validation_labels,
        "test_data" : test_data,
        "test_labels" : test_labels,
        "num_epochs" : num_epochs
        }

    flags = kwargs
    y = train_mnist(dataset, flags)
    print("Result: " + str(y))
    
    return y
        
if __name__ == '__main__':
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    result = main(params, **args)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))

