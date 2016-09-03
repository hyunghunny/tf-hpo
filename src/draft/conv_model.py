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
from six.moves import xrange    # pylint: disable=redefined-builtin

import argparse

import tensorflow as tf
from util import PerformanceCSVLogger

SEED = 66478    # Set to None for random seed.    
NUM_EPOCHS = 3

BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100    # Number of steps between evaluations.

# EXECUTION FLAGS
CPU_DEVICE_ID = "/cpu:0"
TRAIN_DEVICE_ID = CPU_DEVICE_ID
EVAL_DEVICE_ID = CPU_DEVICE_ID

# LOG LEVEL
SHOW_DEBUG=True
SHOW_ERR=True
    
# initialize tensorflow variables which are required to learning
def initialize_variables(dataset, **hyperparams):
    with tf.device(TRAIN_DEVICE_ID):    
        #debug(str(hyperparams))
        # These hyperparams will be passed from kwargs
        var_init_value = hyperparams["var_init_value"]

        filter_size = hyperparams["filter_size"]
        conv1_depth = hyperparams["conv1_depth"]
        conv2_depth = hyperparams["conv2_depth"]
        fc_depth = hyperparams["fc_depth"]

        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.

        train_data_node = tf.placeholder(
            tf.float32,
            shape=(BATCH_SIZE, dataset["image_size"], dataset["image_size"], dataset["num_channels"]))

        train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))

        eval_data = tf.placeholder(
            tf.float32,
            shape=(EVAL_BATCH_SIZE, dataset["image_size"], dataset["image_size"], dataset["num_channels"]))


        # The variables below hold all the trainable weights. They are passed an
        # initial value which will be assigned when we call:
        # {tf.initialize_all_variables().run()}
        conv1_weights = tf.Variable(
                tf.truncated_normal([filter_size, filter_size, dataset["num_channels"], conv1_depth],
                                    stddev=var_init_value,
                                    seed=SEED))

        conv1_biases = tf.Variable(tf.zeros([conv1_depth]))

        conv2_weights = tf.Variable(
                tf.truncated_normal([filter_size, filter_size, conv1_depth, conv2_depth],
                                                        stddev=var_init_value,
                                                        seed=SEED))

        conv2_biases = tf.Variable(tf.constant(var_init_value, shape=[conv2_depth]))

        fc1_weights = tf.Variable(    # fully connected
                tf.truncated_normal(
                        [dataset["image_size"] // 4 * dataset["image_size"] // 4 * conv2_depth, fc_depth],
                        stddev=var_init_value,
                        seed=SEED))

        fc1_biases = tf.Variable(tf.constant(var_init_value, shape=[fc_depth]))

        fc2_weights = tf.Variable(tf.truncated_normal([fc_depth, dataset["num_labels"]],
                                                        stddev=var_init_value,
                                                        seed=SEED))
        fc2_biases = tf.Variable(tf.constant(var_init_value, shape=[dataset["num_labels"]]))    

        return {
            "train_data_node" : train_data_node,
            "train_labels_node" : train_labels_node,
            "eval_data" : eval_data,
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
    with tf.device(TRAIN_DEVICE_ID):
        """The Model definition."""

        DROPOUT_RATE = 0.5

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


    
def train_neural_net(dataset, params, logger=None, progress=False, opt='Momentum'):
    
    #debug(str(params))
    
    # Small utility function to evaluate a dataset by feeding batches of data to
    # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.
    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = numpy.ndarray(shape=(size, dataset["num_labels"]), dtype=numpy.float32)
        for begin in xrange(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict={vars['eval_data']: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                   eval_prediction,
                   feed_dict={vars['eval_data']: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        
        return predictions

    train_size = dataset["train_labels"].shape[0]
    num_epochs = dataset["num_epochs"]
    
    
    # This is where training samples and labels are fed to the graph.
    
    # parameter type casting 
    var_init_value = float(params["var_init_value"])
    filter_size = int(float(params["filter_size"]))
    conv1_depth = int(float(params["conv1_depth"])) 
    conv2_depth = int(float(params["conv2_depth"]))
    fc_depth = int(float(params["fc_depth"]))
                    
    # Initialize required variables
    vars = initialize_variables(dataset,
                    var_init_value = var_init_value,
                    filter_size = filter_size,
                    conv1_depth = conv1_depth, 
                    conv2_depth = conv2_depth,
                    fc_depth = fc_depth)
    
    # Training computation: logits + cross-entropy loss.
    logits = model(vars, vars["train_data_node"], True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, vars["train_labels_node"]))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(vars["fc1_weights"]) + tf.nn.l2_loss(vars["fc1_biases"]) +
                                 tf.nn.l2_loss(vars["fc2_weights"]) + tf.nn.l2_loss(vars["fc2_biases"]))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    with tf.device(CPU_DEVICE_ID):
        batch = tf.Variable(0)

    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
            0.01,                                # Base learning rate.
            batch * BATCH_SIZE,    # Current index into the dataset.
            train_size,                    # Decay step.
            0.95,                                # Decay rate.
            staircase=True)

    # Use simple momentum for the optimization.
    if opt is 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
    elif 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=batch)
    else:
        error('No ' + opt + ' optimizer implemented')
    
    debug("optimizer: " + opt)         

    if logger:
        setting = "filter:" + str(filter_size) + "-conv1:" + str(conv1_depth) + \
            "-conv2:" + str(conv2_depth) + "-fc:" + str(fc_depth) + "-" + opt
        
        logger.setSetting(setting)    
    
    #with tf.device(TRAIN_DEVICE_ID): 
    #Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)

    with tf.device(EVAL_DEVICE_ID): 
        # Predictions for the test and validation, which we'll compute less often.
        eval_prediction = tf.nn.softmax(model(vars, vars["eval_data"]))

    # Create a local session to run the training.
    start_time = time.time()
    config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    
    with tf.Session(config = config) as sess:
        # Run all the initializers to prepare the trainable parameters.
        init = tf.initialize_all_variables()
        sess.run(init)
        #debug('Initialized!')
        
        # Loop through training steps.
        #debug("epoch num: " + str(num_epochs) + ", total steps: " + str(int(num_epochs * train_size) // BATCH_SIZE))
        for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
            
            if logger:
                logger.setTimer()
                logger.setParamColumns(filter_size, conv1_depth, conv2_depth, fc_depth)
            
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = dataset["train_data"][offset:(offset + BATCH_SIZE), ...]
            batch_labels = dataset["train_labels"][offset:(offset + BATCH_SIZE)]
            
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {vars["train_data_node"]: batch_data,
                                     vars["train_labels_node"]: batch_labels}
            
            
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = sess.run(
                [optimizer, loss, learning_rate, train_prediction],
                feed_dict=feed_dict)
            
            if step % EVAL_FREQUENCY == 0:
                if progress: 
                    with tf.device(EVAL_DEVICE_ID):
                    
                        elapsed_time = time.time() - start_time
                        start_time = time.time()
                        
                        debug('Step %d (epoch %.2f), %.1f ms' %
                            (step, float(step) * BATCH_SIZE / train_size,
                             1000 * elapsed_time / EVAL_FREQUENCY))
                        debug('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        minibatch_err = error_rate(predictions, batch_labels)
                        debug('Minibatch error: %.1f%%' % minibatch_err)
                        validation_err = error_rate(eval_in_batches(dataset["validation_data"], sess), dataset["validation_labels"])
                        debug('Validation error: %.1f%%' % validation_err)
                        sys.stdout.flush()
                
                        if logger:
                            # log test accuracy 
                            test_err = error_rate(eval_in_batches(dataset["test_data"], sess), dataset["test_labels"])
                            debug('Test error: %.1f%%' % test_err)
                            

                            logger.measure(step, (100.0 - test_err), (100.0 - validation_err), (100.0 - minibatch_err))
              
           
        with tf.device(EVAL_DEVICE_ID):
            test_error = error_rate(eval_in_batches(dataset["test_data"], sess), dataset["test_labels"])
            test_accuracy = 100.0 - test_error
            logger.measure(str(NUM_EPOCHS) + "_epoch", test_accuracy)
            debug('Test error: %.1f%%' % test_error)

        sess.close()
        
        return test_error    

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
            100.0 *
            numpy.sum(numpy.argmax(predictions, 1) == labels) /
            predictions.shape[0])

def debug(*args):
    if SHOW_DEBUG:
        print(args)
        
def error(*args):
    if SHOW_ERR:
        print(args)
    raise Exception(args)
              
              
def download_dataset():
    
    import mnist_data as mnist
    
    VALIDATION_SIZE = 5000    # Size of the validation set.
    
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
         
    return {
        "image_size" : mnist.IMAGE_SIZE,
        "num_channels" : mnist.NUM_CHANNELS,
        "pixel_depth" : mnist.PIXEL_DEPTH,
        "num_labels" : mnist.NUM_LABELS,
        
        "train_data" : train_data,
        "train_labels" : train_labels,
        "validation_data" : validation_data,
        "validation_labels" : validation_labels,
        "test_data" : test_data,
        "test_labels" : test_labels,
        "num_epochs" : num_epochs
    }
    

def learn(dataset, params, **kwargs):    # pylint: disable=unused-argument
    starttime = time.time()
    debug('Params: '+ str(params))
    debug('KVParams: ' + str(kwargs))
    
    if 'progress' in kwargs:
        show_progress = bool(kwargs['progress'])        
    else:
        show_progress=False
        
    debug("show progress: " + str(show_progress))
    
    if 'opt' in kwargs:
        # TODO:validation required
        optimizer=kwargs['opt']
    else:
        optimizer='Momentum'

    if 'log_path' in kwargs:
        LOG_PATH = kwargs('log_path')
    else:
        LOG_PATH = "test.csv"
    debug("Logging test errors at " + LOG_PATH)
    
    logger = PerformanceCSVLogger(LOG_PATH)
    logger.create(4, 3) # create log with 4 hyperparams and 3 accuracy metrics        
    
    if 'train_dev' in kwargs:
        TRAIN_DEVICE_ID = kwargs['train_dev']
        
    if 'eval_dev' in kwargs:    
        EVAL_DEVICE_ID = kwargs['eval_dev']
        
    if 'epoch' in kwargs:
        NUM_EPOCHS = int(kwargs['epoch'])
   
    debug('Training device id: ' + TRAIN_DEVICE_ID)
    debug('Evaluation device id: '  + EVAL_DEVICE_ID)
    
    
    if not ("var_init_value" in params):
        params["var_init_value"] = 0.1 # additional hyperparameter for variable initial value
    
    with tf.device(TRAIN_DEVICE_ID):
        y = train_neural_net(dataset, params, logger, progress=show_progress, opt=optimizer)
    duration = time.time() - starttime
    debug("Result: " + str(y) + ', Duration: ' + str(abs(duration)))
    
    return y

# prevent running directly

if __name__ == '__main__':
    print("Direct learning is not supported. Run through train_main.py or hpolib_main.py with HPOLib-run.")
'''
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    debug("Starting to learn directly with: " + str(args))
    params = args
    starttime = time.time()
    #args, params = benchmark_util.parse_cli()
    result = learn(params, **args)
    duration = time.time() - starttime
    print ("Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__)))
'''
