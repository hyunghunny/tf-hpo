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

# code refactored from https://github.com/tensorflow/tensorflow/blob/r0.9/tensorflow/models/image/mnist/convolutional.py

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import traceback

import numpy
from six.moves import xrange    # pylint: disable=redefined-builtin

import argparse
import pickle
import datetime

import tensorflow as tf
from modules.logger import PerformanceCSVLogger 
from modules.predictor import PerformancePredictor

SEED = 66478    # Set to None for random seed.    
NUM_EPOCHS = 10

BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100    # Number of steps between evaluations.

LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = True #XXX: If False, TypeError raised
BASE_LEARNING_RATE = 0.01

# EXECUTION FLAGS
CPU_DEVICE_ID = "/cpu:0"
TRAIN_DEVICE_ID = CPU_DEVICE_ID
EVAL_DEVICE_ID = CPU_DEVICE_ID

# FOR EARLY TERMINATION
EARLY_STOP_CHECK = False
EARLY_STOP_CHECK_EPOCHS = 1

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
def model(tf_vars, data, train=False):
    with tf.device(TRAIN_DEVICE_ID):
        """The Model definition."""

        DROPOUT_RATE = 0.5

        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            tf_vars["conv1_weights"],
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, tf_vars["conv1_biases"]))

        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv = tf.nn.conv2d(pool,
                            tf_vars["conv2_weights"],
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        relu = tf.nn.relu(tf.nn.bias_add(conv, tf_vars["conv2_biases"]))

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

        #debug(tf_vars["fc1_weights"].get_shape())
        #debug(reshape.get_shape())
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, tf_vars["fc1_weights"]) + tf_vars["fc1_biases"])

        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, DROPOUT_RATE, seed=SEED)

        return tf.matmul(hidden, tf_vars["fc2_weights"]) + tf_vars["fc2_biases"]


    
def train_neural_net(dataset, params, logger=None, predictor=None, eval=False, opt='Momentum'):
    
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
                    feed_dict={tf_vars['eval_data']: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                   eval_prediction,
                   feed_dict={tf_vars['eval_data']: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        
        return predictions

    
    train_size = dataset["train_labels"].shape[0]
        
    
    # This is where training samples and labels are fed to the graph.
    
    # parameter type casting 
    var_init_value = float(params["var_init_value"])
    filter_size = int(float(params["filter_size"]))
    conv1_depth = int(float(params["conv1_depth"])) 
    conv2_depth = int(float(params["conv2_depth"]))
    fc_depth = int(float(params["fc_depth"]))
                    
    # Initialize required variables
    tf_vars = initialize_variables(dataset,
                    var_init_value = var_init_value,
                    filter_size = filter_size,
                    conv1_depth = conv1_depth, 
                    conv2_depth = conv2_depth,
                    fc_depth = fc_depth)
    
    # Training computation: logits + cross-entropy loss.
    logits = model(tf_vars, tf_vars["train_data_node"], True)
    debug(tf_vars["train_data_node"].get_shape())
    debug(logits.get_shape())
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, tf_vars["train_labels_node"]))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(tf_vars["fc1_weights"]) + tf.nn.l2_loss(tf_vars["fc1_biases"]) +
                                 tf.nn.l2_loss(tf_vars["fc2_weights"]) + tf.nn.l2_loss(tf_vars["fc2_biases"]))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    
    if LEARNING_RATE_DECAY:
        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        with tf.device(CPU_DEVICE_ID):
            batch = tf.Variable(0)

        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
                BASE_LEARNING_RATE,    # Base learning rate.
                batch * BATCH_SIZE,    # Current index into the dataset.
                train_size,                    # Decay step.
                0.95,                                # Decay rate.
                staircase=True)
        global_step = batch
    else:
        learning_rate = LEARNING_RATE
        global_step = None
        
    # Use simple momentum for the optimization.
    if opt is 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)
    elif 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    else:
        error('No ' + opt + ' optimizer implemented')
    
    debug("optimizer: " + opt)         

    if logger:
        setting = "filter:" + str(filter_size) + "-conv1:" + str(conv1_depth) + \
            "-conv2:" + str(conv2_depth) + "-fc:" + str(fc_depth) + "-" + opt
        
        logger.setSetting(setting)
        logger.setTimer("total")
    
    #Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)

    with tf.device(EVAL_DEVICE_ID): 
        # Predictions for the test and validation, which we'll compute less often.
        eval_prediction = tf.nn.softmax(model(tf_vars, tf_vars["eval_data"]))

    # Create a local session to run the training.
    start_time = time.time()
    
    if logger:
        logger.setParamColumns(filter_size, conv1_depth, conv2_depth, fc_depth)
        logger.setTimer("total")
        logger.setTimer("epoch")
        if eval:
            logger.setTimer("eval")    
    
    config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    
    with tf.Session(config = config) as sess:
        # Run all the initializers to prepare the trainable parameters.
        init = tf.initialize_all_variables()
        sess.run(init)
        #debug('Initialized!')
        
        # Loop through training steps.
        debug("epoch num: " + str(NUM_EPOCHS) + ", total steps: " + str(int(NUM_EPOCHS * train_size) // BATCH_SIZE))
        if logger:
            steps_per_epoch = int(train_size // self.hp["BATCH_SIZE"])
            logger.setStepsPerEpoch(steps_per_epoch)
        
        for i in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
            step = i + 1 # XXX: step MUST start with 1 not 0
            
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = dataset["train_data"][offset:(offset + BATCH_SIZE), ...]
            batch_labels = dataset["train_labels"][offset:(offset + BATCH_SIZE)]
            
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {tf_vars["train_data_node"]: batch_data,
                                     tf_vars["train_labels_node"]: batch_labels}
            
            
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = sess.run(
                [optimizer, loss, learning_rate, train_prediction],
                feed_dict=feed_dict)
            
            if step % EVAL_FREQUENCY == 0:
                
                if eval: 
                    with tf.device(EVAL_DEVICE_ID):
                    
                        elapsed_time = time.time() - start_time
                        start_time = time.time()
                        
                        debug('Step %d (epoch %.2f), %.1f ms' %
                            (step, float(step) * BATCH_SIZE / train_size,
                             1000 * elapsed_time / EVAL_FREQUENCY))
                        debug('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        minibatch_err = error_rate(predictions, batch_labels)
                        debug('Minibatch error: %.1f%%' % minibatch_err)
                        validation_err = error_rate(eval_in_batches(dataset["validation_data"], sess),\
                                                    dataset["validation_labels"])
                        debug('Validation error: %.1f%%' % validation_err)
                                        
                        if logger:
                            # log test accuracy 
                            test_err = error_rate(eval_in_batches(dataset["test_data"], sess), dataset["test_labels"])
                            debug('Test error: %.1f%%' % test_err)                            

                            logger.measure("eval", step, 
                                           {"Test Accuracy" : (100.0 - test_err), 
                                            "Validation Accuarcy" : (100.0 - validation_err), 
                                            "Training Accuracy" : (100.0 - minibatch_err)})
                        
                else:
                    debug(str(step))
                    
                sys.stdout.flush()
            
            # when each epochs calculate test accuracy 
            if step % (int(train_size) // BATCH_SIZE) == 0:
                num_epoch = step // (int(train_size) // BATCH_SIZE)
                with tf.device(EVAL_DEVICE_ID):
                    debug("step number when " +  str(num_epoch) + " epochs ended : " + str(step))
                    test_error = error_rate(eval_in_batches(dataset["test_data"], sess), dataset["test_labels"])
                    test_accuracy = 100.0 - test_error
                    validation_err = error_rate(eval_in_batches(dataset["validation_data"], sess),\
                                                    dataset["validation_labels"])                    
                    valid_accuracy = 100.0 - validation_err
                    if logger:
                        logger.measure("epoch", step, 
                                       {"Test Accuracy" : test_accuracy, 
                                        "Validation Accuarcy" : valid_accuracy})
                    debug('Test error: %.1f%%' % test_error)
                    sys.stdout.flush()
                # Check early termination
                if predictor:                    
                    if num_epoch >= EARLY_STOP_CHECK_EPOCHS:
                        if predictor.load() is False:
                            debug("Unable to load training log")
                        else:
                            debug("Predicting whether keep learning or not " )
                            result = predictor.predict(Param1=filter_size, Param2=conv1_depth, Param3=conv2_depth, Param4=fc_depth)
                            debug("Prediction result: " + str(result))
                            if result is False:
                                debug("Early termination")
                                break
                    
        
        debug("step number when training ended : " + str(step))
        with tf.device(EVAL_DEVICE_ID):
            test_error = error_rate(eval_in_batches(dataset["test_data"], sess), dataset["test_labels"])
            test_accuracy = 100.0 - test_error
            validation_err = error_rate(eval_in_batches(dataset["validation_data"], sess),\
                                        dataset["validation_labels"])                    
            valid_accuracy = 100.0 - validation_err
            if logger:
                logger.measure("total", step, 
                               {"Test Accuracy" : test_accuracy, 
                                "Validation Accuarcy" : valid_accuracy})
            
            debug('Test error: %.1f%%' % test_error)
            sys.stdout.flush()

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
    
    import models.mnist_data as mnist
    
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
        "test_labels" : test_labels
    }
    

def learn(dataset, params, **kwargs):    # pylint: disable=unused-argument
    global TRAIN_DEVICE_ID
    global EVAL_DEVICE_ID
    global NUM_EPOCHS
    global SEED
    
    starttime = time.time()
    debug('Params: '+ str(params))
    debug('KVParams: ' + str(kwargs))
    
    if 'do_eval' in kwargs:
        do_eval = bool(kwargs['do_eval'])        
    else:
        do_eval=False
        
    debug("do evaluation: " + str(do_eval))
    
    if 'opt' in kwargs:
        # TODO:validation required
        optimizer=kwargs['opt']
    else:
        optimizer='Momentum'
    
    if 'logger' in kwargs:
        logger = kwargs['logger']        
    else:
        logger = None

    if 'predictor' in kwargs:
        predictor = kwargs['predictor']
    else:
        predictor = None        
    
    if 'train_dev' in kwargs:
        TRAIN_DEVICE_ID = kwargs['train_dev']
        
    if 'eval_dev' in kwargs:    
        EVAL_DEVICE_ID = kwargs['eval_dev']
        
    if 'epochs' in kwargs:
        NUM_EPOCHS = int(kwargs['epochs'])
        
    if 'seed' in kwargs:
        SEED = int(kwargs['seed'])
   
    debug('Training device id: ' + TRAIN_DEVICE_ID)
    debug('Evaluation device id: ' + EVAL_DEVICE_ID)
    debug('Epochs: ' + str(NUM_EPOCHS))
    
    if not ("var_init_value" in params):
        params["var_init_value"] = 0.1 # additional hyperparameter for variable initial value
    
    with tf.device(TRAIN_DEVICE_ID):
        try:
            y = train_neural_net(dataset, params, logger, predictor, eval=do_eval, opt=optimizer)
            duration = time.time() - starttime
            debug("Result: " + str(y) + ', Duration: ' + str(abs(duration)))
            return y
        
        except:
            e = sys.exc_info()
            traceback.print_exc()
            print(e)
            error('Training failed: ' + str(params))
            date_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
            file_name = './temp/failed_params_' + date_str + '.pickle'
            with open(file_name, 'wb') as f:
                pickle.dump(params, f)


# prevent running directly

if __name__ == '__main__':
    print("Direct learning is not supported. Run with *_main.py.")
