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


# global dictionaries which contain many parameters
hyperparams = None
execution = None
data_info = None

# For learning rate decay
CPU_DEVICE_ID = '/cpu:0'


# LOG LEVEL
SHOW_DEBUG = True
SHOW_ERR = True
    
# initialize tensorflow variables which are required to learning
def initialize_tf_variables():
    with tf.device(execution["TRAIN_DEVICE_ID"]):    

        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.

        train_data_node = tf.placeholder(
            tf.float32,
            shape=(hyperparams["BATCH_SIZE"], dataset["image_size"], data_info["image_size"], data_info["num_channels"]))

        train_labels_node = tf.placeholder(tf.int64, shape=(hyperparams["BATCH_SIZE"],))

        eval_data = tf.placeholder(
            tf.float32,
            shape=(hyperparams["EVAL_BATCH_SIZE"], data_info["image_size"], data_info["image_size"], data_info["num_channels"]))

        # The variables below hold all the trainable weights. They are passed an
        # initial value which will be assigned when we call:
        # {tf.initialize_all_variables().run()}
        conv1_weights = tf.Variable(
                tf.truncated_normal([hyperparams["FILTER_SIZE"], hyperparams["FILTER_SIZE"], data_info["num_channels"], hyperparams["CONV1_DEPTH"]],
                                    stddev=hyperparams["INIT_STDDEV"],
                                    seed=hyperparams["SEED"]))

        conv1_biases = tf.Variable(tf.zeros([hyperparams["CONV1_DEPTH"]]))
        
        conv2_weights = tf.Variable(
                tf.truncated_normal([hyperparams["FILTER_SIZE"], hyperparams["FILTER_SIZE"], hyperparams["CONV1_DEPTH"], hyperparams["CONV2_DEPTH"]],
                                                        stddev=hyperparams["INIT_STDDEV"],
                                                        seed=hyperparams["SEED"]))
        conv2_biases = tf.Variable(tf.constant(hyperparams["INIT_WEIGHT_VALUE"], shape=[hyperparams["CONV2_DEPTH"]]))
        
        fc1_weights = tf.Variable(    # fully connected
                tf.truncated_normal(                          
                        [data_info["image_size"] // (2 * hyperparams["NUM_POOLING"]) * \
                         data_info["image_size"] // (2 * hyperparams["NUM_POOLING"]) * \
                         hyperparams["CONV2_DEPTH"], 
                         hyperparams["FC1_WIDTH"]],
                        stddev=hyperparams["INIT_STDDEV"],
                        seed=hyperparams["SEED"]))

        fc1_biases = tf.Variable(tf.constant(hyperparams["INIT_WEIGHT_VALUE"], shape=[hyperparams["FC1_WIDTH"]]))

        # output layer
        output_weights = tf.Variable(tf.truncated_normal([hyperparams["FC1_WIDTH"], 
                                                          data_info["num_labels"]],
                                                        stddev=hyperparams["INIT_STDDEV"],
                                                        seed=hyperparams["SEED"]))
        
        output_biases = tf.Variable(tf.constant(hyperparams["INIT_WEIGHT_VALUE"], shape=[data_info["num_labels"]]))    

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
            "output_weights" : output_weights,
            "output_biases" : output_biases
            }
            

# We will replicate the model structure for the training subgraph, as well
# as the evaluation subgraphs, while sharing the trainable parameters.
def leNet5(tf_vars, data, train=False):
    with tf.device(execution["TRAIN_DEVICE_ID"]):
        """The Model definition."""


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
                              ksize=[1, hyperparams["POOLING_SIZE"], hyperparams["POOLING_SIZE"], 1],
                              strides=[1, hyperparams["STRIDE_SIZE"], hyperparams["STRIDE_SIZE"], 1],
                              padding='SAME')

        conv = tf.nn.conv2d(pool,
                            tf_vars["conv2_weights"],
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        relu = tf.nn.relu(tf.nn.bias_add(conv, tf_vars["conv2_biases"]))

        pool = tf.nn.max_pool(relu,
                              ksize=[1, hyperparams["POOLING_SIZE"], hyperparams["POOLING_SIZE"], 1],
                              strides=[1, hyperparams["STRIDE_SIZE"], hyperparams["STRIDE_SIZE"], 1],
                              padding='SAME')
        
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]]) # XXX:I couldn't understand what this operation did

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        #debug(tf_vars["fc1_weights"].get_shape())
        #debug(reshape.get_shape())
        
        hidden = tf.nn.relu(tf.matmul(reshape, tf_vars["fc1_weights"]) + tf_vars["fc1_biases"])

        # Add a dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, hyperparams["DROPOUT_RATE"], seed=hyperparams["SEED"])

        return tf.matmul(hidden, tf_vars["output_weights"]) + tf_vars["output_biases"]

    
def train_model(logger=None, predictor=None):
    
    # Small utility function to evaluate a dataset by feeding batches of data to
    # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.
    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < hyperparams["EVAL_BATCH_SIZE"]:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = numpy.ndarray(shape=(size, data_info["num_labels"]), dtype=numpy.float32)
        for begin in xrange(0, size, hyperparams["EVAL_BATCH_SIZE"]):
            end = begin + hyperparams["EVAL_BATCH_SIZE"]
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict={tf_vars['eval_data']: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                   eval_prediction,
                   feed_dict={tf_vars['eval_data']: data[-hyperparams["EVAL_BATCH_SIZE"]:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        
        return predictions

    
    train_size = data_info["train_labels"].shape[0]
    
    # This is where training samples and labels are fed to the graph.
    # parameter type casting 
    
    # Initialize required variables
    tf_vars = initialize_tf_variables()
    
    # Training computation: logits + cross-entropy loss.
    logits = leNet5(tf_vars, tf_vars["train_data_node"], True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf_vars["train_labels_node"]))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(tf_vars["fc1_weights"]) + tf.nn.l2_loss(tf_vars["fc1_biases"]) + \
                                 tf.nn.l2_loss(tf_vars["output_weights"]) + tf.nn.l2_loss(tf_vars["output_biases"]))
    # Add the regularization term to the loss.
    loss += hyperparams["REGULARIZER_FACTOR"] * regularizers

    if hyperparams["USE_LEARNING_RATE_DECAY"]:
        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        with tf.device(CPU_DEVICE_ID):
            batch = tf.Variable(0)

        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
                hyperparams["BASE_LEARNING_RATE"],    # Base learning rate.
                batch * hyperparams["BATCH_SIZE"],    # Current index into the dataset.
                train_size,            # Decay step.
                hyperparams["DECAY_RATE"],                  # Decay rate.
                staircase=True)
        global_step = batch
    else:
        learning_rate = hyperparams["STATIC_LEARNING_RATE"]
        global_step = None
        
    
    if hyperparams["OPTIMIZATION"] == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, hyperparams["MOMENTUM_VALUE"]).minimize(loss, global_step=global_step)
    elif hyperparams["OPTIMIZATION"] == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    else:
        error('No ' + optimizer_name + ' optimizer implemented')
    
    debug("optimizer: " + hyperparams["OPTIMIZATION"])         

    if logger:
       
        logger.setTimer("total")
    
    #Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)

    with tf.device(execution["TEST_DEVICE_ID"]): 
        # Predictions for the test and validation, which we'll compute less often.
        eval_prediction = tf.nn.softmax(leNet5(tf_vars, tf_vars["eval_data"]))

    # Create a local session to run the training.
    start_time = time.time()
    if logger:                
        logger.setTimer("epoch")
        #logger.setParamColumns(hyperparams["FILTER_SIZE"], hyperparams["CONV1_DEPTH"], hyperparams["CONV2_DEPTH"], hyperparams["FC1_WIDTH"])
        logger.setHyperparamSetting({
            "Filter size": hyperparams["FILTER_SIZE"],
            "Conv1 depth" : hyperparams["CONV1_DEPTH"],
            "Conv2 depth" : hyperparams["CONV2_DEPTH"],
            "FC neurons" : hyperparams["FC1_WIDTH"]})
        
        if execution["VALIDATION"]:
            logger.setTimer("eval")
            
    config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    
    with tf.Session(config = config) as sess:
        # Run all the initializers to prepare the trainable parameters.
        init = tf.initialize_all_variables()
        sess.run(init)
        #debug('Initialized!')
        
        # Loop through training steps.
        debug("epoch num: " + str(hyperparams["NUM_EPOCHS"]) + ", total steps: " + str(int(hyperparams["NUM_EPOCHS"] * train_size) // hyperparams["BATCH_SIZE"]))
        for step in xrange(int(hyperparams["NUM_EPOCHS"] * train_size) // hyperparams["BATCH_SIZE"]):
          
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * hyperparams["BATCH_SIZE"]) % (train_size - hyperparams["BATCH_SIZE"])
            batch_data = data_info["train_data"][offset:(offset + hyperparams["BATCH_SIZE"]), ...]
            batch_labels = data_info["train_labels"][offset:(offset + hyperparams["BATCH_SIZE"])]
            
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {tf_vars["train_data_node"]: batch_data,
                tf_vars["train_labels_node"]: batch_labels}
            
            
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = sess.run(
                [optimizer, loss, learning_rate, train_prediction],
                feed_dict=feed_dict)
            
            if step % hyperparams["EVAL_FREQUENCY"] == 0:
                
                if execution["VALIDATION"]: 
                    with tf.device(execution["TEST_DEVICE_ID"]):
                    
                        elapsed_time = time.time() - start_time
                        start_time = time.time()
                        
                        debug('Step %d (epoch %.2f), %.1f ms' %
                            (step, float(step) * hyperparams["BATCH_SIZE"] / train_size,
                             1000 * elapsed_time / hyperparams["EVAL_FREQUENCY"]))
                        debug('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                        minibatch_err_rate = error_rate(predictions, batch_labels)
                        debug('Minibatch error rate: %.1f%%' % minibatch_err_rate)
                        validation_err_rate = error_rate(eval_in_batches(data_info["validation_data"], sess),\
                                                    data_info["validation_labels"])
                        debug('Validation error rate: %.1f%%' % validation_err_rate)
                                        
                        if logger:
                            # log test accuracy 
                            test_err_rate = error_rate(eval_in_batches(data_info["test_data"], sess), data_info["test_labels"])
                            debug('Test error rate per epoch: %.1f%%' % test_err_rate)                            

                            logger.measure("eval", step, 
                                           {"Test Error" : test_err_rate / 100.0, 
                                            "Validation Error" : validation_err_rate / 100.0, 
                                            "Training Error" : minibatch_err_rate / 100.0})
                else:
                    debug(str(step))
                    
                sys.stdout.flush()
            
            # when each epochs calculate test accuracy 
            if step % (int(train_size) // hyperparams["BATCH_SIZE"]) == 0:
                num_epoch = step // (int(train_size) // hyperparams["BATCH_SIZE"])
                with tf.device(execution["TEST_DEVICE_ID"]):
                    test_error_rate = error_rate(eval_in_batches(data_info["test_data"], sess), data_info["test_labels"])
                    
                    validation_err_rate = error_rate(eval_in_batches(dataset["validation_data"], sess),\
                                                    dataset["validation_labels"])                    

                    logger.measure("epoch", num_epoch, 
                                   {"Test Error" : test_error_rate / 100.0, 
                                    "Validation Error" : validation_err_rate / 100.0})
                    debug('Test error rate of total: %.1f%%' % test_error_rate)
                    sys.stdout.flush()
                # Check early termination
                if predictor:                    
                    if num_epoch >= execution["EARLY_STOP_CHECK_EPOCHS"]:
                        if predictor.load() is False:
                            debug("Unable to load training log")
                        else:
                            debug("Predicting whether keep learning or not " )
                            result = predictor.predict(Param1=hyperparams["FILTER_SIZE"], Param2=hyperparams["CONV1_DEPTH"], Param3=hyperparams["CONV2_DEPTH"], Param4=hyperparams["FC1_WIDTH"])
                            debug("Prediction result: " + str(result))
                            if result is False:
                                debug("Early termination")
                                break
                    
           
        with tf.device(execution["TEST_DEVICE_ID"]):
            test_error_rate = error_rate(eval_in_batches(data_info["test_data"], sess), data_info["test_labels"])             
            validation_err_rate = error_rate(eval_in_batches(dataset["validation_data"], sess),\
                                        dataset["validation_labels"])                    
                      
            logger.measure("total", hyperparams["NUM_EPOCHS"], 
                           {"Test Error" : test_error_rate / 100.0, 
                            "Validation Error" : validation_err_rate / 100.0 })
            
            debug('Test error rate of total: %.1f%%' % test_error_rate)
            sys.stdout.flush()

        sess.close()
        
        return test_error_rate    

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

def learn(dataset, configuration, **kwargs):    # pylint: disable=unused-argument
    global hyperparams
    global execution
    global data_info
    
    hyperparams = configuration.getSectionMap('Hyperparameters')
    execution = configuration.getSectionMap('Execution')
    data_info = dataset
    
    debug(hyperparams)
    debug(execution)
    
    if 'logger' in kwargs:
        logger = kwargs['logger']        
    else:
        logger = None

    if 'predictor' in kwargs:
        predictor = kwargs['predictor']
    else:
        predictor = None        
    
    
    starttime = time.time()
    with tf.device(execution["TRAIN_DEVICE_ID"]):
        try:
            y = train_model(logger, predictor)
            duration = time.time() - starttime
            debug("Result: " + str(y) + ', Duration: ' + str(abs(duration)))
            return y
        
        except:
            e = sys.exc_info()
            traceback.print_exc()
            print(e)
            error('Training failed: ' + str(hyperparams))
            date_str = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
            file_name = 'temp/failed_params_' + date_str + '.pickle'
            with open(file_name, 'wb') as f:
                pickle.dump(hyperparams, f)


if __name__ == '__main__':
    
    import mnist_data as mnist
    dataset = mnist.import_dataset()                                    
    
    from modules.hpmgr import HPVManager 
    
    cfg = HPVManager('HPV_002.ini')
    log_file = cfg.getOption('Execution', 'output_log_file')
    
    hyperparams = ['Filter size', 'Conv1 depth', 'Conv2 depth', 'FC neurons']
    metrics = ['Test Error', 'Validation Error', 'Training Error']                                 
    
    # create logger
    logger = PerformanceCSVLogger(log_file)
    logger.create(hyperparams, metrics) 
    logger.setSetting(cfg.getConfigPath())
    
    learn(dataset, cfg, logger=logger)
    