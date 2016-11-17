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
from modules.interface import ModelInterface

# For learning rate decay
CPU_DEVICE_ID = '/cpu:0'

# LOG LEVEL
SHOW_DEBUG = False
SHOW_ERR = True

class CNN3C1F(ModelInterface):
    def __init__(self, dataset):
        self.data = dataset
        self.logger = None
        self.predictor = None
    
    def setLogger(self, logger):
        self.logger = logger
        
    def setPredictor(self, predictor):
        self.predictor = predictor
        
    def learn(self, hpv):

        self.hp = hpv.getSectionMap('Hyperparameters')
        self.flag = hpv.getSectionMap('Execution')

        debug(self.hp)
        debug(self.flag)

        starttime = time.time()
        with tf.device(self.flag["TRAIN_DEVICE_ID"]):
            try:
                y = self.train()
                duration = time.time() - starttime
                debug("Result: " + str(y) + ', Duration: ' + str(abs(duration)))
                return y

            except:
                e = sys.exc_info()
                traceback.print_exc()
                print(e)
                    
    def train(self):

        train_size = self.data["TRAIN_LABELS"].shape[0]

        # This is where training samples and labels are fed to the graph.
        # parameter type casting 

        # Initialize required variables
        self.initialize()

        # Training computation: logits + cross-entropy loss.
        logits = self.createModel(self.train_data_node, True)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.train_labels_node))

        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(self.fc1_weights) + tf.nn.l2_loss(self.fc1_biases) + \
                                     tf.nn.l2_loss(self.output_weights) + tf.nn.l2_loss(self.output_biases))
        # Add the regularization term to the loss.
        loss += self.hp["REGULARIZER_FACTOR"] * regularizers

        if self.hp["USE_LEARNING_RATE_DECAY"]:
            # Optimizer: set up a variable that's incremented once per batch and
            # controls the learning rate decay.
            with tf.device(CPU_DEVICE_ID):
                batch = tf.Variable(0)

            # Decay once per epoch, using an exponential schedule starting at 0.01.
            learning_rate = tf.train.exponential_decay(
                    self.hp["BASE_LEARNING_RATE"],    # Base learning rate.
                    batch * self.hp["BATCH_SIZE"],    # Current index into the dataset.
                    train_size,            # Decay step.
                    self.hp["DECAY_RATE"],                  # Decay rate.
                    staircase=True)
            global_step = batch
        else:
            learning_rate = tf.Variable(self.hp["BASE_LEARNING_RATE"]) # self.hp["STATIC_LEARNING_RATE"]
            global_step = None

        
        if self.hp["OPTIMIZATION"] == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, self.hp["MOMENTUM_VALUE"]).minimize(loss, global_step=global_step)
        elif self.hp["OPTIMIZATION"] == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        else:
            error('No ' + optimizer_name + ' optimizer implemented')
            raise NotImplementedError()

        debug("Adaptive Learning Rate Algorithm: " + self.hp["OPTIMIZATION"])       

        #Predictions for the current training minibatch.
        train_prediction = tf.nn.softmax(logits)
        
        with tf.device(self.flag["TEST_DEVICE_ID"]): 
            # Predictions for the test and validation, which we'll compute less often.
            self.eval_prediction = tf.nn.softmax(self.createModel(self.eval_data))
        
        start_time = time.time()
        if self.logger:                
            self.logger.setTimer("total")
            self.logger.setTimer("epoch")
            
            setting = {}
            for param in self.logger.getParamList():
                setting[param] = self.hp[param]
                
            self.logger.setHyperparamSetting(setting)

            if self.flag["VALIDATION"]:
                self.logger.setTimer("eval")

        config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)

        # Create a local session to run the training.
        with tf.Session(config = config) as sess:
            # Run all the initializers to prepare the trainable parameters.
            init = tf.initialize_all_variables()
            sess.run(init)
            
            # Loop through training steps.
            debug("total epoch num: " + str(self.hp["NUM_EPOCHS"]) + 
                  ", total steps: " + str(int(self.hp["NUM_EPOCHS"] * train_size) // self.hp["BATCH_SIZE"]))            
            
            steps_per_epoch = int(train_size // self.hp["BATCH_SIZE"])
            
            if self.logger:                
                self.logger.setStepsPerEpoch(steps_per_epoch)
            
            for i in xrange(self.hp["NUM_EPOCHS"] * steps_per_epoch):
                step = i + 1 # XXX: step MUST start with 1 not 0
                
                # Compute the offset of the current minibatch in the data.
                # Note that we could use better randomization across epochs.
                offset = (step * self.hp["BATCH_SIZE"]) % (train_size - self.hp["BATCH_SIZE"])
                batch_data = self.data["TRAIN_DATA"][offset:(offset + self.hp["BATCH_SIZE"]), ...]
                batch_labels = self.data["TRAIN_LABELS"][offset:(offset + self.hp["BATCH_SIZE"])]

                # This dictionary maps the batch data (as a numpy array) to the node in the graph it should be fed to.
                feed_dict = {self.train_data_node: batch_data,
                    self.train_labels_node: batch_labels}

                # Run the graph and fetch some of the nodes.
                _, l, lr, predictions = sess.run(
                    [optimizer, loss, learning_rate, train_prediction],
                    feed_dict=feed_dict)
                
                num_epoch = float(step) * self.hp["BATCH_SIZE"] / train_size 
                # training, validation and test error calculation at each evaluation frequency
                if self.flag["VALIDATION"]:
                    if step % self.hp["EVAL_FREQUENCY"] == 0: 
                        with tf.device(self.flag["TEST_DEVICE_ID"]):

                            elapsed_time = time.time() - start_time
                            start_time = time.time()

                            debug('Step %d (epoch %.2f), %.1f ms' %
                                (step, num_epoch, 1000 * elapsed_time / self.hp["EVAL_FREQUENCY"]))
                            debug('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                            minibatch_err_rate = error_rate(predictions, batch_labels)
                            debug('Minibatch error rate: %.1f%%' % minibatch_err_rate)
                            validation_err_rate = error_rate(self.evaluateInBatches(self.data["VALIDATION_DATA"], sess),\
                                                        self.data["VALIDATION_LABELS"])
                            debug('Validation error rate: %.1f%%' % validation_err_rate)

                            if self.logger:                                
                                test_err_rate = error_rate(self.evaluateInBatches(self.data["TEST_DATA"], sess), self.data["TEST_LABELS"])
                                debug('Test error rate per evaluation: %.1f%%' % test_err_rate)                            

                                self.logger.measure("eval", step, 
                                               {"Test Error" : test_err_rate / 100.0, 
                                                "Validation Error" : validation_err_rate / 100.0, 
                                                "Training Error" : minibatch_err_rate / 100.0})
                    sys.stdout.flush()

                
                # test and validation error calcuation at each epochs 
                if step % steps_per_epoch == 0:
                    
                    with tf.device(self.flag["TEST_DEVICE_ID"]):                        
                        validation_err_rate = error_rate(self.evaluateInBatches(self.data["VALIDATION_DATA"], sess),\
                                                        self.data["VALIDATION_LABELS"])
                        test_error_rate = error_rate(self.evaluateInBatches(self.data["TEST_DATA"], sess), self.data["TEST_LABELS"])

                        if self.logger:
                            self.logger.measure("epoch", step, 
                                       {"Test Error" : test_error_rate / 100.0, 
                                        "Validation Error" : validation_err_rate / 100.0})
                        debug('Test error rate per epoch: %.1f%%' % test_error_rate)
                        sys.stdout.flush()
                    
                    # Check early termination
                    if self.checkEarlyStop():
                        debug("Early stopping happened")
                        break
                   

            # test error calculation after being trained
            with tf.device(self.flag["TEST_DEVICE_ID"]):
                          
                validation_err_rate = error_rate(self.evaluateInBatches(self.data["VALIDATION_DATA"], sess),\
                                            self.data["VALIDATION_LABELS"])                    
                test_error_rate = error_rate(self.evaluateInBatches(self.data["TEST_DATA"], sess), self.data["TEST_LABELS"])   
                if self.logger:
                    self.logger.measure("total", step, 
                               {"Test Error" : test_error_rate / 100.0, 
                                "Validation Error" : validation_err_rate / 100.0 })

                debug('Test error rate of total: %.1f%%' % test_error_rate)
            
            sys.stdout.flush()
            sess.close()

        return test_error_rate   
        
    def checkEarlyStop(self):
        result = False
        # Check early termination
        
        if self.predictor:                    
            if num_epoch >= self.flag["EARLY_STOP_CHECK_EPOCHS"]:
                if self.predictor.load() is False:
                    debug("Unable to load training log")
                else:
                    debug("Predicting whether keep learning or not " )
                    togo = self.predictor.check(self.hp)                    
                    result = not togo 
                    debug("Prediction result: " + str(result))
        return result                       
    
    # initialize tensorflow variables which are required to learning
    def initialize(self):
        with tf.device(self.flag["TRAIN_DEVICE_ID"]):    

            # These placeholder nodes will be fed a batch of training data at each
            # training step using the {feed_dict} argument to the Run() call below.

            self.train_data_node = tf.placeholder(
                tf.float32,
                shape=(self.hp["BATCH_SIZE"], self.data["IMAGE_SIZE"], self.data["IMAGE_SIZE"], self.data["NUM_CHANNELS"]))

            self.train_labels_node = tf.placeholder(tf.int64, shape=(self.hp["BATCH_SIZE"],))

            self.eval_data = tf.placeholder(
                tf.float32,
                shape=(self.hp["EVAL_BATCH_SIZE"], self.data["IMAGE_SIZE"], self.data["IMAGE_SIZE"], self.data["NUM_CHANNELS"]))

            # The variables below hold all the trainable weights. They are passed an initial value which will be assigned when we call:
            # {tf.initialize_all_variables().run()}
            self.conv1_weights = tf.Variable(
                    tf.truncated_normal([self.hp["FILTER_SIZE"], self.hp["FILTER_SIZE"], self.data["NUM_CHANNELS"], self.hp["CONV1_DEPTH"]],
                                        stddev=self.hp["INIT_STDDEV"],
                                        seed=self.hp["SEED"]))

            self.conv1_biases = tf.Variable(tf.zeros([self.hp["CONV1_DEPTH"]]))

            self.conv2_weights = tf.Variable(
                    tf.truncated_normal([self.hp["FILTER_SIZE"], self.hp["FILTER_SIZE"], self.hp["CONV1_DEPTH"], self.hp["CONV2_DEPTH"]],
                                        stddev=self.hp["INIT_STDDEV"],
                                        seed=self.hp["SEED"]))
            self.conv2_biases = tf.Variable(tf.constant(self.hp["INIT_WEIGHT_VALUE"], shape=[self.hp["CONV2_DEPTH"]]))

            self.conv3_weights = tf.Variable(
                    tf.truncated_normal([self.hp["FILTER_SIZE"], self.hp["FILTER_SIZE"], self.hp["CONV2_DEPTH"], self.hp["CONV3_DEPTH"]],
                                        stddev=self.hp["INIT_STDDEV"],
                                        seed=self.hp["SEED"]))
            self.conv3_biases = tf.Variable(tf.constant(self.hp["INIT_WEIGHT_VALUE"], shape=[self.hp["CONV3_DEPTH"]]))            
            
            resized = int(round(self.data["IMAGE_SIZE"] / self.hp["STRIDE_SIZE"] / self.hp["STRIDE_SIZE"] / self.hp["STRIDE_SIZE"]))
            self.fc1_weights = tf.Variable(    # fully connected
                    tf.truncated_normal(                          
                            [resized * resized * self.hp["CONV3_DEPTH"], 
                             self.hp["FC1_WIDTH"]],
                            stddev=self.hp["INIT_STDDEV"],
                            seed=self.hp["SEED"]))

            self.fc1_biases = tf.Variable(tf.constant(self.hp["INIT_WEIGHT_VALUE"], shape=[self.hp["FC1_WIDTH"]]))

            # output layer
            self.output_weights = tf.Variable(tf.truncated_normal([self.hp["FC1_WIDTH"], 
                                                              self.data["NUM_LABELS"]],
                                                            stddev=self.hp["INIT_STDDEV"],
                                                            seed=self.hp["SEED"]))

            self.output_biases = tf.Variable(tf.constant(self.hp["INIT_WEIGHT_VALUE"], shape=[self.data["NUM_LABELS"]]))    
            

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def createModel(self, data, train=False):
        with tf.device(self.flag["TRAIN_DEVICE_ID"]):
            """The Model definition."""

            # 2D convolution, with 'SAME' padding (i.e. the output feature map has
            # the same size as the input). Note that {strides} is a 4D array whose
            # shape matches the data layout: [image index, y, x, depth].
            conv = tf.nn.conv2d(data,
                                self.conv1_weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME')

            # Bias and rectified linear non-linearity.
            relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))

            # Max pooling. The kernel size spec {ksize} also follows the layout of
            # the data. Here we have a pooling window of 2, and a stride of 2.
            pool = tf.nn.max_pool(relu,
                                  ksize=[1, self.hp["POOLING_SIZE"], self.hp["POOLING_SIZE"], 1],
                                  strides=[1, self.hp["STRIDE_SIZE"], self.hp["STRIDE_SIZE"], 1],
                                  padding=self.hp["PADDING"])

            conv = tf.nn.conv2d(pool,
                                self.conv2_weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME')

            relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv2_biases))

            pool = tf.nn.max_pool(relu,
                                  ksize=[1, self.hp["POOLING_SIZE"], self.hp["POOLING_SIZE"], 1],
                                  strides=[1, self.hp["STRIDE_SIZE"], self.hp["STRIDE_SIZE"], 1],
                                  padding=self.hp["PADDING"])
            
            
            conv = tf.nn.conv2d(pool,
                                self.conv3_weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME')

            relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv3_biases))

            pool = tf.nn.max_pool(relu,
                                  ksize=[1, self.hp["POOLING_SIZE"], self.hp["POOLING_SIZE"], 1],
                                  strides=[1, self.hp["STRIDE_SIZE"], self.hp["STRIDE_SIZE"], 1],
                                  padding=self.hp["PADDING"])
            
            # Reshape the feature map cuboid into a 2D matrix to feed it to the
            # fully connected layers.
            pool_shape = pool.get_shape().as_list()
            debug("pool shape: "+ str(pool_shape))
            reshape = tf.reshape(pool,
                [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]]) # XXX:I couldn't understand what this operation did

            # Fully connected layer. Note that the '+' operation automatically broadcasts the biases.
            debug("reshaped: " + str(reshape.get_shape()))
            debug("fc1 shape: " + str(self.fc1_weights.get_shape()))
            

            hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)

            # Add a dropout during training only. Dropout also scales
            # activations such that no rescaling is needed at evaluation time.
            if train:
                hidden = tf.nn.dropout(hidden, 1.0 - self.hp["DROPOUT_RATE"], seed=self.hp["SEED"])

        return tf.matmul(hidden, self.output_weights) + self.output_biases

    
    # Small utility function to evaluate a dataset by feeding batches of data to
    # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.
    def evaluateInBatches(self, data, sess):
        """Get all predictions for a dataset by running it in small batches."""
              
        size = data.shape[0]
        if size < self.hp["EVAL_BATCH_SIZE"]:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = numpy.ndarray(shape=(size, self.data["NUM_LABELS"]), dtype=numpy.float32)
        for begin in xrange(0, size, self.hp["EVAL_BATCH_SIZE"]):
            end = begin + self.hp["EVAL_BATCH_SIZE"]
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    self.eval_prediction,
                    feed_dict={self.eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                   self.eval_prediction,
                   feed_dict={self.eval_data: data[-self.hp["EVAL_BATCH_SIZE"]:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]

        return predictions  
    
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


if __name__ == '__main__':
    
    import mnist_data as mnist
    dataset = mnist.import_dataset()                                    
    
    from modules.hpmgr import HPVManager 
   
    hpv = HPVManager('HPV_002.ini', ini_dir='../config/')
    model = CNN3C1F(dataset)
    model.learn(hpv)
