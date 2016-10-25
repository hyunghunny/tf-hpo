from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import time
import traceback

import models.mnist_conv_model as model
from modules.logger import PerformanceCSVLogger 
from modules.predictor import PerformancePredictor

import tensorflow as tf

import HPOlib.benchmark_util as benchmark_util
import HPOlib.benchmark_functions as benchmark_functions

LOG_PATH = '../../log/hpolib.csv'
NUM_EPOCHS = 1

def main(params, **kwargs):    # pylint: disable=unused-argument
    
    print('Params: '+ str(params))
    
    eval_device_id = '/gpu:0'
    train_device_id = '/gpu:0'
    
    dataset = model.download_dataset()
    logger = PerformanceCSVLogger(LOG_PATH)
    logger.create(['Filter size', 'Conv1 depth', 'Conv2 depth', 'FC neurons'], ['Test Accuracy', 'Validation Accuarcy', 'Training Accuracy'])  
    print("Logging test errors at " + LOG_PATH)        
    test_error = model.learn(dataset, params, train_dev=train_device_id, \
                       eval_dev=eval_device_id, do_eval=False, \
                       epochs=NUM_EPOCHS, logger=logger)
    return test_error    

if __name__ == '__main__':
    print("hyperparameter optimization for tensorflow is being started")
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    result = main(params, **args)
    duration = time.time() - starttime
    print ("Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__)))