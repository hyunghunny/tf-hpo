from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import time
import traceback

import models.mnist_data as mnist
from models.cnn_model import CNN
from modules.hpvconf import HPVGenerator
from modules.hpmgr import HPVManager
from modules.trainmgr import TrainingManager

import HPOlib.benchmark_util as benchmark_util
import HPOlib.benchmark_functions as benchmark_functions

LOG_PATH = '../../log/hpolib-history.csv'
TEMPLATE_PATH = '../../src/CNN_HPV.ini'
CONFIG_PATH = '../../src/config/'
NUM_GPUS = 1 # only one gpu can be used for HPOlib

def main(params, **kwargs):    # pylint: disable=unused-argument
    
    print('Params: '+ str(params))
    
    try:        
        generator = HPVGenerator()
        generator.setTemplate(TEMPLATE_PATH)
        hpv_file = generator.generate(params, output_dir=CONFIG_PATH)
                
        train_manager = TrainingManager(CNN(mnist.import_dataset()), LOG_PATH)
        train_manager.setTrainingDevices('gpu', NUM_GPUS)
        train_manager.setLoggingParams([ hyperparam for hyperparam in params])
        hpv = HPVManager(hpv_file, ini_dir=CONFIG_PATH)
        return train_manager.run(hpv)
           
    except:
        e = sys.exc_info()
        print("Error: " + str(e))
        traceback.print_exc()

if __name__ == '__main__':
    print("hyperparameter optimization for tensorflow is being started")
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    result = main(params, **args)
    duration = time.time() - starttime
    print ("Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__)))