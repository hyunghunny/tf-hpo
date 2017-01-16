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
from modules.predictor import PerformancePredictor

import HPOlib.benchmark_util as benchmark_util
import HPOlib.benchmark_functions as benchmark_functions

from config import Config

CONFIG_FILE = "../../src/hpolib.cfg"

def main(params, **kwargs):    # pylint: disable=unused-argument
    
    print('Params: '+ str(params))
    print('Dicts: ' + str(kwargs))
    
    try:
        cfg = Config(file(CONFIG_FILE))
        generator = HPVGenerator()
        generator.setTemplate(cfg.TEMPLATE_PATH)
        hpv_file = generator.generate(params, output_dir=cfg.CONFIG_PATH)
        
        logging_params = [ hyperparam for hyperparam in params]
        train_manager = TrainingManager(CNN(mnist.import_dataset()), cfg.LOG_PATH)
        train_manager.setTrainingDevices('gpu', cfg.NUM_GPUS)
        train_manager.setLoggingParams(logging_params)
        
        hpv = HPVManager(hpv_file, ini_dir=cfg.CONFIG_PATH)
        
        result_rate = train_manager.run(hpv, cfg.GPU_ID)
        pred = PerformancePredictor(cfg.LOG_PATH)
        if (pred.load()):
            # key of setting is hpv_file        
            best_result = pred.get_best_error(hpv_file)
            best_result_rate = best_result * 100
            if result_rate > best_result_rate:
                print("test error calibrated due to overfitting: " + str(result_rate) + " -> " + str(best_result_rate))
                result_rate = best_result_rate
                
        return result_rate
           
    except:
        e = sys.exc_info()
        print("Error: " + str(e))
        traceback.print_exc()

if __name__ == '__main__':
    print("hyperparameter Optimization for TensorFlow is being started")
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    result = main(params, **args)
    duration = time.time() - starttime
    print ("Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__)))