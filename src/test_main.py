from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import getopt
import traceback
from multiprocessing import Pool, Lock, Manager, Process

import models.cnn_model as model

import tensorflow as tf

from config import Config
import pickle
from modules.logger import PerformanceCSVLogger
from modules.predictor PerformancePredictor 
from random import shuffle


NUM_PROCESSES = 4
NUM_EPOCHS = 1
OUTPUT_LOG_PATH = "train_log.csv"
LOG_DB_PATH = "../log/test.csv"
CFG_PATH="test.cfg"
PARAMS_LIST_PICKLE_FILE = "test.pickle"

# Whether evaluation at each evaluation frequency
DO_VALIDATION=False

# Whether using stop training early 
EARLY_STOP=False

# Whether using prevalidation for hyperparameter vectors
PREVALIDATE=False

# CLI interface definition
tf.app.flags.DEFINE_string("config", CFG_PATH, "set config file path. default is " + CFG_PATH)
tf.app.flags.DEFINE_integer("concurrent", NUM_PROCESSES, "set the number of concurrent operations. default is " + str(NUM_PROCESSES))


# get ascending order grid search parameter list
def get_all_params_list(filter_sizes=[2], conv1_depths=[2], fc_depths=[2]):
    params_list = []

    for f in range(len(filter_sizes)):
        filter_size = filter_sizes[f]
        for conv1 in range(len(conv1_depths)):
            conv1_depth = conv1_depths[conv1]
            for fc in range(len(fc_depths)):
                fc_depth = fc_depths[fc]
                params = {"filter_size": filter_size,\
                          "conv1_depth" : conv1_depth,\
                          "fc1_depth": fc_depth}
                params_list.append(params)    
    
    return params_list           


def prevalidate(params_list):
    predictor = PerformancePredictor(DB_LOB_PATH)
    if predictor.load() is False:
        print("Unable to load training log")
        return params_list
    else:        
        validated_list = []
        for i in range(len(params_list)):
            params = params_list[i]
            #print ("Show your content: " + str(params))
            result = predictor.validate(Param1=params["filter_size"], Param2=params["conv1_depth"],\
                                      Param3=params["conv2_depth"], Param4=params["fc1_depth"])
            if result is True:
                validated_list.append(params)
            '''
            else:
                print(str(params) + " is not appropriate to learn. Therefore, skip it")
            '''
        return validated_list        

    
         

            
def train_model(process_index, dataset, params, num_epochs, logger, predictor):
    
    # Set the same GPU device to run concurrently 
    eval_device_id = '/gpu:' + str(process_index)
    train_device_id = '/gpu:' + str(process_index)

    return model.learn(dataset, params,\
                       train_dev=train_device_id, \
                       eval_dev=eval_device_id, \
                       do_validation=DO_VALIDATION, \
                       epochs=num_epochs, \
                       logger=logger, predictor=predictor)


def train_all(dataset, params_list, epochs, logger, predictor):
    
    #TODO: If Lock is required, create here and pass
    print("Training parameter sets: " + str(len(params_list)))
    try:
        working_params_list = []
        while len(params_list) > 0:
            processes = []
            for p in range(FLAGS.concurrent):
                if len(params_list) is 0:
                    break
                else:
                    params = params_list.pop(0) # for FIFO
                    working_params_list.append(params)
                    
                    processes.append(Process(target=train_model, args=(p, dataset, params, epochs, logger, predictor)))
            
            # start processes at the same time
            for k in range(len(processes)):
                processes[k].start()
            # wait until processes done
            for j in range(len(processes)):
                processes[j].join()
            # XXX: to prepare shutdown
            save_remains(params_list)
            
    except:
        # save undone params list to pickle file
        remains_list = params_list + working_params_list
        save_remains(remains_list)
        sys.exit(-1)

        
        
def save_remains(params_list):        
    print(str(len(params_list)) + " params remained to learn")
    try:
        with open(PARAMS_LIST_PICKLE_FILE, 'wb') as f:
            pickle.dump(params_list, f)
            print(PARAMS_LIST_PICKLE_FILE + " saved properly")
    except:
        e = sys.exc_info()
        traceback.print_exc()
        print(e)    

FLAGS = tf.app.flags.FLAGS

def get_pending_params_list():
    try:
        params_list = None
        if os.path.exists(PARAMS_LIST_PICKLE_FILE):
            with open(PARAMS_LIST_PICKLE_FILE, 'rb') as f:
                params_list = pickle.load(f)
                print("Try to resume learnings by remained trains :" + str(len(params_list)))
                #os.remove(PARAMS_LIST_PICKLE_FILE)
                
    except:
        e = sys.exc_info()
        print("Params list pickle file error: " + str(e))
        traceback.print_exc()
        
    finally:
        if params_list is None:
            return []
        else:
            return params_list

def main(argv=None):

    global OUTPUT_LOG_PATH
    global DO_EVAL
    global EARLY_STOP
    
    try:
        print("Using settings in " + FLAGS.config) 
        cfg = Config(file(FLAGS.config))        

        if cfg.log_path:
            OUTPUT_LOG_PATH = cfg.log_path
        
        if cfg.do_eval:
            DO_EVAL = cfg.do_eval

        if cfg.early_stop:
            EARLY_STOP = cfg.early_stop            
            
        
        if cfg.dataset is "mnist":
            import mnist_data as mnist
            dataset = mnist.import_dataset()           
        else:
            print("No such dataset is implemented yet: " + cfg.dataset)
            return
        
        # Check pending paramters are exited
        params_list = get_pending_params_list()
        
        if len(params_list) is 0:
            print("Generating parameters list")
            params_list = get_all_params_list(filter_sizes=cfg.filter_sizes, \
                                         conv1_depths=cfg.conv1_depths, \
                                         fc_depths=cfg.fc_depths)
        
        # eliminate unsuccessful parameter sets in history
        if cfg.prevalidate:
            params_list = prevalidate(params_list)
            
        # create logger
        logger = PerformanceCSVLogger(OUTPUT_LOG_PATH)
        logger.create(3, 3) # create log with 3 hyperparams and 3 accuracy metrics  
        print("Logging test errors at " + OUTPUT_LOG_PATH)        
        
        # create predictor
        if EARLY_STOP:
            # XXX: change if you want to other log DB
            LOG_DB_PATH = OUTPUT_LOG_PATH
            predictor = PerformancePredictor(LOG_DB_PATH)
        else:
            predictor = None
        
        trainMgr = TrainingManager(dataset, params_list)
        trainMgr.setTrainingDevices('gpu', cfg.
        # determine hyperparameter optimization algorithm
        if cfg.algorithm is "random":            
            shuffle(params_list)
            train_all(dataset, params_list, cfg.num_epochs, logger, predictor)

        elif cfg.algorithm is "grid":           
            train_all(dataset, params_list, cfg.num_epochs, logger, predictor)
        else:
            print("No such method is implemented yet: " + FLAGS.method)
            return 
        
    except:
        e = sys.exc_info()
        print("Configuration file error: " + str(e))
        traceback.print_exc()
        
        return

    
if __name__ == '__main__':
    tf.app.run()
    

