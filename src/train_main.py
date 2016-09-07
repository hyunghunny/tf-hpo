from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import getopt
import traceback
from multiprocessing import Pool, Lock, Manager, Process
import mnist_conv_model as model
import tensorflow as tf
from config import Config
import pickle
from util import PerformancePredictor
from random import shuffle


NUM_PROCESSES = 2
NUM_EPOCHS = 1
OUTPUT_LOG_PATH = "grid.csv"
DB_LOB_PATH = "../log/grid.csv"
PARAMS_LIST_PICKLE_FILE = "remained_params.pickle"
CFG_PATH="train.cfg"

DO_EVAL = False
EARLY_STOP=True
PREVALIDATE=True

# CLI interface definition
tf.app.flags.DEFINE_string("config", CFG_PATH, "set config file path. default is train.cfg")
tf.app.flags.DEFINE_integer("concurrent", 4, "set the number of concurrent operations. default is 4")


# get grid search parameter list
def get_all_params_list(filter_sizes=[2], conv1_depths=[2], conv2_depths=[2], fc_depths=[2]):
    params_list = []

    for f in range(len(filter_sizes)):
        filter_size = filter_sizes[f]
        for conv1 in range(len(conv1_depths)):
            conv1_depth = conv1_depths[conv1]
            for conv2 in range(len(conv2_depths)):
                conv2_depth = conv2_depths[conv2]
                for fc in range(len(fc_depths)):
                    fc_depth = fc_depths[fc]
                    params = {"filter_size": filter_size, "conv1_depth" : conv1_depth,\
                              "conv2_depth": conv2_depth, "fc_depth": fc_depth}
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
                                      Param3=params["conv2_depth"], Param4=params["fc_depth"])
            if result is True:
                validated_list.append(params)
            '''
            else:
                print(str(params) + " is not appropriate to learn. So, skip it")
            '''
        return validated_list        
                

def learn_cnn(process_index, dataset, params, num_epochs):
    
    eval_device_id = '/gpu:' + str(process_index)
    train_device_id = '/gpu:' + str(process_index)

    return model.learn(dataset, params, train_dev=train_device_id, \
                       eval_dev=eval_device_id, progress=DO_EVAL, \
                       epochs=num_epochs, log_path=OUTPUT_LOG_PATH, breakout=EARLY_STOP)


def train_seq(dataset, params_list, model_func, epochs, num_processes):
    
    print("Training parameter sets: " + str(len(params_list)))
    try:
        working_params_list = []
        while len(params_list) > 0:
            processes = []
            for p in range(num_processes):
                if len(params_list) is 0:
                    break
                else:
                    params = params_list.pop(0) # for FIFO
                    working_params_list.append(params)
                    
                    processes.append(Process(target=learn_cnn, args=(p, dataset, params, epochs)))
            
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
                os.remove(PARAMS_LIST_PICKLE_FILE)
                
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

    global NUM_REPEATS
    global OUTPUT_LOG_PATH
    global DO_EVAL
    
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
            dataset = model.download_dataset()           
        else:
            print("No such dataset is implemented yet: " + cfg.dataset)
            return
        
        # Check pending paramters are exited
        params_list = get_pending_params_list()
        
        if len(params_list) is 0:
            print("Generating parameters list")
            params_list = get_all_params_list(filter_sizes=cfg.filter_sizes, \
                                         conv1_depths=cfg.conv1_depths, \
                                         conv2_depths=cfg.conv2_depths, \
                                         fc_depths=cfg.fc_depths)
        
        # drop out unsuccessful parameter sets in history
        if cfg.prevalidate:
            params_list = prevalidate(params_list)    

        if cfg.algorithm is "random":            
            shuffle(params_list)
            train_seq(dataset, params_list, learn_cnn, cfg.num_epochs, FLAGS.concurrent)

        elif cfg.algorithm is "grid":
           
            train_seq(dataset, params_list, learn_cnn, cfg.num_epochs, FLAGS.concurrent)
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
    

