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

NUM_PROCESSES = 2
NUM_EPOCHS = 1
LOG_PATH = "grid.csv"
DO_EVAL = False

NUM_REPEATS = 0
# global variables
DATASET = {} # empty dataset
PARAMS_LIST = [] # create dummy params list

PARAMS_LIST_PICKLE_FILE = "remained_params.pickle"


# CLI interface definition
tf.app.flags.DEFINE_string("config", "grid.cfg", "set config file path. default is train.cfg")
tf.app.flags.DEFINE_integer("concurrent", 2, "set the number of concurrent operations. default is 2")


# get grid search parameter list
def get_all_params(filter_sizes=[2], conv1_depths=[2], conv2_depths=[2], fc_depths=[2]):
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


def mnist_cnn(process_index, params, num_epochs):
    
    print (str(params) + "-" + str(len(PARAMS_LIST)) + " remains")
    eval_device_id = '/gpu:' + str(process_index)
    train_device_id = '/gpu:' + str(process_index)

    return model.learn(DATASET, params, train_dev=train_device_id, \
                       eval_dev=eval_device_id, progress=DO_EVAL, \
                       epochs=num_epochs, log_path=LOG_PATH)


def grid_search(num_processes, model_func, epochs):
    global PARAMS_LIST
    
    print("grid search: " + str(len(PARAMS_LIST)))
    try:
        while len(PARAMS_LIST) > 0:
            processes = []
            for p in range(num_processes):
                if len(PARAMS_LIST) is 0:
                    break
                else:
                    params = PARAMS_LIST.pop(0) # for FIFO
                    processes.append(Process(target=mnist_cnn, args=(p, params, epochs)))
            
            # start processes at the same time
            for k in range(len(processes)):
                processes[k].start()
            # wait until processes done
            for j in range(len(processes)):
                processes[j].join()
            
    except:
        # save undone params list to pickle file
        save_remains(PARAMS_LIST)
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

def main(argv=None):
    global DATASET
    global PARAMS_LIST
    global NUM_REPEATS
    global LOG_PATH
    global DO_EVAL
    
    
    try:
        print("Using settings in " + FLAGS.config) 
        cfg = Config(file(FLAGS.config))        
        params_list = PARAMS_LIST
        try:
            if os.path.exists(PARAMS_LIST_PICKLE_FILE):
                with open(PARAMS_LIST_PICKLE_FILE, 'rb') as f:
                    params_list = pickle.load(f)
                    print("Try to resume learnings by remained trains :" + str(len(params_list)))
                    os.remove(PARAMS_LIST_PICKLE_FILE)

        except:
            e = sys.exc_info()
            print("Params list pickle file error: " + str(e))
            traceback.print_exc()
        
        if len(params_list) is 0:
            print("Generating parameters list")
            params_list = get_all_params(filter_sizes=cfg.filter_sizes, \
                                         conv1_depths=cfg.conv1_depths, \
                                         conv2_depths=cfg.conv2_depths, \
                                         fc_depths=cfg.fc_depths)
        NUM_REPEATS = len(params_list)
        PARAMS_LIST = params_list
        
        
        print("Starting grid search: " + str(NUM_REPEATS) + " parameter sets")
        
        
        if cfg.log_path:
            LOG_PATH = cfg.log_path
        
        if cfg.do_eval:
            DO_EVAL = cfg.do_eval
            
        if cfg.dataset is "mnist":
            DATASET = model.download_dataset()           
        else:
            print("No such dataset is implemented yet: " + cfg.dataset)
            return

        if cfg.algorithm is "grid":
            grid_search(FLAGS.concurrent, mnist_cnn, cfg.num_epochs)
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
    

