from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import getopt
import traceback
from config import Config

import models.cnn_model as model

import tensorflow as tf

from modules.trainmgr import TrainingManager
from modules.hpvconf import HPVGridGenerator
import models.mnist_data as mnist
from models.cnn_model import CNN

CFG_PATH = "main.cfg"
HPV_TEMPLATE_FILE = 'CNN_HPV.ini'
PICKLE_DIR = 'temp/'
REMAIN_PICKLE="remain.pickle"
IN_PROGRESS_PICKLE="ongoing.pickle"
ERROR_PICKLE="error.pickle"
NUM_GPUS = 4
MAX_COUNT = 10 # limits the number of each hyperparameter values (use for debugging purpose)
RESUME = False

# CLI interface definition
tf.app.flags.DEFINE_string("config", CFG_PATH, "set config file path. default is " + CFG_PATH)
tf.app.flags.DEFINE_integer("concurrent", NUM_GPUS, "set the number of concurrent operations. default is " + str(NUM_GPUS))
tf.app.flags.DEFINE_boolean("resume", RESUME, "set whether resumes the previous trainings. default is " + str(RESUME))
tf.app.flags.DEFINE_string("pickle", REMAIN_PICKLE, "set pickle to restore if resume flag is True. default is " + REMAIN_PICKLE)

FLAGS = tf.app.flags.FLAGS

def main(argv=None):
    
    try:
        cfg = Config(file(FLAGS.config))
        # generate hyperparameter vectors from configurator's configuration
        generator = HPVGridGenerator(cfg)
        generator.setTemplate(cfg.ini_template_path)
        hpv_list = generator.generate(MAX_COUNT)
                
        train_manager = TrainingManager(CNN(mnist.import_dataset()), cfg.train_log_path)
        train_manager.setTrainingDevices('gpu', NUM_GPUS)
        train_manager.setPickleDir(PICKLE_DIR)
        train_manager.enableValidation(cfg.enable_validation)
        
        train_manager.setLoggingParams([ hyperparam for hyperparam in cfg.hyperparameters])
        
        if FLAGS.resume:
            # merge HPV in progress into error
            working_remains = train_manager.restore(IN_PROGRESS_PICKLE)
            error_remains = train_manager.restore(ERROR_PICKLE)
            merge_remains = list(set(working_remains + error_remains)) # delete duplicates
            train_manager.backup(merge_remains, ERROR_PICKLE)
            print ("undone HPV list merged to error list: " + str(len(merge_remains)))
            
            restore_list = train_manager.restore(FLAGS.pickle) # reload from remained 
            
            if len(restore_list) > 0:
                hpv_list = restore_list
                print ("previous HPV list is restored: " + str(len(hpv_list)))
        #print (hpv_list)
        train_manager.runAll(hpv_list, num_processes=FLAGS.concurrent)
           
    except:
        e = sys.exc_info()
        print("Error: " + str(e))
        traceback.print_exc()
        
        return

    
if __name__ == '__main__':
    tf.app.run()
    

