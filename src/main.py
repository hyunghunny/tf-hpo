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
from modules.hpvconf import HPVGenerator
import models.mnist_data as mnist
from models.cnn_model import CNN

CFG_PATH = "main.cfg"
HPV_TEMPLATE_FILE = 'CNN_HPV.ini'
PICKLE_PATH = 'temp/backup.pickle'
NUM_GPUS = 4
MAX_COUNT = 10 # limits the number of each hyperparameter values 
RESUME = False

# CLI interface definition
tf.app.flags.DEFINE_string("config", CFG_PATH, "set config file path. default is " + CFG_PATH)
tf.app.flags.DEFINE_integer("concurrent", NUM_GPUS, "set the number of concurrent operations. default is " + str(NUM_GPUS))
tf.app.flags.DEFINE_integer("resume", RESUME, "set whether resumes the previous trainings. default is " + str(RESUME))

FLAGS = tf.app.flags.FLAGS

def main(argv=None):
    
    try:
        cfg = Config(file(FLAGS.config))
        # generate hyperparameter vectors from configurator's configuration
        generator = HPVGenerator(cfg)
        generator.loadTemplate(cfg.ini_template_path)
        grid_search_list = generator.grid(MAX_COUNT)
                
        train_manager = TrainingManager(CNN(mnist.import_dataset()), cfg)
        train_manager.setTrainingDevices('gpu', NUM_GPUS)
        train_manager.setPickle(PICKLE_PATH)
        
        train_manager.setLoggingParams([ hyperparam for hyperparam in cfg.hyperparameters])
        
        if FLAGS.resume:
            restore_list = train_manager.restore()
            
            if len(restore_list) > 0:
                grid_search_list = restore_list
                print ("previous HPV list is restored")
        print (grid_search_list)
        train_manager.runAll(grid_search_list, num_processes=FLAGS.concurrent)
           
    except:
        e = sys.exc_info()
        print("Error: " + str(e))
        traceback.print_exc()
        
        return

    
if __name__ == '__main__':
    tf.app.run()
    

