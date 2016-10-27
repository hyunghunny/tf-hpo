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

import pickle
from modules.logger import PerformanceCSVLogger
from modules.predictor import PerformancePredictor
from modules.hpvconf import HPVGenerator
from modules.hpmgr import HPVManager

from random import shuffle


class TrainingManager:
    
    def __init__(self, model, config):        
        self.model = model
        
        self.cfg = config
        
        self.dev_type = 'cpu'
        self.num_devs = 1
        
        self.validation = False
        self.logging = True
        
        self.hyperparams = []
        self.metrics = ['Test Error', 'Validation Error', 'Training Error']
        
        self.early_stop = False
        self.early_stop_check_epochs = 1

        self.pickle_file = "backup.pickle"
        
        self.hpv_file_list = []
        
    def setPickle(self, pickle_file):
        self.pickle_file = pickle_file
    
    
    def setTrainingDevices(self, dev_type, num_devs):
        self.dev_type = dev_type            
        self.num_devs = num_devs

    def enableValidation(self, flag):
        self.validation = flag
    
    def enableLogging(self, flag):
        self.logging = flag
    
    def setLoggingParams(self, hyperparams):
        self.hyperparams = hyperparams
 
    def setLoggingMetrics(self, metrics):
        self.metrics = metrics

    def enableEarlyStop(self, flag, db_log_path=None, min_epochs=1):
        self.early_stop = flag
        if self.early_stop is True:
            self.validation = self.early_stop
            self.early_stop_check_epochs = min_epochs
    

    def train(self, hpv, process_index = 0):
        #print(self.dev_type)
        test_device_id = '/' + self.dev_type + ':' + str(process_index)
        train_device_id = '/' + self.dev_type + ':' + str(process_index)

        
        if self.cfg is not None:
            hpv.setOption('Execution', 'output_log_file', self.cfg.train_log_path)
        hpv.setOption('Execution', 'train_device_id', train_device_id)
        hpv.setOption('Execution', 'test_device_id', test_device_id)
        hpv.setOption('Execution', 'validation', self.validation)
        hpv.setOption('Execution', 'early_stop_check_epochs', self.early_stop_check_epochs)

        if self.logging:
            logger = PerformanceCSVLogger(self.cfg.train_log_path)
            logger.create(self.hyperparams, self.metrics)    
            logger.setSetting(hpv.getPath())
            self.model.setLogger(logger)
        
        if self.early_stop:
            predictor = PerformancePredictor(self.cfg.train_log_path)
            self.model.setPredictor(predictor)
        
        hpv.save()
        
        self.model.learn(hpv)    
    
    def runAll(self, hpv_file_list = None, num_processes=1):
        try:
            self.hpv_file_list = hpv_file_list
            working_hpv_list = []
            while len(self.hpv_file_list) > 0:
                processes = []
                for p in range(num_processes):
                    if len(hpv_file_list) is 0:
                        break
                    else:
                        hpv_file = self.hpv_file_list.pop(0) # for FIFO
                        working_hpv_list.append(hpv_file)
                        hpv = HPVManager(hpv_file)
                        processes.append(Process(target=self.train, args=(hpv, p)))

                # start processes at the same time
                for k in range(len(processes)):
                    processes[k].start()
                # wait until processes done
                for j in range(len(processes)):
                    processes[j].join()
                # XXX: to prepare shutdown
                self.backup()

        except:
            # save undone params list to pickle file
            self.hpv_file_list = self.hpv_file_list + working_hpv_list
            self.backup()
            traceback.print_exc()
            sys.exit(-1)
        
    def backup(self):        
        
        try:
            with open(self.pickle_file, 'wb') as f:
                pickle.dump(self.hpv_file_list, f)
                print(str(len(self.hpv_file_list)) + " HPV files remained.")
        except:
            e = sys.exc_info()
            traceback.print_exc()
            print(e)
            
    def restore(self):
        try:
            hpv_file_list = []
            if os.path.exists(self.pickle_file):
                with open(self.pickle_file, 'rb') as f:
                    hpv_file_list = pickle.load(f)
                    print("restore remained HPV files :" + str(len(self.hpv_file_list)))

        except:
            e = sys.exc_info()
            print("Pickle file error: " + str(e))
            traceback.print_exc()
            hpv_file_list = []

        finally:
            self.hpv_file_list = hpv_file_list
            return self.hpv_file_list