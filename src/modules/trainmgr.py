from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import getopt
import traceback
from multiprocessing import Lock, Manager, Process

import models.cnn_model as model

import tensorflow as tf

import pickle
from modules.logger import PerformanceCSVLogger
from modules.predictor import PerformancePredictor
from modules.hpvconf import HPVGenerator
from modules.hpmgr import HPVManager

from random import shuffle

IN_PROGRESS_PICKLE="ongoing.pickle"
ERROR_PICKLE="error.pickle"
REMAIN_PICKLE="remain.pickle"

class TrainingManager:
    
    def __init__(self, model, train_log_path = None):        
        self.model = model
        
        
        self.dev_type = 'cpu'
        self.num_devs = 1
        
        self.validation = False

        self.train_log_path = train_log_path
        
        if train_log_path is None:
            self.logging = False
        else:
            self.logging = True
            
        self.hyperparams = []
        self.metrics = ['Test Error', 'Validation Error', 'Training Error']
        
        self.early_stop = False
        self.early_stop_check_epochs = 1

        self.pickle_dir = "./"
        self.hpv_file_list = []
        self.working_hpv_list = []
        
        self.lock = Lock()
        
    def setPickleDir(self, pickle_dir):
        self.pickle_dir = pickle_dir    
    
    def setTrainingDevices(self, dev_type, num_devs):
        self.dev_type = dev_type            
        self.num_devs = num_devs

    def enableValidation(self, flag):
        self.validation = flag
    
    def enableLogging(self, flag, train_log_path = None):
        self.logging = flag
        self.train_log_path = train_log_path
    
    def setLoggingParams(self, hyperparams):
        self.hyperparams = hyperparams
 
    def setLoggingMetrics(self, metrics):
        self.metrics = metrics

    def enableEarlyStop(self, flag, db_log_path=None, min_epochs=1):
        self.early_stop = flag
        if self.early_stop is True:
            self.validation = self.early_stop
            self.early_stop_check_epochs = min_epochs

    def run(self, hpv, process_index = 0):
        #print(self.dev_type)
        test_device_id = '/' + self.dev_type + ':' + str(process_index)
        train_device_id = '/' + self.dev_type + ':' + str(process_index)        
                
        hpv.setOption('Execution', 'output_log_file', self.train_log_path)
        hpv.setOption('Execution', 'train_device_id', train_device_id)
        hpv.setOption('Execution', 'test_device_id', test_device_id)
        hpv.setOption('Execution', 'validation', self.validation)
        hpv.setOption('Execution', 'early_stop_check_epochs', self.early_stop_check_epochs)

        if self.logging:
            logger = PerformanceCSVLogger(self.train_log_path, lock=self.lock)
            logger.create(self.hyperparams, self.metrics)    
            logger.setSetting(hpv.getPath())
            self.model.setLogger(logger)
        
        if self.early_stop:
            predictor = PerformancePredictor(self.train_log_path)
            self.model.setPredictor(predictor)
        
        hpv.save()
        
        result = self.model.learn(hpv)
        
        if result:
            # saving working HPV list after learning terminated
            self.working_hpv_list = self.restore(IN_PROGRESS_PICKLE)
            hpv_file = hpv.getPath()
            print ("Training with " + hpv_file + " is terminated properly")
            if hpv_file in self.working_hpv_list:
                self.working_hpv_list.remove(hpv_file) 
                self.backup(self.working_hpv_list, IN_PROGRESS_PICKLE) 
        
        return result
    
    def runAll(self, hpv_file_list, num_processes=1):
        try:
            self.hpv_file_list = hpv_file_list
            dev_ids = [i for i in range(self.num_devs)]
            dev_ids.reverse() # use higher device id first           
            while len(self.hpv_file_list) > 0:                
                processes = []
                for p in range(num_processes):
                    
                    if len(self.hpv_file_list) > 0:
                        hpv_file = self.hpv_file_list.pop(0) # for FIFO
                        self.working_hpv_list.append(hpv_file)                        
                        hpv = HPVManager(hpv_file)
                        processes.append(Process(target=self.run, args=(hpv, dev_ids[p])))
                
                self.backup(self.hpv_file_list, REMAIN_PICKLE)
                self.backup(self.working_hpv_list, IN_PROGRESS_PICKLE) # saving working HPV list before learning is terminated 
                # start processes at the same time
                for k in range(len(processes)):
                    processes[k].start()
                
                # wait until processes done
                for j in range(len(processes)):
                    processes[j].join()
                
                self.working_hpv_list = []

        except:
            # save undone params list to pickle file            
            error_list = self.restore(ERROR_PICKLE)
            error_list = error_list + self.working_hpv_list
            self.backup(error_list, ERROR_PICKLE)
            traceback.print_exc()
            sys.exit(-1)
        
    def backup(self, file_list, pickle_file):        
        
        try:
            pickle_path = self.pickle_dir + pickle_file
            with open(pickle_path, 'wb') as f:
                pickle.dump(file_list, f)
                print("Backup " + str(len(file_list)) + " HPV files to " + pickle_file)
        except:
            e = sys.exc_info()
            traceback.print_exc()
            print(e)
            
    def restore(self, pickle_file):
        hpv_file_list = []
        pickle_path = self.pickle_dir + pickle_file
        try:
            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as f:
                    hpv_file_list = pickle.load(f)
                    print("restore " + str(len(hpv_file_list)) + " HPV files at " + pickle_path)
        except:
            e = sys.exc_info()            
            traceback.print_exc()
            hpv_file_list = []

        finally:
            return hpv_file_list
