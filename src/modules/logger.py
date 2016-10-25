#!/usr/bin/python
"""Module Utilities for TF-HPO

Useful classes and functions such as:
- to measure the performance and logging
- predict whether to learn or not

Author: Hyunghun Cho
"""

import os
import sys
import time
import logging
import logging.handlers
import traceback


class PerformanceCSVLogger:
    def __init__(self, path):
        self.path = path
        self.setting = ""
        
    def __del__(self):
        self.delete()
    
    def create(self, params_list, metrics_list):
        num_params = len(params_list)
        num_metrics = len(metrics_list)
        self.csv_format = '%(asctime)s,%(message)s'
        self.csv_header = "Timestamp,Msec,Setting,Measure Type,Step/Epoch,Elapsed Time"
        self.num_metrics = num_metrics
        self.num_params = num_params
        self.timers = {}
        self.elapsed_times_dict = {}
        self.params_list = params_list
        self.metrics_list = metrics_list
        
        for m in metrics_list:
            self.csv_header = self.csv_header + "," + m
            
        for p in params_list:
            self.csv_header = self.csv_header + "," + p
       
        
        # set timezone as GMT+9 
        
        os.environ['TZ'] = 'Asia/Seoul'
        # Create logger instance
        logger = logging.getLogger('tflogger')

        # Create logger formatter
        #fomatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
        file_formatter = logging.Formatter(self.csv_format)
        
        # Create handles to redirect the log to each stream and file
        fileHandler = logging.FileHandler(self.path)
        #streamHandler = logging.StreamHandler()

        # Set the formatter for each handlers
        fileHandler.setFormatter(file_formatter)
        #streamHandler.setFormatter(file_formatter)

        # Attach the handlers to logger instance
        logger.addHandler(fileHandler)
        #logger.addHandler(streamHandler)

        logger.setLevel(logging.DEBUG)

        # add the head if it doesn't existed yet
        if os.path.getsize(self.path) < 10 :
            f = open(self.path, 'w')
            f.writelines([self.csv_header + os.linesep])
            f.close()
            #logger.info(self.csv_header)
            
        self.logger = logger

    def setParamColumns(self, *params):
        self.params = params
        hyperparams = ""
        for l in params:
            hyperparams += "," + str(l)
        self.hyperparams_vector = hyperparams
        
    def setHyperparamSetting(self, dict):
        hyperparams = ""
        for param in self.params_list:
            hyperparams += "," + str(dict[param])
        self.hyperparams_vector = hyperparams
        
    def delete(self) :
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
            
    def setTimer(self, measure_type):
        self.timers[measure_type] = time.time()
        
        # add empty list if key did not existed in accumulated
        if not measure_type in self.elapsed_times_dict:
            self.elapsed_times_dict[measure_type] = []
        
    def setSetting(self, setting):
        self.setting = setting

    def measure(self, measure_type, step, metrics_dict):        
        # measure elapsed time
        timegap = time.time() - self.timers[measure_type]
        
        self.elapsed_times_dict[measure_type].append(timegap)
        
        #print(timegap)
        accumulated_time = sum(self.elapsed_times_dict[measure_type])
        #print(self.elapsed_times_dict[measure_type])
        msg = self.setting + "," + measure_type + "," + str(step) + ",{0:.3g}".format(accumulated_time)
        #print (metrics_dict)        
        for metric in self.metrics_list:
            if metric in metrics_dict:
                value = metrics_dict[metric]                
                msg = msg + ",{:.3f}".format(value)
            else:
                msg = msg + ",NA"
                
        msg = msg + self.hyperparams_vector
        self.logger.debug(msg) # print out to log file
        self.setTimer(measure_type) # reset timer        

     
        
        

        