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
from multiprocessing import Lock

class PerformanceCSVLogger:
    def __init__(self, path, lock=None):
        self.path = path
        self.setting = ""
        self.lock = lock
        
    def __del__(self):
        self.delete()
    
    def create(self, params_list, metrics_list):
        num_params = len(params_list)
        num_metrics = len(metrics_list)
        column_names = "Timestamp,Msec,Setting,Measure Type,Step,Epoch,Elapsed Time"
        self.csv_format = '%(asctime)s,%(message)s'
        self.csv_header = column_names
        self.num_metrics = num_metrics
        self.num_params = num_params
        self.timers = {}
        self.elapsed_times_dict = {}
        self.params_list = params_list
        self.metrics_list = metrics_list
        self.steps_epoch = 0
        
        metric_names = ""
        for m in metrics_list:
            metric_names = metric_names + "," + m
        self.csv_header = self.csv_header + metric_names
        
        params_names = ""
        for p in params_list:
            params_names = params_names + "," + p
        self.csv_header = self.csv_header + params_names
        
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
            if self.lock:
                self.lock.acquire()
            f = open(self.path, 'w')
            f.writelines([self.csv_header + os.linesep])
            f.close()
            if self.lock:
                self.lock.release()
        else:
            f = open(self.path, 'r')
            csv_header = f.readline()
            csv_header = csv_header.rstrip()
            f.close()
            if csv_header != self.csv_header:
                print "CAUTION! the logging column number or order is different"
                metrics_params_list = csv_header.replace(column_names, "").split(",")
                adjusted_metrics = []
                adjusted_params = []
                for col in metrics_params_list:
                    if col is "":
                        print metrics_params_list
                        pass
                    elif col in metrics_list:
                        adjusted_metrics.append(col)
                    elif col in params_list:
                        adjusted_params.append(col)
                    else:
                        print "missing column is existed: " + col
                        raise ValueError("Compatiblity error: missing column is found in existed log file")
                        
                self.params_list = adjusted_params
                self.metrics_list = adjusted_metrics 
            
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
        
    
    def getParamList(self):
        return self.params_list
    
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

    def setStepsPerEpoch(self, steps):
        self.steps_epoch = steps
    
    def measure(self, measure_type, step, metrics_dict):        
        # measure elapsed time
        timegap = time.time() - self.timers[measure_type]
        
        self.elapsed_times_dict[measure_type].append(timegap)
        
        if self.steps_epoch == 0:
            epochs = "NA"
        else:
            epochs = "{0:.2f}".format(float(step) / float(self.steps_epoch))
            
        #print(timegap)
        accumulated_time = sum(self.elapsed_times_dict[measure_type])
        #print(self.elapsed_times_dict[measure_type]) 
        msg = self.setting + "," + measure_type + "," + str(step) + "," + epochs + ",{0:.3g}".format(accumulated_time)
        #print (metrics_dict)        
        for metric in self.metrics_list:
            if metric in metrics_dict:
                value = metrics_dict[metric]                
                msg = msg + ",{:.5f}".format(value)
            else:
                msg = msg + ",NA"
                
        msg = msg + self.hyperparams_vector
        if self.lock:
                self.lock.acquire()
        self.logger.debug(msg) # print out to log file
        if self.lock:
                self.lock.release()
        self.setTimer(measure_type) # reset timer        

     
        
        

        