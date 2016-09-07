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

import math
import pandas as pd
import numpy as np
from scipy.stats import linregress

class PerformanceCSVLogger:
    def __init__(self, path):
        self.path = path
        self.setting = ""
        
    def __del__(self):
        self.delete()
    
    def create(self, num_params, num_metrics):
                
        self.csv_format = '%(asctime)s,%(message)s'
        self.csv_header = "Setting,Measure Type,Step,Elapsed Time"
        self.num_metrics = num_metrics
        self.num_params = num_params
        self.timers = {}
        
        for a in range(num_metrics):
            self.csv_header = self.csv_header + ",Metric"+str(a+1)
            
        hyperparams = []
        for l in range(num_params):
            self.csv_header = self.csv_header + ",Param" + str(l+1)
            hyperparams.append("NA")
       
        self.setParamColumns(*hyperparams)
        
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
            logger.info(self.csv_header)
            
        self.logger = logger

    def setParamColumns(self, *params):
        self.params = params
        hyperparams = ""
        for l in params:
            hyperparams += "," + str(l)
        self.hyperparams = hyperparams
        
    def delete(self) :
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
            
    def setTimer(self, type):
        self.timers[type] = time.time()
        
    def setSetting(self, setting):
        self.setting = setting
    
    def measure(self, type, step, *metrics):        
        # measure elapsed time
        timegap = time.time() - self.timers[type]
        
        msg = self.setting + "," + type + "," + str(step) + ",{0:.3g}".format(timegap)
        num_metric_remains = self.num_metrics - len(metrics)
        
        for i in range(self.num_metrics):
            if i < len(metrics):
                metric = metrics[i]
                msg = msg + ",{:.3f}".format(metric)
            else:
                msg = msg + ",NA"
                
        msg = msg + self.hyperparams
        self.logger.debug(msg)
        self.setTimer(type) # reset timer

        
class PerformancePredictor:
    def __init__(self, log_path, acc_type="Metric1", threshold=50.0):
        self.acc_type = acc_type
        self.threshold = threshold
        self.log_path = log_path
        self.acc_table = None
    
    def load(self):
        try:
            #print "read " + self.log_path
            self.acc_table = pd.read_csv(self.log_path, header=0)
            #print "Check accuracy table:"
            #print self.acc_table.describe()
            #print len(self.acc_table)
            if len(self.acc_table) is 0:
                return False
            return True
        except:
            self.acc_table = None
            print "Couldn't load log file: " + self.log_path
            return False
    
    def reload(self, log_path):
        if log_path is None:
            log_path = self.log_path
        return load(log_path)

    
    def validate(self, **params):
        #print "Try to lookup best accuracy with given conditions: " + str(params)
        try:
            if self.acc_table is not None:
                
                subset_table = self.acc_table.copy()
                #print subset_table.describe()
                
                for k in params:
                    col = k
                    val = params[k]
                    #print "column name: " + str(col) + ", search value: " + str(val)
                    subset_table = subset_table[subset_table[col] == val]
                    #print subset_table.head(5)
                    #print len(subset_table)
                
                selected = subset_table[self.acc_type]
                print subset_table.head(1)
            else:            
                selected = df.DataFrame([]) # return empty list
            
            if len(selected) is 0:
                print "Unable to load accuracy table properly"
                return False

            #print str("Available accuracy values: " + str(selected.values))
            max_perf = selected.values.max()
            #print "Current MAX performance: " + str(max_perf) + ", threshold: "  + str(self.threshold)
            if max_perf > self.threshold:
                # Good to keep learning
                return True
            else:
                # Bad learner, stop waste!
                return False
        except:
            e = sys.exc_info()
            traceback.print_exc()
            print(e) 

    def predict(self, **params):
        ''' predict increase or decrease using linear regression '''
        try:
            if self.acc_table is not None:
                
                #print self.acc_table.head(5)
                subset_table = self.acc_table.copy()
                #print subset_table.head(5)
                
                for k in params:
                    col = k
                    val = params[k]
                    #print "column name: " + str(col) + ", search value: " + str(val)
                    subset_table = subset_table[subset_table[col] == val]
                    #print subset_table.head(5)
                    #print len(subset_table)
                
                selected = subset_table[self.acc_type]
                #print subset_table.head(1)
            else:            
                selected = df.DataFrame([]) # return empty list
            
            if len(selected) is 0:
                print "Unable to load accuracy table properly"
                return False

            # choose only 5 
            if len(selected) > 5:
                selected = selected.tail(5)
                
            x = range(len(selected))
            y = selected.values
            
            #clean_data = pd.concat([x, y], 1).dropna(0) # row-wise
            #(_, x), (_, y) = clean_data.iteritems()
            slope, intercept, r, p, stderr = linregress(x, y)
            
            print "Accuracy slop: " + str(slope)
            
            if math.isnan(float(slope)):
                # keep going
                return True
            elif float(slope) < 0.0:
                # Bad learner, stop wasting!
                return False
            else:
                # Good! keep learning                
                return True
        except:
            e = sys.exc_info()
            traceback.print_exc()
            print(e) 
        
        
        
        

        