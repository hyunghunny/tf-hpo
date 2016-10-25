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
'''    
    def measure(self, measure_type, step, *metrics):        
        # measure elapsed time
        timegap = time.time() - self.timers[measure_type]
        
        self.elapsed_times_dict[measure_type].append(timegap)
        print(timegap)
        accumulated_time = sum(self.elapsed_times_dict[measure_type])
        print(accumulated_time)
        msg = self.setting + "," + measure_type + "," + str(step) + ",{0:.3g}".format(accumulated_time)
        num_metric_remains = self.num_metrics - len(metrics)
        
        for i in range(self.num_metrics):
            if i < len(metrics):
                metric = metrics[i]
                msg = msg + ",{:.3f}".format(metric)
            else:
                msg = msg + ",NA"
                
        msg = msg + self.hyperparams_vector
        self.logger.debug(msg) # print out to log file
        self.setTimer(measure_type) # reset timer
'''
        
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
                #print subset_table.head(1)
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
                tag = "hyper params: "
                for k in params:
                    col = k
                    val = params[k]
                    tag += (", " + str(k) + "=" + str(val))
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
            
            print tag + ", accuracy slop: " + str(slope)
            
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
        
        
        
        

        