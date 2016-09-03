#!/usr/bin/python
"""Module Utilities for TF-HPO

Useful classes and functions to measure the performance and logging

Author: Hyunghun Cho
"""

import os
import time
import logging
import logging.handlers

class PerformanceCSVLogger:
    def __init__(self, path):
        self.path = path
        self.setting = ""
        
    def __del__(self):
        self.delete()
    
    def create(self, num_params, num_metrics):
                
        self.csv_format = '%(asctime)s,%(message)s'
        self.csv_header = "Setting,Step,Elapsed Time"
        self.num_metrics = num_metrics
        self.num_params = num_params
        
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
            
    def setTimer(self):
        self.timestamp = time.time()
        
    def setSetting(self, setting):
        self.setting = setting
    
    def measure(self, step, *metrics):        
        # measure elapsed time
        timegap = time.time() - self.timestamp
        
        msg = self.setting +"," + str(step) + ",{0:.3g}".format(timegap)
        num_metric_remains = self.num_metrics - len(metrics)
        
        for i in range(self.num_metrics):
            if i < len(metrics):
                metric = metrics[i]
                msg = msg + ",{:.3f}".format(metric)
            else:
                msg = msg + ",NA"
                
        msg = msg + self.hyperparams
        self.logger.debug(msg)
        self.setTimer() # reset timer
