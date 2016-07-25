#!/usr/bin/python
"""Module Utilities for TF-HPO

Useful classes and functions to measure the performance and logging

Author: Hyunghun Cho
"""

import os
import time
import logging
import logging.handlers

class CSVLogger:
    def __init__(self, path, layers):
        self.path = path
        self.csv_format = '%(asctime)s,%(message)s'
        self.csv_header = "Tag Name,Iteration,Testing Accuracy(256),Testing Accuracy(512),Testing Accuracy(2048),Elapsed Time"
        emptyLayers = []
        for l in range(layers):
            self.csv_header = self.csv_header + ",L" + str(l+1)
            emptyLayers.append("NA")
        self.create()
        
        self.setLayers(*emptyLayers)
        
    def __del__(self):
        self.delete()
        
    def create(self) :
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

        # add head tag if it doesn't existed
        if os.path.getsize(self.path) < 10 :
            logger.info(self.csv_header)
            
        self.logger = logger
        
    def delete(self) :
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
            
    def setTimer(self):
        self.timestamp = time.time()
        
    def setLayers(self, *layers):
        self.layers = layers
        tails = ""
        for l in layers:
            tails += "," + str(l)
        self.tails = tails
        
    def measure(self, tag, iteration, acc1, acc2, acc3):
        timegap = time.time() - self.timestamp
        msg = tag +"," + str(iteration) + "," + \
            "{:.6f}".format(acc1) + "," + \
            "{:.5f}".format(acc2) + "," + "{:.5f}".format(acc3) + "," + \
            "{0:.3g}".format(timegap) + self.tails
        self.logger.debug(msg)
        self.setTimer() # reset timer
      
        