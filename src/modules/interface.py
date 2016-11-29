import os
import sys
import time
import traceback

from modules.debug import debug, error

# Abstract interface for models
class ModelInterface:
    
    def __init__(self, dataset):
        self.data = dataset
        self.logger = None
        self.predictor = None
        self.hp = {}
        self.flag = {}

    def setLogger(self, logger):
        self.logger = logger
        
    def setPredictor(self, predictor):
        self.predictor = predictor
        
    def learn(self, hpv):

        self.hp = hpv.getSectionMap('Hyperparameters')
        self.flag = hpv.getSectionMap('Execution')

        debug(self.hp)
        debug(self.flag)

        starttime = time.time()        
        try:
            y = self.train()
            duration = time.time() - starttime
            debug("Result: " + str(y) + ', Duration: ' + str(abs(duration)))
            return y

        except:
            e = sys.exc_info()
            traceback.print_exc()
            print(e)
        

    def checkEarlyStop(self):
        result = False
        # Check early termination
        
        if self.predictor:                    
            if num_epoch >= self.flag["EARLY_STOP_CHECK_EPOCHS"]:
                if self.predictor.load() is False:
                    debug("Unable to load training log")
                else:
                    debug("Predicting whether keep learning or not " )
                    togo = self.predictor.check(self.hp)                    
                    result = not togo 
                    debug("Prediction result: " + str(result))
        return result             
            
    def train(self):
        raise NotImplementedError()
