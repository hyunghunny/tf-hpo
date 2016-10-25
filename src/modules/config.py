#!/usr/bin/python
"""Module Utilities for TF-HPO

Useful classes and functions such as:
- to generate hyperparameter vectors with given ranges
- to prevalidate the performance with given train log

Author: Hyunghun Cho
"""
from logger import PerformanceCSVLogger, 
from predictor import PerformancePredictor 

# get ascending order grid search parameter list
def get_all_params_list(filter_sizes=[2], conv1_depths=[2], fc_depths=[2]):
    params_list = []

    for f in range(len(filter_sizes)):
        filter_size = filter_sizes[f]
        for conv1 in range(len(conv1_depths)):
            conv1_depth = conv1_depths[conv1]
            for fc in range(len(fc_depths)):
                fc_depth = fc_depths[fc]
                params = {"filter_size": filter_size, "conv1_depth" : conv1_depth,\
                          "fc_depth": fc_depth}
                params_list.append(params)    
    
    return params_list           


def prevalidate(params_list):
    predictor = PerformancePredictor(DB_LOB_PATH)
    if predictor.load() is False:
        print("Unable to load training log")
        return params_list
    else:        
        validated_list = []
        for i in range(len(params_list)):
            params = params_list[i]
            #print ("Show your content: " + str(params))
            result = predictor.validate(Param1=params["filter_size"], Param2=params["conv1_depth"],\
                                      Param3=params["conv2_depth"], Param4=params["fc_depth"])
            if result is True:
                validated_list.append(params)
            '''
            else:
                print(str(params) + " is not appropriate to learn. Therefore, skip it")
            '''
        return validated_list        

# TODO: produce hyperparameter vectos by cartesian product
class HyConfigurator:
    def __init__(self, path):
        self.path = path
        self.setting = ""
        
    def __del__(self):
        self.delete()