#!/usr/bin/python
"""Module Utilities for TF-HPO

Author: Hyunghun Cho
"""
import traceback
import math
from modules.hpmgr import HPVManager
import itertools

class HPVGenerator:
    def __init__(self):
        self.template_file = "CNN_HPV.ini"
    
    def setTemplate(self, template_file):
        self.template_file = template_file
        
    def generate(self, param_dict, prefix="HPV_", output_dir="./config/"):        
        ''' produce a hyperparameter vector with given param_dict '''        
        try:
            hpv = HPVManager(self.template_file, output_dir)
            for key in param_dict:
                value = param_dict[key]
                hpv.setOption('Hyperparameters', key, value)
            new_hpv_file = hpv.saveAs(prefix=prefix)            
            
        except:
            traceback.print_exc()
           
        return new_hpv_file

    
class HPVGridGenerator(HPVGenerator):
    def __init__(self, config, cfg_dir = "./"):
        self.cfg = config 
        self.hpv_file_list = []
 
    
    def generate(self, max_count=100, output_dir="./config/"):
        '''produce hyperparameter vectors by cartesian product'''
        hp_values_dict = {}
        
        try:
            for key in self.cfg.hyperparameters:
                hp = (self.cfg.hyperparameters[key])
                hp_values_dict[key] = self.parse(hp, max_count)
        except:
            print "Invalid configuration"
            traceback.print_exc()
        #print hp_values_dict
        
        hpv_list = []
        hp_values_lists = [hp_values_dict[key] for key in hp_values_dict]        
        for element in itertools.product(*hp_values_lists):
            hpv_list.append(element)
        #print hpv_list        
        
        hpv = HPVManager(self.template_file, output_dir)
        for hpv_tuple in hpv_list:
            i = 0
            for key in hp_values_dict:
                value = hpv_tuple[i]
                hpv.setOption('Hyperparameters', key, value)
                i += 1
            
            new_hpv_file = hpv.saveAs(prefix='Grid_')
            self.hpv_file_list.append(new_hpv_file)

        return self.hpv_file_list
    
    def getHPVlist(self):
        return self.hpv_file_list

    def parse(self, hp, count):
        values = []
        boundary = []
        spans = [1 for _ in range(count)] # default spans
        if hasattr(hp,"range"):
            boundary = hp.range
        elif hasattr(hp,"border"):
            boundary = hp.border

        boundary = sorted(boundary)
        lower = boundary[0]
        upper = boundary[1]

        if hasattr(hp,"variation"):
            if hp.variation.type is 'fixed' :
                spans = [hp.variation.span for _ in range(count)]
            elif hp.variation.type is 'exponentiation':
                base = hp.variation.span
                spans = [pow(hp.variation.span, i) for i in range(count)]

        item = lower
        i = 0
        for i in range(len(spans)):               
            if item <= upper:            
                values.append(item)
            else:
                break
            item = item + spans[i] 

        return values

