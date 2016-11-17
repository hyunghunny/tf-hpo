import math
import pandas as pd
import numpy as np
from scipy.stats import linregress

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

    def check(self, params):
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
        
   