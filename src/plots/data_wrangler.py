# import required libraries to munge data
import pandas as pd
import numpy as np
import os
from math import log

from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

#TODO:grand refactoring required!

def log_scale(input_list, base=2):
    """apply log2 function to all list item"""
    for i in range(len(input_list)):
        input_list[i] = log(list[i], base)
    return input_list

def select_case(table, kv):
    subset_table = table
    for k in kv:
        subset_table = subset_table[subset_table[k] == kv[k]]
       
    return (subset_table)  

def create_subset_table(grid_table, measureType):
    
    subset_table = grid_table[grid_table["Measure Type"] == measureType]
    
    subset_table = pd.DataFrame(subset_table[["Step", "Epoch", "filter_size", "conv1_depth", "conv2_depth", "fc1_width", "Elapsed Time", "Test Error"]])
    subset_table = subset_table.sort_values(["Step", "filter_size", "conv1_depth", "conv2_depth", "fc1_width"], \
                                                    ascending=[True, True, True, True, True])
    subset_table = subset_table.reset_index(drop=True)

    #subset_table.describe()
    return subset_table

def create_best_error_table(best_list, total_only_table):
    merged_list = []
    for best_condition in best_list:
        selection = {"filter_size": best_condition["filter_size"],\
                     "conv1_depth" : best_condition["conv1_depth"],\
                     "conv2_depth" : best_condition["conv2_depth"],\
                     "fc1_width" : best_condition["fc1_width"]}
        row = select_case(total_only_table, selection)        
        items = row.values.tolist()
        
        if len(items) >= 1:            
            item = items[0]            
            best_condition["Elapsed Time"] = row["Elapsed Time"].values.tolist()[0] 
            if item[5] > best_condition["Best Test Error"]:
                best_condition["Best Test Error"] = row["Test Error"].values.tolist()[0]  
        else:
            print str(selection) + ": " + str(best_condition) # print out missing conditions in total_only_table            
        merged_list.append(best_condition)
    merged_table = pd.DataFrame(merged_list)
    #merged_table.head(10)
    return merged_table

# Extract an one dimension
def get_one_dim(table, kv, selected_dim, output, log_x=True, log_y= True, plot=False):
    subset_table = table
    x_col = selected_dim
    for k in kv:
        subset_table = subset_table[subset_table[k] == kv[k]]
    
    if log_x: # log 2 transformation
        logscale_x = "log2(" + selected_dim + ")"
        subset_table[logscale_x] = log_scale(subset_table[selected_dim].values.tolist())
        x_col = logscale_x

    if log_y:  # log 10 transformation
        logscale_y = "log10(" + output + ")"
        subset_table[logscale_y] = log_scale(subset_table[output].values.tolist(), 10)
        output = logscale_y
        
    if plot:
        subset_table.plot(x=x_col, y=[output], figsize=(8, 8))
        
    return (subset_table)

def create_best_error_list(grid_table):
    # munge table to record best accuracy of hyperparameter vector which measured at each epochs
    epoch_table = grid_table[grid_table["Measure Type"] == "epoch"]

    best_list = []
    filter_sizes = set(grid_table['filter_size'].values.tolist()) 
    conv1_depths = set(grid_table['conv1_depth'].values.tolist()) 
    conv2_depths = set(grid_table['conv2_depth'].values.tolist()) 
    fc_depths = set(grid_table['fc1_width'].values.tolist()) 
    
    for filter_size in filter_sizes:
        for conv1_depth in conv1_depths:
            for conv2_depth in conv2_depths:
                for fc_depth in fc_depths:
                    selection = {"filter_size": filter_size, "conv1_depth" : conv1_depth, "conv2_depth" : conv2_depth, "fc1_width" : fc_depth}
                    case = select_case(epoch_table, selection)
                    if len(case) == 0:
                        print "missing case: " + str(selection)
                    else:
                        best_test_error =  min(case["Test Error"])
                        best_case_row = case[case["Test Error"] == best_test_error]
                        epoch = best_case_row["Epoch"].tail(1).values.tolist()[0]
                        elapsed_time = best_case_row["Elapsed Time"].tail(1).values.tolist()[0]
                        best_case = {"filter_size" : filter_size, "conv1_depth": conv1_depth, \
                                 "conv2_depth" : conv2_depth, "fc1_width" : fc_depth, \
                                     "Best Test Error" : best_test_error,\
                                    "Epoch" : epoch, "Elapsed Time" : elapsed_time}
                        best_list.append(best_case)
    
    return best_list