# import required libraries to munge data
import pandas as pd
import numpy as np

from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# define log function
from math import log

def log_scale(list, base=2):
    """apply log2 function to all list item"""
    for i in range(len(list)):
        list[i] = log(list[i], base)
    return list

# read data from csv
accuracy_table = pd.read_csv("test_accuracy.csv", header=0)
all_logs = pd.read_csv("../../log/all_logs_20160810.csv", header=0) 

def plot_by_fc(iteration, test_data_num, to_log_scale=False):
    """This figure shows the accuracy after iterations when the neurons in conv1, conv2 and fully connected layers varies."""

    tables = get_df_by_iter(iteration, test_data_num)

    # reset plot
    plt.clf() 
    
    
    # set figure size
    fig = plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

    cov1_neurons = [2, 4, 8, 16, 32, 64, 128]
    
    if to_log_scale is True:
        cov1_neurons = log_scale(cov1_neurons)
    
    # create sub plots
    cov1_neurons = [2, 4, 8, 16, 32, 64, 128]
    ax = []
    for i in range(len(cov1_neurons)) :
        subplot = fig.add_subplot(2, 4, i+1)
        ax.append(subplot)
    
    # show all accuacy change trends figure
    markers = ["r--", "g-", "b-", "m-"]
    legends = ["fc:128", "fc:256", "fc:512", "fc:1024"]
    
    cov2_neurons = [2, 4, 8, 16, 32, 64, 128, 256, 512]

    if to_log_scale is True:
        cov2_neurons = log_scale(cov2_neurons)

    # number of accuracy observation count for each conv2
    obs_counts = [0, 9, 18, 26, 35, 44, 53, 62] # XXX: dragon lives here
    fully_size = 4

    for i in range(fully_size):
        for j in range(len(cov1_neurons)):
            start_index = obs_counts[j]
            end_index = obs_counts[j+1]
            #exiprint "cov1=" + str(cov1_neurons[j]) + ", " + str(start_index) + ":" + str(end_index) # for debug purpose
            
            # XXX:if conv1 is 8 or log(conv1) is 3, missing condition exists
            if ( int(cov1_neurons[j]) is 8 or int(cov1_neurons[j]) is 3)  :
                
                leaky_cov2_neurons = [2, 4, 8, 16, 32, 64, 256, 512]
                
                if to_log_scale is True:
                    leaky_cov2_neurons = log_scale(leaky_cov2_neurons)
                    
                ax[j].plot(leaky_cov2_neurons, tables[i][start_index:end_index], markers[i], label=legends[i])
            else:
                ax[j].plot(cov2_neurons, tables[i][start_index:end_index], markers[i], label=legends[i])
                
            ax[j].set_title("cov1 size:" + str(cov1_neurons[j]))
            ax[j].set_xlabel("cov2 size")
            ax[j].set_ylim([0.0, 1.0])
    plt.ylabel("test accuracy")
    title = "Accuracy trends at " + str(iteration) + " iterations with " + str(test_data_num) + " test data"
    
    if to_log_scale is True:
        title += " (log scale)"
    
    plt.suptitle(title)
    plt.legend(loc="best")
    plt.show()
    
    return fig

def get_acc():
    return accuracy_table


def get_df_by_iter(iteration, test_data_num):
    """ get accuracy table with a specific condition """
    # followings are observed conditions
    observed_iterations = [6400, 12800, 199680]    
    observed_test_data = [256, 512, 2048, 8092]
    
    # observed fully connected layers
    L3_neurons = [128, 256, 512, 1024] 
    
    if not test_data_num in observed_test_data :
        return [] # return empty array if not observed
    
    if not iteration in observed_iterations :
        return [] # return empty array if not observed
    
    selected_acc_table = accuracy_table[accuracy_table["Iteration"] == iteration]
    
    tables = []

    for j in L3_neurons:
        table = selected_acc_table[selected_acc_table["L3"] == j]
        table = table["Testing Accuracy(" + str(test_data_num) +")"]
        tables.append(table)
    
    return tables


