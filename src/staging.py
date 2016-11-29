import sys
import time
import traceback

import models.mnist_data as mnist
from models.cnn import LeNet5
from modules.hpvconf import HPVGenerator
from modules.hpmgr import HPVManager
from modules.trainmgr import TrainingManager
import numpy
import numpy.random as nr
import random
import os
import pickle

import pandas as pd
import numpy as np

INDEX = 2

def generate_ini_file(template, hpv):
    generator = HPVGenerator()
    generator.setTemplate(template)
    hpv_file = generator.generate(hpv, output_dir='.config/')
    return hpv_file

def save_as_pickle(hpv_list, pickle_file):
    try:
        with open(pickle_file, 'wb') as f:
            pickle.dump(hpv_list, f)
            print("Backup " + str(len(hpv_list)) + " HPVs to " + pickle_file)
    except:
        e = sys.exc_info()
        traceback.print_exc()
        print(e)

def restore_hpy_list(pickle_file):
    try:
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as f:
                hpv_list = pickle.load(f)
                print("restore " + str(len(hpv_list)) + " HPVs at " + pickle_file)
    except:
        e = sys.exc_info()            
        traceback.print_exc()
        hpv_list = []

    finally:
        return hpv_list


def random_generation(num_epoch):
    ''' randomly generated for same ranges which were defined in spearmint configuration. '''
    hyperparams = {"NUM_EPOCHS" : num_epoch} # coarse optimization condition (given by domain) 
    hyperparams["FILTER_SIZE"] = nr.randint(1, 14, 1)[0]
    hyperparams["CONV1_DEPTH"] = nr.randint(1, 512, 1)[0]
    hyperparams["CONV2_DEPTH"] = nr.randint(1, 512, 1)[0]
    hyperparams["FC1_WIDTH"] = nr.randint(1, 1024, 1)[0]
    hyperparams["BASE_LEARNING_RATE"] = random.uniform(0.0001, 0.1)
    hyperparams["DROPOUT_RATE"] = random.uniform(0.1, 1)
    hyperparams["REGULARIZER_FACTOR"] = nr.random(1)[0]
    return hyperparams 

def runAll(hpv_file_list, log_path):
    try:        
        train_manager = TrainingManager(LeNet5(mnist.import_dataset()), log_path)
        train_manager.setTrainingDevices('gpu', 4)
        hyperparams = random_generation(0.1)
        train_manager.setLoggingParams([ key for key in hyperparams])
        train_manager.setPickleDir('temp/')
        train_manager.setIniDir('.config/')
        train_manager.runAll(hpv_file_list, num_processes=3)
           
    except:
        e = sys.exc_info()
        print("Error: " + str(e))
        traceback.print_exc()     
        
def run_opt(index, num_coarses, num_coarse_epoch, select_rate, num_fine_epoch):    
    PICKLE_COARSE = "coarse"+ str(num_coarse_epoch) + "-" + str(index) + ".pickle"
    
    COARSE_LOG_FILE = "../log/random-coarse" + str(index) + "-" + str(num_coarse_epoch) + "epochs.csv"
    FINE_LOG_FILE = "../log/random-fine" + str(index) + "-" + str(num_fine_epoch) + "epochs.csv"

    TOP_RATE = select_rate

    hyperparams_list = []
    for i in range(num_coarses):
        generated = random_generation(0.1)
        hyperparams_list.append(generated)
    
    #for params in hyperparams_list:
    #    print str(params)

    save_as_pickle(hyperparams_list, PICKLE_COARSE)
    hpv_list = restore_hpy_list(PICKLE_COARSE)
    
    #for params in hpv_list:
    #    print str(params)

    hpv_file_list = []
    for hpv in hpv_list:
        hpv_file = generate_ini_file('CNN_HPV.ini', hpv)
        hpv_file_list.append(hpv_file)

    #for hpv_file in hpv_file_list:
    #    print hpv_file
    
    print ("run coarse optimization with " + str(num_coarse_epoch))
    runAll(hpv_file_list, COARSE_LOG_FILE)

    # We will conduct wangling from now
    coarse_table = pd.read_csv(COARSE_LOG_FILE, header=0)
    #print (coarse_table.describe())

    # sorting by test error
    total_table = coarse_table[coarse_table["Measure Type"] == 'total']
    num_total = len(total_table)
    acc_sorted_table = total_table.sort_values(['Test Error']).reset_index(drop=True)
    
    num_selected = int(num_total * select_rate)
    selected_table = acc_sorted_table[:num_selected]
    print (selected_table.describe())

    # sorting by elapsed time
    time_sorted_table = selected_table.sort_values(['Elapsed Time']).reset_index(drop=True)

    # create 20 epoch HPVs
    hpv_file_list = []
    for i in range(num_selected):
        template_file = time_sorted_table.loc[i, "Setting"]
        hpv_file = generate_ini_file(template_file, {"NUM_EPOCHS" : 20})
        print (hpv_file)
        hpv_file_list.append(hpv_file)

    # fine optimization
    print("run fine optimization with " + str(num_fine_epoch))
    runAll(hpv_file_list, FINE_LOG_FILE)

    
if __name__ == '__main__':    
    for i in range(3):
        run_opt(INDEX, 400, 0.1, 0.2, 20)
        INDEX += 1
