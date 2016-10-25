#!/bin/bash
HPO_LIB_DIR=~/tf-hpolib/HPOlib
MODEL_NAME=LeNet-5_EPOCHS_1
OUTPUT_PATH=`pwd`/plots/

if [[ $# != 2 ]]
then
    echo "Usage: ./run_benchmarks.sh {model name} {optimizer name}"
fi

if [[ $2 == "smac" ]]  
then
    echo "Hyperparameter Optimizing $MODEL_NAME with SMAC algorithm..."
    HPOlib-run -o $HPO_LIB_DIR/optimizers/smac/smac -s 23
fi

if [[ $2 == "tpe" ]]  
then
    echo "Hyperparameter Optimizing $MODEL_NAME with Tree Parzen Estimator (TPE) algorithm..."
    HPOlib-run -o $HPO_LIB_DIR/optimizers/tpe/h -s 23
fi

if [[ $2 == "spearmint" ]]  
then
    echo "Hyperparameter Optimizing $MODEL_NAME with Spearmint algorithm..."
    HPOlib-run -o $HPO_LIB_DIR/optimizers/spearmint/spearmint_april2013_mod -s 23
fi

if [[ $2 == "plot" ]]  
then
    echo "Performance benchmarking with above three algorithms..."
    HPOlib-plot SMAC smac_2_06_01-dev_23_*/smac_*.pkl TPE hyperopt_august2013_mod_23_*/hyp*.pkl SPEARMINT spearmint_april2013_mod_23_*/spear*.pkl -s $OUTPUT_PATH
    echo "Performance visualization for $MODEL_NAME..."
    HPOlib-plot $MODEL_NAME smac_2_06_01-dev_23_*/smac_*.pkl hyperopt_august2013_mod_23_*/hyp*.pkl spearmint_april2013_mod_23_*/spear*.pkl -s $OUTPUT_PATH
fi