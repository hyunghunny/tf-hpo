#!/bin/bash
nohup HPOlib-run -o ~/tf-hpolib/HPOlib/optimizers/smac/smac -s 23
nohup HPOlib-run -o ~/tf-hpolib/HPOlib/optimizers/tpe/h -s 23
nohup HPOlib-run -o ~/tf-hpolib/HPOlib/optimizers/spearmint/spearmint_april2013_mod -s 23
HPOlib-plot SMAC smac_2_06_01-dev_23_*/smac_*.pkl TPE hyperopt_august2013_mod_23_*/hyp*.pkl SPEARMINT spearmint_april2013_mod_23_*/spear*.pkl -s `pwd`/plots/
HPOlib-plot LeNet5 smac_2_06_01-dev_23_*/smac_*.pkl hyperopt_august2013_mod_23_*/hyp*.pkl spearmint_april2013_mod_23_*/spear*.pkl -s `pwd`/plots/
