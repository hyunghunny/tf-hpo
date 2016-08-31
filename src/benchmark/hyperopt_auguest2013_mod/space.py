from hyperopt import hp

space = {'filter_size': hp.quniform('filter_size', 1, 14, 1),
         'conv1_depth': hp.quniform('conv1_depth', 1, 512, 2),
         'conv1_depth': hp.quniform('conv2_depth', 1, 512, 2),
         'fc_depth': hp.quniform('fc_depth', 1, 512, 1)}
