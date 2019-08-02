## @package gdayf.common.constants

'''
Copyright (C) e2its - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 *
 * This file is part of gDayF project.
 *
 * Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2019
'''

POC = 0
FAST = 1
NORMAL = 2
FAST_PARANOIAC = 3
PARANOIAC = 4
ANOMALIES = 5
CLUSTERING = 6
BEST = 0
BEST_3 = 1
EACH_BEST = 2
ALL = 3
NONE = -1
DTYPES = ['int', 'float', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
FTYPES = ['float', 'float16', 'float32', 'float64']
ITYPES = ['int', 'int16', 'int32', 'int64']
METRICS_TYPES = ['train_accuracy', 'test_accuracy', 'combined_accuracy', 'train_rmse', 'test_rmse', 'cdistance',
                 'train_r2', 'test_r2']
ACCURACY_METRICS = ['train_accuracy', 'test_accuracy', 'combined_accuracy']
REGRESSION_METRICS = ['train_rmse', 'test_rmse', 'train_r2', 'test_r2']
CLUSTERING_METRICS = ['cdistance']
NO_STANDARDIZE = ['']
#NO_STANDARDIZE = ['H2ORandomForestEstimator', 'H2OGradientBoostingEstimator', '']
cdistance= 'cdistance'
train_accuracy = 'train_accuracy'
test_accuracy = 'test_accuracy'
combined_accuracy = 'combined_accuracy'
train_rmse = 'train_rmse'
test_rmse = 'test_rmse'
train_r2 = 'train_rmse'
test_r2 = 'test_rmse'
atypes = ['binomial', 'multinomial', 'regression']