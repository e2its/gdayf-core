## @package gdayf.common.constants

'''
 * This file is part of the gDayF AutoML Core Framework project
 * distribution (https://github.com/e2its/gdayf-core).
 * Copyright (c) 2016-2019 Jose Luis Sanchez del Coso <e2its.es@gmail.com>.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 ** Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2019
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