## @package gdayf.core.adviserastar
# Define all objects, functions and structured related to manage and execute Smart analysis based on A* Algorithm
# and defined heuristic
# Main class AdviserAStarAvg. Lets us execute analysis, make recommendations over optimizing on selected algoritms

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


from gdayf.core.adviserbase import Adviser
from gdayf.common.utils import get_model_fw
import importlib
''' Don't delete used for dynamic optimizer methods'''
from gdayf.conf.loadconfig import LoadConfig
from gdayf.common.utils import decode_ordered_dict_to_dataframe
from gdayf.models.parametersmetadata import ParameterMetadata


## Class focused on execute A* based analysis on three modalities of working
# Fast: 1 level analysis over default parameters
# Normal: One A* analysis for all models based until max_deep with early_stopping
# Paranoiac: One A* algorithm per model analysis until max_deep without early stoping
class AdviserAStar(Adviser):

    ## Constructor
    # @param self object pointer
    # @param e_c context pointer
    # @param deep_impact A* max_deep
    # @param metric metrict for priorizing models ['train_accuracy', 'test_rmse', 'train_rmse', 'test_accuracy', 'combined_accuracy'] on train
    # @param dataframe_name dataframe_name or id
    # @param hash_dataframe MD5 hash value
    def __init__(self, e_c, deep_impact=3, metric='train_accuracy', dataframe_name='', hash_dataframe=''):
        super(AdviserAStar, self).__init__(e_c=e_c, deep_impact=deep_impact, metric=metric,
                                           dataframe_name=dataframe_name, hash_dataframe=hash_dataframe)

    ## Method mangaing the generation of possible optimized models
    # @param armetadata ArMetadata Model
    # @return list of possible optimized models to execute return None if nothing to do
    def optimize_models(self, armetadata):
        metric_value, _, objective = eval('self.get_' + self.metric + '(armetadata)')
        engine = get_model_fw(armetadata)
        for engines in [*self._frameworks]:
            if engine == engines:
                optimizer_engine = importlib.import_module(self._frameworks[engine]['conf']['optimization_method'])
                model_list = optimizer_engine.Optimizer(self._ec).optimize_models(armetadata=armetadata,
                                                                                  metric_value=metric_value,
                                                                                  objective=objective,
                                                                                  deepness=self.deepness,
                                                                                  deep_impact=self.deep_impact)
                optimized_model_list = list()
                for model in model_list:
                    self.safe_append(optimized_model_list, model)

                return (optimized_model_list)


