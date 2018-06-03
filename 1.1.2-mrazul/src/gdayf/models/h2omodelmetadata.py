## @package gdayf.models.h2omodelmetadata
# Define Base Model for H2OFramework
#  on an unified way. Base for all Models

'''
Copyright (C) e2its - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 *
 * This file is part of gDayF project.
 *
 * Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2018
'''

from gdayf.models.modelmetadata import ModelMetadata
from gdayf.common.constants import *
from collections import OrderedDict
from time import time


## Generate H2O Model base Class and initialize base members
class H2OModelMetadata(ModelMetadata):
    ## Constructor
    def __init__(self):
        ModelMetadata.__init__(self)
        # @var _config
        # Initialized _config to h2o all models default values
        self._optimizable_scale_params = self._config['h2o']['conf']['optimizable_scale_params']
        self._models = self._config['h2o']['models']
    ## Generate H2O models
    # This method is used to load config parameters adapting its to specific analysis requirements
    # @param model_type catalogued H2O model
    # @param atype AtypeMetadata
    # @param amode Analysis mode. Define early_stopping parameters inclusion
    # @param increment increment x size
    # @return H2O model json compatible (OrderedDict())
    def generate_models(self, model_type, atype, amode=POC, increment=1):
        if atype[0]['type'] == 'binomial':
            distribution = 'binomial'
        elif atype[0]['type'] == 'multinomial':
            distribution = 'multinomial'
        else:
            distribution = 'default'
        ts = round(time(), 0)
        for each_model in self._models:
            if each_model['model'] == model_type:
                for key, value in each_model.items():
                    if key == 'parameters':
                        self.model[key] = OrderedDict()
                        for subkey, subvalue in value.items():
                            if subkey not in ['stopping', 'distribution', 'effort']:
                                for parm, parm_value in subvalue.items():
                                    if parm_value['seleccionable']:
                                        self.model[key][parm] = parm_value
                            elif subkey == 'stopping':
                                if amode in [POC, FAST, FAST_PARANOIAC]:
                                    for parm, parm_value in subvalue.items():
                                        if parm_value['seleccionable']:
                                            self.model[key][parm] = parm_value
                            elif subkey == 'effort':
                                    for parm, parm_value in subvalue.items():
                                        if parm_value['seleccionable']:
                                            self.model[key][parm] = parm_value
                                            if isinstance(self.model[key][parm]['value'], list)\
                                                    and parm in self._optimizable_scale_params:
                                                for counter in range(0, len(self.model[key][parm]['value'])):
                                                    if model_type != 'H2OAutoEncoderEstimator':
                                                        self.model[key][parm]['value'][counter] = \
                                                            int(self.model[key][parm]['value'][counter] * increment)
                                            elif self.model[key][parm]['type'] in DTYPES \
                                                    and parm in self._optimizable_scale_params:
                                                if self.model[key][parm]['type'] in ITYPES:
                                                    self.model[key][parm]['value'] = \
                                                        int(self.model[key][parm]['value'] * increment)
                                                else:
                                                    self.model[key][parm]['value'] *= increment
                            elif subkey == 'distribution':
                                for parm, parm_value in subvalue.items():
                                    if parm_value['seleccionable']:
                                        self.model[key][parm] = parm_value
                                    if parm == 'family':
                                        if distribution in ['binomial', 'multinomial']:
                                            self.model[key][parm]['value'] = distribution
                                            self.model[key][parm]['type'] = list()

                                    elif parm == 'distribution':
                                        if distribution == 'binomial':
                                            distribution = 'bernoulli'
                                        if distribution in ['bernoulli', 'multinomial']:
                                            self.model[key][parm]['value'] = distribution
                                            self.model[key][parm]['type'] = list()
                    elif key == 'types':
                        self.model[key] = atype
                    else:
                        self.model[key] = value
        # Fijamos semilla
        self.model['parameters']['seed']['value'] = int(ts)
        return self.model


if __name__ == "__main__":
    from json import dumps
    m = H2OModelMetadata()
    models = ['H2ODeepLearningEstimator', 'H2OGradientBoostingEstimator',
              'H2OGeneralizedLinearEstimator', 'H2ORandomForestEstimator']
    amodes = [POC, NORMAL]
    atypes = [
                [
                    {
                      "type": "binomial",
                      "active": True,
                      "valued": "enum"
                    }
                  ],
                [
                    {
                        "type": "multinomial",
                        "active": True,
                        "valued": "enum"
                    }
                ],
                [
                    {
                        "type": "regression",
                        "active": True,
                        "valued": "float64"
                    }
                ]
    ]
    for each_model in models:
        for atype in atypes:
            for amode in amodes:
                modelbase = H2OModelMetadata()
                print (amode)
                print(dumps(modelbase.generate_models(each_model, atype, amode),indent=4))

