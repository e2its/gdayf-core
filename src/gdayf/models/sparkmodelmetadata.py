## @package gdayf.models.sparkmodelmetadata
# Define Base Model for sparkFramework
#  on an unified way. Base for all Models

'''
Copyright (C) e2its - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 *
 * This file is part of gDayF project.
 *
 * Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2019
'''

from gdayf.models.modelmetadata import ModelMetadata
from gdayf.common.constants import *
from collections import OrderedDict
from time import time


## Generate spark Model base Class and initialize base members
class sparkModelMetadata(ModelMetadata):
    ## Constructor
    def __init__(self, e_c):
        ModelMetadata.__init__(self, e_c)
        # @var _config
        # Initialized _config to spark all models default values
        self._optimizable_scale_params = self._config['spark']['conf']['optimizable_scale_params']
        self._models = self._config['spark']['models']
    ## Generate spark models
    # This method is used to load config parameters adapting its to specific analysis requirements
    # @param model_type catalogued spark model
    # @param atype AtypeMetadata
    # @param amode Analysis mode. Define early_stopping parameters inclusion
    # @param increment increment x size
    # @return spark model json compatible (OrderedDict())
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
                                    if parm == 'modelType':
                                        if distribution == 'binomial':
                                            distribution = 'bernoulli'
                                        elif distribution == 'multinomial':
                                            distribution = 'multinomial'
                                        if distribution in ['bernoulli', 'multinomial']:
                                            self.model[key][parm]['value'] = distribution
                                            self.model[key][parm]['type'] = list()
                    elif key == 'types':
                        self.model[key] = atype
                    else:
                        self.model[key] = value
        try:
            # Fijamos semilla
            self.model['parameters']['seed']['value'] = int(ts)
        except KeyError:
             pass
        return self.model


if __name__ == "__main__":
    from json import dumps
    m = sparkModelMetadata()
    models = ['sparkDeepLearningEstimator', 'sparkGradientBoostingEstimator',
              'sparkGeneralizedLinearEstimator', 'sparkRandomForestEstimator']
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
                modelbase = sparkModelMetadata()
                print (amode)
                print(dumps(modelbase.generate_models(each_model, atype, amode), indent=4))