## @package gdayf.models.h2omodelmetadata
# Define Base Model for H2OFramework
#  on an unified way. Base for all Models

from gdayf.models.modelmetadata import ModelMetadata
from collections import OrderedDict


## Generate H2O Model base Class and initialize base members
class H2OModelMetadata(ModelMetadata):
    ## Constructor
    def __init__(self):
        ModelMetadata.__init__(self)
        # @var _config
        # Initialized _config to h2o all models default values
        self._config = self._config['h2o']['models']

    ## Generate H2O models
    # This method is used to load config parameters adapting its to specific analysis requirements
    # @param model_type catalogued H2O model
    # @param atype AtypeMetadata
    # @param amode Analysis mode FAST=0 NORMAL=1 PARANOIAC= 2. Define early_stopping parameters inclusion
    # @return H2O model json compatible (OrderedDict())
    def generate_models(self, model_type, atype, amode=1):
        if atype[0]['type'] == 'binomial':
            distribution = 'binomial'
        elif atype[0]['type'] == 'multinomial':
            distribution = 'multinomial'
        else:
            distribution = 'default'

        for each_model in self._config:
            if each_model['model'] == model_type:
                for key, value in each_model.items():
                    if key == 'parameters':
                        self.model[key] = OrderedDict()
                        for subkey, subvalue in value.items():
                            if subkey not in ['stopping', 'distribution']:
                                for parm, parm_value in subvalue.items():
                                    if parm_value['seleccionable']:
                                        self.model[key][parm] = parm_value

                            elif subkey == 'stopping':
                                if amode in [0, 3]: #[FAST, POC]
                                    for parm, parm_value in subvalue.items():
                                        if parm_value['seleccionable']:
                                            self.model[key][parm] = parm_value

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
        return self.model


if __name__ == "__main__":
    from json import dumps
    m = H2OModelMetadata()
    models = ['H2ODeepLearningEstimator', 'H2OGradientBoostingEstimator',
              'H2OGeneralizedLinearEstimator', 'H2ORandomForestEstimator']
    amodes = [2, 3]
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

