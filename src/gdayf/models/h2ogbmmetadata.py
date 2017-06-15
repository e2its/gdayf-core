from gdayf.models.h2omodelmetadata import H2OModelMetadata

##  Define GBM Model for H2OFramework

## Generate H2O Model base Class and initialize base members
# Load default parameters for Models H2OGbmMetadata
# Initialized model to default parameters for Model H2OGbmMetadata

class H2OGbmMetadata (H2OModelMetadata):
    ## Constructor
    def __init__(self):
        H2OModelMetadata.__init__(self)
        for each_model in self._config['models']:
            if each_model['model'] == 'H2OGradientBoostingEstimator':
                self.model = each_model
        self._config = self.get_copy_model()
