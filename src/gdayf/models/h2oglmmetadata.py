from gdayf.models.h2omodelmetadata import H2OModelMetadata

##  Define GLM Model for H2OFramework

## Generate H2O Model base Class and initialize base members
# Load default parameters for Models H2OGlmMetadata
# Initialized model to default parameters for Model H2OGlmMetadata

class H2OGlmMetadata (H2OModelMetadata):

    ## Constructor
    def __init__(self):
        H2OModelMetadata.__init__(self)
        for each_model in self._config['models']:
            if each_model['model'] == 'H2OGeneralizedLinearEstimator':
                self.model = each_model
        self._config = self.get_copy_model()


if __name__ == "__main__":
    m = H2OGlmMetadata()
    print (m.get_model())