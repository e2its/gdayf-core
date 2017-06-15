## @package gdayf.models.h2odrfmetadata
# Define DRF Model for H2OFramework

from gdayf.models.h2omodelmetadata import H2OModelMetadata


## Generate H2O Model base Class and initialize base members
# Load default parameters for Models H2OGbmMetadata
# Initialized model to default parameters for Model H2ODrfMetadata
class H2ODrfMetadata (H2OModelMetadata):
    ## Constructor
    def __init__(self):
        H2OModelMetadata.__init__(self)
        for each_model in self._config['models']:
            if each_model['model'] == 'H2ORandomForestEstimator':
                self.model = each_model
            self._config = self.get_copy_model()

if __name__ == "__main__":
    m = H2ODrfMetadata()
    print(m.get_model())

