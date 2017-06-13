from gdayf.conf.loadconfig import LoadConfig
from gdayf.models.h2omodelmetadata import H2OModelMetadata


class H2ODrfMetadata (H2OModelMetadata):
    def __init__(self):
        super(H2ODrfMetadata, self).__init__()
        self._config = LoadConfig().get_config()
        for each_model in self._config['frameworks']['h2o']['models']:
            if each_model['model'] == 'H2ORandomForestEstimator':
                self.model = each_model
