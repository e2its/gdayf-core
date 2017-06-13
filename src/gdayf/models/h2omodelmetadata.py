from gdayf.conf.loadconfig import LoadConfig
import copy


class H2OModelMetadata(object):
    def __init__(self):
        self._config = None
        self.model = None

    def get_model(self):
        return self.model

    def get_copy_model(self):
        return copy.deepcopy(self.model)
