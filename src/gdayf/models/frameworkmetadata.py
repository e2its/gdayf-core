## @package gdayf.models.frameworkmetadata
# Define Base Framework methods and members
#  on an unified way.

from collections import OrderedDict
from gdayf.conf.loadconfig import LoadConfig


## Generate Framework base Class and base members
class FrameworkMetadata (OrderedDict):
    ## Constructor
    def __init__(self):
        # @var _config
        # Load default parameters for Models
        self._config = LoadConfig().get_config()['frameworks']

if __name__ == "__main__":
    m = FrameworkMetadata()
    print (m._config)