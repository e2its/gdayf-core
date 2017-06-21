## @package gdayf.models.frameworkmetadata
# Define Base Framework methods and members
#  on an unified way.

from collections import OrderedDict
from gdayf.conf.loadconfig import LoadConfig


## Generate Framework base Class and base members
class FrameworkMetadata (OrderedDict):
    ## Constructor
    def __init__(self):
        # Load default parameters for Models as OrderedDict
        config = LoadConfig().get_config()['frameworks']
        for key, value in config.items():
            self[key] = value


if __name__ == "__main__":
    m = FrameworkMetadata()
    print (m['h2o'])