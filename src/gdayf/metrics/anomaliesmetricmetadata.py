## @package gdayf.metrics.anomaliesmetricmetadata
# Define Regression Metric object as OrderedDict() of common measures for all frameworks
#  on an unified way

from gdayf.metrics.regressionmetricmetadata import RegressionMetricMetadata

# Class Base for Regression metrics as OrderedDict
#
# Base Metrics for Anomalies-Regression
# [No expanded metrics]
class AnomaliesMetricMetadata(RegressionMetricMetadata):
    ## Method constructor
    # @param self object pointer
    def __init__(self):
        RegressionMetricMetadata.__init__(self)


