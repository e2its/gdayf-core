## @package gdayf.core.adviserastar
# Define all objects, functions and structured related to manage and execute Smart analysis based on A* Algorithm
# and defined heuristic
# Main class AdviserAStarAvg. Lets us execute analysis, make recommendations over optimizing on selected algoritms

from gdayf.common.queues import PriorityQueue
from gdayf.models.frameworkmetadata import FrameworkMetadata
from gdayf.common.armetadata import ArMetadata
from copy import deepcopy
from gdayf.common.dfmetada import DFMetada
from gdayf.common.normalizationset import NormalizationSet
from gdayf.models.atypesmetadata import ATypesMetadata
from gdayf.models.h2oframeworkmetadata import H2OFrameworkMetadata
from gdayf.common.utils import dtypes
from collections import OrderedDict
from time import time
import json

## Class focused on execute A* based analysis on three modalities of working
# Fast: 1 level analysis over default parameters
# Normal: One A* analysis for all models based until max_deep with early_stopping
# Paranoiac: One A* algorithm per model analysis until max_deep without early stoping
class AdviserAStar(object):
    _NORMAL = 1
    _FAST = 0
    _PARANOIAC = 2
    deepness = 0

    ## Constructor
    # @param self object pointer
    # @param analysis_id main id traceability code
    # @param deep_impact A* max_deep
    def __init__(self, analysis_id, deep_impact=5):
        self.analysis_id = analysis_id
        self.deep_impact = deep_impact
        self.analysis_recommendation_order = PriorityQueue()
        self.ar_execution_models = list()

    ## Main method oriented to execute smart analysis
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param atype [NORMAL, FAST, PARANOIAC]
    # @param objective_column string indicating objective column
    # @param analysis_id string indicating primary model or box identification
    # @param an_objective atype[type]
    # @return ArMetadata()'s Prioritized queue
    def set_recommendations(self, dataframe_metadata, objective_column, analysis_id, an_objective, atype=_NORMAL):
        if atype == self._FAST:
            return self.analysisfast(dataframe_metadata, objective_column, analysis_id, an_objective)
        elif atype == self._NORMAL:
            return self.analysisnormal(dataframe_metadata, objective_column, analysis_id, an_objective)
        elif atype == self._PARANOIAC:
            return self.analysisparanoiac(dataframe_metadata, objective_column, analysis_id, an_objective)

    ## Method oriented to execute smart fast analysis
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param objective_column string indicating objective column
    # @param analysis_id string indicating primary model or box identification
    # @param an_objective atype[type]
    # @return analysis_id, Ordered[(algorithm_metadata.json, normalizations_sets.json)]
    def analysisfast(self, dataframe_metadata, objective_column, analysis_id, an_objective):
        ar_structure = ArMetadata()
        ar_structure['model_id'] = analysis_id
        ar_structure['version'] = '0.0.1'
        ar_structure['type'] = None
        ar_structure['objective_column'] = objective_column
        ar_structure['timestamp'] = time()
        ar_structure['normalizations_set'] = None
        ar_structure['data_initial'] = dataframe_metadata
        ar_structure['data_normalized'] = None
        ar_structure['model_parameters'] = self.get_candidate_models(an_objective)
        ar_structure['ignored_parameters'] = None
        ar_structure['full_parameters_stack'] = None
        ar_structure['status'] = -1

        self.ar_execution_models.append((ar_structure, None))
        return analysis_id, self.ar_execution_models
        

    ## Method oriented to execute smart normal analysis
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param objective_column string indicating objective column
    # @param analysis_id string indicating primary model or box identification
    # @param an_objective atype[type]
    # @return analysis_id, Ordered[(algorithm_metadata.json, normalizations_sets.json)]
    def analysisnormal(self, dataframe_metadata, objective_column, analysis_id, an_objective):
        pass

    ## Method oriented to execute smart paranoiac analysis
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param objective_column string indicating objective column
    # @param analysis_id string indicating primary model or box identification
    # @param an_objective atype[type]
    # @return analysis_id, Ordered[(algorithm_metadata.json, normalizations_sets.json)]
    def analysisparanoiac(self, dataframe_metadata, objective_column, analysis_id, an_objective):
        pass

    ## Method oriented to get frameworks default values from config
    # @param self object pointer
    # @return FrameWorkMetadata
    def load_frameworks(self):
        return FrameworkMetadata()

    ## Method oriented to analyze DFmetadata and select analysis objective
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param objective_column string indicating objective column
    # @return ArType or None if objective_column not found
    def get_analysis_objective(self, dataframe_metadata, objective_column):
        for each_column in dataframe_metadata['columns']:
            if each_column['name'] == objective_column:
                if int(each_column['cardinality']) == 2:
                    return ATypesMetadata(binomial=True)
                if each_column['type'] not in dtypes:
                    if int(each_column['cardinality']) > 2:
                        return ATypesMetadata(multinomial=True)
                elif int(each_column['cardinality']) <= round(dataframe_metadata['rowcount']/5000):
                    return ATypesMetadata(multinomial=True)
                else:
                    return ATypesMetadata(regression=True)
        return None

    ## Method oriented to analyze choose models candidate and select analysis objective
    # @param self object pointer
    # @param atype ATypesMetadata
    # @return FrameworkMetadata()
    def get_candidate_models(self, atype):
        defaultframeworks = self.load_frameworks()
        ar_models = OrderedDict()
        for fw, parameters in defaultframeworks.items():
            if fw == 'h2o':
                wfw = H2OFrameworkMetadata(defaultframeworks)
                for each_model in wfw.get_default():

                    for each_type in each_model['types']:
                        if each_type['type'] == atype[0]['type'] and each_type['active']:
                            each_model['types'] = [each_type]
                            wfw.append(each_model)
                    print(json.dumps(wfw.models, indent=4))
        return ar_models


if __name__ == '__main__':
    from gdayf.handlers.inputhandler import inputHandlerCSV
    from pandas import concat
    import operator
    import json
    source_data = list()
    source_data.append("D:/Dropbox/DayF/Technology/Python-DayF-adaptation-path/")
    source_data.append("Oreilly.Practical.Machine.Learning.with.H2O.149196460X/")
    source_data.append("CODE/h2o-bk/datasets/")

    pd_train_dataset = concat([inputHandlerCSV().inputCSV(''.join(source_data) + "football.train2.csv"),
                               inputHandlerCSV().inputCSV(''.join(source_data) + "football.valid2.csv")],
                              axis=0)

    m = DFMetada()
    adv = AdviserAStar('football_csv')

    df = m.getDataFrameMetadata(pd_train_dataset, 'pandas')
    an_objective = adv.get_analysis_objective(df, objective_column='HomeWin')
    print(an_objective)
    ana, lista = adv.set_recommendations(dataframe_metadata=df,
                                  objective_column='HomeWin',
                                  analysis_id='POC-binomial-SOC',
                                  an_objective=an_objective,
                                  atype=adv._FAST
                                  )

    print (json.dumps(lista, indent=4))


