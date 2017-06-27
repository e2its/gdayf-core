## @package gdayf.core.adviserastar
# Define all objects, functions and structured related to manage and execute Smart analysis based on A* Algorithm
# and defined heuristic
# Main class AdviserAStarAvg. Lets us execute analysis, make recommendations over optimizing on selected algoritms

from gdayf.models.frameworkmetadata import FrameworkMetadata
from gdayf.common.armetadata import ArMetadata
from copy import deepcopy
from gdayf.common.dfmetada import DFMetada
from gdayf.common.normalizationset import NormalizationSet
from gdayf.models.atypesmetadata import ATypesMetadata
from gdayf.models.h2oframeworkmetadata import H2OFrameworkMetadata
from gdayf.models.h2omodelmetadata import H2OModelMetadata
from gdayf.common.utils import decode_json_to_dataframe
from gdayf.common.utils import dtypes
from gdayf.common.utils import compare_dict
from collections import OrderedDict
from time import time
from json import dumps

## Class focused on execute A* based analysis on three modalities of working
# Fast: 1 level analysis over default parameters
# Normal: One A* analysis for all models based until max_deep with early_stopping
# Paranoiac: One A* algorithm per model analysis until max_deep without early stoping
class AdviserAStar(object):
    NORMAL = 1
    FAST = 0
    PARANOIAC = 2
    POC = 3
    deepness = 0

    ## Constructor
    # @param self object pointer
    # @param analysis_id main id traceability code
    # @param deep_impact A* max_deep
    # @param metric metrict for priorizing models ['accuracy', 'rmse'] on train
    def __init__(self, analysis_id, deep_impact=5, metric='accuracy'):
        self.analysis_id = analysis_id
        self.timestamp = time()
        self.an_objective = None
        self.deep_impact = deep_impact
        self.analysis_recommendation_order = list()
        self.analyzed_models = list()
        self.metric = metric

    ## Main method oriented to execute smart analysis
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param atype [NORMAL, FAST, PARANOIAC]
    # @param objective_column string indicating objective column
    # @param an_objective atype[type]
    # @return ArMetadata()'s Prioritized queue
    def set_recommendations(self, dataframe_metadata, objective_column, atype=NORMAL):
        self.an_objective = self.get_analysis_objective(dataframe_metadata, objective_column=objective_column)
        if atype == self.POC:
            return self.analysispoc(dataframe_metadata, objective_column)
        if atype == self.FAST:
            return self.analysisfast(dataframe_metadata, objective_column)
        elif atype == self.NORMAL:
            return self.analysisnormal(dataframe_metadata, objective_column)
        elif atype == self.PARANOIAC:
            return self.analysisparanoiac(dataframe_metadata, objective_column)

    ## Method oriented to execute smart fast analysis
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param objective_column string indicating objective column
    # @return analysis_id, Ordered[(algorithm_metadata.json, normalizations_sets.json)]
    def analysisfast(self, dataframe_metadata, objective_column):
        if self.deepness == 0:
            ar_structure = ArMetadata()
            ar_structure['model_id'] = self.analysis_id
            ar_structure['version'] = '0.0.1'
            ar_structure['objective_column'] = objective_column
            ar_structure['timestamp'] = self.timestamp
            ar_structure['normalizations_set'] = None
            ar_structure['data_initial'] = dataframe_metadata
            ar_structure['data_normalized'] = None
            ar_structure['model_parameters'] = self.get_candidate_models(self.an_objective, self.FAST)
            ar_structure['ignored_parameters'] = None
            ar_structure['full_parameters_stack'] = None
            ar_structure['status'] = -1

            self.deepness += 1
            return self.analysis_id, [(ar_structure, None)]
        else:
            return self.analysis_id, (None, None)

    ## Method oriented to execute smart PoC analysis
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param objective_column string indicating objective column
    # @return analysis_id, Ordered[(algorithm_metadata.json, normalizations_sets.json)]
    def analysispoc(self, dataframe_metadata, objective_column):
        if self.deepness == 0:
            ar_structure = ArMetadata()
            ar_structure['model_id'] = self.analysis_id
            ar_structure['version'] = '0.0.1'
            ar_structure['objective_column'] = objective_column
            ar_structure['timestamp'] = self.timestamp
            ar_structure['normalizations_set'] = None
            ar_structure['data_initial'] = dataframe_metadata
            ar_structure['data_normalized'] = None
            ar_structure['model_parameters'] = self.get_candidate_models(self.an_objective, self.POC)
            ar_structure['ignored_parameters'] = None
            ar_structure['full_parameters_stack'] = None
            ar_structure['status'] = -1

            self.deepness += 1
            return self.analysis_id, [(ar_structure, None)]
        else:
            return self.analysis_id, (None, None)

    ## Method oriented to execute smart normal analysis
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param objective_column string indicating objective column
    # @return analysis_id, Ordered[(algorithm_metadata.json, normalizations_sets.json)]
    def analysisnormal(self, dataframe_metadata, objective_column):
        if self.deepness == 0:
            ar_structure = ArMetadata()
            ar_structure['model_id'] = self.analysis_id
            ar_structure['version'] = '0.0.1'
            ar_structure['objective_column'] = objective_column
            ar_structure['timestamp'] = self.timestamp
            ar_structure['normalizations_set'] = None
            ar_structure['data_initial'] = dataframe_metadata
            ar_structure['data_normalized'] = None
            ar_structure['model_parameters'] = self.get_candidate_models(self.an_objective, self.NORMAL)
            ar_structure['ignored_parameters'] = None
            ar_structure['full_parameters_stack'] = None
            ar_structure['status'] = -1

            self.analyzed_models.append((ar_structure, None))
            self.deepness += 1
        elif self.deepness > self.deep_impact:
            self.analyzed_models = None
        else:
            pass
            self.deepness += 1
        print(type(self.analyzed_models))
        return self.analysis_id, self.analyzed_models

    ## Method oriented to execute smart paranoiac analysis
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param objective_column string indicating objective column
    # @return analysis_id, Ordered[(algorithm_metadata.json, normalizations_sets.json)]
    def analysisparanoiac(self, dataframe_metadata, objective_column):
        if self.deepness == 0:
            ar_structure = ArMetadata()
            ar_structure['model_id'] = self.analysis_id
            ar_structure['version'] = '0.0.1'
            ar_structure['objective_column'] = objective_column
            ar_structure['timestamp'] = self.timestamp
            ar_structure['normalizations_set'] = None
            ar_structure['data_initial'] = dataframe_metadata
            ar_structure['data_normalized'] = None
            ar_structure['model_parameters'] = self.get_candidate_models(self.an_objective, self.PARANOIAC)
            ar_structure['ignored_parameters'] = None
            ar_structure['full_parameters_stack'] = None
            ar_structure['status'] = -1

            self.analyzed_models.append((ar_structure, None))
            self.deepness += 1
        elif (self.deepness - 1) == self.deep_impact:
            pass
            self.deepness += 1
        elif self.deepness == self.deep_impact:
            self.analyzed_models = None
        else:
            pass
            self.deepness += 1
        print(type(self.analyzed_models))
        return self.analysis_id, self.analyzed_models

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
    # @param amode Analysismode
    # @return FrameworkMetadata()
    def get_candidate_models(self, atype, amode):
        defaultframeworks = self.load_frameworks()
        ar_models = OrderedDict()
        for fw, parameters in defaultframeworks.items():
            if fw == 'h2o':
                wfw = H2OFrameworkMetadata(defaultframeworks)
                for each_model in wfw.get_default():
                    modelbase = H2OModelMetadata()
                    model = modelbase.generate_models(each_model['model'], atype, amode)
                    wfw.append(model)
                    # individualize models on partial structure and generate vectors for analyzed_models list population
                    auxiliary_structure_for_vectors = OrderedDict()
                    auxiliary_structure_for_vectors['model_parameters'] = OrderedDict()
                    auxiliary_structure_for_vectors['model_parameters'][fw] = list()
                    auxiliary_structure_for_vectors['model_parameters'][fw].append(deepcopy(model))
                    auxiliary_structure_for_vectors['normalizations_set'] = None
                    vector = self.generate_vectors(model=auxiliary_structure_for_vectors,
                                                   normalization_set= \
                                                       auxiliary_structure_for_vectors['normalizations_set'])
                    self.analyzed_models.append(vector)
                ar_models['h2o'] = wfw.get_models()
        return ar_models

    ##Method get accuracy for generic model
    @staticmethod
    def get_accuracy(model):
        try:
            print('Accuracy:' + str(model['metrics']['accuracy']))
            return float(model['metrics']['accuracy'])
        except KeyError:
            return 0.0

    ##Method get rmse for generic model
    @staticmethod
    def get_rmse(model):
        try:
            print('RMSE:' + str(model['metrics']['execution']['train']['RMSE']))
            return float(model['metrics']['execution']['train']['RMSE'])
        except KeyError:
            return 0.0

    ## Method managing scoring algorithm results
    # params: results for Handlers (gdayf.handlers)
    # @param analysis_id
    # @param list for models analized
    # @return (fw,model_list) (ArMetadata, normalization_set)
    def priorize_models(self, analysis_id, model_list):
        if self.metric == 'accuracy':
            return sorted(model_list, key=self.get_accuracy, reverse=True)
        elif self.metric == 'rmse':
            return sorted(model_list, key=self.get_rmse)
        else:
            return model_list

    ## Store executed model base parameters to check past executions
    # @param model - ArMetadata to be stored as executed
    # @param normalization set
    # @return model_vector (fw, vector, normalizaton_set)
    def generate_vectors(self, model, normalization_set):
        vector = list()
        if 'h2o' in model['model_parameters'].keys():
            for parm, parm_value in model['model_parameters']['h2o'][0]['parameters'].items():
                vector.append(parm_value['value'])
            return 'h2o', model['model_parameters']['h2o'][0]['model'], vector, normalization_set
        else:
            return None

    ## Check if model has benn executed or is planned to execute
    # @param vector - model vector
    # @return True if executed False in other case
    def is_executed(self, vector):
        aux_analized_models = deepcopy(self.analyzed_models)
        analyzed = False
        while not analyzed and len(aux_analized_models) > 0:
            analyzed = analyzed or self.compare_vectors(vector, aux_analized_models.pop())
        return analyzed

    ## Compare to execution vectors
    # @param vector1 - model_execution vector
    # @param vector2 - model_execution vector
    # @return True if equal False if inequal
    @ staticmethod
    def compare_vectors(vector1, vector2):
        #print(vector1)
        print(vector2)
        return vector1[0] == vector2[0] and vector1[1] == vector2[1] \
               and vector1[2] == vector2[2] and compare_dict(vector1[3], vector2[3])

    ## Check if model is previously executed. If it not append to list
    # @param model_list
    # @param  model json compatible
    def safe_append(self, model_list, model):
        vector = self.generate_vectors(model, model['normalizations_set'])
        print(vector)
        if not self.is_executed(vector):
            model_list.append(model)
            self.analyzed_models.append(vector)

    ## Method manging generation of possible optimized models
    # params: results for Handlers (gdayf.handlers)
    # @param fw framework
    # @param model ArMetadata
    # @return list of possible optimized models to execute return None if nothing to do
    def optimize_models(self, model):
        model_metric = decode_json_to_dataframe(['metrics']['model'])
        model_list = list()

        if model['model_parameters'].has_key['h2o']:
            if model['model_parameters']['h2o'][0]['model'] == 'H2OGradientBoostingEstimator':
                if (self.deepness + 1 == self.deep_impact) and model['type'] == 'regression':
                    for distribution in model['model_parameters']['h2o'][0]['parameters']['max_depth']['type']:
                        model_aux = deepcopy(model['model_parameters']['h2o'][0])
                        model_aux['model_parameters']['h2o'][0]['parameters']['distribution']['value'] = distribution
                        self.safe_append(model_list, model_aux)
                if model_metric['number_of_trees'] == model['model_parameters']['h2o'][0]\
                                                           ['parameters']['ntrees']['value']:
                    model_aux = deepcopy(model)
                    model_aux['model_parameters']['h2o'][0]['parameters']['ntrees']['value'] *= 2
                    if model_metric['number_of_trees']['max_depth'] == \
                       model['model_parameters']['h2o'][0]['parameters']['max_depth']['value']:
                        model_aux['model_parameters']['h2o'][0]['parameters']['max_depth']['value'] *= 2
                    self.safe_append(model_list, model_aux)
                if model['model_parameters']['h2o'][0]['parameters']['c']['value'] < 5:
                    model_aux = deepcopy(model['model_parameters']['h2o'][0])
                    model_aux['model_parameters']['h2o'][0]['parameters']['nfolds']['value'] += 2
                    self.safe_append(model_list, model_aux)
                if model['model_parameters']['h2o'][0]['parameters']['min_rows']['value'] < 9:
                    model_aux = deepcopy(model['model_parameters']['h2o'][0])
                    model_aux['model_parameters']['h2o'][0]['parameters']['min_rows']['value'] += 2
                    self.safe_append(model_list, model_aux)
                if len(model_list) == 0:
                    return 'h2o', None
                else:
                    return 'h2o', model_list
            elif model['model_parameters']['h2o'][0]['model'] == 'H2OGeneralizedLinearEstimator':
                pass
            elif model['model_parameters']['h2o'][0]['model'] == 'H2ODeepLearningEstimator':
                pass
            elif model['model_parameters']['h2o'][0]['model'] == 'H2ORandomForestEstimator':
                pass
        else:
            return None


if __name__ == '__main__':
    from gdayf.handlers.inputhandler import inputHandlerCSV
    from pandas import concat
    source_data = list()
    source_data.append("D:/Dropbox/DayF/Technology/Python-DayF-adaptation-path/")
    source_data.append("Oreilly.Practical.Machine.Learning.with.H2O.149196460X/")
    source_data.append("CODE/h2o-bk/datasets/")

    pd_train_dataset = concat([inputHandlerCSV().inputCSV(''.join(source_data) + "football.train2.csv"),
                               inputHandlerCSV().inputCSV(''.join(source_data) + "football.valid2.csv")],
                              axis=0)

    m = DFMetada()
    adv = AdviserAStar('football_csv', metric='accuracy')

    df = m.getDataFrameMetadata(pd_train_dataset, 'pandas')
    ana, lista = adv.set_recommendations(dataframe_metadata=df,
                                         objective_column='HomeWin',
                                         atype=adv.FAST
                                         )
    print(adv.analyzed_models)


