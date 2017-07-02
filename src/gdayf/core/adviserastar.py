## @package gdayf.core.adviserastar
# Define all objects, functions and structured related to manage and execute Smart analysis based on A* Algorithm
# and defined heuristic
# Main class AdviserAStarAvg. Lets us execute analysis, make recommendations over optimizing on selected algoritms

from gdayf.models.frameworkmetadata import FrameworkMetadata
from gdayf.common.armetadata import ArMetadata
from copy import deepcopy
from gdayf.common.dfmetada import DFMetada
from gdayf.common.normalizationset import NormalizationSet
from gdayf.conf.loadconfig import LoadConfig
from gdayf.models.atypesmetadata import ATypesMetadata
from gdayf.models.h2oframeworkmetadata import H2OFrameworkMetadata
from gdayf.models.h2omodelmetadata import H2OModelMetadata
from gdayf.common.utils import decode_json_to_dataframe
from gdayf.common.utils import dtypes
from gdayf.common.utils import compare_dict
from gdayf.common.utils import get_model_fw
from gdayf.common.utils import get_model_ns
from collections import OrderedDict
from time import time
from json import dumps
import pprint

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
    # @param metric metrict for priorizing models ['accuracy', 'rmse', 'test_accuracy', 'combined'] on train
    def __init__(self, analysis_id, deep_impact=2, metric='accuracy'):
        self.analysis_id = analysis_id
        self.timestamp = time()
        self.an_objective = None
        self.deep_impact = deep_impact
        self.analysis_recommendation_order = list()
        self.analyzed_models = list()
        self.excluded_models = list()
        self.next_analysis_list = list()
        self.metric = metric

    ## Main method oriented to execute smart analysis
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param atype [NORMAL, FAST, PARANOIAC]
    # @param objective_column string indicating objective column
    # @return ArMetadata()'s Prioritized queue
    def set_recommendations(self, dataframe_metadata, objective_column, atype=NORMAL):
        self.an_objective = self.get_analysis_objective(dataframe_metadata, objective_column=objective_column)
        if atype == self.POC:
            return self.analysispoc(dataframe_metadata, objective_column)
        if atype == self.FAST:
            return self.analysisnormal(dataframe_metadata, objective_column)
        elif atype == self.NORMAL:
            return self.analysisnormal(dataframe_metadata, objective_column)
        elif atype == self.PARANOIAC:
            return self.analysisparanoiac(dataframe_metadata, objective_column)


    ## Method oriented to execute smart PoC analysis
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param objective_column string indicating objective column
    # @return analysis_id, Ordered[(algorithm_metadata.json, normalizations_sets.json)]
    def analysispoc(self, dataframe_metadata, objective_column):
        self.next_analysis_list.clear()
        if self.deepness == 0:
            fw_model_list = self.get_candidate_models(self.an_objective, self.NORMAL)
            for fw, model_params, norm_sets in fw_model_list:
                ar_structure = ArMetadata()
                ar_structure['model_id'] = self.analysis_id
                ar_structure['version'] = '0.0.1'
                ar_structure['objective_column'] = objective_column
                ar_structure['timestamp'] = self.timestamp
                ar_structure['normalizations_set'] = norm_sets
                ar_structure['data_initial'] = dataframe_metadata
                ar_structure['data_normalized'] = None
                ar_structure['model_parameters'] = OrderedDict()
                ar_structure['model_parameters'][fw] = model_params
                ar_structure['ignored_parameters'] = None
                ar_structure['full_parameters_stack'] = None
                ar_structure['status'] = -1
                self.next_analysis_list.append(ar_structure)
            self.deepness += 1
        else:
            return self.analysis_id, None
        return self.analysis_id, self.next_analysis_list

    ## Method oriented to execute smart normal and fast analysis
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param objective_column string indicating objective column
    # @return analysis_id, Ordered[(algorithm_metadata.json, normalizations_sets.json)]
    def analysisnormal(self, dataframe_metadata, objective_column):
        self.next_analysis_list.clear()
        if self.deepness == 0:
            fw_model_list = self.get_candidate_models(self.an_objective, self.NORMAL)
            for fw, model_params, norm_sets in fw_model_list:
                ar_structure = ArMetadata()
                ar_structure['model_id'] = self.analysis_id
                ar_structure['version'] = '0.0.1'
                ar_structure['objective_column'] = objective_column
                ar_structure['timestamp'] = self.timestamp
                ar_structure['normalizations_set'] = norm_sets
                ar_structure['data_initial'] = dataframe_metadata
                ar_structure['data_normalized'] = None
                ar_structure['model_parameters'] = OrderedDict()
                ar_structure['model_parameters'][fw] = model_params
                ar_structure['ignored_parameters'] = None
                ar_structure['full_parameters_stack'] = None
                ar_structure['status'] = -1
                self.next_analysis_list.append(ar_structure)
            self.deepness += 1
        elif self.deepness > self.deep_impact:
            self.next_analysis_list = None
        elif self.deepness == 1:
            # Get all models
            fw_model_list = list()
            aux_loop_controller = len(self.analysis_recommendation_order)
            for indexer in range(0, aux_loop_controller):
                try:
                    fw_model_list.extend(self.optimize_models(self.analysis_recommendation_order[indexer]))
                except TypeError:
                    pass
            #if fw_model_list is not None:
            self.next_analysis_list.extend(fw_model_list)
            if len(self.next_analysis_list) == 0:
                    self.next_analysis_list = None
            self.deepness += 1
        elif self.next_analysis_list is not None:
            # Get two most potential best models
            fw_model_list = list()
            for indexer in range(0, 2):
                try:
                    fw_model_list.extend(self.optimize_models(self.analysis_recommendation_order[indexer]))
                except TypeError:
                    pass
            #if fw_model_list is not None:
            self.next_analysis_list.extend(fw_model_list)
            if len(self.next_analysis_list) == 0:
                    self.next_analysis_list = None
            self.deepness += 1
        return self.analysis_id, self.next_analysis_list

        ## Method oriented to execute smart normal and fast analysis
        # @param self object pointer
        # @param dataframe_metadata DFMetadata()
        # @param objective_column string indicating objective column
        # @return analysis_id,(framework, Ordered[(algorithm_metadata.json, normalizations_sets.json)])

    def analysisparanoiac(self, dataframe_metadata, objective_column):
        self.next_analysis_list.clear()
        if self.deepness == 0:
            fw_model_list = self.get_candidate_models(self.an_objective, self.NORMAL)
            for fw, model_params, norm_sets in fw_model_list:
                ar_structure = ArMetadata()
                ar_structure['model_id'] = self.analysis_id
                ar_structure['version'] = '0.0.1'
                ar_structure['objective_column'] = objective_column
                ar_structure['timestamp'] = self.timestamp
                ar_structure['normalizations_set'] = norm_sets
                ar_structure['data_initial'] = dataframe_metadata
                ar_structure['data_normalized'] = None
                ar_structure['model_parameters'] = OrderedDict()
                ar_structure['model_parameters'][fw] = model_params
                ar_structure['ignored_parameters'] = None
                ar_structure['full_parameters_stack'] = None
                ar_structure['status'] = -1
                self.next_analysis_list.append(ar_structure)
            self.deepness += 1
        elif self.deepness > self.deep_impact:
            self.next_analysis_list = None
        elif self.next_analysis_list is not None:
            fw_model_list = list()
            aux_loop_controller = len(self.analysis_recommendation_order)
            for indexer in range(0, aux_loop_controller):
                try:
                    fw_model_list.extend(self.optimize_models(self.analysis_recommendation_order[indexer]))
                except TypeError:
                    pass
            #if fw_model_list is not None:
            self.next_analysis_list.extend(fw_model_list)
            if len(self.next_analysis_list) == 0:
                    self.next_analysis_list = None
            self.deepness += 1
        return self.analysis_id, self.next_analysis_list

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
        model_list = list()
        for fw in defaultframeworks.keys():
            if fw == 'h2o':
                wfw = H2OFrameworkMetadata(defaultframeworks)
                for each_model in wfw.get_default():
                    modelbase = H2OModelMetadata()
                    model = modelbase.generate_models(each_model['model'], atype, amode)
                    wfw.models.append(model)
                    model_list.append((fw, model, None))
        return model_list

    ##Method get train accuracy for generic model
    # @param model
    # @return accuracy metric or None if not exists
    @staticmethod
    def get_accuracy(model):
        try:
            print('Train Accuracy:' + str(model['metrics']['accuracy']['train']))
            return float(model['metrics']['accuracy']['train'])
        except KeyError:
            return 0.0

    ##Method get test accuracy for generic model
    # @param model
    # @return accuracy metric or None if not exists
    @staticmethod
    def get_test_accuracy(model):
        try:
            print('Test Accuracy:' + str(model['metrics']['accuracy']['test']))
            return float(model['metrics']['accuracy']['test'])
        except KeyError:
            return 0.0

    ##Method get averaged train and test  accuracy for generic model
    # @param model
    # @return accuracy metric or None if not exists
    @staticmethod
    def get_combined(model):
        try:
            print('Combined Accuracy:' + str(model['metrics']['accuracy']['combined']))
            return float(model['metrics']['accuracy']['combined'])
        except KeyError:
            return 0.0

    ##Method get rmse for generic model
    # @param model
    # @return rsme metric or None if not exists
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
    # @param model_list for models analyzed
    # @return (fw,model_list) (ArMetadata, normalization_set)
    def priorize_models(self, analysis_id, model_list):
        if self.metric == 'accuracy':
            return sorted(model_list, key=self.get_accuracy, reverse=True)
        elif self.metric == 'rmse':
            return sorted(model_list, key=self.get_rmse)
        elif self.metric == 'test_accuracy':
            return sorted(model_list, key=self.get_test_accuracy, reverse=True)
        elif self.metric == 'combined':
            return sorted(model_list, key=self.get_combined, reverse=True)
        else:
            return model_list

    ## Store executed model base parameters to check past executions
    # @param model - ArMetadata to be stored as executed
    # @param normalization_set
    # @return model_vector (fw, vector, normalizaton_set)
    def generate_vectors(self, model, normalization_set):
        vector = list()
        if 'h2o' in model['model_parameters'].keys():
            for parm, parm_value in model['model_parameters']['h2o']['parameters'].items():
                if isinstance(parm_value, OrderedDict) and parm != 'model_id':
                    vector.append(parm_value['value'])
            return 'h2o', model['model_parameters']['h2o']['model'], vector, normalization_set
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
        return vector1[0] == vector2[0] and vector1[1] == vector2[1] \
               and vector1[2] == vector2[2] and compare_dict(vector1[3], vector2[3])

    ## Check if model is previously executed. If it not append to list
    # @param model_list
    # @param  model json compatible
    def safe_append(self, model_list, model):
        vector = self.generate_vectors(model, model['normalizations_set'])
        if not self.is_executed(vector):
            model_list.append(model)
            self.analyzed_models.append(vector)
        else:
            self.excluded_models.append(vector)

    ## Method manging generation of possible optimized models
    # params: results for Handlers (gdayf.handlers)
    # @param armetadata ArMetadata Model
    # @return list of possible optimized models to execute return None if nothing to do
    def optimize_models(self, armetadata):
        model_metric = decode_json_to_dataframe(armetadata['metrics']['model'])
        scoring_metric = decode_json_to_dataframe(armetadata['metrics']['scoring'])
        model_list = list()
        model = armetadata['model_parameters']['h2o']

        if get_model_fw(armetadata) == 'h2o' and eval('self.get_' + self.metric + '(armetadata)') != 1.0:
            config = LoadConfig().get_config()['optimizer']['AdviserStart_rules']['h2o']
            nfold_limit = config['nfold_limit']
            min_rows_limit = config['min_rows_limit']
            cols_breakdown = config['cols_breakdown']
            nfold_increment = config['nfold_increment']
            min_rows_increment = config['min_rows_increment']
            max_interactions_rows_breakdown = config['max_interactions_rows_breakdown']
            max_interactions_increment = config['max_interactions_increment']
            max_depth_increment = config['max_depth_increment']
            ntrees_increment = config['ntrees_increment']
            dpl_rcount_limit = config['dpl_rcount_limit']
            dpl_divisor = config['dpl_divisor']
            h_dropout_ratio = config['h_dropout_ratio']
            epochs_increment = config['epochs_increment']
            dpl_min_batch_size = config['dpl_min_batch_size']
            dpl_batch_divisor = config['dpl_batch_divisor']
            dpl_batch_reduced_divisor = config['dpl_batch_reduced_divisor']
            hidden_increment = config['hidden_increment']

            if model['model'] == 'H2OGradientBoostingEstimator':
                if (self.deepness + 1 == self.deep_impact) and model['types'][0]['type'] == 'regression':
                    for distribution in model['parameters']['distribution']['type']:
                        new_armetadata = armetadata.copy_template(deepness=self.deepness)
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['distribution']['value'] = distribution
                        self.safe_append(model_list, new_armetadata)
                if model_metric['number_of_trees'][0] >= model['parameters']['ntrees']['value']:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['ntrees']['value'] *= ntrees_increment
                    self.safe_append(model_list, new_armetadata)
                if model_metric['max_depth'][0] >= model['parameters']['max_depth']['value']:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['max_depth']['value'] *= max_depth_increment
                    self.safe_append(model_list, new_armetadata)
                if model['parameters']['nfolds']['value'] < nfold_limit:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['nfolds']['value'] += nfold_increment
                    self.safe_append(model_list, new_armetadata)
                if model['parameters']['min_rows']['value'] < min_rows_limit:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['min_rows']['value'] += min_rows_increment
                    self.safe_append(model_list, new_armetadata)

            elif model['model'] == 'H2OGeneralizedLinearEstimator':
                if (self.deepness + 1 == self.deep_impact) and model['types'][0]['type'] == 'regression':
                    for family in model['parameters']['family']['type']:
                        new_armetadata = armetadata.copy_template(deepness=self.deepness)
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['family']['value'] = family
                        self.safe_append(model_list, new_armetadata)
                if self.deepness == 1:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['alpha']['value'] = 0.0
                    self.safe_append(model_list, new_armetadata)
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['alpha']['value'] = 1.0
                    self.safe_append(model_list, new_armetadata)
                    if armetadata['data_initial']['cols'] > cols_breakdown:
                        new_armetadata = armetadata.copy_template(deepness=self.deepness)
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['solver']['value'] = 'L_BGFS'
                        self.safe_append(model_list, new_armetadata)
                if self.deepness == 1:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['balance_classes']['value'] = \
                        not model_aux['parameters']['balance_classes']['value']
                    self.safe_append(model_list, new_armetadata)
                if model['parameters']['nfolds']['value'] < nfold_limit:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['nfolds']['value'] += nfold_increment
                    self.safe_append(model_list, new_armetadata)
                if model_metric['number_of_iterations'][0] >= model['parameters']['max_iterations']['value']:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    if self.deepness == 1:
                        model_aux['parameters']['max_interactions']['value'] = \
                            max(round(armetadata['data_initial']['rowcount']/max_interactions_rows_breakdown), 1)
                    else:
                        model_aux['parameters']['max_interactions']['value'] *= max_interactions_increment
                    self.safe_append(model_list, new_armetadata)

            elif model['model'] == 'H2ODeepLearningEstimator':
                if (self.deepness + 1 == self.deep_impact) and model['types'][0]['type'] == 'regression':
                    for distribution in model['parameters']['distribution']['type']:
                        new_armetadata = armetadata.copy_template(deepness=self.deepness)
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['distribution']['value'] = distribution
                        self.safe_append(model_list, new_armetadata)
                if self.deepness == 1:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['sparse']['value'] = not model_aux['parameters']['sparse']['value']
                    self.safe_append(model_list, new_armetadata)
                if self.deepness == 1 and model['parameters']['initial_weight_distribution']['value'] == "normal":
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['initial_weight_distribution']['value'] = "uniform"
                    self.safe_append(model_list, new_armetadata)
                elif self.deepness == 1:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['initial_weight_distribution']['value'] = "normal"
                    self.safe_append(model_list, new_armetadata)

                if self.deepness == 1 and armetadata['data_initial']['rowcount'] > dpl_rcount_limit:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['hidden']['value'] = \
                        [round(armetadata['data_initial']['rowcount']/(dpl_divisor*(self.deep_impact - self.deepness))),
                        round(armetadata['data_initial']['rowcount']/(dpl_divisor*self.deep_impact))]
                    model_aux['parameters']['hidden_dropout_ratios']['value'] = [h_dropout_ratio, h_dropout_ratio]
                    self.safe_append(model_list, new_armetadata)
                elif self.deepness <= self.deep_impact:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['hidden']['value'].insert(0, \
                                round(model_aux['parameters']['hidden']['value'][0] * hidden_increment))
                    model_aux['parameters']['hidden_dropout_ratios']['value'].insert(0, h_dropout_ratio)
                    self.safe_append(model_list, new_armetadata)
                if scoring_metric.shape[0] == 0 or \
                        (scoring_metric['epochs'].max() >= \
                        model['parameters']['epochs']['value']):
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['epochs']['value'] *= epochs_increment
                    self.safe_append(model_list, new_armetadata)
                if self.deepness == 1 and model['parameters']['mini_batch_size']['value'] >= dpl_min_batch_size:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['mini_batch_size']['value'] = \
                        max(round(armetadata['data_initial']['rowcount'] / dpl_batch_divisor), dpl_min_batch_size)
                    self.safe_append(model_list, new_armetadata)
                elif model['parameters']['mini_batch_size']['value'] >= dpl_min_batch_size:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['mini_batch_size']['value'] = \
                        round(model_aux['parameters']['mini_batch_size']['value'] / dpl_batch_reduced_divisor)
                    self.safe_append(model_list, new_armetadata)

            elif model['model'] == 'H2ORandomForestEstimator':
                if model_metric['number_of_trees'][0] == model['parameters']['ntrees']['value']:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['ntrees']['value'] *= ntrees_increment
                    self.safe_append(model_list, new_armetadata)
                if model_metric['max_depth'][0] == model['parameters']['max_depth']['value']:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['max_depth']['value'] *= max_depth_increment
                    self.safe_append(model_list, new_armetadata)
                if model['parameters']['nfolds']['value'] < nfold_limit:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['nfolds']['value'] += nfold_increment
                    self.safe_append(model_list, new_armetadata)
                if model['parameters']['mtries']['value'] not in [round(armetadata['data_initial']['cols']/2),
                                                                  round(armetadata['data_initial']['cols']*3/4)]:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['mtries']['value'] = round(armetadata['data_initial']['cols']/2)
                    self.safe_append(model_list, new_armetadata)
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['mtries']['value'] = round(armetadata['data_initial']['cols']*3/4)
                    self.safe_append(model_list, new_armetadata)
                if model['parameters']['min_rows']['value'] < min_rows_limit:
                    new_armetadata = armetadata.copy_template(deepness=self.deepness)
                    model_aux = new_armetadata['model_parameters']['h2o']
                    model_aux['parameters']['min_rows']['value'] += min_rows_increment
                    self.safe_append(model_list, new_armetadata)
        else:
            return None
        if len(model_list) == 0:
            return None
        else:
            return model_list


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
                                         atype=adv.POC
                                         )
    for each_model in adv.next_analysis_list:
        print(dumps(each_model, indent=4))


