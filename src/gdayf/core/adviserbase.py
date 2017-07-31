## @package gdayf.core.adviserbase
# Define all objects, functions and structured related to manage and execute Smart analysis based on A* Algorithm
# and defined heuristic
# Main class Adviser. Lets us execute analysis, make recommendations over optimizing on selected algoritms

from gdayf.models.frameworkmetadata import FrameworkMetadata
from gdayf.common.armetadata import ArMetadata
from copy import deepcopy
from gdayf.normalizer.normalizer import Normalizer
from gdayf.conf.loadconfig import LoadConfig
from gdayf.models.atypesmetadata import ATypesMetadata
from gdayf.models.h2oframeworkmetadata import H2OFrameworkMetadata
from gdayf.models.h2omodelmetadata import H2OModelMetadata
from gdayf.conf.loadconfig import LoadLabels
from gdayf.logs.logshandler import LogsHandler
from gdayf.common.utils import ftypes
from gdayf.common.utils import compare_dict
from gdayf.common.utils import get_model_fw
from gdayf.common.constants import *
from collections import OrderedDict
from time import time

## Class focused on execute A* based analysis on three modalities of working
# Fast: 1 level analysis over default parameters
# Normal: One A* analysis for all models based until max_deep with early_stopping
# Paranoiac: One A* algorithm per model analysis until max_deep without early stoping
class Adviser(object):
    deepness = 1

    ## Constructor
    # @param self object pointer
    # @param analysis_id main id traceability code
    # @param deep_impact A* max_deep
    # @param metric metrict for priorizing models ['accuracy', 'rmse', 'test_accuracy', 'combined'] on train
    def __init__(self, analysis_id, deep_impact=2, metric='accuracy'):
        self._labels = LoadLabels().get_config()['messages']['adviser']
        self._logging = LogsHandler()
        self.timestamp = time()
        self.analysis_id = analysis_id + '_' + str(self.timestamp)
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
    # @param atype [POC, NORMAL, FAST, PARANOIAC, FAST_PARANOIAC]
    # @param objective_column string indicating objective column
    # @return ArMetadata()'s Prioritized queue
    def set_recommendations(self, dataframe_metadata, objective_column, atype=POC):
        self._logging.log_exec(self.analysis_id, 'AdviserAStar',
                               self._labels["ana_type"],
                               str(atype) + ' (' + str(self.deepness) + ')')
        self.an_objective = self.get_analysis_objective(dataframe_metadata, objective_column=objective_column)
        if atype == POC:
            return self.analysispoc(dataframe_metadata, objective_column, amode=FAST)
        if atype in [FAST, NORMAL]:
            return self.analysisnormal(dataframe_metadata, objective_column, amode=atype)
        elif atype in [FAST_PARANOIAC, PARANOIAC]:
            return self.analysisparanoiac(dataframe_metadata, objective_column, amode=atype)


    ## Method oriented to execute smart normal and fast analysis
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param objective_column string indicating objective column
    # @param amode [POC, NORMAL, FAST, PARANOIAC, FAST_PARANOIAC]
    # @return analysis_id, Ordered[(algorithm_metadata.json, normalizations_sets.json)]
    def analysisnormal(self, dataframe_metadata, objective_column, amode):
        self.next_analysis_list.clear()
        if self.deepness == 1:
            self.base_iteration(amode, dataframe_metadata, objective_column)
        elif self.deepness > self.deep_impact:
            self.next_analysis_list = None
        elif self.deepness == 2:
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

    ## Method oriented to execute poc analysis
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param objective_column string indicating objective column
    # @param amode [POC, NORMAL, FAST, PARANOIAC, FAST_PARANOIAC]
    # @return analysis_id, Ordered[(algorithm_metadata.json, normalizations_sets.json)]
    def analysispoc(self, dataframe_metadata, objective_column, amode):
        self.next_analysis_list.clear()
        if self.deepness == 1:
            self.base_iteration(amode, dataframe_metadata, objective_column)
        elif self.deepness > self.deep_impact:
            self.next_analysis_list = None
        elif self.next_analysis_list is not None:
            # Get two most potential best models
            fw_model_list = list()
            for indexer in range(0, 1):
                try:
                    fw_model_list.extend(self.optimize_models(self.analysis_recommendation_order[indexer]))
                except TypeError:
                    pass
            # if fw_model_list is not None:
            self.next_analysis_list.extend(fw_model_list)
            if len(self.next_analysis_list) == 0:
                self.next_analysis_list = None
        self.deepness += 1
        return self.analysis_id, self.next_analysis_list

    ## Method oriented to execute smart normal and fast analysis
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param amode [POC, NORMAL, FAST, PARANOIAC, FAST_PARANOIAC]
    # @param objective_column string indicating objective column
    # @return analysis_id,(framework, Ordered[(algorithm_metadata.json, normalizations_sets.json)])

    def analysisparanoiac(self, dataframe_metadata, objective_column, amode):
        self.next_analysis_list.clear()
        if self.deepness == 1:
            self.base_iteration(amode, dataframe_metadata, objective_column)
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

    ## Method oriented to select initial candidate models
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param amode [POC, NORMAL, FAST, PARANOIAC, FAST_PARANOIAC]
    # @param objective_column string indicating objective column
    def base_iteration(self, amode, dataframe_metadata, objective_column):
        fw_model_list = self.get_candidate_models(self.an_objective, amode)
        aux_model_list = list()
        norm = Normalizer()
        minimal_nmd = [norm.define_minimal_norm(objective_column=objective_column)]
        for fw, model, _ in fw_model_list:
            aux_model_list.append((fw, model, minimal_nmd))
        fw_model_list = aux_model_list
        nmd = norm.define_normalizations(dataframe_metadata=dataframe_metadata,
                                               objective_column=objective_column,
                                               an_objective=self.an_objective)

        self.applicability(fw_model_list, nrows=dataframe_metadata['rowcount'])
        if nmd is not None:
            nmdlist = list()
            for fw, model, _ in fw_model_list:
                nmdlist.append((fw, model, nmd))
            fw_model_list.extend(nmdlist)

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
            ar_structure['predecessor'] = 'root'
            ar_structure['status'] = -1
            self.next_analysis_list.append(ar_structure)
            self.analyzed_models.append(self.generate_vectors(ar_structure, norm_sets))


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
        config = LoadConfig().get_config()['optimizer']['AdviserStart_rules']['common']
        for each_column in dataframe_metadata['columns']:
            if each_column['name'] == objective_column:
                if int(each_column['cardinality']) == 2:
                    return ATypesMetadata(binomial=True)
                if each_column['type'] not in ftypes:
                    if int(each_column['cardinality']) > 2:
                        return ATypesMetadata(multinomial=True)
                elif int(each_column['cardinality']) <= (dataframe_metadata['rowcount']*config['multi_limit']):
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
        for fw, fw_value in defaultframeworks.items():
            if fw == 'h2o' and fw_value['conf']['enabled']:
                wfw = H2OFrameworkMetadata(defaultframeworks)
                for each_base_model in wfw.get_default():
                    for each_type in each_base_model['types']:
                        if each_type['active'] and each_type['type'] == atype[0]['type']:
                            modelbase = H2OModelMetadata()
                            model = modelbase.generate_models(each_base_model['model'], atype, amode)
                            wfw.models.append(model)
                            model_list.append((fw, model, None))
        return model_list

    ## Method oriented to select applicability of models over min_rows_limit
    # @param self object pointer
    # @param model_list List[ArMetadata]
    # @param nrows number of rows of dataframe
    # @return implicit List[ArMetadata]
    def applicability(self, model_list, nrows):
        fw_config = LoadConfig().get_config()['frameworks']
        exclude_model = list()
        for iterator in range(0, len(model_list)):
            fw = model_list[iterator][0]
            model = model_list[iterator][1]
            if fw_config[fw]['conf']['min_rows_enabled'] and (nrows < model['min_rows_applicability']):
                self._logging.log_exec(self.analysis_id, 'AdviserAStar', self._labels["exc_applicability"],
                                       model['model'] + ' - ' +
                                       str(model['min_rows_applicability']))
                exclude_model.append(model_list[iterator])
        for model in exclude_model:
           model_list.remove(model)

    ##Method get train accuracy for generic model
    # @param model
    # @return accuracy metric, inverse rmse, objective or 0.0, 10e+8, objective if not exists
    @staticmethod
    def get_accuracy(model):
        try:
            return float(model['metrics']['accuracy']['train']),\
                   1/float(model['metrics']['execution']['train']['RMSE']),\
                   1.0
        except ZeroDivisionError:
            return float(model['metrics']['accuracy']['train']), \
                   10e+308, \
                   1.0
        except KeyError:
            return 0.0, 10e+8, 1.0

    ##Method get test accuracy for generic model
    # @param model
    # @return accuracy metric, inverse rmse, objective or 0.0, 10e+308, objective if not exists
    @staticmethod
    def get_test_accuracy(model):
        try:
            return float(model['metrics']['accuracy']['test']),\
                   1/float(model['metrics']['execution']['train']['RMSE']),\
                   1.0
        except ZeroDivisionError:
            return float(model['metrics']['accuracy']['test']), \
                   10e+308, \
                   1.0
        except KeyError:
            return 0.0, 10e+8, 1.0

    ##Method get averaged train and test  accuracy for generic model
    # @param model
    # @return accuracy metric, inverse rmse, objective or 0.0, 10e+308, objective if not exists
    @staticmethod
    def get_combined(model):
        try:
            return float(model['metrics']['accuracy']['combined']),\
                   1/float(model['metrics']['execution']['train']['RMSE']),\
                   1.0
        except ZeroDivisionError:
            return float(model['metrics']['accuracy']['combined']), \
                   10e+308, \
                   1.0
        except KeyError:
            return 0.0, 10e+308, 1.0

    ##Method get rmse for generic model
    # @param model
    # @return rsme metric, inverse combined accuracy, objective or 10e+308, 0.0, objective if not exists
    @staticmethod
    def get_rmse(model):
        try:
            return float(model['metrics']['execution']['train']['RMSE']),\
                   1/float(model['metrics']['accuracy']['combined']),\
                   0.0
        except ZeroDivisionError:
            return float(model['metrics']['execution']['train']['RMSE']),\
                   10e+308, \
                   0.0
        except KeyError:
            return 10e+308, 0.0, 0.0

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
    # @return model_vector (fw, model_id, vector, normalizaton_set)
    def generate_vectors(self, model, normalization_set):
        vector = list()
        fw = get_model_fw(model)
        for parm, parm_value in model['model_parameters'][fw]['parameters'].items():
            if isinstance(parm_value, OrderedDict) and parm != 'model_id':
                vector.append(parm_value['value'])
        return fw, model['model_parameters'][fw]['model'], vector, normalization_set


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
            self._logging.log_exec(self.analysis_id, 'AdviserAStar', self._labels["new_vector"], str(vector))
        else:
            self.excluded_models.append(vector)
            self._logging.log_exec(self.analysis_id, 'AdviserAStar', self._labels["exc_vector"], str(vector))

