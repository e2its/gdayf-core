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
from gdayf.models.sparkframeworkmetadata import sparkFrameworkMetadata
from gdayf.models.sparkmodelmetadata import sparkModelMetadata
from gdayf.conf.loadconfig import LoadLabels
from gdayf.logs.logshandler import LogsHandler
from gdayf.common.constants import FTYPES
from gdayf.common.dfmetada import compare_dict
from gdayf.common.utils import compare_list_ordered_dict
from gdayf.common.utils import get_model_fw
from gdayf.common.constants import *
from collections import OrderedDict
from time import time
from hashlib import md5 as md5
from json import dumps
from copy import deepcopy

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
    # @param dataframe_name dataframe_name or id
    # @param hash_dataframe MD5 hash value
    def __init__(self, analysis_id, deep_impact=5, metric='accuracy', dataframe_name='', hash_dataframe=''):
        self._labels = LoadLabels().get_config()['messages']['adviser']
        self._config = LoadConfig().get_config()['optimizer']
        self._logging = LogsHandler()
        self.timestamp = time()
        self.analysis_id = analysis_id
        self.an_objective = None
        self.deep_impact = deep_impact
        self.analysis_recommendation_order = list()
        self.analyzed_models = list()
        self.excluded_models = list()
        self.next_analysis_list = list()
        self.metric = metric
        self.dataframe_name = dataframe_name
        self.hash_dataframe = hash_dataframe

    ## Main method oriented to execute smart analysis
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param atype [POC, NORMAL, FAST, PARANOIAC, FAST_PARANOIAC]
    # @param objective_column string indicating objective column
    # @return ArMetadata()'s Prioritized queue
    def set_recommendations(self, dataframe_metadata, objective_column, atype=POC):
        supervised = True
        if objective_column is None:
            supervised = False
        self._logging.log_exec(self.analysis_id, 'AdviserAStar',
                               self._labels["ana_type"],
                               str(atype) + ' (' + str(self.deepness) + ')')
        if supervised:
            self.an_objective = self.get_analysis_objective(dataframe_metadata, objective_column=objective_column)
            if atype == POC:
                return self.analysispoc(dataframe_metadata, objective_column, amode=FAST)
            if atype in [FAST, NORMAL]:
                return self.analysisnormal(dataframe_metadata, objective_column, amode=atype)
            elif atype in [FAST_PARANOIAC, PARANOIAC]:
                return self.analysisparanoiac(dataframe_metadata, objective_column, amode=atype)
        else:
            if atype in [ANOMALIES]:
                self.an_objective = ATypesMetadata(anomalies=True)
                return self.analysisanomalies(dataframe_metadata, objective_column, amode=atype)
            elif atype in [CLUSTERING]:
                self.an_objective = ATypesMetadata(clustering=True)
                return self.analysisclustering(dataframe_metadata, objective_column, amode=atype)

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
            fw_model_list = list()
            # Added 31/08/2017
            best_models = list()
            # End - Added 31/08/2017
            aux_loop_controller = len(self.analysis_recommendation_order)
            for indexer in range(0, aux_loop_controller):
                try:
                    model = self.analysis_recommendation_order[indexer]
                    model_type = model['model_parameters'][get_model_fw(model)]['model']
                    if model_type not in best_models and len(best_models) < self._config['adviser_L2_wide']:
                        fw_model_list.extend(self.optimize_models(self.analysis_recommendation_order[indexer]))
                        best_models.append(model_type)
                except TypeError:
                    ''' If all optimize_models doesn't return new models 
                    register it as evaluated and seleted'''
                    best_models.append(model_type)
            self.next_analysis_list.extend(fw_model_list)
            if len(self.next_analysis_list) == 0:
                    self.next_analysis_list = None
        elif self.next_analysis_list is not None:
            fw_model_list = list()
            # Added 31/08/2017
            best_models = list()
            # End - Added 31/08/2017
            aux_loop_controller = len(self.analysis_recommendation_order)
            for indexer in range(0, aux_loop_controller):
                try:
                    model = self.analysis_recommendation_order[indexer]
                    model_type = model['model_parameters'][get_model_fw(model)]['model']
                    if model_type not in best_models and len(best_models) < self._config['adviser_normal_wide']:
                        fw_model_list.extend(self.optimize_models(self.analysis_recommendation_order[indexer]))
                        #print("Trace:%s-%s" % (model_type, best_models))
                        best_models.append(model_type)
                except TypeError:
                    ''' If all optimize_models doesn't return new models 
                    register it as evaluated and seleted'''
                    best_models.append(model_type)

            '''' Modified 20/09/2017
            # Get two most potential best models
            fw_model_list = list()
            for indexer in range(0, 2):
                try:
                    fw_model_list.extend(self.optimize_models(self.analysis_recommendation_order[indexer]))
                except TypeError:
                    pass
            #if fw_model_list is not None:'''
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

    ## Method oriented to execute new analysis
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()j
    # @param list_ar_metadata List of ar json compatible model's descriptors
    # @return analysis_id, Ordered[(algorithm_metadata.json, normalizations_sets.json)]
    def analysis_specific(self, dataframe_metadata, list_ar_metadata):
        self.next_analysis_list.clear()
        if self.deepness == 1:
            #Check_dataframe_metadata compatibility
            self.base_specific(dataframe_metadata, list_ar_metadata)
        # Added 22/09/1974
        elif self.deepness > self.deep_impact:
            self.next_analysis_list = None
        elif self.next_analysis_list is not None:
            fw_model_list = list()
            # Added 31/08/2017
            best_models = list()
            # End - Added 31/08/2017
            aux_loop_controller = len(self.analysis_recommendation_order)
            for indexer in range(0, aux_loop_controller):
                try:
                    # Modified 31/08/2017
                    model = self.analysis_recommendation_order[indexer]
                    model_type = model['model_parameters'][get_model_fw(model)]['model']
                    if model_type not in best_models:
                        fw_model_list.extend(self.optimize_models(self.analysis_recommendation_order[indexer]))
                        #print("Trace:%s-%s" % (model_type, best_models))
                        best_models.append(model_type)
                        # End - Modified 31/08/2017
                except TypeError:
                    ''' If all optimize_models doesn't return new models 
                    pass and look for next best model on this type'''
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
            # Added 31/08/2017
            best_models = list()
            # End - Added 31/08/2017
            aux_loop_controller = len(self.analysis_recommendation_order)
            for indexer in range(0, aux_loop_controller):
                try:
                    # Modified 31/08/2017
                    model = self.analysis_recommendation_order[indexer]
                    model_type = model['model_parameters'][get_model_fw(model)]['model']
                    if model_type not in best_models:
                        fw_model_list.extend(self.optimize_models(self.analysis_recommendation_order[indexer]))
                        #print("Trace:%s-%s" % (model_type, best_models))
                        best_models.append(model_type)
                        # End - Modified 31/08/2017
                except TypeError:
                    ''' If all optimize_models doesn't return new models 
                    pass and look for next best model on this type'''
                    pass
            #if fw_model_list is not None:
            self.next_analysis_list.extend(fw_model_list)
            if len(self.next_analysis_list) == 0:
                    self.next_analysis_list = None
        self.deepness += 1
        return self.analysis_id, self.next_analysis_list

    ## Method oriented to execute unsupervised anomalies models
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param amode [ANOMALIES]
    # @param objective_column string indicating objective column
    # @return analysis_id,(framework, Ordered[(algorithm_metadata.json, normalizations_sets.json)])

    def analysisanomalies(self, dataframe_metadata, objective_column, amode):
        self.next_analysis_list.clear()
        if self.deepness == 1:
            self.base_iteration(amode, dataframe_metadata, objective_column)
        elif self.deepness > self.deep_impact:
            self.next_analysis_list = None
        elif self.next_analysis_list is not None:
            fw_model_list = list()
            # Added 31/08/2017
            best_models = list()
            # End - Added 31/08/2017
            aux_loop_controller = len(self.analysis_recommendation_order)
            for indexer in range(0, aux_loop_controller):
                try:
                    # Modified 31/08/2017
                    model = self.analysis_recommendation_order[indexer]
                    model_type = model['model_parameters'][get_model_fw(model)]['model']
                    if model_type not in best_models:
                        #print("Trace:%s-%s"%(model_type, best_models))
                        fw_model_list.extend(self.optimize_models(self.analysis_recommendation_order[indexer]))
                        best_models.append(model_type)
                        # End - Modified 31/08/2017
                except TypeError:
                    ''' If all optimize_models doesn't return new models 
                    pass and look for next best model on this type'''
                    pass
            #if fw_model_list is not None:
            self.next_analysis_list.extend(fw_model_list)
            if len(self.next_analysis_list) == 0:
                    self.next_analysis_list = None
        self.deepness += 1
        return self.analysis_id, self.next_analysis_list

    ## Method oriented to execute unsupervised clustering models
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param amode [CLUSTERING]
    # @param objective_column string indicating objective column
    # @return analysis_id,(framework, Ordered[(algorithm_metadata.json, normalizations_sets.json)])

    def analysisclustering(self, dataframe_metadata, objective_column, amode):
        self.next_analysis_list.clear()
        if self.deepness == 1:
            self.base_iteration(amode, dataframe_metadata, objective_column)
        elif self.deepness > self.deep_impact:
            self.next_analysis_list = None
        elif self.next_analysis_list is not None:
            fw_model_list = list()
            # Added 31/08/2017
            best_models = list()
            # End - Added 31/08/2017
            aux_loop_controller = len(self.analysis_recommendation_order)
            for indexer in range(0, aux_loop_controller):
                try:
                    # Modified 31/08/2017
                    model = self.analysis_recommendation_order[indexer]
                    model_type = model['model_parameters'][get_model_fw(model)]['model']
                    if model_type not in best_models:
                        #print("Trace:%s-%s"%(model_type, best_models))
                        fw_model_list.extend(self.optimize_models(self.analysis_recommendation_order[indexer]))
                        best_models.append(model_type)
                        # End - Modified 31/08/2017
                except TypeError:
                    ''' If all optimize_models doesn't return new models 
                    pass and look for next best model on this type'''
                    best_models.append(model_type)
            #if fw_model_list is not None:
            self.next_analysis_list.extend(fw_model_list)
            if len(self.next_analysis_list) == 0:
                    self.next_analysis_list = None
        self.deepness += 1
        return self.analysis_id, self.next_analysis_list

    ## Method oriented to generate specific candidate metadata
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param list_ar_metadata
    def base_specific(self, dataframe_metadata, list_ar_metadata):
        version = LoadConfig().get_config()['common']['version']
        for ar_metadata in list_ar_metadata:

            ar_structure = ArMetadata()
            if ar_metadata['dataset_hash_value'] == self.hash_dataframe:
                self.analysis_id = ar_metadata['model_id']
                ar_structure['predecessor'] = ar_metadata['model_parameters'][get_model_fw(ar_metadata)] \
                    ['parameters']['model_id']['value']
                ar_structure['round'] = int(ar_metadata['round']) + 1
            else:
                ar_structure['predecessor'] = 'root'

            ar_structure['model_id'] = self.analysis_id
            ar_structure['version'] = version
            ar_structure['objective_column'] = ar_metadata['objective_column']
            ar_structure['timestamp'] = self.timestamp
            ar_structure['normalizations_set'] = ar_metadata['normalizations_set']
            ar_structure['dataset'] = self.dataframe_name
            ar_structure['dataset_hash_value'] = self.hash_dataframe
            ar_structure['data_initial'] = dataframe_metadata
            ar_structure['data_normalized'] = None
            ar_structure['model_parameters'] = ar_metadata['model_parameters']
            ar_structure['ignored_parameters'] = None
            ar_structure['full_parameters_stack'] = None
            ar_structure['status'] = -1
            self.next_analysis_list.append(ar_structure)
            self.analyzed_models.append(self.generate_vectors(ar_structure, ar_metadata['normalizations_set']))

    ## Method oriented to select initial candidate models
    # @param self object pointer
    # @param dataframe_metadata DFMetadata()
    # @param amode [POC, NORMAL, FAST, PARANOIAC, FAST_PARANOIAC]
    # @param objective_column string indicating objective column
    def base_iteration(self, amode, dataframe_metadata, objective_column):
        version = LoadConfig().get_config()['common']['version']
        supervised = True
        if objective_column is None:
            supervised = False

        increment = self.get_size_increment(dataframe_metadata)
        fw_model_list = self.get_candidate_models(self.an_objective, amode, increment=increment)

        aux_model_list = list()
        norm = Normalizer()
        #modified 11/09/2017
        #minimal_nmd = [norm.define_minimal_norm(objective_column=objective_column)]
        minimal_nmd = norm.define_minimal_norm(dataframe_metadata=dataframe_metadata,
                                               objective_column=objective_column,
                                               an_objective=self.an_objective)
        for fw, model, _ in fw_model_list:
            aux_model_list.append((fw, model, deepcopy(minimal_nmd)))
        fw_model_list = aux_model_list

        self.applicability(fw_model_list, nrows=dataframe_metadata['rowcount'], ncols=dataframe_metadata['cols'])

        nmd = norm.define_normalizations(dataframe_metadata=dataframe_metadata,
                                         objective_column=objective_column,
                                         an_objective=self.an_objective)

        if nmd is not None:
            nmdlist = list()
            for fw, model, _ in fw_model_list:
                if minimal_nmd[0] is not None:
                    whole_nmd = deepcopy(minimal_nmd)
                    whole_nmd.extend(deepcopy(nmd))
                    nmdlist.append((fw, model, whole_nmd))
                else:
                    nmdlist.append((fw, model, deepcopy(nmd)))

            fw_model_list.extend(nmdlist)

        for fw, model_params, norm_sets in fw_model_list:
            ar_structure = ArMetadata()
            ar_structure['model_id'] = self.analysis_id
            ar_structure['version'] = version
            ar_structure['objective_column'] = objective_column
            ar_structure['timestamp'] = self.timestamp
            ar_structure['normalizations_set'] = norm_sets
            ar_structure['dataset'] = self.dataframe_name
            ar_structure['dataset_hash_value'] = self.hash_dataframe
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
                if each_column['type'] not in FTYPES:
                    if int(each_column['cardinality']) > 2:
                        return ATypesMetadata(multinomial=True)
                elif int(each_column['cardinality']) <= (dataframe_metadata['rowcount']*config['multi_limit']):
                    return ATypesMetadata(multinomial=True)
                else:
                    return ATypesMetadata(regression=True)
        return None

    ## Method oriented to analyze get increments on effort based on DF_metadata structure
    # @param self object pointer
    # @param df_metadata DfMetada
    # @return float increment
    def get_size_increment(self, df_metadata):
        base = LoadConfig().get_config()['optimizer']['common']['base_increment']
        increment = 1.0
        variabilizations = df_metadata['rowcount'] * df_metadata['cols']
        for _ , pvalue in base.items():
            if variabilizations > pvalue['base'] and increment < pvalue['increment']:
                increment = pvalue['increment']
        self._logging.log_info(self.analysis_id, 'AdviserAStar', self._labels["inc_application"],
                               increment)
        return increment

    ## Method oriented to analyze choose models candidate and select analysis objective
    # @param self object pointer
    # @param atype ATypesMetadata
    # @param amode Analysismode
    # @param increment increment x size
    # @return FrameworkMetadata()
    def get_candidate_models(self, atype, amode, increment=1.0):
        defaultframeworks = self.load_frameworks()
        model_list = list()
        for fw, fw_value in defaultframeworks.items():
            if fw == 'h2o' and fw_value['conf']['enabled']:
                wfw = H2OFrameworkMetadata(defaultframeworks)
                for each_base_model in wfw.get_default():
                    if each_base_model['enabled']:
                        for each_type in each_base_model['types']:
                            if each_type['active'] and each_type['type'] == atype[0]['type']:
                                modelbase = H2OModelMetadata()
                                model = modelbase.generate_models(each_base_model['model'], atype, amode, increment)
                                wfw.models.append(model)
                                model_list.append((fw, model, None))
            if fw == 'spark' and fw_value['conf']['enabled']:
                wfw = sparkFrameworkMetadata(defaultframeworks)
                for each_base_model in wfw.get_default():
                    if each_base_model['enabled']:
                        for each_type in each_base_model['types']:
                            if each_type['active'] and each_type['type'] == atype[0]['type']:
                                modelbase = sparkModelMetadata()
                                model = modelbase.generate_models(each_base_model['model'], atype, amode, increment)
                                wfw.models.append(model)
                                model_list.append((fw, model, None))
        return model_list

    ## Method oriented to select applicability of models over min_rows_limit
    # @param self object pointer
    # @param model_list List[ArMetadata]
    # @param nrows number of rows of dataframe
    # @param ncols number of cols of dataframe
    # @return implicit List[ArMetadata]
    def applicability(self, model_list, nrows, ncols):
        fw_config = LoadConfig().get_config()['frameworks']
        exclude_model = list()
        for iterator in range(0, len(model_list)):
            fw = model_list[iterator][0]
            model = model_list[iterator][1]
            if fw_config[fw]['conf']['min_rows_enabled'] and (nrows < model['min_rows_applicability']):
                self._logging.log_info(self.analysis_id, 'AdviserAStar', self._labels["exc_applicability"],
                                       model['model'] + ' - ' + 'rows < ' +
                                       str(model['min_rows_applicability']))
                exclude_model.append(model_list[iterator])
            if fw_config[fw]['conf']['max_cols_enabled'] and model['max_cols_applicability'] is not None \
                    and(ncols > model['max_cols_applicability']):
                self._logging.log_info(self.analysis_id, 'AdviserAStar', self._labels["exc_applicability"],
                                       model['model'] + ' - ' + 'cols > ' +
                                       str(model['max_cols_applicability']))
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
                   1/float(model['metrics']['execution']['test']['RMSE']),\
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

    ##Method get clustering distance for generic model
    # @param model
    # @return The Total Within Cluster Sum-of-Square Error metric, inverse The Between Cluster Sum-of-Square Error,
    # objective or 10e+308, 0.0, objective if not exists
    @staticmethod
    def get_cdistance(model):
        try:
            return float(model['metrics']['execution']['train']['tot_withinss']), \
                   1/float(model['metrics']['execution']['train']['betweenss']), \
                   0.0
        except ZeroDivisionError:
            return float(model['metrics']['execution']['train']['tot_withinss']), \
                   10e+308, \
                   0.0
        except TypeError:
            return float(model['metrics']['execution']['train']['tot_withinss']), \
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
        elif self.metric == 'cdistance':
            return sorted(model_list, key=self.get_cdistance)
        else:
            return model_list

    ## Store executed model base parameters to check past executions
    # @param model - ArMetadata to be stored as executed
    # @param normalization_set
    # @return model_vector (fw, model_id, vector, normalizaton_set)
    def generate_vectors(self, model, normalization_set):
        vector = list()
        norm_vector = list()
        fw = get_model_fw(model)
        for parm, parm_value in model['model_parameters'][fw]['parameters'].items():
            if isinstance(parm_value, OrderedDict) and parm != 'model_id':
                vector.append(parm_value['value'])
        #added 31/08/2017
        if normalization_set == [None]:
            norm_vector = normalization_set
        else:
            for normalization in normalization_set:
                norm_vector.append(md5(dumps(normalization).encode('utf8')).hexdigest())
        #print("Trace:%s-%s-%s-%s"%(fw, model['model_parameters'][fw]['model'], vector, norm_vector))
        return fw, model['model_parameters'][fw]['model'], vector, norm_vector
        ''' Deleted 31/08/2017. Change vector model
        return fw, model['model_parameters'][fw]['model'], vector, normalization_set'''


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
    # @return True if equal False if inequity
    @ staticmethod
    def compare_vectors(vector1, vector2):
        return vector1[0] == vector2[0] and vector1[1] == vector2[1] \
               and vector1[2] == vector2[2] and vector1[3] == vector2[3]
               # Deleted 31/08/2017
               # and vector1[2] == vector2[2] and compare_list_ordered_dict(vector1[3], vector2[3])

    ## Check if model is previously executed. If it not append to list
    # @param model_list
    # @param  model json compatible
    def safe_append(self, model_list, model):
        vector = self.generate_vectors(model, model['normalizations_set'])
        if not self.is_executed(vector):
            model_list.append(model)
            self.analyzed_models.append(vector)
            self._logging.log_info(self.analysis_id, 'AdviserAStar', self._labels["new_vector"], str(vector))
        else:
            self.excluded_models.append(vector)
            self._logging.log_info(self.analysis_id, 'AdviserAStar', self._labels["exc_vector"], str(vector))

