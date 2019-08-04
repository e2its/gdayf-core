## @package gdayf.core.controller
# Define all objects, functions and structured related to manage and execute actions over DayF core
# and expose API to users

'''
 * This file is part of the gDayF AutoML Core Framework project
 * distribution (https://github.com/e2its/gdayf-core).
 * Copyright (c) 2016-2019 Jose Luis Sanchez del Coso <e2its.es@gmail.com>.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 ** Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2019
'''

from gdayf.handlers.inputhandler import inputHandlerCSV
from gdayf.common.dfmetada import DFMetada

from gdayf.logs.logshandler import LogsHandler
from gdayf.common.utils import get_model_fw
from gdayf.common.constants import *
from gdayf.common.utils import pandas_split_data, hash_key
from gdayf.common.armetadata import ArMetadata
from gdayf.common.armetadata import deep_ordered_copy
from gdayf.persistence.persistencehandler import PersistenceHandler
from gdayf.common.storagemetadata import StorageMetadata
from collections import OrderedDict
from pathlib import Path
from pandas import DataFrame
import importlib
from json.decoder import JSONDecodeError
from time import time
from pymongo import  MongoClient
from pymongo.errors import *
# import bson
# from  bson.codec_options import CodecOptions
from hashlib import md5
from gdayf.core.experiment_context import Experiment_Context as E_C

## Core class oriented to manage the comunication and execution messages pass for all components on system
# orchestrating the execution of actions activities (train and prediction) on specific frameworks
class Controller(object):

    ## Constructor
    def __init__(self, e_c=None, user_id='PoC_gDayF', workflow_id='default'):
        self.timestamp = str(time())
        if e_c is None:
            if workflow_id == 'default':
                self._ec = E_C(user_id=user_id, workflow_id=workflow_id + '_' + self.timestamp)
            else:
                self._ec = E_C(user_id=user_id, workflow_id=workflow_id)
        else:
            self._ec = e_c
        self._config = self._ec.config.get_config()
        self._labels = self._ec.labels.get_config()['messages']['controller']
        self._frameworks = self._ec.config.get_config()['frameworks']
        self._logging = LogsHandler(self._ec)
        self.analysis_list = OrderedDict()  # For future multi-analysis uses
        self.model_handler = OrderedDict()
        self.adviser = importlib.import_module(self._config['optimizer']['adviser_classpath'])
        self._logging.log_info('gDayF', "Controller", self._labels["loading_adviser"],
                               self._config['optimizer']['adviser_classpath'])

    ## Method leading configurations coherence checks
    # @param self object pointer
    # @return True if OK / False if wrong
    def config_checks(self):
        storage_conf = self._config['storage']
        grants = storage_conf['grants']
        localfs = (storage_conf['localfs'] is not None) \
                  and self._coherence_fs_checks(storage_conf['localfs'], grants=grants)
        hdfs = (storage_conf['hdfs'] is not None) \
                  and self._coherence_fs_checks(storage_conf['hdfs'], grants=grants)
        mongoDB = (storage_conf['mongoDB'] is not None) \
                  and self._coherence_db_checks(storage_conf['mongoDB'])
        self._logging.log_info('gDayF', "Controller", self._labels["primary_path"],
                               str(storage_conf['primary_path']))

        ''' Checking primary Json storage Paths'''
        primary = False
        #if storage_conf['primary_path'] in ['localfs', 'hdfs']:
        for storage in StorageMetadata(self._ec).get_load_path():
            if storage_conf['primary_path'] == storage['type'] and storage['type'] != 'mongoDB':
                primary = True
            if storage['type'] == 'mongoDB':
                self._logging.log_critical('gDayF', "Controller", self._labels["failed_file_storage"],
                                           str(storage))
                return False
            elif storage['type'] == 'localfs':
                if not localfs:
                    self._logging.log_critical('gDayF', "Controller", self._labels["failed_load"],
                                           str(storage))
                    return False
            elif storage['type'] == 'hdfs':
                if not hdfs:
                    self._logging.log_critical('gDayF', "Controller", self._labels["failed_load"],
                                           str(storage))
                    return False

        if not primary:
            self._logging.log_critical('gDayF', "Controller", self._labels["no_primary"],
                                   str(storage_conf[storage_conf['primary_path']]))
            return False

        ''' Checking Load storage Paths'''
        at_least_on = False
        for storage in StorageMetadata(self._ec).get_json_path():
            if storage['type'] == 'mongoDB':
                if not mongoDB:
                    self._logging.log_critical('gDayF', "Controller", self._labels["failed_json"],
                                           str(storage))
                    return False
                else:
                    at_least_on = at_least_on or True
            elif storage['type'] == 'localfs':
                if not localfs:
                    self._logging.log_critical('gDayF', "Controller", self._labels["failed_json"],
                                           str(storage))
                    return False
                else:
                    at_least_on = at_least_on or True
            elif storage['type'] == 'hdfs':
                if not hdfs:
                    self._logging.log_critical('gDayF', "Controller", self._labels["failed_json"],
                                           str(storage))
                    return False
                else:
                    at_least_on = at_least_on or True

        if not at_least_on:
            self._logging.log_critical('gDayF', "Controller", self._labels["no_primary"],
                                   str(storage_conf[storage_conf['primary_path']]))
            return False

        ''' Checking log storage Paths'''
        at_least_on = False
        for storage in StorageMetadata(self._ec).get_log_path():
            if storage['type'] == 'mongoDB':
                self._logging.log_critical('gDayF', "Controller", self._labels["failed_file_storage"],
                                       str(storage))
                return False
            elif storage['type'] == 'localfs':
                if not localfs:
                    self._logging.log_critical('gDayF', "Controller", self._labels["failed_log"],
                                           str(storage))
                    return False
                else:
                    at_least_on = at_least_on or True
            elif storage['type'] == 'hdfs':
                if not hdfs:
                    self._logging.log_critical('gDayF', "Controller", self._labels["failed_log"],
                                                str(storage))
                    return False
                else:
                    at_least_on = at_least_on or True
        if not at_least_on:
            self._logging.log_critical('gDayF', "Controller", self._labels["no_primary"],
                                   str(storage_conf[storage_conf['primary_path']]))
            return False

        ''' If all things OK'''
        return True

    ## Method leading configurations coherence checks on fs engines
    # @param self object pointer
    # @param storage StorageMetadata
    # @param grants Octal grants format
    # @return True if OK / False if wrong
    def _coherence_fs_checks(self, storage, grants):
        persistence = PersistenceHandler(self._ec)
        try:
            if persistence.mkdir(type=storage['type'], path=str(storage['value']), grants=grants):
                return False
        except OSError:
            self._logging.log_critical('gDayF', "Controller", self._labels["failed_json_path"],
                                   str(storage['value']))
            return False
        if storage['hash_type'] not in ['MD5', 'SHA256']:
            self._logging.log_critical('gDayF', "Controller", self._labels["failed_hash_method"],
                                   str(storage))
            return False
        return True

    ## Method leading configurations coherence checks on fs engines
    # @param self object pointer
    # @param storage StorageMetadata
    # @return True if OK / False if wrong
    def _coherence_db_checks(self, storage):
        if storage['type'] == 'mongoDB':
            try:
                client = MongoClient(host=storage['url'],
                                     port=int(storage['port']),
                                     document_class=OrderedDict)
            except ConnectionFailure as cexecution_error:
                print(repr(cexecution_error))
                return False
            try:
                db = client[storage['value']]
                collection = db[self._ec.get_id_user()]
                test_insert = collection.insert_one({'test': 'connection.check.dot.bson'}).inserted_id
                collection.delete_one({"_id": test_insert})
            except PyMongoError as wexecution_error:
                print(repr(wexecution_error))
                return False
            finally:
                client.close()
        return True

    ## Method leading and controlling prediction's executions on all frameworks
    # @param self object pointer
    # @param datapath String Path indicating file to be analyzed or Dataframe
    # @param armetadata
    # @param model_file String Path indicating model_file ArMetadata.json structure
    def exec_prediction(self, datapath, armetadata=None, model_file=None):

        self._logging.log_info('gDayF', "Controller", self._labels["ana_mode"], 'prediction')
        if armetadata is None and model_file is None:
            self._logging.log_critical('gDayF', "Controller", self._labels["failed_model"], datapath)
            return self._labels["failed_model"]
        elif armetadata is not None:
            try:
                assert isinstance(armetadata, ArMetadata)
                base_ar = deep_ordered_copy(armetadata)
            except AssertionError:
                self._logging.log_critical('gDayF', "Controller", self._labels["failed_model"], armetadata)
                return self._labels["failed_model"]
        elif model_file is not None:
            try:
                #json_file = open(model_file)
                persistence = PersistenceHandler(self._ec)
                invalid, base_ar = persistence.get_ar_from_engine(model_file)
                del persistence

                if invalid:
                    self._logging.log_critical('gDayF', "Controller", self._labels["failed_model"], model_file)
                    return self._labels["failed_model"]
            except IOError as iexecution_error:
                print(repr(iexecution_error))
                self._logging.log_critical('gDayF', "Controller", self._labels["failed_model"], model_file)
                return self._labels["failed_model"]
            except OSError as oexecution_error:
                print(repr(oexecution_error))
                self._logging.log_critical('gDayF', "Controller", self._labels["failed_model"], model_file)
                return self._labels["failed_model"]

        if isinstance(datapath, str):
            try:
                self._logging.log_info('gDayF', "Controller", self._labels["input_param"], datapath)
                pd_dataset = inputHandlerCSV().inputCSV(filename=datapath)
            except [IOError, OSError, JSONDecodeError]:
                self._logging.log_critical('gDayF', "Controller", self._labels["failed_input"], datapath)
                return self._labels['failed_input']
        elif isinstance(datapath, DataFrame):
            pd_dataset = datapath
            self._logging.log_info('gDayF', "Controller", self._labels["input_param"], str(datapath.shape))
        else:
            self._logging.log_critical('gDayF', "Controller", self._labels["failed_input"], datapath)
            return self._labels['failed_input']

        fw = get_model_fw(base_ar)

        self.init_handler(fw)

        prediction_frame = None
        try:
            prediction_frame, _ = self.model_handler[fw]['handler'].predict(predict_frame=pd_dataset,
                                                                            base_ar=base_ar)
        except TypeError:
            self._logging.log_critical('gDayF', "Controller", self._labels["failed_model"], model_file)

        self.clean_handler(fw)

        self._logging.log_info('gDayF', 'controller', self._labels["pred_end"])

        return prediction_frame

    ## Method focus on cleaning handler objects
    # @param fw framework
    def clean_handler(self, fw):
        if self.model_handler[fw]['handler'] is not None:
            self.model_handler[fw]['handler'].delete_frames()
            self.model_handler[fw]['handler'] = None

    ## Method oriented to init handler objects
    # @param fw framework
    def init_handler(self, fw):
        try:
            if self.model_handler[fw]['handler'] is None:
                handler = importlib.import_module(self._frameworks[fw]['conf']['handler_module'])
                self.model_handler[fw]['handler'] = \
                    eval('handler.' + self._frameworks[fw]['conf']['handler_class'] + '(e_c=self._ec)')
        except KeyError:
            self.model_handler[fw] = OrderedDict()
            handler = importlib.import_module(self._frameworks[fw]['conf']['handler_module'])
            self.model_handler[fw]['handler'] = \
                eval('handler.' + self._frameworks[fw]['conf']['handler_class'] + '(e_c=self._ec)')
            self.model_handler[fw]['initiated'] = False
        if not self.model_handler[fw]['handler'].is_alive():
            initiated = self.model_handler[fw]['handler'].connect()
            self.model_handler[fw]['initiated'] = (self.model_handler[fw]['initiated'] or initiated)

    ## Method oriented to shutdown localClusters
    def clean_handlers(self):
        for fw, each_handlers in self.model_handler.items():
            if each_handlers['handler'] is not None:
                #self.model_handler[fw][each_handlers['handler']].clean_handler(fw)
                self.clean_handler(fw)
                self._logging.log_exec('gDayF', "Controller", self._labels["cleaning"], fw)
                if each_handlers['initiated']:
                    handler = importlib.import_module(self._frameworks[fw]['conf']['handler_module'])
                    self.model_handler[fw]['handler'] = \
                        eval('handler.' + self._frameworks[fw]['conf']['handler_class']
                             + '(e_c=self._ec).shutdown_cluster()')
                    self._logging.log_exec('gDayF', "Controller", self._labels["shuttingdown"], fw)

    ## Method leading and controlling analysis's executions on all frameworks
    # @param self object pointer
    # @param datapath String Path indicating file to be analyzed or DataFrame
    # @param objective_column string indicating objective column
    # @param amode Analysis mode of execution [0,1,2,3,4,5,6]
    # @param metric to evalute models ['train_accuracy', 'train_rmse', 'test_accuracy', 'combined_accuracy', 'test_rmse', 'cdistance']
    # @param deep_impact  deep analysis
    # @return status, adviser.analysis_recommendation_order
    def exec_analysis(self, datapath, objective_column, amode=POC, metric='test_accuracy', deep_impact=3, **kwargs):
        # Clustering variables
        k = None
        estimate_k = False

        #Force analysis variable
        atype = None

        hash_dataframe = ''

        for pname, pvalue in kwargs.items():
            if pname == 'k':
                assert isinstance(pvalue, int)
                k = pvalue
            elif pname == 'estimate_k':
                assert isinstance(pvalue, bool)
                estimate_k = pvalue
            elif pname == 'atype':
                assert pvalue in atypes
                atype = pvalue


        supervised = True
        if objective_column is None:
            supervised = False

        self._logging.log_info('gDayF', "Controller", self._labels["start"])
        self._logging.log_info('gDayF', "Controller", self._labels["ana_param"], metric)
        self._logging.log_info('gDayF', "Controller", self._labels["dep_param"], deep_impact)
        self._logging.log_info('gDayF', "Controller", self._labels["ana_mode"], amode)


        if isinstance(datapath, str):
            try:
                self._logging.log_info('gDayF', "Controller", self._labels["input_param"], datapath)
                pd_dataset = inputHandlerCSV().inputCSV(filename=datapath)
                id_datapath = Path(datapath).name
                hash_dataframe = hash_key('MD5', datapath)
            except IOError:
                self._logging.log_critical('gDayF', "Controller", self._labels["failed_input"], datapath)
                return self._labels['failed_input']
            except OSError:
                self._logging.log_critical('gDayF', "Controller", self._labels["failed_input"], datapath)
                return self._labels['failed_input']
            except JSONDecodeError:
                self._logging.log_critical('gDayF', "Controller", self._labels["failed_input"], datapath)
                return self._labels['failed_input']
        elif isinstance(datapath, DataFrame):
            self._logging.log_info('gDayF', "Controller", self._labels["input_param"], str(datapath.shape))
            pd_dataset = datapath
            id_datapath = 'Dataframe' + \
                          '_' + str(pd_dataset.size) + \
                          '_' + str(pd_dataset.shape[0]) + \
                          '_' + str(pd_dataset.shape[1])
            hash_dataframe = md5(datapath.to_msgpack()).hexdigest()
        else:
            self._logging.log_critical('gDayF', "Controller", self._labels["failed_input"], datapath)
            return self._labels['failed_input'], None

        pd_test_dataset = None
        ''' Changed 05/04/2018
        if metric == 'combined_accuracy' or 'test_accuracy':'''
        if self._config['common']['minimal_test_split'] <= len(pd_dataset.index) \
                and (metric in ACCURACY_METRICS or metric in REGRESSION_METRICS):
            pd_dataset, pd_test_dataset = pandas_split_data(pd_dataset,
                                                            train_perc=self._config['common']['test_frame_ratio'])

        df = DFMetada().getDataFrameMetadata(pd_dataset, 'pandas')
        
        self._ec.set_id_analysis(self._ec.get_id_user() + '_' + id_datapath + '_' + str(time()))
        adviser = self.adviser.AdviserAStar(e_c=self._ec,
                                            metric=metric,
                                            deep_impact=deep_impact, dataframe_name=id_datapath,
                                            hash_dataframe=hash_dataframe)

        adviser.set_recommendations(dataframe_metadata=df, objective_column=objective_column, amode=amode, atype=atype)

        while adviser.next_analysis_list is not None:
            for each_model in adviser.next_analysis_list:
                fw = get_model_fw(each_model)

                if k is not None:
                    try:
                        each_model["model_parameters"][fw]["parameters"]["k"]["value"] = k
                        each_model["model_parameters"][fw]["parameters"]["k"]["seleccionable"] = True
                        each_model["model_parameters"][fw]["parameters"]["estimate_k"]["value"] = estimate_k
                        each_model["model_parameters"][fw]["parameters"]["estimate_k"]["seleccionable"] = True
                    except KeyError:
                        pass

                self.init_handler(fw)
                if pd_test_dataset is not None:
                    _, analyzed_model = self.model_handler[fw]['handler'].order_training(training_pframe=pd_dataset,
                                                                                         base_ar=each_model,
                                                                                         test_frame=pd_test_dataset,
                                                                                         filtering='STANDARDIZE')
                else:
                    _, analyzed_model = self.model_handler[fw]['handler'].order_training(training_pframe=pd_dataset,
                                                                                         base_ar=each_model,
                                                                                         test_frame=pd_dataset,
                                                                                         filtering='STANDARDIZE')

                if analyzed_model is not None:
                    adviser.analysis_recommendation_order.append(analyzed_model)
            adviser.next_analysis_list.clear()
            adviser.analysis_recommendation_order = adviser.priorize_models(model_list=
                                                                            adviser.analysis_recommendation_order)
            adviser.set_recommendations(dataframe_metadata=df, objective_column=objective_column, amode=amode)

        self._logging.log_info(self._ec.get_id_analysis(), 'controller',
                               self._labels["ana_models"], str(len(adviser.analyzed_models)))
        self._logging.log_info(self._ec.get_id_analysis(), 'controller',
                               self._labels["exc_models"], str(len(adviser.excluded_models)))

        self._logging.log_exec(self._ec.get_id_analysis(), 'controller', self._labels["end"])

        self.clean_handlers()

        adviser.analysis_recommendation_order = adviser.priorize_models(model_list=
                                                                        adviser.analysis_recommendation_order)

        return self._labels['success_op'], adviser.analysis_recommendation_order

    ## Method oriented to log leaderboard against selected metrics
    # @param ar_list List of AR models Execution Data
    # @param metric to execute order ['train_accuracy', 'train_rmse', 'test_accuracy', 'combined_accuracy', 'test_rmse', 'cdistance']
    def log_model_list(self, ar_list, metric):
        best_check = True
        ordered_list = self.priorize_list(arlist=ar_list, metric=metric)
        for model in ordered_list:
            if best_check:
                self._logging.log_info(self._ec.get_id_analysis(), 'controller', self._labels["best_model"],
                                       model['model_parameters'][get_model_fw(model)]['parameters']['model_id']['value'])
                best_check = False
            else:
                self._logging.log_info(self._ec.get_id_analysis(), 'controller', self._labels["res_model"],
                                       model['model_parameters'][get_model_fw(model)]['parameters']['model_id']['value'])

            self._logging.log_info(self._ec.get_id_analysis(), 'controller', self._labels["round_reach"], model['round'])
            if model["normalizations_set"] is None:
                self._logging.log_info(self._ec.get_id_analysis(), 'controller', self._labels["norm_app"], [])
            else:
                self._logging.log_info(self._ec.get_id_analysis(), 'controller', self._labels["norm_app"],
                                       model["normalizations_set"])

            if metric in ACCURACY_METRICS or metric in REGRESSION_METRICS:
                self._logging.log_info(self._ec.get_id_analysis(), 'controller', self._labels["ametric_order"],
                                       model['metrics']['accuracy'])
                self._logging.log_info(self._ec.get_id_analysis(), 'controller', self._labels["pmetric_order"],
                                       model['metrics']['execution']['train']['RMSE'])
                self._logging.log_info(self._ec.get_id_analysis(), 'controller', self._labels["pmetric_order"],
                                       model['metrics']['execution']['test']['RMSE'])
                self._logging.log_info(self._ec.get_id_analysis(), 'controller', self._labels["rmetric_order"],
                                       model['metrics']['execution']['train']['r2'])
                self._logging.log_info(self._ec.get_id_analysis(), 'controller', self._labels["rmetric_order"],
                                       model['metrics']['execution']['test']['r2'])
            if metric in CLUSTERING_METRICS:
                try:
                    self._logging.log_info(self._ec.get_id_analysis(), 'controller', self._labels["ckmetric_order"],
                                       model['metrics']['execution']['train']['k'])
                except KeyError:
                    self._logging.log_info(self._ec.get_id_analysis(),'controller', self._labels["ckmetric_order"],
                                       "0")
                self._logging.log_info(self._ec.get_id_analysis(), 'controller', self._labels["ctmetric_order"],
                                       model['metrics']['execution']['train']['tot_withinss'])
                self._logging.log_info(self._ec.get_id_analysis(), 'controller', self._labels["cbmetric_order"],
                                       model['metrics']['execution']['train']['betweenss'])

    ## Method oriented to log leaderboard against selected metrics on dataframe
    # @param analysis_id
    # @param ar_list List of AR models Execution Data
    # @param metric to execute order ['train_accuracy', 'train_rmse', 'test_accuracy', 'combined_accuracy', 'test_rmse', 'cdistance']
    # @return Dataframe performance model list

    def table_model_list(self, ar_list, metric):
        dataframe = list()
        normal_cols = ['Model', 'Train_accuracy', 'Test_accuracy', 'Combined_accuracy', 'train_rmse', 'test_rmse']
        cluster_cols = ['Model', 'k', 'tot_withinss', 'betweenss']

        ordered_list = self.priorize_list(arlist=ar_list, metric=metric)
        for model in ordered_list:
            if metric in ACCURACY_METRICS or metric in REGRESSION_METRICS:
                try:
                    dataframe.append(
                        {'Model': model['model_parameters'][get_model_fw(model)]['parameters']['model_id']['value'],
                         'Round': model['round'],
                         'train_accuracy': model['metrics']['accuracy']['train'],
                         'test_accuracy': model['metrics']['accuracy']['test'],
                         'combined_accuracy': model['metrics']['accuracy']['combined'],
                         'train_rmse': model['metrics']['execution']['train']['RMSE'],
                         'test_rmse': model['metrics']['execution']['test']['RMSE'],
                         'train_r2': model['metrics']['execution']['train']['r2'],
                         'test_r2': model['metrics']['execution']['test']['r2'],
                         'path': model['json_path'][0]['value']
                         }
                    )
                # AutoEncoders metrics
                except KeyError:
                    dataframe.append(
                        {'Model': model['model_parameters'][get_model_fw(model)]['parameters']['model_id']['value'],
                         'Round': model['round'],
                         'train_accuracy': model['metrics']['accuracy']['train'],
                         'test_accuracy': model['metrics']['accuracy']['test'],
                         'combined_accuracy': model['metrics']['accuracy']['combined'],
                         'train_rmse': model['metrics']['execution']['train']['RMSE'],
                         'path': model['json_path'][0]['value']
                         }
                    )

            if metric in CLUSTERING_METRICS:
                try:
                    aux = model['metrics']['execution']['train']['k']
                except KeyError:
                    aux = 0

                dataframe.append(
                    {'Model': model['model_parameters'][get_model_fw(model)]['parameters']['model_id']['value'],
                     'Round': model['round'],
                     'k': aux,
                     'tot_withinss':model['metrics']['execution']['train']['tot_withinss'],
                     'betweenss':model['metrics']['execution']['train']['betweenss'],
                     'path': model['json_path'][0]['value']
                     }
                )
        return DataFrame(dataframe)

    ## Method leading and controlling analysis's executions on specific analysis
    # @param self object pointer
    # @param datapath String Path indicating file to be analyzed or DataFrame
    # @param list_ar_metadata list of models to execute
    # @param metric to evalute models
    # @param deep_impact  deep analysis
    # @return status, adviser.analysis_recommendation_order
    def exec_sanalysis(self, datapath, list_ar_metadata, metric='combined_accuracy', deep_impact=1, **kwargs):

        self._logging.log_info('gDayF', "Controller", self._labels["start"])
        self._logging.log_info('gDayF', "Controller", self._labels["ana_param"], metric)
        self._logging.log_info('gDayF', "Controller", self._labels["dep_param"], deep_impact)

        if isinstance(datapath, str):
            try:
                self._logging.log_info('gDayF', "Controller", self._labels["input_param"], datapath)
                pd_dataset = inputHandlerCSV().inputCSV(filename=datapath)
                id_datapath = Path(datapath).name
                hash_dataframe = hash_key('MD5', datapath)
            except IOError:
                self._logging.log_critical('gDayF', "Controller", self._labels["failed_input"], datapath)
                return self._labels['failed_input']
            except OSError:
                self._logging.log_critical('gDayF', "Controller", self._labels["failed_input"], datapath)
                return self._labels['failed_input']
            except JSONDecodeError:
                self._logging.log_critical('gDayF', "Controller", self._labels["failed_input"], datapath)
                return self._labels['failed_input']
        elif isinstance(datapath, DataFrame):
            hash_dataframe = None
            self._logging.log_critical('gDayF', "Controller", self._labels["input_param"], str(datapath.shape))
            pd_dataset = datapath
            id_datapath = 'Dataframe' + \
                          '_' + str(pd_dataset.size) + \
                          '_' + str(pd_dataset.shape[0]) + \
                          '_' + str(pd_dataset.shape[1])
        else:
            self._logging.log_critical('gDayF', "Controller", self._labels["failed_input"], datapath)
            return self._labels['failed_input'], None

        pd_test_dataset = None
        if self._config['common']['minimal_test_split'] <= len(pd_dataset.index) \
                and (metric in ACCURACY_METRICS or metric in REGRESSION_METRICS):
            pd_dataset, pd_test_dataset = pandas_split_data(pd_dataset,
                                                            train_perc=self._config['common']['test_frame_ratio'])

        df = DFMetada().getDataFrameMetadata(pd_dataset, 'pandas')
        self._ec.set_id_analysis(self._ec.get_id_user() + '_' + id_datapath + '_' + str(time()))
        adviser = self.adviser.AdviserAStar(e_c=self._ec,
                                            metric=metric,
                                            deep_impact=deep_impact, dataframe_name=id_datapath,
                                            hash_dataframe=hash_dataframe)

        adviser.analysis_specific(dataframe_metadata=df, list_ar_metadata=list_ar_metadata)

        while adviser.next_analysis_list is not None:

            for each_model in adviser.next_analysis_list:
                fw = get_model_fw(each_model)

                self.init_handler(fw)

                if pd_test_dataset is not None:
                    _, analyzed_model = self.model_handler[fw]['handler'].order_training(
                        training_pframe=pd_dataset,
                        base_ar=each_model,
                        test_frame=pd_test_dataset, filtering='NONE')
                else:
                    _, analyzed_model = self.model_handler[fw]['handler'].order_training(
                        training_pframe=pd_dataset,
                        base_ar=each_model, filtering='NONE')
                if analyzed_model is not None:
                    adviser.analysis_recommendation_order.append(analyzed_model)

            adviser.next_analysis_list.clear()
            adviser.analysis_recommendation_order = adviser.priorize_models(model_list=
                                                                            adviser.analysis_recommendation_order)
            adviser.analysis_specific(dataframe_metadata=df, list_ar_metadata=adviser.analysis_recommendation_order)

        self._logging.log_info(self._ec.get_id_analysis(), 'controller',
                               self._labels["ana_models"], str(len(adviser.analyzed_models)))
        self._logging.log_info(self._ec.get_id_analysis(), 'controller',
                               self._labels["exc_models"], str(len(adviser.excluded_models)))

        self.log_model_list(adviser.analysis_recommendation_order, metric)

        self._logging.log_info(self._ec.get_id_analysis(), 'controller', self._labels["end"])

        self.clean_handlers()

        adviser.analysis_recommendation_order = adviser.priorize_models(model_list=
                                                                        adviser.analysis_recommendation_order)

        return self._labels['success_op'], adviser.analysis_recommendation_order

    ## Method leading and controlling coversion to java model
    # @param self object pointer
    # @param armetadata Armetada object
    # @param type base type if is possible
    # @return download_path, hash MD5 key
    def get_external_model(self, armetadata, type='pojo'):
        fw = get_model_fw(armetadata)
        self.init_handler(fw)
        results = self.model_handler[fw]['handler'].get_external_model(armetadata, type)
        self.clean_handler(fw)
        return results

    ## Method leading and controlling model savings
    # @param self object pointer
    # @param mode [BEST, BEST_3, EACH_BEST, ALL]
    # @param  arlist List of armetadata
    # @param  metric ['accuracy', 'combined', 'test_accuracy', 'rmse']
    def save_models(self, arlist, mode=BEST, metric='accuracy'):
        if mode == BEST:
            model_list = [arlist[0]]
        elif mode == BEST_3:
            model_list = arlist[0:3]
        elif mode == EACH_BEST:
            exclusion = list()
            model_list = list()
            for model in arlist:
                if (get_model_fw(model), model['model_parameters'][get_model_fw(model)]['model'],
                        model['normalizations_set']) not in exclusion:
                        model_list.append(model)
                        exclusion.append((get_model_fw(model), model['model_parameters'][get_model_fw(model)]['model'],
                                          model['normalizations_set']))
        elif mode == ALL:
            model_list = arlist
        elif mode == NONE:
            model_list = list()
        for fw in self._config['frameworks'].keys():
                self.init_handler(fw)
                for each_model in model_list:
                    if fw in each_model['model_parameters'].keys():
                        self.model_handler[fw]['handler'].store_model(each_model, user=self._ec.get_id_user())
                self.clean_handler(fw)

    ## Method leading and controlling model loads
    # @param self object pointer
    # @param  arlist List of armetadata
    # @return  list of ar_descriptors of models correctly loaded
    def load_models(self, arlist):
        model_loaded = list()
        for fw in self._config['frameworks'].keys():
                self.init_handler(fw)
                for each_model in arlist:
                    if fw in each_model['model_parameters'].keys():
                        model_load = self.model_handler[fw]['handler'].load_model(each_model)
                        if model_load is not None:
                            model_loaded.append(model_load)
                self.clean_handler(fw)
        return model_loaded

    ## Method leading and controlling model removing from server
    # @param self object pointer
    # @param mode to be keeped in memory [BEST, BEST_3, EACH_BEST, ALL,NONE]
    # @param  arlist List of armetadata
    def remove_models(self, arlist, mode=ALL):
        if mode == BEST:
            model_list = arlist[1:]
        elif mode == BEST_3:
            model_list = arlist[3:]
        elif mode == EACH_BEST:
            exclusion = list()
            model_list = list()
            for model in arlist:
                if (get_model_fw(model), model['model_parameters'][get_model_fw(model)]['model'],
                        model['normalizations_set']) not in exclusion:
                        exclusion.append((get_model_fw(model), model['model_parameters'][get_model_fw(model)]['model'],
                                          model['normalizations_set']))
                else:
                    model_list.append(model)
        elif mode == ALL:
            model_list = arlist
        elif mode == NONE:
            model_list = list()
        fw_list = list()
        for models in model_list:
            if get_model_fw(models) not in fw_list:
                fw_list.append(get_model_fw(models))

        for fw in fw_list:
            self.init_handler(fw)
            self.model_handler[fw]['handler'].remove_models(model_list)
            self.clean_handler(fw)

    ##Method oriented to generate execution tree for visualizations and analysis issues
    # @param arlist Priorized ArMetadata list
    # @param  metric ['accuracy', 'combined', 'test_accuracy', 'rmse']
    # @param  store True/False
    # @param experiment analysys_id for mongoDB recovery
    # @param user user_id for mongoDB recovery
    # @return OrderedDict() with execution tree data Analysis
    def reconstruct_execution_tree(self, arlist=None, metric='combined', store=True):
        if (arlist is None or len(arlist) == 0) and self._ec.get_id_analysis() is None:
            self._logging.log_critical('gDayF', 'controller', self._labels["failed_model"])
            return None
        elif self._ec.get_id_analysis() is not None and self._ec.get_id_user() != 'guest':
            new_arlist = PersistenceHandler(self._ec).recover_experiment_mongoDB()
        else:
            analysis_id = arlist[0]['model_id']
            new_arlist = arlist

        ordered_list = self.priorize_list(arlist=new_arlist, metric=metric)

        root = OrderedDict()
        root['data'] = None
        root['ranking'] = 0
        root['successors'] = OrderedDict()
        variable_dict = OrderedDict()
        variable_dict[0] = {'root': root}

        ranking = 1
        for new_tree_structure in ordered_list:
            new_model = deep_ordered_copy(new_tree_structure)
            model_id = new_tree_structure['model_parameters'][get_model_fw(new_tree_structure)]\
                                         ['parameters']['model_id']['value']
            level = new_tree_structure['round']
            if level not in variable_dict.keys():
                variable_dict[level] = OrderedDict()

            new_tree_structure = OrderedDict()
            new_tree_structure['ranking'] = ranking
            new_tree_structure['data'] = new_model
            new_tree_structure['successors'] = OrderedDict()
            variable_dict[level][model_id] = new_tree_structure

            ranking += 1

        level = 1
        max_level = max(variable_dict.keys())
        while level in range(1, max_level+1):
            for model_id, new_tree_structure in variable_dict[level].items():
                counter = 1
                found = False
                while not found or (level - counter) == 0:
                    if new_tree_structure['data']['predecessor'] in variable_dict[level-counter].keys():
                        container = eval('variable_dict[level-counter][new_tree_structure[\'data\'][\'predecessor\']]')
                        container['successors'][model_id] = new_tree_structure
                        found = True
                    counter += 1
                if not found:
                    self._logging.log_debug(self._ec.get_id_analysis(), 'controller', self._labels['fail_reconstruct'],
                                            model_id)
            level += 1

        #Store_json on primary path
        if store and self._config['storage']['primary_path'] != 'mongoDB':
            primary_path = self._config['storage']['primary_path']
            fstype = self._config['storage'][primary_path]['type']

            datafile = list()
            datafile.append(self._config['storage'][primary_path]['value'])
            datafile.append('/')
            datafile.append(self._ec.get_id_user())
            datafile.append('/')
            datafile.append(self._ec.get_id_workflow())
            datafile.append('/')
            datafile.append(self._config['common']['execution_tree_dir'])
            datafile.append('/')
            datafile.append(self._ec.get_id_analysis())
            datafile.append('.json')

            if self._config['persistence']['compress_json']:
                datafile.append('.gz')

            storage = StorageMetadata(self._ec)
            storage.append(value=''.join(datafile), fstype=fstype)
            PersistenceHandler(self._ec).store_json(storage, root)
        return root

    ##Method oriented to priorize ARlist
    # @param self object pointer
    # @param analysis_id
    # @param arlist Priorized ArMetadata list
    # @param  metric ['accuracy', 'combined', 'test_accuracy', 'rmse']
    # @return OrderedDict() with execution tree data Analysis
    def priorize_list(self, arlist, metric):
        adviser = self.adviser.AdviserAStar(e_c=self._ec, metric=metric)
        ordered_list = adviser.priorize_models(arlist)
        del adviser
        return ordered_list

    ## Method base to get an ArMetadata Structure from file
    # @param self object pointer
    # @param path FilePath
    # @return operation status (0 success /1 error, ArMetadata/None)
    def get_ar_from_engine(self, path):
        persistence = PersistenceHandler(self._ec)
        failed, armetadata = persistence.get_ar_from_engine(path=path)
        del persistence
        return failed, armetadata




