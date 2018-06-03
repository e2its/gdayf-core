## @package gdayf.workflow.workflow
# Define all objects, functions and structured related to manage and execute actions over DayF core
# and expose API to users

'''
Copyright (C) e2its - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 *
 * This file is part of gDayF project.
 *
 * Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2018
'''

from gdayf.core.controller import Controller
from gdayf.logs.logshandler import LogsHandler
from gdayf.conf.loadconfig import LoadConfig
from gdayf.conf.loadconfig import LoadLabels
from gdayf.common.storagemetadata import StorageMetadata
from gdayf.persistence.persistencehandler import PersistenceHandler
from gdayf.common.utils import decode_ordered_dict_to_dataframe, xstr
from gdayf.common.constants import *
from gdayf.handlers.inputhandler import inputHandlerCSV
from json.decoder import JSONDecodeError
from json import load, dumps
from collections import OrderedDict
from pandas import DataFrame, set_option
from time import time
from pathlib import Path
from gdayf.core import global_var

## Core class oriented to manage pipeline of workflows execution
# orchestrating the execution of actions activities
class Workflow(object):
    ## Constructor
    def __init__(self, user_id='PoC_gDayF'):
        global_var.set_id_user(user_id)
        self._config = LoadConfig().get_config()
        self._labels = LoadLabels().get_config()['messages']['workflow']
        self._logging = LogsHandler()
        self.user_id = user_id
        self.timestamp = str(time())


    ## Method leading workflow executions
    # @param self object pointer
    # @param datapath String Path indicating file to be analyzed or Dataframe
    # @param workflow String Path indicating train workflow definition path
    # @param workflow_id Workflow_file_name + workflow_timestamp execution
    # @param save_models [BEST, BEST_3, EACH_BEST, ALL]
    # @param prefix value

    def workflow(self, datapath, workflow, prefix=None, workflow_id='default', save_models=EACH_BEST):

        if isinstance(workflow, str):
            file = open(workflow, 'r')
            wf = load(file, object_hook=OrderedDict)
            if workflow_id == 'default':
                workflow_id = Path(workflow).stem + '_' + self.timestamp
        else:
            wf = workflow
            if workflow_id == 'default':
                workflow_id = workflow_id + '_' + self.timestamp

        for wkey, wvalue in wf.items():
            if prefix is None:
                prefix = xstr(wvalue['parameters']['objective_column'])
            if wvalue['parameters']['mode'] == "train":
                self.train_workflow(datapath=datapath, wkey=wkey, workflow=wvalue,
                                    prefix=prefix, workflow_id=workflow_id, save_models=save_models)
            elif wvalue['parameters']['mode'] == "predict":
                self.predict_workflow(datapath=datapath, wkey=wkey, workflow=wvalue,
                                      prefix=prefix, workflow_id=workflow_id, save_models=save_models)
            else:
                self._logging.log_info('gDayF', "Workflow", self._labels["nothing_to_do"])

    ## Method leading train workflow executions
    # @param self object pointer
    # @param datapath String Path indicating file to be analyzed or Dataframe
    # @param wkey Step name
    # @param workflow String Path indicating train workflow definition path
    # @param workflow_id Workflow_file_name + workflow_timestamp execution
    # @param prefix value
    # @param save_models [BEST, BEST_3, EACH_BEST, ALL]

    def train_workflow(self, datapath, wkey, workflow, prefix='main', workflow_id='default', save_models=EACH_BEST):
        set_option('display.max_rows', 500)
        set_option('display.max_columns', 500)
        set_option('display.width', 1000)

        wf = workflow
        pfix = prefix

        error, dataset = self.check_path(datapath)
        if dataset is None:
            return error

        controller = Controller(user_id=self.user_id, workflow_id=workflow_id)
        if controller.config_checks():
            variables = dataset.columns.tolist()

            #for wkey, wvalue in wf.items():
            if wf["data"]["filtered_columns"] is not None:
                for delete in wf["data"]["filtered_columns"]:
                    try:
                        variables.remove(delete)
                    except Exception:
                        self._logging.log_info('gDayF', "Workflow", self._labels["failed_var"], delete)
            self._logging.log_info('gDayF', "Workflow", self._labels["variables_desc"], variables)
            if wf["data"]["for_each"] is not None:
                fe_column = wf["data"]["for_each"]
                fe_data_exclusions = wf["data"]["for_each_exclusions"]
                fe_filtered_data = wf["data"]["filtered_data"]
                fe_parameters = wf["parameters"]
                fe_next = wf["Next"]

                for each in eval('dataset.'+fe_column+'.unique()'):
                    if fe_data_exclusions is None or each not in fe_data_exclusions:
                        aux_dataset = eval('dataset[dataset.' + fe_column + '== each]')
                        pfix = xstr(prefix + '_' + str(each))


                        if fe_filtered_data is not None:
                            qcolumn = fe_filtered_data["column"]
                            quantile = aux_dataset[qcolumn].quantile(q=fe_filtered_data["quantile"])
                            aux_dataset = eval('aux_dataset.loc[aux_dataset.' + qcolumn + '<= ' + str(quantile) + ']')
                            pfix = xstr(pfix + '_' + str(fe_filtered_data["quantile"]))


                        if fe_parameters is not None:
                            source_parameters = list()
                            source_parameters.append('controller.exec_analysis(')
                            source_parameters.append('datapath=aux_dataset.loc[:, variables]')
                            for ikey, ivalue in fe_parameters.items():
                                source_parameters.append(',')
                                source_parameters.append(ikey)
                                source_parameters.append('=')
                                if isinstance(ivalue, str) and ikey != "amode":
                                    source_parameters.append('\'')
                                    source_parameters.append(ivalue)
                                    source_parameters.append('\'')
                                else:
                                    source_parameters.append(str(ivalue))
                            source_parameters.append(')')

                            self._logging.log_info('gDayF', "Workflow", self._labels["desc_operation"],
                                                   ''.join(source_parameters))
                            status, recomendations = eval(''.join(source_parameters))
                            controller.remove_models(recomendations, mode=save_models)
                            controller.reconstruct_execution_tree(recomendations, metric=fe_parameters['metric'],
                                                                  store=True)


                            model_id = recomendations[0]['model_id']
                            table_model_list = controller.table_model_list(model_id, recomendations,
                                                                           metric=eval(fe_parameters['metric']))
                            self._logging.log_info('gDayF', 'workflow', self._labels["results"]+'\n',
                                                   table_model_list.to_string(justify="left"))

                            filename = self.storage_path('train', wkey + '_' + str(pfix) + '_' + 'train_performance'
                                                         , 'xls', workflow_id)
                            table_model_list.to_excel(filename, index=False, sheet_name='performance')
                            self.replicate_file('train', filename=filename, workflow_id=workflow_id)

                            prediction_frame = controller.exec_prediction(datapath=aux_dataset,
                                                                          model_file=recomendations[0]['json_path'][0]['value'])
                            try:
                                if 'predict' in prediction_frame.columns.values:
                                    prediction_frame.rename(columns={"predict": wkey}, inplace=True)
                                elif 'prediction' in prediction_frame.columns.values:
                                    prediction_frame.rename(columns={"prediction": wkey}, inplace=True)

                                self._logging.log_info('gDayF', 'workflow', self._labels["results"]+'\n',
                                                       prediction_frame.to_string(index_names=False, justify="left"))

                                filename = self.storage_path('train', wkey + '_'
                                                             + str(pfix) + '_' + 'prediction', 'xls', workflow_id)
                                prediction_frame.to_excel(filename, index=False, sheet_name='train_prediction')
                                self.replicate_file('train', filename=filename, workflow_id=workflow_id)

                            except AttributeError as oexecution_error:
                                self._logging.log_info('gDayF', "Workflow", self._labels["failed_model"],
                                                       str(repr(oexecution_error)))

                            try:
                                if fe_next is not None and prediction_frame is not None:
                                    self.workflow(prediction_frame, fe_next, pfix, workflow_id, save_models=save_models)
                            except Exception as oexecution_error:
                                self._logging.log_info('gDayF', "Workflow", self._labels["failed_wf"], str(fe_next))
            else:
                aux_dataset = dataset

                if wf["data"]["filtered_data"] is not None:
                    qcolumn = wf["data"]["filtered_data"]["column"]
                    quantile = aux_dataset[[qcolumn]].quatile([wf["data"]["filtered_data"]["quantile"]])
                    aux_dataset = aux_dataset.query('%s <= %s' % (qcolumn, quantile))

                if wf['parameters'] is not None:
                    source_parameters = list()
                    source_parameters.append('controller.exec_analysis(')
                    source_parameters.append('datapath=aux_dataset.loc[:, variables]')
                    for ikey, ivalue in wf['parameters'].items():
                        source_parameters.append(',')
                        source_parameters.append(ikey)
                        source_parameters.append('=')
                        if isinstance(ivalue, str) and ikey != "amode":
                            source_parameters.append('\'')
                            source_parameters.append(ivalue)
                            source_parameters.append('\'')
                        else:
                            source_parameters.append(str(ivalue))
                    source_parameters.append(')')
                    self._logging.log_info('gDayF', "Workflow", self._labels["desc_operation"],
                                           ''.join(source_parameters))
                    status, recomendations = eval(''.join(source_parameters))
                    controller.remove_models(recomendations, mode=save_models)
                    controller.reconstruct_execution_tree(recomendations, metric=wf['parameters']['metric'], store=True)

                    model_id = recomendations[0]['model_id']
                    table_model_list = controller.table_model_list(model_id, recomendations,
                                                      metric=eval(wf['parameters']['metric']))
                    self._logging.log_info('gDayF', 'workflow', self._labels["results"]+'\n',
                                           table_model_list.to_string(justify="left"))

                    filename = self.storage_path('train', wkey + '_' + str(pfix) + '_'
                                                 + 'train_performance', 'xls', workflow_id)
                    table_model_list.to_excel(filename, index=False, sheet_name="performace")
                    self.replicate_file('train', filename=filename, workflow_id=workflow_id)

                    prediction_frame = controller.exec_prediction(datapath=aux_dataset,
                                                                  model_file=recomendations[0]['json_path'][0]['value'])
                    try:
                        if 'predict' in prediction_frame.columns.values:
                            prediction_frame.rename(columns={"predict": wkey}, inplace=True)
                        elif 'prediction' in prediction_frame.columns.values:
                            prediction_frame.rename(columns={"prediction": wkey}, inplace=True)

                        self._logging.log_info('gDayF', 'workflow', self._labels["results"]+'\n',
                                               prediction_frame.to_string(index_names=False, justify="left"))

                        filename = self.storage_path('train', wkey + '_' + str(pfix) + '_'
                                                     + 'prediction', 'xls', workflow_id)
                        prediction_frame.to_excel(filename, index=False, sheet_name="train_prediction")
                        self.replicate_file('train', filename=filename, workflow_id=workflow_id)

                    except AttributeError as oexecution_error:
                            self._logging.log_info('gDayF', "Workflow", self._labels["failed_model"],
                                                   str(repr(oexecution_error)))

                    if wf['Next'] is not None and prediction_frame is not None:
                        try:
                            self.workflow(prediction_frame, wf['Next'], pfix, workflow_id, save_models=save_models)
                        except Exception as oexecution_error:
                            self._logging.log_info('gDayF', "Workflow", self._labels["failed_wf"], str(wf['Next']))


        controller.clean_handlers()
        del controller

    ## Method leading predict workflow executions
    # @param datapath String Path indicating file to be analyzed or Dataframe
    # @param wkey Step name
    # @param workflow String Path indicating test workflow definition path
    # @param workflow_id Workflow_file_name + workflow_timestamp execution
    # @param save_models [BEST, BEST_3, EACH_BEST, ALL]
    # @param prefix value

    def predict_workflow(self, datapath, wkey, workflow, prefix='main', workflow_id='default', save_models=EACH_BEST):
        set_option('display.height', 1000)
        set_option('display.max_rows', 500)
        set_option('display.max_columns', 500)
        set_option('display.width', 1000)

        error, dataset = self.check_path(datapath)
        if dataset is None:
            return error

        if isinstance(workflow, str):
            file = open(workflow, 'r')
            wf = load(file, object_hook=OrderedDict)
        else:
            wf = workflow
        pfix = xstr(prefix)
        controller = Controller(user_id=self.user_id, workflow_id=workflow_id)
        if controller.config_checks():
            variables = dataset.columns.tolist()

            #for wkey, wvalue in wf.items():
            if wf["model"] is not None and \
                    (isinstance(wf["model"], str) or isinstance(wf["model"], dict)):

                if wf["data"]["filtered_columns"] is not None:
                    for delete in wf["data"]["filtered_columns"]:
                        try:
                            variables.remove(delete)
                        except Exception:
                            self._logging.log_info('gDayF', "Workflow", self._labels["failed_var"], delete)

                self._logging.log_info('gDayF', "Workflow", self._labels["variables_desc"], variables)

                if wf["data"]["for_each"] is not None:
                    fe_column = wf["data"]["for_each"]
                    fe_data_exclusions = wf["data"]["for_each_exclusions"]
                    fe_filtered_data = wf["data"]["filtered_data"]
                    fe_next = wf["Next"]

                    for each in eval('dataset.' + fe_column + '.unique()'):
                        if fe_data_exclusions is None or each not in fe_data_exclusions:
                            aux_dataset = eval('dataset[dataset.' + fe_column + '== each]')
                            pfix = xstr(prefix + '_' + str(each))

                            if fe_filtered_data is not None:
                                qcolumn = fe_filtered_data["column"]
                                quantile = aux_dataset[qcolumn].quantile(q=fe_filtered_data["quantile"])
                                aux_dataset = eval('aux_dataset.loc[aux_dataset.' + qcolumn + '<= ' + str(quantile) + ']')
                                pfix = xstr(pfix + '_' + str(fe_filtered_data["quantile"]))

                            prediction_frame = controller.exec_prediction(datapath=aux_dataset,
                                                                          model_file=wf["model"][str(each)])
                            try:
                                if 'predict' in prediction_frame.columns.values:
                                    prediction_frame.rename(columns={"predict": wkey}, inplace=True)
                                elif 'prediction' in prediction_frame.columns.values:
                                    prediction_frame.rename(columns={"prediction": wkey}, inplace=True)
                            except AttributeError:
                                self._logging.log_info('gDayF', "Workflow", self._labels["anomalies_operation"])

                            self._logging.log_info('gDayF', 'workflow', self._labels["results"]+'\n',
                                                   prediction_frame.to_string(index_names=False, justify="left"))

                            try:
                                if isinstance(prediction_frame, DataFrame):
                                    filename = self.storage_path('predict', wkey + '_'
                                                        + str(pfix) + '_' + 'prediction', 'xls', workflow_id)
                                    prediction_frame.to_excel(filename, index=False, sheet_name="prediction")
                                    self.replicate_file('predict', filename=filename, workflow_id=workflow_id)
                                else:
                                    for ikey, ivalue in prediction_frame['columns'].items():
                                        ppDF = decode_ordered_dict_to_dataframe(ivalue)
                                        if isinstance(ppDF, DataFrame):
                                            filename = self.storage_path('predict', wkey + '_'
                                                          + str(pfix) + '_' + 'prediction_' + ikey, 'xls', workflow_id)
                                            ppDF.to_excel(filename, index=False, sheet_name="prediction")
                                            self.replicate_file('predict', filename=filename, workflow_id=workflow_id)
                                    filename = self.storage_path('predict', wkey + '_'
                                              + str(each) + '_' + 'prediction', 'json', workflow_id)
                                    with open(filename, 'w') as f:
                                        f.write(dumps(prediction_frame['global_mse']))
                                    self.replicate_file('predict', filename=filename, workflow_id=workflow_id)
                            except AttributeError:
                                self._logging.log_info('gDayF', "Workflow", self._labels["anomalies_operation"],
                                                       prediction_frame)
                            try:
                                if fe_next is not None and prediction_frame is not None:
                                    self.workflow(prediction_frame, fe_next, pfix, workflow_id, save_models=save_models)
                            except Exception as oexecution_error:
                                self._logging.log_info('gDayF', "Workflow", self._labels["failed_wf"], str(fe_next))
                else:
                    aux_dataset = dataset

                    prediction_frame = controller.exec_prediction(datapath=aux_dataset, model_file=wf["model"])
                    if 'predict' in prediction_frame.columns.values:
                        prediction_frame.rename(columns={"predict": wkey}, inplace=True)
                    elif 'prediction' in prediction_frame.columns.values:
                        prediction_frame.rename(columns={"prediction": wkey}, inplace=True)

                    self._logging.log_info('gDayF', 'workflow', self._labels["results"]+'\n',
                                           prediction_frame.to_string(index_names=False, justify="left"))
                    if isinstance(prediction_frame, DataFrame):
                        filename = self.storage_path('predict', wkey + '_prediction', 'xls', workflow_id)
                        prediction_frame.to_excel(filename, index=False, sheet_name="prediction")
                        self.replicate_file('predict', filename=filename, workflow_id=workflow_id)
                    else:
                        for ikey, ivalue in prediction_frame['columns'].items():
                            ppDF = decode_ordered_dict_to_dataframe(ivalue)
                            if isinstance(ppDF, DataFrame):
                                filename = self.storage_path('predict', wkey + '_'
                                                             + str(pfix) + '_' + 'prediction_' + ikey, 'xls',
                                                             workflow_id)
                                ppDF.to_excel(filename, index=False, sheet_name="prediction")
                                self.replicate_file('predict', filename=filename, workflow_id=workflow_id)

                        filename = self.storage_path('predict', wkey + '_'
                                                     + str(pfix) + '_' + '_prediction', 'json', workflow_id)
                        with open(filename, 'w') as f:
                            f.write(dumps(prediction_frame))
                        self.replicate_file('predict', filename=filename, workflow_id=workflow_id)
                    if wf['Next'] is not None and prediction_frame is not None:
                        try:
                            self.workflow(prediction_frame, wf['Next'], pfix, workflow_id, save_models=save_models)
                        except Exception as oexecution_error:
                            self._logging.log_info('gDayF', "Workflow", self._labels["failed_wf"], str(wf['Next']))

        controller.clean_handlers()
        del controller

    ## Method managing dataset load from datapath:
    # @param datapath String Path indicating file to be analyzed or Dataframe
    # @return  None, Dataframe if no load errors, Error Message/None if load errors
    def check_path(self, datapath):
        if isinstance(datapath, str):
            try:
                self._logging.log_info('gDayF', "Workflow", self._labels["input_param"], datapath)
                pd_dataset = inputHandlerCSV().inputCSV(filename=datapath)
                return None, pd_dataset.copy()
            except [IOError, OSError, JSONDecodeError]:
                self._logging.log_critical('gDayF', "Workflow", self._labels["failed_input"], datapath)
                return self._labels['failed_input'], None
        elif isinstance(datapath, DataFrame):
            self._logging.log_info('gDayF', "Controller", self._labels["input_param"], str(datapath.shape))
            return None, datapath
        else:
            self._logging.log_critical('gDayF', "Workflow", self._labels["failed_input"], datapath)
            return self._labels['failed_input'], None

    ## Method managing storage path
    # @param mode ['train','predict']
    # @param filename filename
    # @param filetype file type
    # @param workflow_id Workflow_file_name + workflow_timestamp execution
    # @return  None if no localfs primary path found . Abosulute path if true
    def storage_path(self, mode, filename, filetype, workflow_id):
        load_storage = StorageMetadata()

        for each_storage_type in load_storage.get_load_path():
            if each_storage_type['type'] == 'localfs':
                source_data = list()
                primary_path = self._config['storage'][each_storage_type['type']]['value']
                source_data.append(primary_path)
                source_data.append('/')
                source_data.append(self.user_id)
                source_data.append('/')
                source_data.append(workflow_id)
                source_data.append('/')
                source_data.append(self._config['common']['workflow_summary_dir'])
                source_data.append('/')
                source_data.append(mode)
                source_data.append('/')

                PersistenceHandler().mkdir(type=each_storage_type['type'],
                                           path=''.join(source_data),
                                           grants=self._config['storage']['grants'])
                source_data.append(filename)
                source_data.append('.' + filetype)

                return ''.join(source_data)
        return None

    ## Method replicate files from primery to others
    # @param mode ['train','predict']
    # @param filename filename
    # @param workflow_id Workflow_file_name + workflow_timestamp execution
    # @return  None if no localfs primary path found . Abosulute path if true
    def replicate_file(self, mode, filename, workflow_id):
        load_storage = StorageMetadata().get_json_path()
        persistence = PersistenceHandler()
        for each_storage_type in load_storage:
            if each_storage_type['type'] in ['localfs', 'hdfs']:
                source_data = list()
                primary_path = self._config['storage'][each_storage_type['type']]['value']
                source_data.append(primary_path)
                source_data.append('/')
                source_data.append(self.user_id)
                source_data.append('/')
                source_data.append(workflow_id)
                source_data.append('/')
                source_data.append(self._config['common']['workflow_summary_dir'])
                source_data.append('/')
                source_data.append(mode)
                source_data.append('/')

                '''if each_storage_type['type'] == 'hdfs':
                    source_data = self._config['storage'][each_storage_type['type']]['uri'] + ''.join(source_data)'''
                each_storage_type['value'] = ''.join(source_data)

                persistence.mkdir(type=each_storage_type['type'], path=each_storage_type['value'],
                                  grants=self._config['storage']['grants'])
                each_storage_type['value'] = each_storage_type['value'] + Path(filename).name

        persistence.store_file(storage_json=load_storage, filename=filename)
        del persistence


