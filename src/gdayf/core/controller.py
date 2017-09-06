from gdayf.handlers.h2ohandler import H2OHandler
from gdayf.handlers.inputhandler import inputHandlerCSV
from gdayf.common.dfmetada import DFMetada

from gdayf.logs.logshandler import LogsHandler
from gdayf.conf.loadconfig import LoadConfig
from gdayf.conf.loadconfig import LoadLabels
from gdayf.common.utils import get_model_fw
from gdayf.common.constants import *
from gdayf.common.utils import pandas_split_data
from gdayf.common.armetadata import ArMetadata
from gdayf.common.armetadata import deep_ordered_copy
from gdayf.persistence.persistencehandler import get_ar_from_file
from gdayf.persistence.persistencehandler import PersistenceHandler
from gdayf.common.storagemetadata import StorageMetadata
from collections import OrderedDict
from pathlib import Path
from pandas import DataFrame
import importlib
from json import load
from json.decoder import JSONDecodeError


## Core class oriented to mange the comunication and execution messages pass for all components on system
# orchestrating the execution of actions activities (train and prediction) on especific frameworks
class Controller(object):

    ## Constructor
    def __init__(self, user_id='PoC-gDayF'):
        self._config = LoadConfig().get_config()
        self._labels = LoadLabels().get_config()['messages']['controller']
        self._logging = LogsHandler()
        self.analysis_list = OrderedDict()  # For future multi-analysis uses
        self.model_handler = OrderedDict()
        self.user_id = user_id
        self.adviser = importlib.import_module(self._config['optimizer']['adviser_classpath'])
        self._logging.log_exec('gDayF', "Controller", self._labels["loading_adviser"],
                               self._config['optimizer']['adviser_classpath'])


    ## Method leading and controlling prediction's executions on all frameworks
    # @param self object pointer
    # @param datapath String Path indicating file to be analyzed or Dataframe
    # @param armetadata
    # @param model_file String Path indicating model_file ArMetadata.json structure

    def exec_prediction(self, datapath, armetadata=None, model_file=None):

        self._logging.log_exec('gDayF', "Controller", self._labels["ana_mode"], 'prediction')
        if armetadata is None and model_file is None:
            self._logging.log_exec('gDayF', "Controller", self._labels["failed_model"], datapath)
            return self._labels["failed_model"]
        elif armetadata is not None:
            try:
                assert isinstance(armetadata, ArMetadata)
                base_ar = deep_ordered_copy(armetadata)
            except AssertionError:
                self._logging.log_exec('gDayF', "Controller", self._labels["failed_model"], armetadata)
                return self._labels["failed_model"]
        elif model_file is not None:
            try:
                #json_file = open(model_file)
                _, base_ar = get_ar_from_file(model_file)
                base_ar = deep_ordered_copy(base_ar)
                '''base_ar = deep_ordered_copy(load(json_file))
                json_file.close()'''
            except [IOError, OSError]:
                self._logging.log_exec('gDayF', "Controller", self._labels["failed_model"], model_file)
                return self._labels["failed_model"]

        if isinstance(datapath, str):
            try:
                self._logging.log_exec('gDayF', "Controller", self._labels["input_param"], datapath)
                pd_dataset = inputHandlerCSV().inputCSV(filename=datapath)
            except [IOError, OSError, JSONDecodeError]:
                self._logging.log_exec('gDayF', "Controller", self._labels["failed_input"], datapath)
                return self._labels['failed_input']
        elif isinstance(datapath, DataFrame):
            pd_dataset = datapath
            self._logging.log_exec('gDayF', "Controller", self._labels["input_param"], str(datapath.shape))
        else:
            self._logging.log_exec('gDayF', "Controller", self._labels["failed_input"], datapath)
            return self._labels['failed_input']

        fw = get_model_fw(base_ar)
        if fw == 'h2o':
            self.init_handler(fw)
            prediction_frame, _ = self.model_handler[fw]['handler'].predict(predict_frame=pd_dataset, base_ar=base_ar)
            self.clean_handler(fw)
        else:
            prediction_frame = None
                
        self._logging.log_exec('gDayF', 'controller', self._labels["pred_end"])

        return prediction_frame

    ## Method focus on cleaning handler objects
    # @param fw framework
    def clean_handler(self, fw):
        if self.model_handler[fw]['handler'] is not None:
            self.model_handler[fw]['handler'].delete_h2oframes()
            self.model_handler[fw]['handler'] = None

    ## Method oriented to init handler objects
    # @param fw framework
    def init_handler(self, fw):
        try:
            if self.model_handler[fw]['handler'] is None:
                self.model_handler[fw]['handler'] = H2OHandler()
        except KeyError:
            self.model_handler[fw] = OrderedDict()
            self.model_handler[fw]['handler'] = H2OHandler()
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
                    if fw == 'h2o':
                        H2OHandler().shutdown_cluster()
                        self._logging.log_exec('gDayF', "Controller", self._labels["shuttingdown"], fw)

    ## Method leading and controlling analysis's executions on all frameworks
    # @param self object pointer
    # @param datapath String Path indicating file to be analyzed
    # @param objective_column string indicating objective column
    # @param analysis_id main id code to be referenced on future predictions
    # @param amode Analysis mode of execution [0,1,2,3] [FAST, NORMAL, PARANOIAC, POC]
    # @param metric to evalute models ['accuracy', 'rmse', 'test_accuracy', 'combined']
    # @param deep_impact  deep analysis
    # @return status, adviser.analysis_recommendation_order
    def exec_sanalysis(self, datapath, objective_column, amode=POC, metric='combined', deep_impact=0, analysis_id='N/A'):
        self._logging.log_exec('gDayF', "Controller", self._labels["start"])
        self._logging.log_exec('gDayF', "Controller", self._labels["ana_param"], metric)
        self._logging.log_exec('gDayF', "Controller", self._labels["dep_param"], deep_impact)
        self._logging.log_exec('gDayF', "Controller", self._labels["ana_mode"], amode)


        if isinstance(datapath, str):
            try:
                self._logging.log_exec('gDayF', "Controller", self._labels["input_param"], datapath)
                pd_dataset = inputHandlerCSV().inputCSV(filename=datapath)
                id_datapath = Path(datapath).name
            except IOError:
                self._logging.log_exec('gDayF', "Controller", self._labels["failed_input"], datapath)
                return self._labels['failed_input']
            except OSError:
                self._logging.log_exec('gDayF', "Controller", self._labels["failed_input"], datapath)
                return self._labels['failed_input']
            except JSONDecodeError:
                self._logging.log_exec('gDayF', "Controller", self._labels["failed_input"], datapath)
                return self._labels['failed_input']
        elif isinstance(datapath, DataFrame):
            self._logging.log_exec('gDayF', "Controller", self._labels["input_param"], str(datapath.shape) )
            pd_dataset = datapath
            id_datapath = 'Dataframe' + \
                          '_' +str(pd_dataset.size) + \
                          '_' + str(pd_dataset.shape[0]) + \
                          '_' + str(pd_dataset.shape[1])
        else:
            self._logging.log_exec('gDayF', "Controller", self._labels["failed_input"], datapath)
            return self._labels['failed_input']

        pd_test_dataset = None
        if metric == 'combined' or 'test_accuracy':
            pd_dataset, pd_test_dataset = pandas_split_data(pd_dataset)

        adviser = self.adviser.AdviserAStar(analysis_id=self.user_id + '_' + id_datapath, metric=metric, deep_impact=deep_impact)
        df = DFMetada().getDataFrameMetadata(pd_dataset, 'pandas')

        adviser.set_recommendations(dataframe_metadata=df, objective_column=objective_column, atype=amode)

        while adviser.next_analysis_list is not None:
            for each_model in adviser.next_analysis_list:
                fw = get_model_fw(each_model)
                if get_model_fw(each_model) == 'h2o':
                    self.init_handler(fw)

                if pd_test_dataset is not None:
                    _, analyzed_model = self.model_handler['h2o']['handler'].order_training(analysis_id=adviser.analysis_id,
                                                                                            training_pframe=pd_dataset,
                                                                                            base_ar=each_model,
                                                                                            test_frame=pd_test_dataset)
                else:
                    _, analyzed_model = self.model_handler['h2o']['handler'].order_training(analysis_id=adviser.analysis_id,
                                                                                            training_frame=pd_dataset,
                                                                                            base_ar=each_model)

                if analyzed_model is not None:
                    adviser.analysis_recommendation_order.append(analyzed_model)
            adviser.next_analysis_list.clear()
            adviser.analysis_recommendation_order = adviser.priorize_models(analysis_id=adviser.analysis_id,
                                                                            model_list=
                                                                            adviser.analysis_recommendation_order)
            adviser.set_recommendations(dataframe_metadata=df, objective_column=objective_column, atype=amode)

        self._logging.log_exec(adviser.analysis_id, 'controller',
                               self._labels["ana_models"], str(len(adviser.analyzed_models)))
        self._logging.log_exec(adviser.analysis_id, 'controller',
                               self._labels["exc_models"], str(len(adviser.excluded_models)))
        best_check = True
        for model in adviser.analysis_recommendation_order:
            if best_check:
                self._logging.log_exec(adviser.analysis_id, 'controller', self._labels["best_model"],
                                    model['model_parameters'][get_model_fw(model)]['parameters']['model_id']['value'])
                best_check = False
            else:
                self._logging.log_exec(adviser.analysis_id, 'controller', self._labels["res_model"],
                                    model['model_parameters'][get_model_fw(model)]['parameters']['model_id']['value'])

            self._logging.log_exec(adviser.analysis_id, 'controller', self._labels["round_reach"], model['round'])
            if model["normalizations_set"] is None:
                self._logging.log_exec(adviser.analysis_id, 'controller', self._labels["norm_app"], [])
            else:
                self._logging.log_exec(adviser.analysis_id, 'controller', self._labels["norm_app"],
                                       model["normalizations_set"])
            self._logging.log_exec(adviser.analysis_id, 'controller', self._labels["ametric_order"],
                                   model['metrics']['accuracy'])
            self._logging.log_exec(adviser.analysis_id, 'controller', self._labels["pmetric_order"],
                                   model['metrics']['execution']['train']['RMSE'])


        self._logging.log_exec(adviser.analysis_id, 'controller', self._labels["end"])

        self.clean_handlers()

        return self._labels['success_op'], adviser.analysis_recommendation_order

    ## Method leading and controlling coversion to java model
    # @param self object pointer
    # @param armetadata Armetada object
    # @param type base type if is possible
    # @return download_path, hash MD5 key
    def get_java_model(self, armetadata, type='pojo'):
        fw = get_model_fw(armetadata)
        if fw == 'h2o':
            self.init_handler(fw)
            results = self.model_handler[fw]['handler'].get_java_model(armetadata, type)
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
                    exclusion.append((get_model_fw(model),model['model_parameters'][get_model_fw(model)]['model'],
                                      model['normalizations_set'])
                                     )
        elif mode == ALL:
            model_list = arlist
        for fw in self._config['frameworks'].keys():
                self.init_handler(fw)
                for each_model in model_list:
                    if fw in each_model['model_parameters'].keys():
                        results = self.model_handler[fw]['handler'].store_model(each_model)
                self.clean_handler(fw)

    ## Method leading and controlling model removing from server
    # @param self object pointer
    # @param mode [BEST, BEST_3, EACH_BEST, ALL]
    # @param  arlist List of armetadata

    def remove_models(self, arlist, mode=ALL):
        if mode == BEST:
            model_list = arlist[1:]
        elif mode == BEST_3:
            model_list = arlist[3:]
        elif mode == ALL:
            model_list = arlist
        for fw in self._config['frameworks'].keys():
            self.init_handler(fw)
            results = self.model_handler[fw]['handler'].remove_models(model_list)
            self.clean_handler(fw)

    ##Method oriented to generate execution tree for visualizations and analysis issues
    # @param arlist Priorized ArMetadata list
    # @param  metric ['accuracy', 'combined', 'test_accuracy', 'rmse']
    # @return OrderedDict() with execution tree data Analysis
    def reconstruct_execution_tree(self, arlist=None, metric='combined'):
        if arlist is None or len(arlist) == 0:
            self._logging.log_exec('gDayF', 'controller', self._labels["failed_model"])
            return None
        else:
            analysis_id = arlist[0]['model_id']

        adviser = self.adviser.AdviserAStar(analysis_id=analysis_id, metric=metric)
        ordered_list = adviser.priorize_models(adviser.analysis_id, arlist)

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
                    self._logging.log_debug(adviser.analysis_id, 'controller', self._labels['fail_reconstruct'],
                                            model_id)
            level += 1

        #Store_json o primary path
        primary_path = self._config['storage']['primary_path']
        fstype = self._config['storage'][primary_path]['type']

        datafile = list()
        datafile.append(self._config['storage'][primary_path]['value'])
        datafile.append('/')
        datafile.append(analysis_id)
        datafile.append('/')
        datafile.append('Execution_tree_')
        datafile.append(analysis_id)
        datafile.append('.json')

        if self._config['persistence']['compress_json']:
            datafile.append('.gz')

        storage = StorageMetadata()
        storage.append(value=''.join(datafile), fstype=fstype)
        print(storage)
        PersistenceHandler().store_json(storage, root)
        return root


if __name__ == '__main__':
    source_data = list()
    source_data.append("D:/Dropbox/DayF/Technology/Python-DayF-adaptation-path/")
    source_data.append("Oreilly.Practical.Machine.Learning.with.H2O.149196460X/")
    source_data.append("CODE/h2o-bk/datasets/")
    source_data.append("ENB2012_data-Y1.csv")
    #Analysis
    controller = Controller()
    controller.exec_sanalysis(datapath=''.join(source_data), objective_column='Y2',
                              amode=FAST, metric='combined', deep_impact=3)

    #Prediction
    source_data = list()
    source_data.append("D:/Dropbox/DayF/Technology/Python-DayF-adaptation-path/")
    source_data.append("Oreilly.Practical.Machine.Learning.with.H2O.149196460X/")
    source_data.append("CODE/h2o-bk/datasets/")
    source_data.append("ENB2012_data-Y1.csv")
    model_source = list()
    model_source.append("D:/Data/models/h2o/ENB2012_data-Y1.csv_1499277062.9702203/train/1499277062.9702203/replica/json/")
    model_source.append('H2OGradientBoostingEstimator_1499277177.3076756.json')
    controller = Controller()
    prediction_frame = controller.exec_prediction(datapath=''.join(source_data), model_file=''.join(model_source))
    print(prediction_frame)

    # Save Pojo
    model_source = list()
    model_source.append("D:/Data/models/h2o/ENB2012_data-Y1.csv_1499277062.9702203/train/1499277062.9702203/replica/json/")
    model_source.append('H2OGradientBoostingEstimator_1499277177.3076756.json')
    controller = Controller()
    fp = open(''.join(model_source))
    result = controller.get_java_model(load(fp, object_hook=OrderedDict), 'pojo')
    fp.close()
    print(result)

    # Save Mojo
    model_source = list()
    model_source.append("D:/Data/models/h2o/ENB2012_data-Y1.csv_1499277062.9702203/train/1499277062.9702203/replica/json/")
    model_source.append('H2OGradientBoostingEstimator_1499277177.3076756.json')
    controller = Controller()
    fp = open(''.join(model_source))
    result = controller.get_java_model(load(fp, object_hook=OrderedDict), 'mojo')
    fp.close()
    print(result)




