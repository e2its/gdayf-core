from gdayf.handlers.h2ohandler import H2OHandler
from gdayf.handlers.inputhandler import inputHandlerCSV
from gdayf.common.dfmetada import DFMetada
from gdayf.core.adviserastar import AdviserAStar
from gdayf.logs.logshandler import LogsHandler
from gdayf.conf.loadconfig import LoadConfig
from gdayf.conf.loadconfig import LoadLabels
from gdayf.common.utils import get_model_fw
from gdayf.common.utils import get_model_ns
from gdayf.common.constants import *
from gdayf.common.utils import pandas_split_data
from gdayf.common.armetadata import ArMetadata
from gdayf.common.armetadata import deep_ordered_copy
from collections import OrderedDict
from pathlib import Path
from copy import deepcopy
from json import load, dumps
from json.decoder import JSONDecodeError


## Core class oriented to mange the comunication and execution messages pass for all components on system
# orchestrating the execution of actions activities (train and prediction) on especific frameworks
class Controller(object):

    ## Constructor
    def __init__(self):
        self._config = LoadConfig().get_config()
        self._labels = LoadLabels().get_config()['messages']['controller']
        self._logging = LogsHandler()
        self.analysis_list = OrderedDict()  # For future multi-analysis uses
        self.h2ohandler = None


    ## Method leading and controlling prediction's executions on all frameworks
    # @param self object pointer
    # @param datapath String Path indicating file to be analyzed
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
        elif model_file is not None:
            try:
                json_file = open(model_file)
                base_ar = deep_ordered_copy(load(json_file))
                json_file.close()
            except [IOError, OSError]:
                self._logging.log_exec('gDayF', "Controller", self._labels["failed_model"], model_file)
        try:
            pd_dataset = inputHandlerCSV().inputCSV(filename=datapath)
        except [IOError, OSError, JSONDecodeError]:
            self._logging.log_exec('gDayF', "Controller", self._labels["failed_input"], datapath)
            return self._labels['failed_input']

        if get_model_fw(base_ar) == 'h2o':
            self.h2ohandler = H2OHandler()
            if not self.h2ohandler.is_alive():
                self.h2ohandler.connect()
            prediction_frame, _ = self.h2ohandler.predict(predict_frame=pd_dataset, base_ar=base_ar)

            if self.h2ohandler is not None:
                self.h2ohandler.delete_h2oframes()
                del self.h2ohandler

        self._logging.log_exec('gDayF', 'controller', self._labels["pred_end"])

        return prediction_frame


    ## Method leading and controlling analysis's executions on all frameworks
    # @param self object pointer
    # @param datapath String Path indicating file to be analyzed
    # @param objective_column string indicating objective column
    # @param analysis_id main id code to be referenced on future predictions
    # @param amode Analysis mode of execution [0,1,2,3] [FAST, NORMAL, PARANOIAC, POC]
    # @param metric to evalute models ['accuracy', 'rmse', 'test_accuracy', 'combined']
    # @param deep_impact  deep analysis
    def exec_analysis(self, datapath, objective_column, amode=POC, metric='combined', deep_impact=0, analysis_id='not used'):
        self._logging.log_exec('gDayF', "Controller", self._labels["start"])
        self._logging.log_exec('gDayF', "Controller", self._labels["ana_param"], metric)
        self._logging.log_exec('gDayF', "Controller", self._labels["dep_param"], deep_impact)
        self._logging.log_exec('gDayF', "Controller", self._labels["ana_mode"], amode)
        self._logging.log_exec('gDayF', "Controller", self._labels["input_param"], datapath)

        try:
            pd_dataset = inputHandlerCSV().inputCSV(filename=datapath)
        except [IOError, OSError]:
            self._logging.log_exec('gDayF', "Controller", self._labels["failed_input"], datapath)
            return self._labels['failed_input']

        pd_test_dataset = None
        if metric == 'combined' or 'test_accuracy':
            pd_dataset, pd_test_dataset = pandas_split_data(pd_dataset)

        adviser = AdviserAStar(analysis_id=Path(datapath).name, metric=metric, deep_impact=deep_impact)
        df = DFMetada().getDataFrameMetadata(pd_dataset, 'pandas')

        adviser.set_recommendations(dataframe_metadata=df, objective_column=objective_column, atype=amode)

        while adviser.next_analysis_list is not None:
            for each_model in adviser.next_analysis_list:
                if get_model_fw(each_model) == 'h2o':
                    if self.h2ohandler is None:
                        self.h2ohandler = H2OHandler()
                    if not self.h2ohandler.is_alive():
                        self.h2ohandler.connect()
                    if pd_test_dataset is not None:
                        _, analyzed_model = self.h2ohandler.order_training(analysis_id=adviser.analysis_id,
                                                                           training_pframe=pd_dataset,
                                                                           base_ar=each_model,
                                                                           test_frame=pd_test_dataset)
                    else:
                        _, analyzed_model = self.h2ohandler.order_training(analysis_id=adviser.analysis_id,
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

            self._logging.log_exec(adviser.analysis_id, 'controller', self._labels["metric_order"],
                                   model['metrics']['accuracy'])
            self._logging.log_exec(adviser.analysis_id, 'controller', self._labels["round_reach"],
                                   model['round'])
        self._logging.log_exec(adviser.analysis_id, 'controller', self._labels["end"])

        if self.h2ohandler is not None:
            self.h2ohandler.delete_h2oframes()
            del self.h2ohandler

        return self._labels['success_op']

if __name__ == '__main__':
    source_data = list()
    source_data.append("D:/Dropbox/DayF/Technology/Python-DayF-adaptation-path/")
    source_data.append("Oreilly.Practical.Machine.Learning.with.H2O.149196460X/")
    source_data.append("CODE/h2o-bk/datasets/")
    source_data.append("ENB2012_data-Y1.csv")
    '''#Analysis
    controller = Controller()
    controller.exec_analysis(datapath=''.join(source_data), objective_column='Y2',
                             amode=FAST, metric='combined', deep_impact=3)'''
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


