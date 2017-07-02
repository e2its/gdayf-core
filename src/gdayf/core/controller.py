from gdayf.handlers.h2ohandler import H2OHandler
from gdayf.handlers.inputhandler import inputHandlerCSV
from gdayf.common.dfmetada import DFMetada
from gdayf.core.adviserastar import AdviserAStar
from gdayf.logs.logshandler import LogsHandler
from gdayf.conf.loadconfig import LoadConfig
from gdayf.common.utils import get_model_fw
from gdayf.common.utils import get_model_ns
from gdayf.common.utils import pandas_split_data
from collections import OrderedDict
from pathlib import Path


## Core class oriented to mange the comunication and execution messages pass for all components on system
# orchestrating the execution of actions activities (train and prediction) on especific frameworks
class Controller(object):

    ## Constructor
    def __init__(self):
        self._config = LoadConfig().get_config()
        self.logger = LogsHandler(__name__)
        self.analysis_list = OrderedDict()  # For future multi-analysis uses


    ## Method leading and controlling analysis's executions on all frameworks
    # @param self object pointer
    # @param datapath String Path indicating file to be analyzed
    # @param objective_column string indicating objective column
    # @param analysis_id main id code to be referenced on future predictions
    # @param amode Analysis mode of execution [0,1,2,3] [FAST, NORMAL, PARANOIAC, POC]
    # @param metric to evalute models ['accuracy', 'rmse', 'test_accuracy', 'combined']
    # @param deep_impact  deep analysis
    def exec_analysis (self, datapath, objective_column, amode=0, metric='combined', deep_impact=2, analysis_id='not used'):
        pd_dataset = inputHandlerCSV().inputCSV(filename=datapath)
        pd_test_dataset = None
        if metric == 'combined' or 'test_accuracy':
            pd_dataset, pd_test_dataset = pandas_split_data(pd_dataset)

        adviser = AdviserAStar(analysis_id=Path(datapath).name, metric=metric, deep_impact=deep_impact)
        df = DFMetada().getDataFrameMetadata(pd_dataset, 'pandas')

        adviser.set_recommendations(dataframe_metadata=df, objective_column=objective_column, atype=amode)

        while adviser.next_analysis_list is not None and amode != adviser.POC:
            for each_model in adviser.next_analysis_list:
                if get_model_fw(each_model) == 'h2o':
                    h2ohadler = H2OHandler()
                    if pd_test_dataset is not None:
                        _, analyzed_model = h2ohadler.order_training(analysis_id=adviser.analysis_id,
                                                                     training_frame=pd_dataset,
                                                                     base_ar=each_model,
                                                                     test_frame=pd_test_dataset)
                    else:
                        _, analyzed_model = h2ohadler.order_training(analysis_id=adviser.analysis_id,
                                                                     training_frame=pd_dataset,
                                                                     base_ar=each_model)

                    if analyzed_model is not None:
                        adviser.analysis_recommendation_order.append(analyzed_model)
            adviser.next_analysis_list.clear()
            adviser.analysis_recommendation_order = adviser.priorize_models(analysis_id=adviser.analysis_id,
                                                                            model_list=
                                                                            adviser.analysis_recommendation_order)
            adviser.set_recommendations(dataframe_metadata=df, objective_column=objective_column, atype=amode)

        print('Modelos_Analizados' + str(len(adviser.analyzed_models)))
        print('Modelos_Excluidos' + str(len(adviser.excluded_models)))
        for model in adviser.analysis_recommendation_order:
            print('ACCURACY ORDER (%s): %s'%(model['round'],
                                             model['model_parameters'][get_model_fw(model)]\
                                                  ['parameters']['model_id']['value']))
            print(model['metrics']['accuracy'])

if __name__ == '__main__':
    source_data = list()
    source_data.append("D:/Dropbox/DayF/Technology/Python-DayF-adaptation-path/")
    source_data.append("Oreilly.Practical.Machine.Learning.with.H2O.149196460X/")
    source_data.append("CODE/h2o-bk/datasets/")
    source_data.append("ENB2012_data-Y1.csv")

    controller = Controller()
    controller.exec_analysis(datapath=''.join(source_data), objective_column='Y2',
                             amode=0, metric='combined', deep_impact=3)
