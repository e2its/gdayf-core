if __name__ == "__main__":

    from gdayf.handlers.h2ohandler import H2OHandler
    from gdayf.handlers.inputhandler import inputHandlerCSV
    from gdayf.core.adviserastar import AdviserAStar
    from gdayf.common.dfmetada import DFMetada
    from pandas import DataFrame as DataFrame
    from pandas import concat as concat
    from time import time
    import os
    from json import dumps

    source_data = list()
    source_data.append("D:/Dropbox/DayF/Technology/Python-DayF-adaptation-path/")
    source_data.append("Oreilly.Practical.Machine.Learning.with.H2O.149196460X/")
    source_data.append("CODE/h2o-bk/datasets/")

    pd_train_dataset = concat([inputHandlerCSV().inputCSV(''.join(source_data) + "football.train2.csv"),
                              inputHandlerCSV().inputCSV(''.join(source_data) + "football.valid2.csv")],
                              axis=0)
    pd_test_dataset = inputHandlerCSV().inputCSV(''.join(source_data) + "football.test2.csv")

    print('Training set', pd_train_dataset.shape)
    print('Test set', pd_test_dataset.shape)

    # Binomial_test

    analysis_id = 'PoC-binomial-SoC'
    adviser = AdviserAStar(analysis_id=analysis_id, metric='accuracy')
    df = DFMetada().getDataFrameMetadata(pd_train_dataset, 'pandas')
    df = DFMetada().getDataFrameMetadata(pd_train_dataset, 'pandas')
    _, analysis_list = adviser.set_recommendations(dataframe_metadata=df, objective_column='HomeWin',
                                                             atype=adviser.FAST)

    analysis_models = H2OHandler()
    analysis_results = analysis_models.order_training(analysis_id=adviser.analysis_id,
                                                      training_frame=pd_train_dataset,
                                                      model=analysis_list)

    sorted_list = adviser.priorize_models(analysis_results[0], analysis_results[1])

    for each_model in sorted_list:
        print(dumps(each_model, indent=4))

    del analysis_models

    '''analysis_models = H2OHandler()
    for file in os.listdir(r'D:\Data\models\h2o\PoC-binomial-SOC\train\json'):
        json_file = open(r'D:\Data\models\h2o\PoC-binomial-SOC\train\json' + '/' + file)
        analysis_results = analysis_models.predict(pd_test_dataset, json_file)
    del analysis_models'''


