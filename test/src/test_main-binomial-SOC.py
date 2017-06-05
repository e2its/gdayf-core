if __name__ == "__main__":

    from gdayf.handlers.h2ohandler import H2OHandler
    from gdayf.handlers.inputhandler import inputHandlerCSV
    from pandas import DataFrame as DataFrame
    from pandas import concat as concat
    import os

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
    json_file = open(r'D:\e2its-dayf.svn\gdayf\branches\0.0.3-team03\test\json\ar-binomial-SOC.json')
    analysis_list = [(json_file, None)]

    analysis_models = H2OHandler()
    analysis_results = analysis_models.order_training(analysis_id='PoC_binomial', training_frame=pd_train_dataset,
                                                      analysis_list=analysis_list)
    del analysis_models

    analysis_models = H2OHandler()
    for file in os.listdir(r'D:\Data\models\h2o\PoC-binomial\train\json'):
        json_file = open(r'D:\Data\models\h2o\PoC-binomial\train\json' + '/' + file)
        analysis_results = analysis_models.predict(pd_test_dataset, json_file)
    del analysis_models


