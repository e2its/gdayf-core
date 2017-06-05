if __name__ == "__main__":

    from gdayf.handlers.h2ohandler import H2OHandler
    from gdayf.handlers.inputhandler import inputHandlerCSV
    import os
    from time import time

    source_data = list()
    source_data.append("D:/Dropbox/DayF/Technology/Python-DayF-adaptation-path/")
    source_data.append("Oreilly.Practical.Machine.Learning.with.H2O.149196460X/")
    source_data.append("CODE/h2o-bk/datasets/")
    source_data.append("ENB2012_data.csv")

    pd_train_dataset = inputHandlerCSV().inputCSV(filename=''.join(source_data))
    print('Training set dimensions:', pd_train_dataset.shape)

    json_file = open(r'D:\e2its-dayf.svn\gdayf\branches\0.0.3-team03\test\json\ar-regression-ENE.json')
    analysis_list = [(json_file, None)]

    print(analysis_list)

    analysis_models = H2OHandler()
    analysis_results = analysis_models.order_training(analysis_id='PoC_regression-ENE' + str(time()),
                                                      training_frame=pd_train_dataset,
                                                      analysis_list=analysis_list)

    for file in os.listdir(r'D:\Data\models\h2o\PoC-regression-ENE\train\json'):
        analysis_models = H2OHandler()
        json_file = open(r'D:\Data\models\h2o\PoC-regression-ENE\train\json' + '/' + file)
        analysis_results = analysis_models.predict(pd_train_dataset, json_file)



