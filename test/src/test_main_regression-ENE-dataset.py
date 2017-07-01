if __name__ == "__main__":

    from gdayf.handlers.h2ohandler import H2OHandler
    from gdayf.handlers.inputhandler import inputHandlerCSV
    from gdayf.common.dfmetada import DFMetada
    from gdayf.core.adviserastar import AdviserAStar
    import os
    from json import dumps
    from time import time

    source_data = list()
    source_data.append("D:/Dropbox/DayF/Technology/Python-DayF-adaptation-path/")
    source_data.append("Oreilly.Practical.Machine.Learning.with.H2O.149196460X/")
    source_data.append("CODE/h2o-bk/datasets/")
    source_data.append("ENB2012_data.csv")

    pd_train_dataset = inputHandlerCSV().inputCSV(filename=''.join(source_data))
    pd_train_dataset.drop('Y1', axis=1, inplace=True)
    pd_train_dataset, pd_test_dataset = pd_train_dataset.sample(frac=0.2)
    print('Training set dimensions:', pd_train_dataset.shape)

    #json_file = open(r'D:\e2its-dayf.svn\gdayf\branches\0.0.3-team03\test\json\ar-regression-ENE.json')
    analysis_id = 'PoC-regression-ENE'
    adviser = AdviserAStar(analysis_id=analysis_id, metric='accuracy')
    df = DFMetada().getDataFrameMetadata(pd_train_dataset, 'pandas')
    _, analysis_list = adviser.set_recommendations(dataframe_metadata=df, objective_column='Y2',
                                                             atype=adviser.POC)

    analysis_models = H2OHandler()
    analysis_results =list()
    for model in analysis_list:
        ana, lista =analysis_models.order_training(analysis_id=adviser.analysis_id,
                                                          training_frame=pd_train_dataset,
                                                          base_ar=model)
        analysis_results.append(lista)

    sorted_list = adviser.priorize_models(ana, analysis_results)

    for each_model in sorted_list:
        print(dumps(each_model, indent=4))
    '''for file in os.listdir(r'D:\Data\models\h2o\PoC-regression-ENE\train\1498066721.7367265\json'):
        analysis_models = H2OHandler()
        json_file = open(r'D:\Data\models\h2o\PoC-regression-ENE\train\1498066721.7367265\json' + '/' + file)
        analysis_results = analysis_models.predict(pd_test_dataset, json_file)'''



