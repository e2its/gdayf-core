if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.handlers.inputhandler import inputHandlerCSV
    from gdayf.common.constants import *
    from pandas import set_option

    source_data = list()
    source_data.append("/Data/Data/datasheets/regression/DM-Metric/")
    source_data.append("DM-Metric-missing-3.csv")

    #Generating missing values

    pd_dataset = inputHandlerCSV().inputCSV(filename=''.join(source_data))
    #Analysis
    controller = Controller()
    if controller.config_checks():
        status, recomendations = controller.exec_analysis(datapath=pd_dataset,
                                                          objective_column='Weather_Temperature',
                                                          amode=POC, metric='test_rmse', deep_impact=2)

        #controller.save_models(recomendations, mode=EACH_BEST)
        controller.reconstruct_execution_tree(arlist=None, metric='test_rmse', store=True)
        controller.remove_models(recomendations, mode=EACH_BEST)

        #Prediction
        source_data = list()
        source_data.append("/Data/Data/datasheets/regression/DM-Metric/")
        source_data.append("DM-Metric-missing-test-3.csv")
        #source_data.append("DM-Metric-missing-test-weather.csv")

        set_option('display.height', 1000)
        set_option('display.max_rows', 500)
        set_option('display.max_columns', 500)
        set_option('display.width', 1000)

        #Prediccion
        prediction_frame = controller.exec_prediction(datapath=''.join(source_data),
                                                      model_file=recomendations[0]['json_path'][0]['value'])
        if 'predict' in prediction_frame.columns.values:
            print(prediction_frame[['Weather_Temperature', 'predict']])
        elif 'prediction' in prediction_frame.columns.values:
            print(prediction_frame[['Weather_Temperature', 'prediction']])

        # Save Pojo
        #result = controller.get_external_model(recomendations[0], 'pojo')

        # Save Mojo
        #result = controller.get_external_model(recomendations[0], 'mojo')

        #controller.log_model_list(recomendations[0]['model_id'], recomendations, metric='test_accuracy')
        print(controller.table_model_list(ar_list=recomendations, metric='test_accuracy'))

        controller.clean_handlers()
    del controller
