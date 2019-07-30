if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *
    from pandas import set_option

    source_data = list()
    source_data.append("/Data/Data/datasheets/Multinomial/ARM/")
    source_data.append("ARM-Metric-train-TS.csv")

    # Generating missing values


    # Analysis
    controller = Controller()
    if controller.config_checks():
        status, recomendations = controller.exec_analysis(datapath=''.join(source_data),
                                                          objective_column='ATYPE',
                                                          amode=FAST, metric='combined_accuracy', deep_impact=3)

        controller.reconstruct_execution_tree(metric='test_accuracy', store=True)
        controller.remove_models(arlist=recomendations, mode=EACH_BEST)

        # Prediction
        source_data = list()
        source_data.append("/Data/Data/datasheets/Multinomial/ARM/")
        source_data.append("ARM-Metric-test-TS.csv")
        # source_data.append("DM-Metric-missing-test-weather.csv")

        # Prediccion
        prediction_frame = controller.exec_prediction(datapath=''.join(source_data),
                                                      model_file=recomendations[0]['json_path'][0]['value'])
        if 'predict' in prediction_frame.columns.values:
            print(prediction_frame[['ATYPE', 'predict']])
        elif 'prediction' in prediction_frame.columns.values:
            print(prediction_frame[['ATYPE', 'prediction']])

        # Save Pojo
        controller = Controller()
        result = controller.get_external_model(recomendations[0], 'pojo')

        # Save Mojo
        controller = Controller()
        result = controller.get_external_model(recomendations[0], 'mojo')

        set_option('display.max_rows', 500)
        set_option('display.max_columns', 5000)

        print(controller.table_model_list(ar_list=recomendations, metric='combined_accuracy'))
        controller.clean_handlers()
    del controller

