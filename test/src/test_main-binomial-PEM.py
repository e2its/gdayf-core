if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *
    from pandas import set_option

    source_data = list()
    source_data.append("/Data/Data/datasheets/binary/PEM/")
    source_data.append("PE-BINARY.csv")
    #Analysis
    controller = Controller()
    if controller.config_checks():
        status, recomendations = controller.exec_analysis(datapath=''.join(source_data), objective_column='ACCION',
                                                          amode=FAST, metric='test_accuracy', deep_impact=3)

        controller.reconstruct_execution_tree(metric='test_accuracy', store=True)
        controller.remove_models(arlist=recomendations, mode=EACH_BEST)


        #Prediction
        source_data = list()
        source_data.append("/Data/Data/datasheets/binary/PEM/")
        source_data.append("PE-BINARY.csv")
        model_source = list()

        # controller = Controller()
        prediction_frame = controller.exec_prediction(datapath=''.join(source_data),
                                                      model_file=recomendations[0]['json_path'][0]['value'])
        if 'predict' in prediction_frame.columns.values:
            print(prediction_frame[['ACCION', 'predict']])
        elif 'prediction' in prediction_frame.columns.values:
            print(prediction_frame[['ACCION', 'prediction']])

        # Save Pojo
        controller = Controller()
        result = controller.get_external_model(recomendations[0], 'pojo')

        # Save Mojo
        controller = Controller()
        result = controller.get_external_model(recomendations[0], 'mojo')

        set_option('display.height', 1000)
        set_option('display.max_rows', 500)
        set_option('display.max_columns', 500)
        set_option('display.width', 1000)

        print(controller.table_model_list(ar_list=recomendations, metric='test_accuracy'))
        controller.clean_handlers()
    del controller

