if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *
    from pprint import pprint

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

        controller.log_model_list(recomendations[0]['model_id'], recomendations, metric='combined_accuracy', accuracy=True)
        #controller.save_models(recomendations, mode=EACH_BEST)
        execution_tree = controller.reconstruct_execution_tree(arlist=None, metric='combined_accuracy', store=False,
                                                               user=controller.user_id,
                                                               experiment=recomendations[0]['model_id'])
        controller.remove_models(recomendations, mode=EACH_BEST)

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
        result = controller.get_external_model(recomendations[0], 'pojo')
        print(result)

        # Save Mojo
        result = controller.get_external_model(recomendations[0], 'mojo')
        print(result)

        controller.clean_handlers()
    del controller
