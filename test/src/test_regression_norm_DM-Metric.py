if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *
    from pprint import pprint

    source_data = list()
    source_data.append("/Data/Data/datasheets/regression/DM-Metric/")
    source_data.append("DM-Metric-missing-3.csv")

    #Generating missing values


    #Analysis
    controller = Controller()
    if controller.config_checks():
        status, recomendations = controller.exec_analysis(datapath=''.join(source_data),
                                                          objective_column='Weather_Temperature',
                                                          amode=FAST, metric='rmse', deep_impact=7)

        controller.log_model_list(recomendations[0]['model_id'], recomendations, metric='rmse', accuracy=True)
        '''controller.save_models(recomendations, mode=EACH_BEST)'''
        controller.reconstruct_execution_tree(arlist=None, metric='rmse', store=True,
                                              user=controller.user_id,
                                              experiment=recomendations[0]['model_id'])
        controller.remove_models(recomendations, mode=BEST)

        #Prediction
        source_data = list()
        source_data.append("/Data/Data/datasheets/regression/DM-Metric/")
        source_data.append("DM-Metric-missing-test-3.csv")
        #source_data.append("DM-Metric-missing-test-weather.csv")

        #Prediccion
        prediction_frame = controller.exec_prediction(datapath=''.join(source_data),
                                                      model_file=recomendations[0]['json_path'][0]['value'])
        if 'predict' in prediction_frame.columns.values:
            pprint(prediction_frame[['Weather_Temperature', 'predict']])
        elif 'prediction' in prediction_frame.columns.values:
            pprint(prediction_frame[['Weather_Temperature', 'prediction']])

        # Save Pojo
        result = controller.get_external_model(recomendations[0], 'pojo')

        # Save Mojo
        result = controller.get_external_model(recomendations[0], 'mojo')

        controller.clean_handlers()
    del controller
