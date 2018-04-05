if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *
    from pprint import pprint

    source_data = list()
    source_data.append("/Data/Data/datasheets/regression/DM-Metric/")
    source_data.append("DM-Metric-missing-3.csv")
    #Analysis
    controller = Controller()
    if controller.config_checks():
        status, recomendations = controller.exec_analysis(datapath=''.join(source_data), objective_column=None,
                                                          amode=ANOMALIES, metric='rmse', deep_impact=7)

        controller.log_model_list(recomendations[0]['model_id'], recomendations, metric='train_rmse', accuracy=True)
        '''controller.save_models(recomendations, mode=BEST)'''
        controller.reconstruct_execution_tree(recomendations, metric='rmse')
        controller.remove_models(recomendations, mode=BEST)

        #Prediction
        source_data = list()
        source_data.append("/Data/Data/datasheets/regression/DM-Metric/")
        source_data.append("DM-Metric-missing-test-3.csv")

        #controller = Controller()
        prediction_frame = controller.exec_prediction(datapath=''.join(source_data),
                                                      model_file=recomendations[0]['json_path'][0]['value'])
        pprint(prediction_frame)

        # Save Pojo
        #controller = Controller()
        result = controller.get_external_model(recomendations[0], 'pojo')

        # Save Mojo
        #controller = Controller()
        result = controller.get_external_model(recomendations[0], 'mojo')

        controller.clean_handlers()
    del controller