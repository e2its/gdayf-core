if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *
    from pandas import set_option

    source_data = list()
    source_data.append("/Data/Data/datasheets/regression/ENB2012/")
    source_data.append("ENB2012_data-Y1.csv")
    #Analysis
    controller = Controller()
    if controller.config_checks():
        status, recomendations = controller.exec_analysis(datapath=''.join(source_data), objective_column='Y2',
                                                          amode=FAST, metric='test_rmse', deep_impact=4)

        '''controller.log_model_list(recomendations[0]['model_id'], recomendations, metric='train_rmse')
        controller.save_models(recomendations, mode=BEST)'''
        controller.reconstruct_execution_tree(metric='train_rmse', store=True,
                                              experiment=recomendations[0]['model_id'],
                                              user=controller.user_id)
        controller.remove_models(recomendations, mode=EACH_BEST)


        #Prediction
        print('Starting Prediction\'s Phase')
        source_data = list()
        source_data.append("/Data/Data/datasheets/regression/ENB2012/")
        source_data.append("ENB2012_data-Y1.csv")

        #controller = Controller()
        prediction_frame = controller.exec_prediction(datapath=''.join(source_data),
                                                      model_file=recomendations[0]['json_path'][0]['value'])
        if 'predict' in prediction_frame.columns.values:
            print(prediction_frame[['Y2', 'predict']])
        elif 'prediction' in prediction_frame.columns.values:
            print(prediction_frame[['Y2', 'prediction']])

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

        print(controller.table_model_list(recomendations[0]['model_id'], recomendations, metric='test_rmse'))
        controller.clean_handlers()
    del controller


