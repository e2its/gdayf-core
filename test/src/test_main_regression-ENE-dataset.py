if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *
    from pandas import set_option
    from gdayf.common.dataload import DataLoad

    #Analysis
    controller = Controller()
    if controller.config_checks():
        data_train, data_test = DataLoad().enb()
        status, recomendations = controller.exec_analysis(datapath=data_train, objective_column='Y2',
                                                          amode=FAST, metric='train_accuracy', deep_impact=5)

        controller.reconstruct_execution_tree(metric='train_accuracy', store=True)
        controller.remove_models(arlist=recomendations, mode=EACH_BEST)

        set_option('display.max_rows', 500)
        set_option('display.max_columns', 50)
        set_option('display.max_colwidth', 100)
        set_option('display.precision', 4)
        set_option('display.width', 1024)

        #Prediction
        print('Starting Prediction\'s Phase')

        prediction_frame = controller.exec_prediction(datapath=data_test,
                                                      model_file=recomendations[0]['json_path'][0]['value'])
        if 'predict' in prediction_frame.columns.values:
            print(prediction_frame[['Y2', 'predict']])

        # Save Pojo
        '''
        controller = Controller()
        result = controller.get_external_model(recomendations[0], 'pojo')

        # Save Mojo
        controller = Controller()
        result = controller.get_external_model(recomendations[0], 'mojo')
        '''

        print(controller.table_model_list(ar_list=recomendations, metric='test_r2'))
        controller.clean_handlers()

    del controller



