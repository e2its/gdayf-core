if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *
    from pandas import set_option
    from gdayf.common.dataload import DataLoad

    #Analysis
    controller = Controller()
    if controller.config_checks():
        data_train, data_test = DataLoad().dm()
        status, recomendations = controller.exec_analysis(datapath=data_train, objective_column=None,
                                                          amode=CLUSTERING, metric='cdistance', deep_impact=4,
                                                          k=8, estimate_k=False)

        controller.reconstruct_execution_tree(recomendations, metric='cdistance')
        controller.remove_models(recomendations, mode=EACH_BEST)

        set_option('display.max_rows', 500)
        set_option('display.max_columns', 50)
        set_option('display.max_colwidth', 100)
        set_option('display.precision', 4)
        set_option('display.width', 1024)

        #Prediction
        print('Starting Prediction\'s Phase')
        prediction_frame = controller.exec_prediction(datapath=data_test,
                                                      model_file=recomendations[0]['json_path'][0]['value'])
        print(prediction_frame)

        '''# Save Pojo
        #controller = Controller()
        result = controller.get_external_model(recomendations[0], 'pojo')

        # Save Mojo
        #controller = Controller()
        result = controller.get_external_model(recomendations[0], 'mojo')
        '''
        print(controller.table_model_list(ar_list=recomendations, metric='cdistance'))

        controller.clean_handlers()

    del controller