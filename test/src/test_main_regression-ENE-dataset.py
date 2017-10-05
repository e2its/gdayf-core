if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *
    from copy import deepcopy

    source_data = list()
    source_data.append("/Data/Data/datasheets/regression/ENB2012/")
    source_data.append("ENB2012_data-Y1.csv")
    #Analysis
    controller = Controller()
    if controller.config_checks():
        status, recomendations = controller.exec_analysis(datapath=''.join(source_data), objective_column='Y2',
                                                          amode=FAST_PARANOIAC, metric='rmse', deep_impact=5)

        controller.log_model_list(recomendations[0]['model_id'], recomendations, metric='combined', accuracy=True)

        controller.save_models(recomendations, mode=BEST_3)
        '''status, recomendations2 = controller.exec_sanalysis(datapath=''.join(source_data),
                                                            list_ar_metadata=recomendations[-4:-2],
                                                            metric='rmse', deep_impact=1)
    
        recomendations.extend(recomendations2)'''
        controller.reconstruct_execution_tree(recomendations, metric='rmse')
        controller.remove_models(recomendations, mode=ALL)


        #Prediction
        source_data = list()
        source_data.append("/Data/Data/datasheets/regression/ENB2012/")
        source_data.append("ENB2012_data-Y1.csv")

        #controller = Controller()
        prediction_frame = controller.exec_prediction(datapath=''.join(source_data),
                                                      model_file=recomendations[0]['json_path'][0]['value'])
        print(prediction_frame)

        # Save Pojo
        #controller = Controller()
        result = controller.get_java_model(recomendations[0], 'pojo')
        print(result)

        # Save Mojo
        #controller = Controller()
        result = controller.get_java_model(recomendations[0], 'mojo')
        print(result)

        controller.remove_models(recomendations, mode=ALL)
        controller.clean_handlers()
    del controller



