if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *

    source_data = list()
    source_data.append("/Data/Data/datasheets/Multinomial/PEM/")
    source_data.append("PE-MULTINOM-3.csv")
    #Analysis
    controller = Controller()
    if controller.config_checks():
        status, recomendations = controller.exec_analysis(datapath=''.join(source_data), objective_column='ACCION',
                                                          amode=FAST_PARANOIAC, metric='test_accuracy', deep_impact=5)

        controller.log_model_list(recomendations[0]['model_id'], recomendations, metric='test_accuracy', accuracy=True)

        '''controller.save_models(recomendations)'''
        controller.reconstruct_execution_tree(recomendations, metric='test_accuracy')
        controller.remove_models(recomendations, mode=EACH_BEST)

        #Prediction
        source_data = list()
        source_data.append("/Data/Data/datasheets/Multinomial/PEM/")
        source_data.append("PE-MULTINOM-3.csv")
        model_source = list()

        #controller = Controller()
        print(recomendations[0]['load_path'][0]['value'])
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

        controller.clean_handlers()
    del controller



