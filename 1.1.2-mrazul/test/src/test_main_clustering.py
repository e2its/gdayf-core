if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *

    source_data = list()
    source_data.append("/Data/Data/datasheets/Anomalies/CCPP/")
    source_data.append("CPP_base_ampliado.csv")
    #Analysis
    controller = Controller()
    if controller.config_checks():
        status, recomendations = controller.exec_analysis(datapath=''.join(source_data), objective_column=None,
                                                          amode=CLUSTERING, metric='cdistance', deep_impact=10,
                                                          k=6, estimate_k=True)

        controller.log_model_list(recomendations[0]['model_id'], recomendations, metric='cdistance', accuracy=False)
        '''controller.save_models(recomendations, mode=EACH_BEST)'''
        controller.reconstruct_execution_tree(recomendations, metric='cdistance')
        controller.remove_models(recomendations, mode=EACH_BEST)

        #Prediction
        source_data = list()
        source_data.append("/Data/Data/datasheets/Anomalies/CCPP/")
        source_data.append("CPP_base_ampliado.csv")

        #controller = Controller()
        prediction_frame = controller.exec_prediction(datapath=''.join(source_data),
                                                      model_file=recomendations[0]['json_path'][0]['value'])
        print(prediction_frame)

        # Save Pojo
        #controller = Controller()
        result = controller.get_external_model(recomendations[0], 'pojo')

        # Save Mojo
        #controller = Controller()
        result = controller.get_external_model(recomendations[0], 'mojo')

        controller.clean_handlers()
    del controller