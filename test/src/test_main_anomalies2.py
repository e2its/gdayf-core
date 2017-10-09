if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *
    from pprint import pprint

    source_data = list()
    source_data.append("/Data/Data/datasheets/Anomalies/CCPP/")
    source_data.append("CPP.csv")
    #Analysis
    controller = Controller()
    if controller.config_checks():
        status, recomendations = controller.exec_analysis(datapath=''.join(source_data), objective_column=None,
                                                          amode=ANOMALIES, metric='rmse', deep_impact=5)

        controller.save_models(recomendations, mode=BEST)
        controller.reconstruct_execution_tree(recomendations, metric='rmse')
        controller.remove_models(recomendations, mode=ALL)

        #Prediction
        source_data = list()
        source_data.append("/Data/Data/datasheets/Anomalies/CCPP/")
        source_data.append("CPP_modificado.csv")

        #controller = Controller()
        prediction_frame = controller.exec_prediction(datapath=''.join(source_data),
                                                      model_file=recomendations[0]['json_path'][0]['value'])
        pprint(prediction_frame)

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