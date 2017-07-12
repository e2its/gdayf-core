if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *

    source_data = list()
    source_data.append("D:/Data/datasheets/binary/PEM/")
    source_data.append("PE-BINARY.csv")
    #Analysis
    controller = Controller()
    status, recomendations = controller.exec_analysis(datapath=''.join(source_data), objective_column='ACCION',
                             amode=FAST_PARANOIAC, metric='combined', deep_impact=3)

    controller.save_models(recomendations, mode=BEST_3)
    controller.remove_models(recomendations, mode=BEST)

    #Prediction
    source_data = list()
    source_data.append("D:/Data/datasheets/binary/PEM/")
    source_data.append("PE-BINARY.csv")
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

    controller.remove_models(recomendations, mode=ALL)
    controller.clean_handlers()
    del controller

