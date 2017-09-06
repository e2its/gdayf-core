if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *
    from pprint import pprint

    source_data = list()
    source_data.append("D:/Data/datasheets/Usa-datasheet/Emission/")
    source_data.append("emissionfactors-missing.csv")


    #Analysis
    controller = Controller()
    status, recomendations = controller.exec_analysis(datapath=''.join(source_data),
                                                      objective_column='gridsubregion',
                             amode=POC, metric='combined', deep_impact=1)

    controller.save_models(recomendations, mode=EACH_BEST)
    controller.reconstruct_execution_tree(recomendations, metric='combined')
    controller.remove_models(recomendations, mode=ALL)

    #Prediction
    source_data = list()
    source_data.append("D:/Data/datasheets/Usa-datasheet/Emission/")
    source_data.append("emissionfactors-missing-test.csv")

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

    controller.remove_models(recomendations, mode=ALL)
    controller.clean_handlers()
    del controller