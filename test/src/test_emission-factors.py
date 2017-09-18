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
                                                      amode=POC, metric='combined', deep_impact=3)

    controller.save_models(recomendations, mode=EACH_BEST)
    status, recomendations2 = controller.exec_sanalysis(datapath=''.join(source_data),
                                                        list_ar_metadata=recomendations[-4:-2],
                                                        metric='combined', deep_impact=1)

    controller.remove_models(recomendations.extend(recomendations2), mode=ALL)
    controller.reconstruct_execution_tree(recomendations.extend(recomendations2), metric='combined')

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