if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *

    source_data = list()
    source_data.append("/Data/Data/datasheets/Anomalies/CCPP/")
    source_data.append("CPP_base_ampliado.csv")
    #Analysis
    controller = Controller()
    status, recomendations = controller.exec_analysis(datapath=''.join(source_data), objective_column=None,
                                                      amode=CLUSTERING, metric='cdistance', deep_impact=10,
                                                      k=6, estimate_k=True)

    controller.save_models(recomendations)
    status, recomendations2 = controller.exec_sanalysis(datapath=''.join(source_data),
                                                        list_ar_metadata=recomendations[-4:-2],
                                                        metric='cdistance', deep_impact=1)

    recomendations.extend(recomendations2)
    controller.reconstruct_execution_tree(recomendations, metric='cdistance')
    controller.remove_models(recomendations, mode=ALL)

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
    result = controller.get_java_model(recomendations[0], 'pojo')
    print(result)

    # Save Mojo
    #controller = Controller()
    result = controller.get_java_model(recomendations[0], 'mojo')
    print(result)

    controller.remove_models(recomendations, mode=ALL)
    controller.clean_handlers()
    del controller