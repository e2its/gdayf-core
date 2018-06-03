if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *

    source_data = list()
    source_data.append("/Data/Data/datasheets/Anomalies/UCI CBM Dataset/")
    source_data.append("UCI-CBM.csv")
    #Analysis
    controller = Controller()
    status, recomendations = controller.exec_analysis(datapath=''.join(source_data), objective_column=None,
                                                      amode=ANOMALIES, metric='test-rmse', deep_impact=5)

    controller.save_models(recomendations)
    status, recomendations2 = controller.exec_sanalysis(datapath=''.join(source_data),
                                                        list_ar_metadata=recomendations[-3:-2],
                                                        metric='train_rmse', deep_impact=3)

    recomendations.extend(recomendations2)
    controller.reconstruct_execution_tree(recomendations, metric='train_rmse')
    controller.remove_models(recomendations, mode=ALL)

    #Prediction
    source_data = list()
    source_data.append("/Data/Data/datasheets/Anomalies/UCI CBM Dataset/")
    source_data.append("UCI-CBM.csv")

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