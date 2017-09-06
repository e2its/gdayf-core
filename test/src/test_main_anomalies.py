if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *

    source_data = list()
    source_data.append("D:/Data/datasheets/Anomalies/CCPP")
    source_data.append("CPP.csv")
    #Analysis
    controller = Controller()
    status, recomendations = controller.exec_sanalysis(datapath=''.join(source_data), objective_column='Y2',
                                                       amode=ANOMALIES, metric='rmse', deep_impact=3)

    controller.save_models(recomendations)
    controller.reconstruct_execution_tree(recomendations, metric='rmse')
    controller.remove_models(recomendations, mode=ALL)

    #Prediction
    source_data = list()
    source_data.append("D:/Data/datasheets/regression/ENB2012/")
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