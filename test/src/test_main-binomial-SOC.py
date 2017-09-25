if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *

    source_data = list()
    source_data.append("/Data/Data/datasheets/binary/FODSET/")
    source_data.append("football.train2-r.csv")
    #Analysis
    controller = Controller()
    status, recomendations = controller.exec_analysis(datapath=''.join(source_data), objective_column='HomeWin',
                                                      amode=FAST, metric='test_accuracy', deep_impact=4)

    controller.log_model_list(recomendations[0]['model_id'], recomendations, metric='combined', accuracy=True)

    controller.save_models(recomendations, mode=EACH_BEST)
    '''status, recomendations2 = controller.exec_sanalysis(datapath=''.join(source_data),
                                                        list_ar_metadata=recomendations[-3:-2],
                                                        metric='test_accuracy', deep_impact=2)

    recomendations.extend(recomendations2)'''
    controller.reconstruct_execution_tree(recomendations, metric='test_accuracy')
    controller.remove_models(recomendations, mode=ALL)

    #Prediction
    source_data = list()
    source_data.append("/Data/Data/datasheets/binary/FODSET/")
    source_data.append("football.test2-r.csv")
    model_source = list()

    #controller = Controller()
    print(recomendations[0]['load_path'][0]['value'])
    prediction_frame = controller.exec_prediction(datapath=''.join(source_data),
                                                  model_file=recomendations[0]['json_path'][0]['value'])
    print(prediction_frame[['HomeWin', 'predict', 'p0', 'p1']])

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

