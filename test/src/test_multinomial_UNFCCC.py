if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.handlers.inputhandler import inputHandlerCSV
    from gdayf.common.constants import *
    from gdayf.common.utils import pandas_split_data
    from pprint import pprint

    source_data = list()
    source_data.append("/Data/Data/datasheets/Europe-datasheet/Pollutant/")
    source_data.append("UNFCCC_v8-missing.csv")

    #Reducing rows
    _, pd_train = pandas_split_data(inputHandlerCSV().inputCSV(filename=''.join(source_data)))


    #Analysis
    controller = Controller()
    status, recomendations = controller.exec_sanalysis(datapath=pd_train, objective_column='Country',
                                                       amode=FAST, metric='combined', deep_impact=3)

    controller.save_models(recomendations, mode=EACH_BEST)
    controller.reconstruct_execution_tree(recomendations, metric='combined')
    controller.remove_models(recomendations, mode=ALL)

    #Prediction
    source_data = list()
    source_data.append("/Data/Data/datasheets/Europe-datasheet/Pollutant/")
    source_data.append("UNFCCC_v8-missing-test.csv")

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