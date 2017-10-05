if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *
    from pprint import pprint

    source_data = list()
    source_data.append("/Data/Data/datasheets/Multinomial/ARM/")
    source_data.append("ARM-Metric-train.csv")

    # Generating missing values


    # Analysis
    controller = Controller()
    if controller.config_checks():
        '''status, recomendations = controller.exec_analysis(datapath=''.join(source_data),
                                                          objective_column='ATYPE',
                                                          amode=FAST, metric='combined', deep_impact=4)

        controller.log_model_list(recomendations[0]['model_id'], recomendations, metric='test-accuracy', accuracy=True)

        controller.save_models(recomendations, mode=EACH_BEST)

        # Analisis especifico
        status, recomendations2 = controller.exec_sanalysis(datapath=''.join(source_data),
                                                            list_ar_metadata=recomendations[-3:-2],
                                                            metric='rmse', deep_impact=3)

        recomendations.extend(recomendations2)


        execution_tree = controller.reconstruct_execution_tree(arlist=None, metric='rmse', store=False,
                                                               user=controller.user_id,
                                                               experiment=recomendations[0]['model_id'])'''

        '''controller.remove_models(recomendations, mode=ALL)'''

        # Prediction
        source_data = list()
        source_data.append("/Data/Data/datasheets/Multinomial/ARM/")
        source_data.append("ARM-Metric-test.csv")
        # source_data.append("DM-Metric-missing-test-weather.csv")

        # Prediccion
        prediction_frame = controller.exec_prediction(datapath=''.join(source_data),
                                                      model_file="/Data/gdayf/experiments/PoC_gDayF/PoC_gDayF_ARM-Metric-train.csv_1507189780.8088667/h2o/train/1507189780.8147936/json/H2OGradientBoostingEstimator_1507192215.0619168.json.gz")
        pprint(prediction_frame[['ATYPE', 'predict']])
        # pprint(prediction_frame)

        '''# Save Pojo
        result = controller.get_java_model(recomendations[0], 'pojo')
        print(result)

        # Save Mojo
        result = controller.get_java_model(recomendations[0], 'mojo')
        print(result)

        controller.remove_models(recomendations, mode=ALL)
        controller.clean_handlers()'''
    del controller
