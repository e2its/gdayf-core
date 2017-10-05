if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *
    from pprint import pprint

    # looking for model
    #db.getCollection('PoC_gDayF').find({$and:[{"model_id":"PoC_gDayF_ARM-Metric-train.csv_1507189780.8088667"},{"model_parameters.h2o.parameters.model_id.value":"H2OGradientBoostingEstimator_1507192215.0619168"}]})


    # Analysis
    controller = Controller()
    if controller.config_checks():

        # Prediction
        source_data = list()
        source_data.append("/Data/Data/datasheets/Multinomial/ARM/")
        source_data.append("ARM-Metric-test.csv")

        # Prediccion
        prediction_frame = controller.exec_prediction(datapath=''.join(source_data),
                                                      model_file="/Data/gdayf/experiments/PoC_gDayF/PoC_gDayF_ARM-Metric-train.csv_1507189780.8088667/h2o/train/1507189780.8147936/json/H2OGradientBoostingEstimator_1507192215.0619168.json.gz")
        pprint(prediction_frame[['ATYPE', 'predict']])


        source_data = list()
        source_data.append("/Data/Data/datasheets/regression/DM-Metric/")
        source_data.append("DM-Metric-missing-test-2.csv")

        # Prediccion
        prediction_frame = controller.exec_prediction(datapath=''.join(source_data),
                                                      model_file="/Data/gdayf/experiments/PoC_gDayF/PoC_gDayF_DM-Metric-missing-2.csv_1507211446.5940318/h2o/train/1507211446.600468/json/H2ODeepLearningEstimator_1507211469.8011096.json.gz")
        pprint(prediction_frame[['Weather_Temperature', 'predict']])


    del controller
