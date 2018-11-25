if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *
    from pandas import set_option
    from os import path

    source_data = list()
    source_data.append(path.join(path.dirname(__file__),
                                 '../../../../../source data/Transformados-PDI/Crulogic/'))
    source_data.append("Crulogic-elevation.csv")

    ignore_columns = [
            "Good_Level",
            "Bad_Level",
            "Engine-worktime",
            "Braking100",
            "Strong-braking100",
            "Strong-Acel100",
            "idling100",
            "Out-speed",
            "Out-engine",
            "Rollout100",
            "SDS",
            "SDS-slopes",
            "SDS-Anticipation",
            "SDS-brakes",
            "SDS-Gear",
            "Fuel",
            "Fuel-idling",
            "idling-time",
            "Distance-remol",
            "Cruise-Control",
            "Avg-speed",
            "Max-speed",
            "Max-rpm",
            "Rollout",
            "Braking",
            "Strong-braking",
            "out-rpm",
            "Strong-Acel100-B",
            "Strong-braking100-B",
            "out-rpm-B",
            "Braking100-N",
            "Rollout100-N",
            "Idling100-N",
            "Cruise-Control-N",
            "V_Cluster",
            "distance_sap",
            "GAINED_ELEV",
            "LOSS_ELEV"
          ]


    model_base = '/Crulogic-r2/CRULOGIC-Avg-speed_1542492337.893231/Crulogic-r2_Dataframe_59332_4238_14_1542492338.0480626/H2ORandomForestEstimator_1542492811.434884'
    model_c2 = '/Crulogic-r2/CRULOGIC-Avg-fuel-speed-based_1537393332.1889584/Crulogic-r2_Dataframe_77256_6438_12_1537396786.269636/H2ORandomForestEstimator_1537397247.3189142'
    model_c4 = '/Crulogic-r2/CRULOGIC-Avg-fuel-speed-based_1537393332.1889584/Crulogic-r2_Dataframe_77256_6438_12_1537394646.6365829/H2OGradientBoostingEstimator_1537395016.7843776'
    controller = Controller(user_id='Crulogic-r2')
    if controller.config_checks():
        prediction_frame = controller.exec_prediction(datapath=''.join(source_data),
                                                      model_file=model_base)
        prediction_frame.drop(labels=ignore_columns, axis=1, inplace=True)

        _, model_ar = controller.get_ar_from_engine(path=model_c2)
        prediction_frame.rename(columns={'predict': 'c2'}, inplace=True)
        status, recomendations = controller.exec_sanalysis(datapath=prediction_frame, list_ar_metadata=[model_ar],
                                                           objective_column='Avg_fuel',
                                                           amode=NORMAL, metric='test_rmse', deep_impact=1)

        _, model_ar = controller.get_ar_from_engine(path=model_c4)
        prediction_frame.rename(columns={'c2': 'c4'}, inplace=True)
        status, recomendations = controller.exec_sanalysis(datapath=prediction_frame, list_ar_metadata=[model_ar],
                                                           objective_column='Avg_fuel',
                                                           amode=NORMAL, metric='test_rmse', deep_impact=1)

    del controller



