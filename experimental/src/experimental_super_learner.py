def modification(a, b):
    for x in b:
        try:
            a.remove(x)
        except ValueError:
            pass
    return a

if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.utils import get_model_fw
    from gdayf.common.constants import *
    from pandas import set_option, DataFrame, read_excel, concat
    from os import path
    from collections import OrderedDict
    from gdayf.handlers.inputhandler import inputHandlerCSV

    source_data = list()
    source_data.append(path.join(path.dirname(__file__),
                                 '../../../../source data/Transformados-PDI/Crulogic-2017/'))
    source_data.append("Crulogic-17-18.csv")
    model_data = inputHandlerCSV().inputCSV(filename=''.join(source_data))

    source_1_data = list()
    source_1_data.append('/Data/gdayf-v1/experiments/Crulogic-r2/CRULOGIC-Avg-fuel-speed-predicted_1537911231.396293/')
    source_1_data.append('summary/predict/')
    source_1_data.append('Avg-speed_a2_p_prediction.xls')
    model_1_data = read_excel(io=''.join(source_1_data))
    
    source_2_data = list()
    source_2_data.append('/Data/gdayf-v1/experiments/Crulogic-r2/CRULOGIC-Avg-fuel-speed-predicted_1537911231.396293/')
    source_2_data.append('summary/predict/')
    source_2_data.append('Avg-speed_a4_p_prediction.xls')
    model_2_data = read_excel(io=''.join(source_2_data))

    #Analysis
    controller = Controller(user_id='Crulogic-r2')
    if controller.config_checks():

        model_1 = '/Data/gdayf-v1/experiments/Crulogic-r2/CRULOGIC-Avg-fuel-speed-predicted_1537911231.396293/Crulogic-r2_Dataframe_77256_6438_12_1537911811.7863178/h2o/train/1537911811.7874267/json/H2ORandomForestEstimator_1537912248.5985134.json'
        model_2 = '/Data/gdayf-v1/experiments/Crulogic-r2/CRULOGIC-Avg-fuel-speed-predicted_1537911231.396293/Crulogic-r2_Dataframe_77256_6438_12_1537913921.5431554/h2o/train/1537913921.544499/json/H2OGradientBoostingEstimator_1537914152.3058493.json'

        dataframe_dict = OrderedDict()
        objective_column = 'Avg-fuel'
        dataframe_dict[objective_column] = model_data[objective_column]
        prediction_frame = controller.exec_prediction(datapath=model_1_data,
                                                      model_file=model_1)
        model_data['model_1'] = prediction_frame['predict']
        prediction_frame = controller.exec_prediction(datapath=model_2_data,
                                                      model_file=model_2)
        model_data['model_2'] = prediction_frame['predict']
        columns = model_data.columns.values.tolist()
        ignore_columns =["Good_Level","Bad_Level","Engine-worktime","Braking100","Strong-braking100",
            "Strong-Acel100","idling100","Out-speed",
            "Out-engine","Rollout100","SDS",
            "SDS-slopes","SDS-Anticipation",
            "SDS-brakes","SDS-Gear","Fuel",
            "Fuel-idling","idling-time","Distance-remol",
            "Cruise-Control","Avg-speed","Max-speed",
            "Max-rpm","Rollout","Braking",
            "Strong-braking","out-rpm","Strong-Acel100-B",
            "Strong-braking100-B","out-rpm-B","Braking100-N",
            "Rollout100-N","Idling100-N","Cruise-Control-N",
            "V_Cluster"]

        model_columns = modification(columns, ignore_columns)
        print(model_columns)

        status, recomendations = controller.exec_analysis(datapath=model_data[model_columns],
                                                          objective_column=objective_column,
                                                          amode=NORMAL, metric='test_rmse', deep_impact=8)

        controller.reconstruct_execution_tree(metric='test_rmse', store=True)
        controller.remove_models(arlist=recomendations, mode=BEST)
        print(controller.table_model_list(ar_list=recomendations, metric='test_rmse'))

        prediction_frame = controller.exec_prediction(datapath=model_data,
                                                      model_file=recomendations[0]['json_path'][0]['value'])

        model_data['predict'] = prediction_frame['predict']
        source_3_data = list()
        source_3_data.append(
            '/Data/gdayf-v1/experiments/Crulogic-r2/CRULOGIC-Avg-fuel-speed-predicted_1537911231.396293/')
        source_3_data.append('summary/predict/')
        source_3_data.append('Ensemble_Avg-fuel_prediction.xls')
        model_data.to_excel(''.join(source_3_data))

        print(controller.table_model_list(ar_list=recomendations, metric='test_rmse'))

        controller.clean_handlers()
    del controller



