if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *
    from copy import deepcopy

    source_data = list()
    source_data.append("/Data/Data/datasheets/regression/DM-Metric/")
    source_data.append("DM-Metric-missing-test.csv")
    # Analysis
    controller = Controller()
    if controller.config_checks():

        path_old ='/PoC_gDayF/PoC_gDayF_DM-Metric-missing-2.csv_1507550027.688733/H2OGradientBoostingEstimator_1507550245.9630692'
        path_new ='/Data/gdayf/experiments/PoC_gDayF/PoC_gDayF_DM-Metric-missing-2.csv_1507550027.688733/h2o/progressive/H2OGradientBoostingEstimator_1507550245.9630692_checkpoint.json'

        failed_1, new_ar = controller.get_ar_from_engine(path_new)
        failed_2, load_ar_metadata = controller.get_ar_from_engine(path_old)

        loaded_models = controller.load_models([load_ar_metadata])

        if failed_1:
            print("Descriptor de analisis no existe")

        elif failed_2:
            print("Error en path de modelo base")
        elif len(loaded_models) == 0:
            print("Error en carga de modelo base")
        else:
            status, recomendations = controller.exec_sanalysis(datapath=''.join(source_data),
                                                               list_ar_metadata=[new_ar],
                                                               deep_impact=1)
            controller.save_models(recomendations, mode=ALL)
            recomendations.extend(loaded_models)
            controller.remove_models(recomendations)
    del controller