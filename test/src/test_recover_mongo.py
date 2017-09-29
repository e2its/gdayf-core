'PoC_gDayF_DM-Metric-missing.csv_1506690460.5805311'


if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *
    from pprint import pprint

    source_data = list()
    source_data.append("/Data/Data/datasheets/regression/DM-Metric/")
    source_data.append("DM-Metric-missing.csv")

    #Generating missing values


    #Analysis
    controller = Controller()

    execution_tree = controller.reconstruct_execution_tree(arlist=None, metric='rmse', store=False,
                                                           user=controller.user_id,
                                                           experiment="PoC_gDayF_DM-Metric-missing.csv_1506690460.5805311")
    pprint(execution_tree)

    del controller