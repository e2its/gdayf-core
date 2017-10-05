'PoC_gDayF_DM-Metric-missing.csv_1506690460.5805311'


if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *
    from pprint import pprint

    #Analysis
    controller = Controller()

    execution_tree = controller.reconstruct_execution_tree(arlist=None, metric='rmse', store=False,
                                                           user=controller.user_id,
                                                           experiment="PoC_gDayF_CPP.csv_1506981336.6569452")
    pprint(execution_tree)

    del controller