if __name__ == "__main__":

    from gdayf.workflow.workflow import Workflow
    from os import path
    from gdayf.common.constants import *

    source_data = list()

    source_data.append(path.join(path.dirname(__file__),
                                 '../../../../../source data/Transformados-PDI/Crulogic/'))
    source_data.append("Crulogic-ALL.csv")



    # DRF NOT ELEVATION
    workflow_data = list()
    workflow_data.append(path.join(path.dirname(__file__), '../workflow/full/'))
    workflow_data.append("CRULOGIC-Avg-fuel-speed-based-opt-drf.json")

    workflow = Workflow(user_id='Crulogic-r2')
    workflow.workflow(datapath=''.join(source_data), workflow=''.join(workflow_data),
                      remove_models=NONE, prefix=None)
    del workflow
    del workflow_data

    '''# GBM NOT ELEVATION

    workflow_data = list()
    workflow_data.append(path.join(path.dirname(__file__), '../workflow/full/'))
    workflow_data.append("CRULOGIC-Avg-fuel-speed-based-opt-gbm.json")

    workflow = Workflow(user_id='Crulogic-r2')
    workflow.workflow(datapath=''.join(source_data), workflow=''.join(workflow_data),
                      remove_models=NONE, prefix=None)
    del workflow
    del workflow_data

    # GLM NOT ELEVATION
    workflow_data = list()
    workflow_data.append(path.join(path.dirname(__file__), '../workflow/full/'))
    workflow_data.append("CRULOGIC-Avg-fuel-speed-based-opt-glm.json")

    workflow = Workflow(user_id='Crulogic-r2')
    workflow.workflow(datapath=''.join(source_data), workflow=''.join(workflow_data),
                      remove_models=NONE, prefix=None)
    del workflow
    del workflow_data

    del source_data'''
