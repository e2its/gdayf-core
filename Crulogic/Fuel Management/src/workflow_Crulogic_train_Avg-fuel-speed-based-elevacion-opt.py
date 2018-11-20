if __name__ == "__main__":

    from gdayf.workflow.workflow import Workflow
    from os import path
    from gdayf.common.constants import *

    source_data = list()

    '''source_data.append(path.join(path.dirname(__file__),
                                 '../../../../../source data/Transformados-PDI/Crulogic-2017/'))
    source_data.append("Crulogic-17-18.csv")'''

    source_data.append(path.join(path.dirname(__file__),
                                 '../../../../../source data/Transformados-PDI/Crulogic/'))
    source_data.append("Crulogic-elevation.csv")

    # DRF ELEVATION
    workflow_data = list()
    workflow_data.append(path.join(path.dirname(__file__), '../workflow/'))
    workflow_data.append("CRULOGIC-Avg-fuel-speed-based-elevacion-opt-drf.json")

    workflow = Workflow(user_id='Crulogic-r2')
    workflow.workflow(datapath=''.join(source_data), workflow=''.join(workflow_data),
                      remove_models=NONE, prefix=None)
    del workflow
    del workflow_data

    # GBM ELEVATION
    '''
    workflow_data = list()
    workflow_data.append(path.join(path.dirname(__file__), '../workflow/'))
    workflow_data.append("CRULOGIC-Avg-fuel-speed-based-elevacion-opt-gbm.json")

    workflow = Workflow(user_id='Crulogic-r2')
    workflow.workflow(datapath=''.join(source_data), workflow=''.join(workflow_data),
                      remove_models=NONE, prefix=None)
    del workflow
    del workflow_data

    # GLM ELEVATION
    workflow_data = list()
    workflow_data.append(path.join(path.dirname(__file__), '../workflow/'))
    workflow_data.append("CRULOGIC-Avg-fuel-speed-based-elevacion-opt-glm.json")

    workflow = Workflow(user_id='Crulogic-r2')
    workflow.workflow(datapath=''.join(source_data), workflow=''.join(workflow_data),
                      remove_models=NONE, prefix=None)
    del workflow
    del workflow_data
    '''

    # DRF NOT ELEVATION
    workflow_data = list()
    workflow_data.append(path.join(path.dirname(__file__), '../workflow/'))
    workflow_data.append("CRULOGIC-Avg-fuel-speed-based-opt-drf.json")

    workflow = Workflow(user_id='Crulogic-r2')
    workflow.workflow(datapath=''.join(source_data), workflow=''.join(workflow_data),
                      remove_models=NONE, prefix=None)
    del workflow
    del workflow_data

    # GBM NOT ELEVATION
    '''
    workflow_data = list()
    workflow_data.append(path.join(path.dirname(__file__), '../workflow/'))
    workflow_data.append("CRULOGIC-Avg-fuel-speed-based-opt-gbm.json")

    workflow = Workflow(user_id='Crulogic-r2')
    workflow.workflow(datapath=''.join(source_data), workflow=''.join(workflow_data),
                      remove_models=NONE, prefix=None)
    del workflow
    del workflow_data

    # GLM NOT ELEVATION
    workflow_data = list()
    workflow_data.append(path.join(path.dirname(__file__), '../workflow/'))
    workflow_data.append("CRULOGIC-Avg-fuel-speed-based-opt-glm.json")

    workflow = Workflow(user_id='Crulogic-r2')
    workflow.workflow(datapath=''.join(source_data), workflow=''.join(workflow_data),
                      remove_models=NONE, prefix=None)
    del workflow
    del workflow_data
    '''

    del source_data
