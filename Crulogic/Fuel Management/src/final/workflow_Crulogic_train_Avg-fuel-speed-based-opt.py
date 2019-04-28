if __name__ == "__main__":

    from gdayf.workflow.workflow import Workflow
    from os import path
    from gdayf.common.constants import *

    source_data = list()
    source_data.append(path.join(path.dirname(__file__),
                                 '../../../../../source data/Transformados-PDI/Crulogic/final/'))
    source_data.append("Crulogic-train.csv")

    workflow_data = list()
    workflow_data.append(path.join(path.dirname(__file__), '../../workflow/final/'))
    workflow_data.append("CRULOGIC-Avg-fuel-speed-based-opt.json")

    workflow = Workflow(user_id='Crulogic-r2')
    workflow.workflow(datapath=''.join(source_data), workflow=''.join(workflow_data),
                      remove_models=NONE, prefix=None)

    del workflow
    del workflow_data

    workflow_data = list()
    workflow_data.append(path.join(path.dirname(__file__), '../../workflow/final/'))
    workflow_data.append("CRULOGIC-Avg-fuel-based.json")

    workflow = Workflow(user_id='Crulogic-r2')
    workflow.workflow(datapath=''.join(source_data), workflow=''.join(workflow_data),
                      remove_models=NONE, prefix=None)

    del workflow
    del workflow_data

    del source_data
