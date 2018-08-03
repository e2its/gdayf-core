if __name__ == "__main__":

    from gdayf.workflow.workflow import Workflow
    from os import path
    from gdayf.common.constants import *

    source_data = list()
    source_data.append(path.join(path.dirname(__file__),
                                 '../../../../../source data/Transformados-PDI/Crulogic-2017/'))
    source_data.append("Crulogic-17-18.csv")


    workflow_data = list()
    workflow_data.append(path.join(path.dirname(__file__), '../workflow/'))
    workflow_data.append("CRULOGIC_workflow-round-2-Avg-fuel-base.json")

    workflow = Workflow(user_id='Crulogic-r2')
    workflow.workflow(datapath=''.join(source_data), workflow=''.join(workflow_data),
                      remove_models=EACH_BEST, prefix='filtered-data')
    del workflow

    source_data = list()
    source_data.append(path.join(path.dirname(__file__),
                                 '../../../../../source data/Transformados-PDI/Crulogic-2017/'))
    source_data.append("Crulogic-17-18-Optima.csv")

    workflow = Workflow(user_id='Crulogic-r2')
    workflow.workflow(datapath=''.join(source_data), workflow=''.join(workflow_data),
                      remove_models=EACH_BEST, prefix='Optima-data')
    del workflow
