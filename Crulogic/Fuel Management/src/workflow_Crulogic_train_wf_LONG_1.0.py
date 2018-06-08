if __name__ == "__main__":

    from gdayf.workflow.workflow import Workflow
    from gdayf.common.constants import *

    source_data = list()
    source_data.append("/Data/Dropbox/DayF/gDayF/Proyectos/Industria4.0/Crulogic/Confidential-Data/Transformados-PDI/Crulogic-2017/")
    source_data.append("Crulogic-17-18.csv")

    workflow_data = list()
    workflow_data.append("/Data/e2its-dayf.svn/gdayf/branches/1.1.2.1-mrazul/Crulogic/Fuel Management/workflow/")
    workflow_data.append("CRULOGIC_workflow-LONG-1.0.json")

    workflow = Workflow(user_id='Crulogic')
    workflow.workflow(datapath=''.join(source_data), workflow=''.join(workflow_data), save_models=BEST_3)
