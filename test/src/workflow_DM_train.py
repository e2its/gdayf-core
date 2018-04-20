if __name__ == "__main__":

    from gdayf.workflow.workflow import Workflow
    from time import time

    source_data = list()
    source_data.append("/Data/Dropbox/DayF/gDayF/Proyectos/Industria 4.0/Crulogic/Data2017.csv")
    source_data.append("DM-Metric-missing-3.csv")

    workflow_data = list()
    workflow_data.append("/Data/e2its-dayf.svn/gdayf/branches/1.1.0-mrazul/Crulogic/Fuel Management/workflow/")
    workflow_data.append("train_CRULOGIC_workflow-1.json")

    workflow = Workflow(user_id='Crulogic_wf1')
    workflow.workflow(datapath=''.join(source_data), workflow=''.join(workflow_data))
