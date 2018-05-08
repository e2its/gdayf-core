if __name__ == "__main__":

    from gdayf.workflow.workflow import Workflow
    from time import time
    from pandas import read_csv

    source_data = list()
    source_data.append("/Data/Dropbox/DayF/gDayF/Proyectos/Industria4.0/Crulogic/Confidential-Data/")
    source_data.append("CRULOGIC_2017_CL.csv")

    workflow_data = list()
    workflow_data.append("/Data/e2its-dayf.svn/gdayf/branches/1.1.0-mrazul/Crulogic/Fuel Management/workflow/")
    workflow_data.append("train_CRULOGIC_workflow-6.json")

    workflow = Workflow(user_id='Crulogic_wf60')
    workflow.workflow(datapath=''.join(source_data), workflow=''.join(workflow_data))
