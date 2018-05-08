if __name__ == "__main__":

    from gdayf.workflow.workflow import Workflow
    from time import time

    source_data = list()
    source_data.append("/Data/Data/datasheets/regression/DM-Metric/")
    source_data.append("DM-Metric-missing-test-3.csv")

    workflow_data = list()
    workflow_data.append("/Data/e2its-dayf.svn/gdayf/branches/1.1.0-mrazul/test/json/")
    workflow_data.append("train_model_workflow.json")

    workflow = Workflow(user_id='WF_POC')
    workflow.workflow(datapath=''.join(source_data), workflow=''.join(workflow_data))
