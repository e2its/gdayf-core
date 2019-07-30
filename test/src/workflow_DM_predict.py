if __name__ == "__main__":

    from gdayf.workflow.workflow import Workflow
    from time import time

    source_data = list()
    source_data.append("/Data/Data/datasheets/regression/DM-Metric/")
    source_data.append("DM-Metric-missing-test-3.csv")

    workflow_data = list()
    workflow_data.append("../json/predict_model_workflow.json")

    workflow = Workflow(user_id='WF_POC')
    workflow.workflow(datapath=''.join(source_data), workflow=''.join(workflow_data))
