if __name__ == "__main__":

    from gdayf.workflow.workflow import Workflow
    from gdayf.common.dataload import DataLoad

    _, data_test = DataLoad().dm()
    del _

    workflow_data = list()
    workflow_data.append("../json/predict_model_workflow.json")

    workflow = Workflow(user_id='WF_POC')
    workflow.workflow(datapath=data_test, workflow=''.join(workflow_data))
