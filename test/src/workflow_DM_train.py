if __name__ == "__main__":

    from gdayf.workflow.workflow import Workflow
    from gdayf.common.constants import *
    from gdayf.common.dataload import DataLoad

    data_train, _ = DataLoad().dm()
    del _

    workflow_data = list()
    workflow_data.append("../json/train_model_workflow.json")

    workflow = Workflow(user_id='WF_POC')
    workflow.workflow(datapath=data_train, workflow=''.join(workflow_data), remove_models=NONE, prefix=None)
