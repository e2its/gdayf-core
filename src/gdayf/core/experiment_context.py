## @package gdayf.core.experiment_context
# Define all global objects, functions and structs related with an specific experiment

'''
Copyright (C) e2its - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 *
 * This file is part of gDayF project.
 *
 * Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2018
'''

from gdayf.conf.loadconfig import LoadConfig
from gdayf.conf.loadconfig import LoadLabels
from time import time


class Experiment_Context(object):
    def __init__(self, user_id='PoC_gDayF', workflow_id='default', lang='en'):
        self.id_user = user_id
        self.timestamp = str(time())
        self.id_workflow = workflow_id
        self.config = LoadConfig(user_id=self.id_user)
        self.labels = LoadLabels(lang=lang)
        self.id_analysis = None


    ## Method used to set global variable id_user used to propagate user_id to all modules
    # @param self object pointer
    # @param value User id value
    def set_id_user(self, value):
        self.id_user = value


    ## Method used to get global variable id_user used to recover user_id in all modules
    # @param self object pointer
    # @return id_user
    def get_id_user(self):
        return self.id_user


    ## Method used to set global variable id_workflow used to propagate workflow_id to all modules
    # @param value User id value
    def set_id_workflow(self, value):
        self.id_workflow = value


    ## Method used to get global variable id_workflow used to recover workflow_id in all modules
    # @param model ArMetadata
    # @return ArMetadata deepcopy
    def get_id_workflow(self):
        return self.id_workflow

    ## Method used to set global variable id_workflow used to propagate workflow_id to all modules
    # @param value User id value
    def set_id_analysis(self, value):
        self.id_analysis = value


    ## Method used to get global variable id_workflow used to recover workflow_id in all modules
    # @param model ArMetadata
    # @return ArMetadata deepcopy
    def get_id_analysis(self):
        return self.id_analysis
