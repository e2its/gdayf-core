## @package gdayf.core.experiment_context
# Define all global objects, functions and structs related with an specific experiment

'''
 * This file is part of the gDayF AutoML Core Framework project
 * distribution (https://github.com/e2its/gdayf-core).
 * Copyright (c) 2016-2019 Jose Luis Sanchez del Coso <e2its.es@gmail.com>.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 ** Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2019
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
        self.spark_temporal_data_frames = dict()

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
