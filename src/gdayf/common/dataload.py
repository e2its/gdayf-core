## @package gdayf.common.data_load
# Define all objects, functions and structured related to Data input for testing proposed

'''
Copyright (C) e2its - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 *
 * This file is part of gDayF project.
 *
 * Written by Jose L. Sanchez del Coso <e2its.es@gmail.com>, 2016-2019
'''

from pandas import read_csv as read_csv


class DataLoad (object):
## Class DataLoad export and load test data to Pandas DataFrame

    ## The constructor
    # Generate an empty DataLoad class with all test uri
    def __init__(self):
        self.data = dict()
        self.data['ARM'] = dict()
        self.data['ARM']['test'] = 'https://raw.githubusercontent.com/e2its/datasets/master/ARM-Metric-test-TS.csv'
        self.data['ARM']['train'] = 'https://raw.githubusercontent.com/e2its/datasets/master/ARM-Metric-train-TS.csv'
        self.data['CPP'] = dict()
        self.data['CPP']['train'] ='https://raw.githubusercontent.com/e2its/datasets/master/CPP_base_ampliado.csv'
        self.data['CPP']['test'] ='https://raw.githubusercontent.com/e2its/datasets/master/CPP_base_ampliado.csv'
        self.data['DM'] = dict()
        self.data['DM']['train'] ='https://raw.githubusercontent.com/e2its/datasets/master/DM-Metric-missing-3.csv'
        self.data['DM']['test'] = 'https://raw.githubusercontent.com/e2its/datasets/master/DM-Metric-missing-test-3.csv'
        self.data['ENB'] = dict()
        self.data['ENB']['train'] = 'https://raw.githubusercontent.com/e2its/datasets/master/ENB2012_data-Y1.csv'
        self.data['ENB']['test'] = 'https://raw.githubusercontent.com/e2its/datasets/master/ENB2012_data-Y1.csv'
        self.data['FOOTSET'] = dict()
        self.data['FOOTSET']['train']= 'https://raw.githubusercontent.com/e2its/datasets/master/football.train2-r.csv'
        self.data['FOOTSET']['test'] = 'https://raw.githubusercontent.com/e2its/datasets/master/football.test2-r.csv'

    def arm(self):
        return read_csv(self.data['ARM']['train']), read_csv(self.data['ARM']['test'])

    def cpp(self):
        return read_csv(self.data['CPP']['train']), read_csv(self.data['CPP']['test'])

    def dm(self):
        return read_csv(self.data['DM']['train']), read_csv(self.data['DM']['test'])

    def enb(self):
        return read_csv(self.data['ENB']['train']), read_csv(self.data['ENB']['test'])

    def footset(self):
        return read_csv(self.data['FOOTSET']['train']), read_csv(self.data['FOOTSET']['test'])
