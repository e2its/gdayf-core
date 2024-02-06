## @package gdayf.common.normalizationset

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

from collections import OrderedDict


## Class NormalizationSet manage the Normalizations metadata as OrderedDict supporting Normalizer Class methods and actions
class NormalizationSet(OrderedDict):
    def __init__(self):
        super().__init__()
        self['type'] = None
        self['class'] = None
        self['objective'] = OrderedDict()

    def reset(self):
        self['type'] = None
        self['class'] = None
        self['objective'] = OrderedDict()

    def set_base(self, datetime=True):
        self['type'] = "normalization"
        self['class'] = "base"
        self['datetime'] = datetime
        self['objective']['value'] = OrderedDict()

    def set_ignore_column(self):
        self['type'] = "drop"
        self['class'] = "ignore_column"
        self['objective']['value'] = OrderedDict()

    def set_stdmean(self, mean=0, std=1):
        self['type'] = "normalization"
        self['class'] = "stdmean"
        self['objective']['value'] = OrderedDict()
        self['objective']['mean'] = mean
        self['objective']['std'] = std

    def set_drop_missing(self):
        self['type'] = "drop"
        self['class'] = "drop_missing"
        self['objective']['value'] = OrderedDict()

    def set_discretize(self, buckets_number=10, fixed_size=True):
        self['type'] = "bucketing"
        self['class'] = "discretize"
        self['objective']['buckets_number'] = buckets_number
        self['objective']['fixed_size'] = fixed_size

    def set_working_range(self, minval=-1.0, maxval=1.0, minrange=-1.0, maxrange=1.0):
        self['type'] = "normalization"
        self['class'] = "working_range"
        self['objective']['minval'] = minval
        self['objective']['maxval'] = maxval
        self['objective']['minrange'] = minrange
        self['objective']['maxrange'] = maxrange

    def set_offset(self, offset=0):
        self['type'] = "normalization"
        self['class'] = "offset"
        self['objective']['offset'] = offset

    def set_aggregation(self, bucket_ratio=0.25):
        self['type'] = "bucketing"
        self['class'] = "working_range"
        self['objective']['bucket_ratio'] = bucket_ratio

    def set_fixed_missing_values(self, value=0.0):
        self['type'] = "imputation"
        self['class'] = "fixed_missing_values"
        self['objective']['value'] = value

    def set_mean_missing_values(self, objective_column, full=False):
        self['type'] = "imputation"
        self['class'] = "mean_missing_values"
        self['objective']['objective_column'] = objective_column
        self['objective']['full'] = full

    def set_progressive_missing_values(self, objective_column):
        self['type'] = "imputation"
        self['class'] = "progressive_missing_values"
        self['objective']['objective_column'] = objective_column

