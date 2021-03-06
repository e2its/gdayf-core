## @package gdayf.normalizer.normalizer

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

import numpy as np
import pandas as pd
from datetime import datetime
from collections import OrderedDict
from gdayf.conf.loadconfig import LoadConfig
from gdayf.conf.loadconfig import LoadLabels
from gdayf.logs.logshandler import LogsHandler
from gdayf.common.constants import DTYPES, NO_STANDARDIZE
from gdayf.common.normalizationset import NormalizationSet
from copy import deepcopy


## Class oriented to manage normalizations on dataframes for improvements on accuracy
class Normalizer (object):

    ## Constructor
    # @param e_c context pointer
    def __init__(self, e_c):
        self._ec = e_c
        self._config = self._ec.config.get_config()['normalizer']
        self._labels = self._ec.labels.get_config()['messages']['normalizer']
        self._logging = LogsHandler(self._ec, __name__)

    ## Method oriented to specificate data_normalizations
    # @param dataframe_metadata DFMetadata()
    # @param an_objective ATypesMetadata
    # @param objective_column string indicating objective column
    # @return None if nothing to DO or Normalization_sets OrderedDict() on other way
    def define_normalizations(self, dataframe_metadata, an_objective, objective_column):
        if not self._config['non_minimal_normalizations_enabled']:
            return None
        else:
            df_type = dataframe_metadata['type']
            rowcount = dataframe_metadata['rowcount']
            #cols = dataframe_metadata['cols']
            columns = dataframe_metadata['columns']
            norms = list()
            normoption = NormalizationSet()
            if df_type == 'pandas':
                for description in columns:
                    col = description['name']
                    if col != objective_column:
                        if int(description['missed']) > 0 and \
                           (int(description['missed'])/rowcount >= self._config['exclusion_missing_threshold']):
                            normoption.set_ignore_column()
                            norms.append({col: normoption.copy()})
                        if self._config['clustering_standardize_enabled'] and an_objective[0]['type'] in ['clustering'] \
                                and description['type'] in DTYPES \
                                and int(description['cardinality']) > 1 and description['mean'] != 0.0 and \
                                description['std'] != 1.0 \
                                and (
                                float(description['std']) / (float(description['max']) - float(description['min']))) \
                                > self._config['std_threshold']:
                            normoption.set_stdmean(description['mean'], description['std'])
                            norms.append({col: normoption.copy()})
                        if self._config['standardize_enabled'] and description['type'] in DTYPES \
                            and an_objective[0]['type'] not in ['clustering']\
                            and int(description['cardinality']) > 1 and description['mean'] != 0.0 and \
                            description['std'] != 1.0 \
                            and(float(description['std']) / (float(description['max']) - float(description['min']))) \
                                 > self._config['std_threshold']:
                            normoption.set_stdmean(description['mean'], description['std'])
                            norms.append({col: normoption.copy()})

                self._logging.log_exec('gDayF', "Normalizer", self._labels["norm_set_establish"], norms)
                if len(norms) != 0:
                    return norms.copy()
                else:
                    return None
            else:
                return None

    ## Method oriented to specificate ignored_columns
    # @param dataframe_metadata DFMetadata()
    # @param objective_column string indicating objective column
    # @return None if nothing to DO or Normalization_sets orderdict() on other way
    def define_ignored_columns(self, dataframe_metadata, objective_column):
        if not self._config['non_minimal_normalizations_enabled']:
            return None
        else:
            df_type = dataframe_metadata['type']
            rowcount = dataframe_metadata['rowcount']
            # cols = dataframe_metadata['cols']
            columns = dataframe_metadata['columns']
            norms = list()
            normoption = NormalizationSet()
            if df_type == 'pandas':
                for description in columns:
                    col = description['name']
                    if col != objective_column:
                        if int(description['cardinality']) == 1:
                            normoption.set_ignore_column()
                            norms.append({col: normoption.copy()})
                        elif self._config['datetime_columns_management'] is not None \
                                and self._config['datetime_columns_management'] \
                                and description['type'] == 'datetime64[ns]':
                            normoption.set_ignore_column()
                            norms.append({col: normoption.copy()})
                self._logging.log_exec('gDayF', "Normalizer", self._labels["ignored_set_establish"], norms)
                if len(norms) != 0:
                    return norms.copy()
                else:
                    return None
            else:
                return None

    ## Method oriented to specificate minimal data_normalizations
    # @param dataframe_metadata DFMetadata()
    # @param an_objective ATypesMetadata
    # @param objective_column string indicating objective column
    # @return [None] if nothing to DO or Normalization_sets orderdict() on other way
    def define_special_spark_naive_norm(self, dataframe_metadata):
        df_type = dataframe_metadata['type']
        if df_type == 'pandas':
            norms = list()
            normoption = NormalizationSet()
            columns = dataframe_metadata['columns']
            norms = list()
            for description in columns:
                col = description['name']
                if description['min'] is not None and float(description['min']) < 0.0:
                    normoption.set_offset(offset=abs(float(description['min']))
                                                 * self._config['special_spark_naive_offset'])
                    norms.append({col: normoption.copy()})

            return norms.copy()
        else:
            return None

    ## Method oriented to specificate special data_normalizations non negative
    # @param dataframe_metadata DFMetadata()
    # @param an_objective ATypesMetadata
    # @param objective_column string indicating objective column
    # @return None if nothing to DO or Normalization_sets orderdict() on other way
    def define_minimal_norm(self, dataframe_metadata, an_objective, objective_column):
        df_type = dataframe_metadata['type']
        if not self._config['minimal_normalizations_enabled']:
            return [None]
        elif objective_column is None:
            norms = list()
            normoption = NormalizationSet()
            columns = dataframe_metadata['columns']
            for description in columns:
                col = description['name']
                if description['type'] == "object" and self._config['base_normalization_enabled']:
                    normoption.set_base(datetime=False)
                    norms.append({col: normoption.copy()})
            return norms.copy()
        else:
            if df_type == 'pandas':
                rowcount = dataframe_metadata['rowcount']
                norms = list()
                normoption = NormalizationSet()
                normoption.set_drop_missing()
                norms.append({objective_column: normoption.copy()})

                columns = dataframe_metadata['columns']
                for description in columns:
                    col = description['name']
                    if col != objective_column:
                        if description['type'] == "object" and self._config['base_normalization_enabled']:
                            normoption.set_base()
                            norms.append({col: normoption.copy()})
                        if int(description['missed']) > 0 and \
                           (int(description['missed'])/rowcount >= self._config['exclusion_missing_threshold']):
                            if an_objective[0]['type'] in ['binomial', 'multinomial'] and self._config['manage_on_train_errors']:
                                normoption.set_mean_missing_values(objective_column, full=False)
                                norms.append({col: normoption.copy()})
                            elif an_objective[0]['type'] in ['regression'] and self._config['manage_on_train_errors']:
                                normoption.set_progressive_missing_values(objective_column)
                                norms.append({col: normoption.copy()})
                            elif an_objective[0]['type'] in ['anomalies']:
                                normoption.set_mean_missing_values(objective_column, full=True)
                                norms.append({col: normoption.copy()})
                            else:
                                normoption.set_mean_missing_values(objective_column, full=True)
                                norms.append({col: normoption.copy()})
                return norms.copy()

    ## Method oriented to filter stdmean operations on non standardize algorithms
    # @param normalizemd OrderedDict() compatible structure
    # @param model_id Model_identification
    # @return normalizemd OrderedDict() compatible structure
    def filter_standardize(self, normalizemd, model_id):
        filter_normalized = list()
        for norm_set in normalizemd:
            if norm_set is not None:
                col = list(norm_set.keys())[0]
                norms = norm_set.get(col)
                if norms['class'] == 'stdmean' and model_id in NO_STANDARDIZE:
                    self._logging.log_info('gDayF', "Normalizer", self._labels["excluding"],
                                           col + ' - ' + norms['class'])
                else:
                    filter_normalized.append(norm_set)
        return filter_normalized

    ## Method oriented to filter drop_missing operations on non standardize algorithms
    # @param normalizemd OrderedDict() compatible structure
    # @return normalizemd OrderedDict() compatible structure
    def filter_drop_missing(self, normalizemd):
        filter_normalized = list()
        for norm_set in normalizemd:
            if norm_set is not None:
                col = list(norm_set.keys())[0]
                norms = norm_set.get(col)
                if norms['class'] == 'drop_missing':
                    self._logging.log_exec('gDayF', "Normalizer", self._labels["excluding"],
                                           col + ' - ' + norms['class'])
                else:
                    filter_normalized.append(norm_set)

        return filter_normalized

    ## Method oriented to filter filling_missing operations dependent of objective_column
    # @param normalizemd OrderedDict() compatible structure
    # @return normalizemd OrderedDict() compatible structure
    def filter_objective_base(self, normalizemd):
        filter_normalized = list()
        for norm_set in normalizemd:
            if norm_set is not None:
                col = list(norm_set.keys())[0]
                norms = norm_set.get(col)
                if norms['class'] == 'progressive_missing_values' or \
                   (norms['class'] == 'mean_missing_values' and not norms['objective']['full']):
                    self._logging.log_exec('gDayF', "Normalizer", self._labels["excluding"],
                                           col + ' - ' + norms['class'])
                else:
                    filter_normalized.append(norm_set)

        return filter_normalized

    ## Main method oriented to define and manage normalizations sets applying normalizations
    # @param self object pointer
    # @param df dataframe
    # @param normalizemd OrderedDict() compatible structure
    # @return dataframe
    def normalizeDataFrame(self, df, normalizemd):
        self._logging.log_info('gDayF', "Normalizer", self._labels["start_data_norm"], str(df.shape))
        if isinstance(df, pd.DataFrame):
            dataframe = df.copy()
            for norm_set in normalizemd:
                if norm_set is not None:
                    col = list(norm_set.keys())[0]
                    norms = norm_set.get(col)
                    if norms['class'] == 'base':
                        dataframe.loc[:, col] = self.normalizeBase(dataframe.loc[:, col])
                        self._logging.log_info('gDayF', "Normalizer", self._labels["applying"],
                                               col + ' - ' + norms['class'])
                        if dataframe[col].dtype == '<M8[ns]' and norms['datetime']:
                            dataframe = self.normalizeDateTime(dataframe=dataframe, date_column=col)
                            if self._config['datetime_columns_management'] is not None \
                                    and self._config['datetime_columns_management']['enable']:
                                self._logging.log_info('gDayF', "Normalizer", self._labels["applying"],
                                                       col + ' - ' + str(self._config['datetime_columns_management']
                                                                         ['filter']))
                    elif norms['class'] == 'drop_missing':
                        try:
                            dataframe = self.normalizeDropMissing(dataframe, col)
                            self._logging.log_info('gDayF', "Normalizer", self._labels["applying"],
                                                   col + ' - ' + norms['class'])
                        except KeyError:
                            self._logging.log_info('gDayF', "Normalizer", self._labels["excluding"],
                                               col + ' - ' + norms['class'])
                    elif norms['class'] == 'stdmean':
                        dataframe.loc[:, col] = self.normalizeStdMean(dataframe.loc[:, col],
                                                                      norms['objective']['mean'],
                                                                      norms['objective']['std']
                                                                      )
                        self._logging.log_info('gDayF', "Normalizer", self._labels["applying"],
                                               col + ' - ' + norms['class'] + ' ( ' +
                                               str(norms['objective']['mean']) + ',' +
                                               str(norms['objective']['std']) + ' ) ')
                    elif norms['class'] == 'working_range':
                        dataframe.loc[:, col] = self.normalizeWorkingRange(dataframe.loc[:, col],
                                                                           norms['objective']['minval'],
                                                                           norms['objective']['maxval'],
                                                                           norms['objective']['minrange'],
                                                                           norms['objective']['maxrange']
                                                                           )
                        self._logging.log_info('gDayF', "Normalizer", self._labels["applying"],
                                               col + ' - ' + norms['class'] + ' ( ' +
                                               str(norms['objective']['minval']) + ',' +
                                               str(norms['objective']['maxval']) + ' ) ')
                    elif norms['class'] == 'offset':
                        dataframe.loc[:, col] = self.normalizeOffset(dataframe.loc[:, col],
                                                                     norms['objective']['offset'])
                        self._logging.log_info('gDayF', "Normalizer", self._labels["applying"],
                                               col + ' - ' + norms['class'] + ' ( ' +
                                               str(norms['objective']['offset']) + ' )')
                    elif norms['class'] == 'discretize':
                        dataframe.loc[:, col] = self.normalizeDiscretize(dataframe.loc[:, col],
                                                                         norms['objective']['buckets_number'],
                                                                         norms['objective']['fixed_size'])
                        self._logging.log_info('gDayF', "Normalizer", self._labels["applying"],
                                               col + ' - ' + norms['class'] + ' ( ' +
                                               str(norms['objective']['buckets_number']) + ',' +
                                               str(norms['objective']['fixed_size']) + ' ) ')
                    elif norms['class'] == 'aggregation':
                        dataframe.loc[:, col] = self.normalizeAgregation(dataframe.loc[:, col],
                                                                         norms['objective']['bucket_ratio'])
                        self._logging.log_info('gDayF', "Normalizer", self._labels["applying"],
                                               col + ' - ' + norms['class'] + ' ( ' +
                                               str(norms['objective']['bucket_ratio']) + ' ) ')
                    elif norms['class'] == 'fixed_missing_values':
                        dataframe.loc[:, col] = self.fixedMissingValues(dataframe.loc[:, col], norms['objective']['value'])
                        self._logging.log_info('gDayF', "Normalizer", self._labels["applying"],
                                               col + ' - ' + norms['class'] + ' ( ' +
                                               str(norms['objective']['value']) + ' ) ')
                    elif norms['class'] == 'mean_missing_values':
                        dataframe = self.meanMissingValues(dataframe,
                                                           col,
                                                           norms['objective']['objective_column'],
                                                           norms['objective']['full']
                                               )
                        if norms['objective']['objective_column'] is None:
                            norms['objective']['objective_column'] = 'None'
                        self._logging.log_info('gDayF', "Normalizer", self._labels["applying"],
                                               col + ' - ' + norms['class'] + ' ( ' +
                                               norms['objective']['objective_column'] + ',' +
                                               str(norms['objective']['full']) + ' ) ')
                    elif norms['class'] == 'progressive_missing_values':
                        dataframe = self.progressiveMissingValues(dataframe,
                                                                  col,
                                                                  norms['objective']['objective_column'])
                        self._logging.log_info('gDayF', "Normalizer", self._labels["applying"],
                                               col + ' - ' + norms['class'] + ' ( ' +
                                               norms['objective']['objective_column'] + ' ) ')
                    elif norms['class'] == 'ignore_column':
                        pass
                    #elif norms['class'] == 'binary_encoding':
                    #self.normalizeBinaryEncoding(dataframe[col])
                else:
                    self._logging.log_info('gDayF', "Normalizer", self._labels["nothing_to_do"])
            return dataframe
        else:
            return df

    ##Method oriented to generate ignored_column_list on issues where missed > exclusion_missing_threshold
    # @param  normalizemd  mormalizations_set_metadata
    # @return ignored_list updated
    def ignored_columns(self, normalizemd):
        ignored_list = list()
        if normalizemd is not None:
            for elements in normalizemd:
                for col, value in elements.items():
                    if value['class'] == 'ignore_column':
                        ignored_list.append(col)
        self._logging.log_info('gDayF', "Normalizer", self._labels["ignored_list"], ignored_list)
        return ignored_list.copy()

    ## Internal method oriented to manage drop NaN values from dataset
    # @param self object pointer
    # @param dataframe single column dataframe
    # @return dataframe
    def normalizeBase(self, dataframe):
        if dataframe.dtype == np.object:
            try:
                return pd.to_numeric(dataframe)
            except ValueError:
                try:
                    return pd.to_datetime(dataframe)
                except ValueError:
                    return pd.Categorical(dataframe)

    ## Internal method oriented to manage base normalizations
    # @param self object pointer
    # @param dataframe single column dataframe
    # @param col column base to reference drop NaN
    # @return dataframe
    def normalizeDropMissing(self, dataframe, col):
        return dataframe.dropna(axis=0, subset=[col])

    ## Internal method oriented to manage Working range normalizations on a [closed, closed] interval
    # @param self object pointer
    # @param dataframe single column dataframe
    # @param minval
    # @param maxval
    # @return dataframe
    def normalizeWorkingRange(self, dataframe, minval=-1.0, maxval=1.0, minrange = -1.0, maxrange = 1.0):
        assert(maxval > minval)
        if dataframe.dtype != np.object:
            if dataframe.dtype != np.object:
                convert_factor = (maxrange - minrange) / (maxval - minval)
                dataframe = dataframe.astype(np.float16)
                dataframe = dataframe.apply(lambda x: (x-minval) * convert_factor + minrange)
            return dataframe.copy()

    ## Internal method oriented to manage Working range normalizations on a [closed, closed] interval
    # @param self object pointer
    # @param dataframe single column dataframe
    # @param minval
    # @param maxval
    # @return dataframe
    def normalizeOffset(self, dataframe, offset=0):
        if dataframe.dtype != np.object:
            dataframe = offset + dataframe
        return dataframe.copy()

    ## Internal method oriented to manage bucket ratio normalizations head - tail
    # @param self object pointer
    # @param dataframe single column dataframe
    # @param br bucket ratio
    # @return dataframe
    def normalizeAgregation(self, dataframe, br=0.25):
        if (dataframe.dtype != np.object):
            buckets = int(1 / (br/2))
            q, bins = pd.qcut(dataframe.iloc[:], buckets, retbins=True)
            if dataframe.dtype != np.int:
                dataframe[dataframe <= bins[1]] = np.int(dataframe[dataframe <= bins[1]].mean().copy())
                dataframe[dataframe >= bins[-2]] = np.int(dataframe[dataframe >= bins[-2]].mean().copy())
            else:
                dataframe[dataframe <= bins[1]] = dataframe[dataframe <= bins[1]].mean().copy()
                dataframe[dataframe <= bins[-2]] = dataframe[dataframe <= bins[-2]].mean().copy()
        return dataframe.copy()

    ## Internal method oriented to manage Binary encodings
    # @param self object pointer
    # @param dataframe single column dataframe
    # @return dataframe
    def normalizeBinaryEncoding(self, dataframe):
        return dataframe.copy()

    ## Internal method oriented to manage mean and std normalizations. Default mean=0 std=1
    # @param self object pointer
    # @param dataframe single column dataframe
    # @param mean mean value to center
    # @param std standard deviation value to be normalized
    # @return dataframe
    def normalizeStdMean(self, dataframe, mean, std):
        if dataframe.dtype != np.object and dataframe.dtype != "datetime64[ns]":
            try:
                dataframe = dataframe.astype(np.float64)
                dataframe = dataframe.apply(lambda x: x - float(mean))
                dataframe = dataframe.apply(lambda x: x / float(std))
            except ZeroDivisionError:
                dataframe = dataframe.apply(lambda x: x + float(mean))
            #dataframe = preprocessing.scale(dataframe)
        return dataframe.copy()

    ## Internal method oriented to manage bucketing for discretize
    # @param self object pointer
    # @param dataframe single column dataframe
    # @param buckets_number Int
    # @param fixed_size Boolean (True=Fixed Size, False Fixed Frecuency
    # @return dataframe
    def normalizeDiscretize(self, dataframe, buckets_number, fixed_size):
        #Un número de buckets de tamaño fixed_size
        if fixed_size:
            return pd.qcut(dataframe, buckets_number)
        else:
            return pd.cut(dataframe, buckets_number)

    ## Internal method oriented to manage imputation for missing values to fixed value
    # @param self object pointer
    # @param dataframe single column dataframe
    # @param value int
    # @return dataframe
    def fixedMissingValues(self, dataframe, value=0.0):
        return dataframe.fillna(value)

    ## Internal method oriented to manage imputation for missing values to mean value
    # @param self object pointer
    # @param dataframe full column dataframe
    # @param col column name for imputation
    # @param objective_col objective_column
    # @param full True means fll_dataframe.mean(), False means objective_col.value.mean()
    # @return dataframe
    def meanMissingValues(self, dataframe, col, objective_col, full=False):
        if full:
            return dataframe.fillna(dataframe.mean())
        else:
            nullfalse = dataframe[dataframe[:][col].notnull()][[col, objective_col]]
            if objective_col in DTYPES:
                nullfalse_gb = nullfalse.groupby(objective_col).mean()
            else:
                nullfalse_gb = nullfalse.groupby(objective_col).agg(lambda x: x.value_counts().index[0])
            for index, row in dataframe[dataframe[:][col].isnull()].iterrows():
                row = row.copy()
                if nullfalse_gb.index.isin([row[objective_col]]).any():
                    dataframe.loc[index, col] = nullfalse_gb.loc[row[objective_col], col]
            return dataframe.copy()

    ## Internal method oriented to manage progressive imputations for missing values.
    # ([right_not_nan] - [left_not_nan])/Cardinality(is_nan)
    # @param self object pointer
    # @param dataframe full column dataframe
    # @param col column name for imputation
    # @param objective_col objective_column
    # @return dataframe
    def progressiveMissingValues(self, dataframe, col, objective_col):
        nullfalse = dataframe[dataframe[:][col].notnull()].sort_values(objective_col,
                                                                       axis=0,
                                                                       ascending=True)[[col, objective_col]]
        nullfalse_gb = nullfalse.groupby(objective_col).mean()
        for index, row in dataframe[dataframe[:][col].isnull()].iterrows():
            row = row.copy()
            if nullfalse_gb.index.isin([row[objective_col]]).any():
                dataframe.loc[index, col] = nullfalse_gb.loc[row[objective_col], col]
            else:
                index_max = nullfalse_gb.index.where(nullfalse_gb.index > row[objective_col]).min()
                index_min = nullfalse_gb.index.where(nullfalse_gb.index < row[objective_col]).max()
                try:
                    if index_min is np.nan and index_max is np.nan \
                       or index_min is None or index_max is None:
                        pass
                    if index_min is np.nan or index_min is None:
                        dataframe.loc[index, col] = nullfalse_gb.loc[index_max, col]
                    elif index_max is np.nan or index_max is None:
                        dataframe.loc[index, col] = nullfalse_gb.loc[index_min, col]
                    else:
                        minimal = min(nullfalse_gb.loc[index_min, col], nullfalse_gb.loc[index_max, col])
                        maximal = max(nullfalse_gb.loc[index_min, col], nullfalse_gb.loc[index_max, col])
                        b = maximal - minimal
                        a = index_max - index_min
                        x = (row[objective_col] - index_min) / a
                        offset = b * x
                        dataframe.loc[index, col] = minimal + offset
                except TypeError:
                    pass
        return dataframe.copy()

    ## Internal method oriented to manage date_time conversions to pattern
    # @param self object pointer
    # @param dataframe full column dataframe to be expanded
    # @param date_column Date_Column name to be transformed
    # @return dataframe
    def normalizeDateTime(self, dataframe, date_column=None):
        datetime_columns_management = self._config['datetime_columns_management']
        if date_column is not None:
            if datetime_columns_management is not None and datetime_columns_management['enable']:
                    for element in datetime_columns_management['filter']:
                        try:
                            if element not in ['weekday', 'weeknumber']:
                                dataframe[date_column + '_' + element] = dataframe.loc[:, date_column]\
                                    .transform(lambda x: eval('x.' + element))
                            elif element == 'weekday':
                                dataframe[date_column + '_' + element] = dataframe.loc[:, date_column]\
                                    .transform(lambda x: x.isoweekday())
                            elif element == 'weeknumber':
                                dataframe[date_column + '_' + element] = dataframe.loc[:, date_column]\
                                    .transform(lambda x: x.isocalendar()[1])
                        except AttributeError:
                            print('TRC: invalid configuration:' + element)
                            pass
        return dataframe.copy()




