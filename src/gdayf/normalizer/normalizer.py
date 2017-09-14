#!/usr/bin/python3
import numpy as np
import pandas as pd
from sklearn import preprocessing
from gdayf.conf.loadconfig import LoadConfig
from gdayf.conf.loadconfig import LoadLabels
from gdayf.logs.logshandler import LogsHandler
from gdayf.common.utils import dtypes
from gdayf.common.normalizationset import NormalizationSet



## Class oriented to manage normalizations on dataframes for improvements on accuracy
class Normalizer (object):
    def __init__(self):
        self._config = LoadConfig().get_config()['normalizer']
        self._labels = LoadLabels().get_config()['messages']['normalizer']
        self._logging = LogsHandler(__name__)

    ## Method oriented to specificate data_normalizations
    # @param dataframe_metadata DFMetadata()
    # @param an_objective ATypesMetadata
    # @param objective_column string indicating objective column
    # @return None if nothing to DO or Normalization_sets orderdict() on other way
    def define_normalizations(self, dataframe_metadata, an_objective, objective_column):
        if not self._config['non_minimal_normalizations_enabled']:
            return None
        else:
            type = dataframe_metadata['type']
            print(dataframe_metadata['type'])
            rowcount = dataframe_metadata['rowcount']
            cols = dataframe_metadata['cols']
            columns = dataframe_metadata['columns']
            norms = list()
            normoption = NormalizationSet()
            if type == 'pandas':
                '''Modified 11/09/2017
                for description in columns:
                    col = description['name']
                    self._logging.log_exec('gDayF', "Normalizer", self._labels["col_analysis"],
                                           description['name'] + ' - ' + description['type'])
                    if objective_column is not None and col == objective_column:
                        norm_aux = self.define_minimal_norm(dataframe_metadata, an_objective, objective_column)
                        if norm_aux is not None:
                            norms.append(norm_aux)
                        if description['type'] == "object" and self._config['base_normalization_enabled']:
                            normoption.set_base()
                            norms.append({col: normoption.copy()})
                '''
                for description in columns:
                    col = description['name']
                    if col != objective_column:
                        if description['type'] == "object" and self._config['base_normalization_enabled']:
                            normoption.set_base()
                            norms.append({col: normoption.copy()})
                        if int(description['missed']) > 0 and \
                           (int(description['missed'])/rowcount <= self._config['exclusion_missing_threshold']):
                            if an_objective[0]['type'] in ['binomial', 'multinomial']:
                                normoption.set_mean_missing_values(objective_column, full=False)
                                norms.append({col: normoption.copy()})
                            elif an_objective[0]['type'] in ['regression']:
                                normoption.set_progressive_missing_values(objective_column)
                                norms.append({col: normoption.copy()})
                            elif an_objective[0]['type'] in ['anomalies']:
                                normoption.set_mean_missing_values(objective_column, full=True)
                                norms.append({col: normoption.copy()})
                        elif int(description['missed']) > 0:
                            normoption.set_ignore_column()
                            norms.append({col: normoption.copy()})
                        if self._config['clustering_standardize_enabled'] and an_objective[0]['type'] in ['clustering']:
                            normoption.set_stdmean()
                            norms.append({col: normoption.copy()})

                self._logging.log_exec('gDayF', "Normalizer", self._labels["norm_set_establish"], norms)
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
    # @return None if nothing to DO or Normalization_sets orderdict() on other way
    def define_minimal_norm(self, dataframe_metadata, an_objective, objective_column):

        type = dataframe_metadata['type']
        if not self._config['minimal_normalizations_enabled']:
            return [None]
        elif objective_column is None:
            columns = dataframe_metadata['columns']
            norms = list()
            normoption = NormalizationSet()
            if type == 'pandas':
                for description in columns:
                    col = description['name']
                    if an_objective[0]['type'] in ['anomalies']:
                        normoption.set_stdmean()
                        norms.append({col: normoption.copy()})
                if len(norms) == 0:
                    return [None]
                else:
                    return norms.copy()
            else:
                return [None]
        else:
            if type == 'pandas':
                norms = list()
                normoption = NormalizationSet()
                normoption.set_drop_missing()
                norms.append({objective_column: normoption.copy()})
                return norms.copy()

    ## Main method oriented to define and manage normalizations sets applying normalizations
    # @param self object pointer
    # @param df dataframe
    # @param normalizemd OrderedDict() compatible structure
    # @return dataframe
    def normalizeDataFrame(self, df, normalizemd):
        self._logging.log_exec('gDayF', "Normalizer", self._labels["start_data_norm"])
        if isinstance(df, pd.DataFrame):
            dataframe = df.copy()
            for norm_set in normalizemd:
                if norm_set is not None:
                    col = list(norm_set.keys())[0]
                    norms = norm_set.get(col)
                #for col, norms in normalizemd['columns'].items():
                    if norms['class'] == 'base':
                        dataframe.loc[:, col] = self.normalizeBase(dataframe.loc[:, col])
                        self._logging.log_exec('gDayF', "Normalizer", self._labels["applying"],
                                               col + ' - ' + norms['class'])
                    elif norms['class'] == 'drop_missing':
                        dataframe = self.normalizeDropMissing(dataframe, col)
                        self._logging.log_exec('gDayF', "Normalizer", self._labels["applying"],
                                               col + ' - ' + norms['class'])
                    elif norms['class'] == 'stdmean':
                        dataframe.loc[:, col] = self.normalizeStdMean(dataframe[col])
                        self._logging.log_exec('gDayF', "Normalizer", self._labels["applying"],
                                               col + ' - ' + norms['class'])
                    elif norms['class'] == 'working_range':
                        dataframe.loc[:, col] = self.normalizeWorkingRange(dataframe.loc[:, col],
                                                                           norms['objective']['minval'],
                                                                           norms['objective']['maxval'])
                        self._logging.log_exec('gDayF', "Normalizer", self._labels["applying"],
                                               col + ' - ' + norms['class'] + ' ( ' +
                                               str(norms['objective']['minval']) + ',' +
                                               str(norms['objective']['maxval']) + ' ) ')
                    elif norms['class'] == 'discretize':
                        dataframe.loc[:, col] = self.normalizeDiscretize(dataframe.loc[:, col],
                                                                         norms['objective']['buckets_number'],
                                                                         norms['objective']['fixed_size'])
                        self._logging.log_exec('gDayF', "Normalizer", self._labels["applying"],
                                               col + ' - ' + norms['class'] + ' ( ' +
                                               str(norms['objective']['buckets_number']) + ',' +
                                               str(norms['objective']['fixed_size']) + ' ) ')
                    elif norms['class'] == 'aggregation':
                        dataframe.loc[:, col] = self.normalizeAgregation(dataframe.loc[:, col],
                                                                         norms['objective']['bucket_ratio'])
                        self._logging.log_exec('gDayF', "Normalizer", self._labels["applying"],
                                               col + ' - ' + norms['class'] + ' ( ' +
                                               str(norms['objective']['bucket_ratio']) + ' ) ')
                    elif norms['class'] == 'fixed_missing_values':
                        dataframe.loc[:, col] = self.fixedMissingValues(dataframe.loc[:, col], norms['objective']['value'])
                        self._logging.log_exec('gDayF', "Normalizer", self._labels["applying"],
                                               col + ' - ' + norms['class'] + ' ( ' +
                                               str(norms['objective']['value']) + ' ) ')
                    elif norms['class'] == 'mean_missing_values':
                        dataframe = self.meanMissingValues(dataframe,
                                                           col,
                                                           norms['objective']['objective_column'],
                                                           norms['objective']['full']
                                               )
                        self._logging.log_exec('gDayF', "Normalizer", self._labels["applying"],
                                               col + ' - ' + norms['class'] + ' ( ' +
                                               norms['objective']['objective_column'] + ',' +
                                               str(norms['objective']['full']) + ' ) ')
                    elif norms['class'] == 'progressive_missing_values':
                        dataframe = self.progressiveMissingValues(dataframe,
                                                                  col,
                                                                  norms['objective']['objective_column'])
                        self._logging.log_exec('gDayF', "Normalizer", self._labels["applying"],
                                               col + ' - ' + norms['class'] + ' ( ' +
                                               norms['objective']['objective_column'] + ' ) ')
                    elif norms['class'] == 'ignore_column':
                        pass
                    #elif norms['class'] == 'binary_encoding':
                    #self.normalizeBinaryEncoding(dataframe[col])
                else:
                    self._logging.log_exec('gDayF', "Normalizer", self._labels["nothing_to_do"])
            return dataframe
        else:
            return df

    ##Method oriented to generate ignored_column_list on issues where missed > exclusion_missing_threshold
    # @param  normalizemd  mormalizations_set_metadata
    # @return ignored_list updated
    def ignored_columns(self, normalizemd):
        ignored_list = list()
        if normalizemd is not None:
            for col, norms in normalizemd['columns'].items():
                if norms['class'] == 'ignore_column':
                    ignored_list.append(col)
        self._logging.log_exec('gDayF', "Normalizer", self._labels["ignore_list"], ignored_list)
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
    def normalizeWorkingRange(self, dataframe, minval=0, maxval=1):
        assert(maxval > minval)
        if dataframe.dtype != np.object:
            dataframe = (maxval - minval) * ((dataframe - dataframe.min()) /
                                             (dataframe.max() - dataframe.min())) + minval
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
    # @return dataframe
    def normalizeStdMean(self, dataframe):
        if (dataframe.dtype != np.object):
            #dataframe = (dataframe - dataframe.mean()) / dataframe.std()
            dataframe = preprocessing.scale(dataframe)
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
            if objective_col in dtypes:
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
                if index_min is np.nan:
                    dataframe.loc[index, col] = nullfalse_gb.loc[index_max, col]
                elif index_max is np.nan:
                    dataframe.loc[index, col] = nullfalse_gb.loc[index_min, col]
                else:
                    minimal = min(nullfalse_gb.loc[index_min, col], nullfalse_gb.loc[index_max, col])
                    maximal = max(nullfalse_gb.loc[index_min, col], nullfalse_gb.loc[index_max, col])
                    b = maximal - minimal
                    a = index_max - index_min
                    x = (row[objective_col] - index_min) / a
                    offset = b * x
                    dataframe.loc[index, col] = minimal + offset
        return dataframe.copy()


