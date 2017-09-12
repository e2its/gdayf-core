## @package gdayf.handlers.h2ohandler
# Define all objects, functions and structures related to executing actions or activities over h2o.ai framework
#
# Main class H2OHandler. Lets us execute analysis, make prediction and execute multi-packet operations structures on
# format [(Analysis_results.json, normalization_sets.json) ]
# Analysis_results.json could contain executions models for various different model or parameters


import copy
import json
import time
from collections import OrderedDict as OrderedDict
from os.path import dirname
from pandas import DataFrame as DataFrame
from hashlib import md5 as md5
from pathlib import Path

from h2o import H2OFrame as H2OFrame
from h2o import cluster as cluster
from h2o import connect as connect
from h2o import connection as connection
from h2o import init as init
from h2o import load_model as load_model
from h2o import save_model as save_model
from h2o.exceptions import H2OError
from h2o.exceptions import H2OConnectionError
from h2o import ls as H2Olist
from h2o import get_model
from h2o import remove as H2Oremove
from h2o import api as H2Oapi
from h2o import get_frame
from h2o import download_pojo

from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator
from h2o.estimators.kmeans import H2OKMeansEstimator

from gdayf.common.normalizationset import NormalizationSet
from gdayf.common.storagemetadata import StorageMetadata
from gdayf.common.storagemetadata import generate_json_path
from gdayf.common.utils import hash_key
from gdayf.logs.logshandler import LogsHandler
from gdayf.metrics.binomialmetricmetadata import BinomialMetricMetadata
from gdayf.metrics.metricmetadata import MetricMetadata
from gdayf.metrics.executionmetriccollection import ExecutionMetricCollection
from gdayf.metrics.regressionmetricmetadata import RegressionMetricMetadata
from gdayf.metrics.multinomialmetricmetadata import MultinomialMetricMetadata
from gdayf.metrics.anomaliesmetricmetadata import AnomaliesMetricMetadata
from gdayf.metrics.clusteringmetricmetadata import ClusteringMetricMetadata
from gdayf.persistence.persistencehandler import PersistenceHandler
from gdayf.conf.loadconfig import LoadConfig
from gdayf.conf.loadconfig import LoadLabels
from gdayf.common.dfmetada import DFMetada
from gdayf.common.utils import get_model_ns
from gdayf.common.armetadata import ArMetadata
from gdayf.models.parametersmetadata import ParameterMetadata
from gdayf.normalizer.normalizer import Normalizer
from gdayf.common.utils import get_model_fw


class H2OHandler(object):

    ## Constructor
    # Initialize all framework variables and starts or connect to h2o cluster
    # Aditionally starts PersistenceHandler and logsHandler
    def __init__(self):
        self._framework = 'h2o'
        self._config = LoadConfig().get_config()
        self._labels = LoadLabels().get_config()['messages']['corehandler']
        self.path_localfs = self._config['frameworks'][self._framework]['conf']['path_localfs']
        self.path_hdfs =self._config['frameworks'][self._framework]['conf']['path_hdfs']
        self.path_mongoDB = self._config['frameworks'][self._framework]['conf']['path_mongoDB']
        self.primary_path = \
            self._config['frameworks'][self._framework]['conf'] \
            [self._config['frameworks'][self._framework]['conf']['primary_path']]
        self.url = self._config['frameworks'][self._framework]['conf']['url']
        self.nthreads = self._config['frameworks'][self._framework]['conf']['nthreads']
        self.ice_root = self._config['frameworks'][self._framework]['conf']['ice_root']
        self.max_mem_size = self._config['frameworks'][self._framework]['conf']['max_mem_size']
        self.start_h2o = self._config['frameworks'][self._framework]['conf']['start_h2o']
        self._debug = self._config['frameworks'][self._framework]['conf']['debug']
        self._save_model = self._config['frameworks'][self._framework]['conf']['save_model']
        self._tolerance = self._config['frameworks'][self._framework]['conf']['tolerance']
        self._anomaly_threshold = self._config['frameworks'][self._framework]['conf']['anomaly_threshold']
        self._model_base = None
        self.analysis_id = None
        self._h2o_session = None
        self._persistence = PersistenceHandler()
        self._logging = LogsHandler(__name__)
        self._frame_list = list()

    ## Destructor
    def __del__(self):
        if self._h2o_session is not None and self.is_alive():
            H2Oapi("POST /3/GarbageCollect")
            self._h2o_session.close()


    ## Class Method for cluster shutdown
    # @param cls class pointer
    @classmethod
    def shutdown_cluster(cls):
        try:
            cluster().shutdown()
        except:
            print('H20-cluster not working')

    ## Connexion_method to cluster
    #If cluster is up connect to cluster on another case start a cluster
    # @return initiatiated True/False
    def connect(self):
        initiated = False
        try:
            self._h2o_session = connect(url=self.url)
        except H2OError:
            try:
                init(url=self.url, nthreads=self.nthreads, ice_root=self.ice_root, max_mem_size=self.max_mem_size)
                self._h2o_session = connection()
                initiated = True
            except H2OError:
                self._logging.log_exec('gDayF', "H2OHandler", self._labels["failed_conn"])
                raise Exception
        finally:
            self._logging.log_exec('gDayF', "H2OHandler", self._labels["start"])
            self._logging.log_exec('gDayF', "H2OHandler", self._labels["framework"], self._framework)
            self._logging.log_exec('gDayF', "H2OHandler", self._labels["sess"], self._h2o_session.session_id())
            return initiated

    ## Is alive_connection method
    def is_alive(self):
        if self._h2o_session is None:
            return False
        else:
            try:
                self._h2o_session.session_id()
            except H2OConnectionError:
                return False
            return self._h2o_session.cluster.is_running()

    ## Generate list of models_id for internal crossvalidation objects_
    # @param self object pointer
    # @param model_id base id_model
    # @param nfols number of cv buckets
    # @return models_ids lst of models_ids
    def _get_cv_ids(self, model_id, nfolds):
        models_ids = list()
        for iter in range(0, nfolds):
            models_ids.append(model_id + '_cv_' + str(iter+1))
        return models_ids

    ## Generate list of models_id for internal crossvalidation objects_
    # @param self object pointer
    # @param ar_metadata ArMetadata stored model
    # @param type ['pojo', 'mojo']
    # @return download_path, MD5 hash_key
    def get_java_model(self, ar_metadata, type):
        remove_model = False
        fw = get_model_fw(ar_metadata)
        model_id = ar_metadata['model_parameters'][fw]['parameters']['model_id']['value']
        self.analysis_id = ar_metadata['model_id']
        analysis_id = self.analysis_id
        config = LoadConfig().get_config()['frameworks'][fw]['conf']
        base_model_id = model_id + '.model'
        load_fails, remove_model = self._get_model(ar_metadata, base_model_id, remove_model)

        if load_fails:
            self._logging.log_exec(self.analysis_id, self._h2o_session.session_id,
                                   self._labels["no_models"], ar_metadata)
            return None

        download_path = Path(config[config['primary_path']])
        download_path = download_path.joinpath(config['download_fs'])
        download_path = download_path.joinpath(model_id)
        self._logging.log_exec(self.analysis_id, self._h2o_session.session_id,
                               self._labels["down_path"], download_path)
        persistence = PersistenceHandler()
        persistence.mkdir(type='localfs', path=str(download_path), grants=0o0777)


        if type.upper() == 'MOJO':
            try:
                file_path = self._model_base.download_mojo(path=str(download_path), get_genmodel_jar=True)
            except H2OError:
                load_fails = True
                self._logging.log_exec(self.analysis_id, self._h2o_session.session_id,
                                       self._labels["failed_op"], download_path)
        else:
            try:
                file_path = download_pojo(self._model_base, path=str(download_path), get_jar=True)
            except H2OError:
                load_fails = True
                self._logging.log_exec(self.analysis_id, self._h2o_session.session_id,
                                       self._labels["failed_op"], download_path)
        try:
            if self._model_base is not None and remove_model:
                H2Oremove(self._model_base.model_id)
        except H2OError:
            self._logging.log_exec(analysis_id,
                                   self._h2o_session.session_id, self._labels["delete_objects"],
                                   self._model_base.model_id)

        if not load_fails:
            return download_path, hash_key('MD5', filename=file_path)
        else:
            return None

    ## Generate list of models_id for internal crossvalidation objects_
    # @param self object pointer
    # @param ar_metadata ArMetadata stored model
    # @return implicit self._model_base / None on Error
    def get_model_from_load_path(self, ar_metadata):
        load_fails = True
        counter_storage = 0
        # Checking file source versus hash_value

        try:
            assert isinstance(ar_metadata['load_path'], list)
        except AssertionError:
            return load_fails

        while ar_metadata['load_path'] is not None and counter_storage < len(ar_metadata['load_path']) and load_fails:

            if hash_key(ar_metadata['load_path'][counter_storage]['hash_type'],
                        ar_metadata['load_path'][counter_storage]['value']) == \
                    ar_metadata['load_path'][counter_storage]['hash_value']:
                try:
                    self._model_base = load_model(ar_metadata['load_path'][counter_storage]['value'])
                    if self._model_base is not None:
                        load_fails = False
                except H2OError:
                    self._logging.log_exec(self.analysis_id, self._h2o_session.session_id,
                                           self._labels["abort"], ar_metadata['load_path'][counter_storage]['value'])
            counter_storage += 1
            self._logging.log_exec(self.analysis_id, self._h2o_session.session_id, self._labels["hk_check"],
                                   ar_metadata['load_path'][counter_storage]['hash_value'] + ' - ' +
                                   hash_key(ar_metadata['load_path'][counter_storage]['hash_type'],
                                            ar_metadata['load_path'][counter_storage]['value'])
                                   )
        return load_fails

    ## Remove used dataframes during analysis execution_
    # @param self object pointer
    def delete_h2oframes (self):
        for frame_id in self._frame_list:
            try:
                H2Oremove(frame_id)
            except H2OError:
                self._logging.log_exec(self.analysis_id,
                                       self._h2o_session.session_id, self._labels["delete_frames"],
                                       frame_id)

    ## Generate base path to store all files [models, logs, json] relative to it
    # @param self object pointer
    # @param base_ar initial ar.json template pass to object instance
    # @param type_ type of analysis to execute
    # @return base path string
    def generate_base_path(self, base_ar, type_):
        assert type_ in ['PoC', 'train', 'predict']
        if self.primary_path == self.path_mongoDB:
            return None
        elif self.primary_path == self.path_hdfs:
            return None
        else:
            # Generating base_path
            load_path = list()
            load_path.append(self.path_localfs)
            load_path.append('/')
            load_path.append(self._framework)
            load_path.append('/')
            load_path.append(base_ar['model_id'])
            load_path.append('/')
            load_path.append(type_)
            load_path.append('/')
            load_path.append(str(base_ar['timestamp']))
            load_path.append('/')
            return ''.join(load_path)

    ## Generate extension for diferente saving modes
    # @param self object pointer
    # @return ['.pojo', '.mojo', '.model']
    def _get_ext(self):
        if self._save_model == 'POJO':
            return '.pojo'
        elif self._save_model == 'MOJO':
            return '.mojo'
        else:
            return '.model'

    ## Generate execution metrics for the correct model
    # @param self object pointer
    # @param dataframe H2OFrame for prediction metrics
    # @param source [train, val, xval]
    # @param  antype Atypemetadata().get_artypes() values allowed
    # @return model_metrics Subclass Metrics Metadata
    def _generate_execution_metrics(self, dataframe, source, antype):
        if antype == 'binomial':
            model_metrics = BinomialMetricMetadata()
        elif antype == 'multinomial':
            model_metrics = MultinomialMetricMetadata()
        elif antype == 'regression':
            model_metrics = RegressionMetricMetadata()
        elif antype == 'anomalies':
            model_metrics = AnomaliesMetricMetadata()
        elif antype == 'clustering':
            model_metrics = ClusteringMetricMetadata()
        else:
            model_metrics = MetricMetadata()

        if dataframe is not None and antype == 'anomalies':
            result1 = self._model_base.predict(dataframe)
            columns = list()
            for item in self._model_base.varimp():
                columns.append(item[0])
            x = list()
            for items in dataframe.columns:
                if items in columns:
                    x.append(items)

            difference = dataframe[x] - result1
            anomalies_threshold = OrderedDict()
            anomalies_threshold['global_mse'] = OrderedDict()
            reconstruction = self._model_base.anomaly(dataframe)
            anomalies_threshold['global_mse']['max'] = reconstruction.max()
            anomalies_threshold['global_mse']['min'] = reconstruction.min()

            anomalies_threshold['columns'] = OrderedDict()
            for col in difference.columns:
                anomalies_threshold['columns'][col] = OrderedDict()
                anomalies_threshold['columns'][col]['max'] = difference[col].max()
                anomalies_threshold['columns'][col]['min'] = difference[col].min()

            return anomalies_threshold

        elif dataframe is not None:
            perf_metrics = self._model_base.model_performance(dataframe)
        else:
            if source == 'valid':
                perf_metrics = self._model_base.model_performance(valid=True)
            elif source == 'xval':
                perf_metrics = self._model_base.model_performance(xval=True)
            else:
                perf_metrics = self._model_base.model_performance(train=True)

        model_metrics.set_h2ometrics(perf_metrics)
        #H2Oremove(perf_metrics)
        return model_metrics

    ## Generate model summary metrics
    # @param self object pointer
    # @return json_pandas_dataframe structure orient=split
    def _generate_model_metrics(self):
        return self._model_base.summary().as_data_frame().drop("", axis=1).to_json(orient='split')

    ## Generate variable importance metrics
    # @param self object pointer
    # @return OrderedDict() for variable importance Key=column name
    def _generate_importance_variables(self):
        aux = OrderedDict()
        try:
            for each_value in self._model_base.varimp():
                aux[each_value[0]] = each_value[2]
        except TypeError:
            pass
        return aux

    ## Generate model scoring_history metrics
    # @param self object pointer
    # @return json_pandas_dataframe structure orient=split
    def _generate_scoring_history(self):
        model_scoring = self._model_base.scoring_history()
        if model_scoring is None:
            return None
        else:
            return model_scoring.drop("", axis=1).to_json(orient='split')

    ## Generate accuracy metrics for model
    #for regression assume tolerance on results equivalent to 2*tolerance % over (max - min) values
    # on dataframe objective's column
    # @param self object pointer
    # @param dataframe H2OFrame
    # @param  antype Atypemetadata().get_artypes() values allowed
    # @param  base type BugFix on regression with range mixing int and float
    # @param tolerance (optional) default value 0.0. Only for regression
    # @return float accuracy of model
    def _accuracy(self, objective, dataframe, antype, base_type, tolerance=0.0):
        accuracy = -1.0
        prediccion = self._model_base.predict(dataframe)

        if antype == 'regression' and prediccion.type('predict') == base_type:
            fmin = eval("lambda x: x - " + str(tolerance/2))
            fmax = eval("lambda x: x + " + str(tolerance/2))
            success = dataframe[objective].apply(fmin) <= \
                       prediccion['predict'] <= \
                       dataframe[objective].apply(fmax)
            accuracy = "Valid"
        elif antype == 'regression':
            accuracy = 0.0
        else:
            tolerance = 0.0
            success = prediccion[0] == dataframe[objective]
            accuracy = "Valid"
        if accuracy not in [0.0, -1.0]:
            accuracy = success.sum() / dataframe.nrows

        self._frame_list.append(prediccion.frame_id)

        self._logging.log_exec(self.analysis_id, self._h2o_session.session_id, self._labels["tolerance"],
                               str(tolerance))
        return accuracy

    ## Generate accuracy metrics for model
    #for regression assume tolerance on results equivalent to 2*tolerance % over (max - min) values
    # on dataframe objective's column
    # @param self object pointer
    # @param dataframe normalized H2OFrame
    # @param  antype Atypemetadata().get_artypes() values allowed
    # @param  base type BugFix on regression with range mixing int and float
    # @param tolerance (optional) default value 0.0. Only for regression
    # @return float accuracy of model, prediction_dataframe
    def _predict_accuracy(self, objective, dataframe, antype, base_type, tolerance=0.0):
        accuracy = -1.0
        dataframe_cols = dataframe.columns
        prediction_dataframe = self._model_base.predict(dataframe)
        self._frame_list.append(prediction_dataframe.frame_id)
        prediccion = dataframe.cbind(prediction_dataframe)
        self._frame_list.append(prediction_dataframe.frame_id)
        prediction_columns = prediccion.columns
        for element in dataframe_cols:
            prediction_columns.remove(element)
        predictor_col = prediction_columns[0]

        if objective in dataframe.columns:
            if antype == 'regression' and prediccion.type(predictor_col) == base_type:
                fmin = eval("lambda x: x - " + str(tolerance/2))
                fmax = eval("lambda x: x + " + str(tolerance/2))
                success = prediccion[objective].apply(fmin).asnumeric() <= \
                          prediccion[predictor_col] <= \
                          prediccion[objective].apply(fmax).asnumeric()
                accuracy = "Valid"
            elif antype == 'regression':
                accuracy = 0.0
            else:
                tolerance = 0.0
                success = prediccion[predictor_col] == prediccion[objective]
                accuracy = "Valid"
        if accuracy not in [0.0, -1.0]:
            accuracy = success.sum() / dataframe.nrows

        self._logging.log_exec(self.analysis_id, self._h2o_session.session_id, self._labels["tolerance"],
                               str(tolerance))

        return accuracy, prediccion

    ## Generate detected anomalies on dataframe
    # @param self object pointer
    # @param odataframe original H2OFrame
    # @param dataframe normalized H2OFrame
    # @param anomalies_thresholds OrderedDict
    # @return OrderedDict with anomalies
    def _predict_anomalies(self, odataframe, dataframe, anomalies_thresholds):

        result1 = self._model_base.predict(dataframe)
        self._frame_list.append(result1.frame_id)

        columns = list()
        for item in self._model_base.varimp():
            columns.append(item[0])
        x = list()
        for items in dataframe.columns:
            if items in columns:
                x.append(items)

        difference = dataframe[x] - result1
        self._frame_list.append(difference.frame_id)

        anomalies = OrderedDict()
        anomalies['columns'] = OrderedDict()
        for col in x:
            if anomalies_thresholds['columns'][col]['max'] < 0:
                max = (1 - self._anomaly_threshold['columns'])
            else:
                max = (1 + self._anomaly_threshold['columns'])
            if anomalies_thresholds['columns'][col]['min'] < 0:
                min = (1 + self._anomaly_threshold['columns'])
            else:
                min = (1 - self._anomaly_threshold['columns'])

            temp_anomalies = odataframe[difference[col] > (anomalies_thresholds['columns'][col]['max'] * max)
                                       or
                                       difference[col] < (anomalies_thresholds['columns'][col]['min'] * min)]
            self._frame_list.append(temp_anomalies.frame_id)
            if temp_anomalies.nrows > 0:
                anomalies['columns'][col] = temp_anomalies.as_data_frame(use_pandas=True)

        anomalies['global_mse'] = OrderedDict()
        if anomalies_thresholds['global_mse']['max'] < 0:
            max = (1 - self._anomaly_threshold['global_mse'])
        else:
            max = (1 + self._anomaly_threshold['global_mse'])
        if anomalies_thresholds['global_mse']['min'] < 0:
            min = (1 + self._anomaly_threshold['global_mse'])
        else:
            min = (1 - self._anomaly_threshold['global_mse'])

        anomalyframe = self._model_base.anomaly(dataframe)
        self._frame_list.append(anomalyframe.frame_id)

        anomalyframe = dataframe.cbind(anomalyframe)
        self._frame_list.append(anomalyframe.frame_id)


        temp_anomalies = odataframe[anomalyframe['Reconstruction.MSE'] > (anomalies_thresholds['global_mse']['max'] * max)
                                   or
                                   anomalyframe['Reconstruction.MSE'] < (anomalies_thresholds['global_mse']['min'] *min)]
        self._frame_list.append(temp_anomalies.frame_id)
        if temp_anomalies.nrows > 0:
            anomalies['global_mse'] = temp_anomalies.as_data_frame(use_pandas=True)

        return anomalies


    ## Generate model full values parameters for execution analysis
    # @param self object pointer
    # @return (status (success 0, error 1) ,OrderedDict())
    def _generate_params(self):
        """
        Generate model params for this model.
        :return (status (success 0, error 1) , OrderedDict(full_stack_parameters))
        """
        params = self._model_base.get_params()
        full_stack_params = OrderedDict()
        for key, values in params.items():
            if key not in ['model_id', 'training_frame', 'validation_frame', 'response_column']:
                full_stack_params[key] = values['actual_value']
        return (0, full_stack_params)

    ## Get one especific metric for execution metrics
    # Not tested yet
    # @param algorithm_description (subclass executionmetricscollection) or compatible OrderedDict()
    # @param metric String metric key name
    # @param source [train, val, xval]
    # @ return (Variable) metrics value or String "Not Found"
    def get_metric(self, algorithm_description, metric, source):  # not tested
        try:
            struct_ar = OrderedDict(json.load(algorithm_description))
        except:
            self._logging.log_error('gDayF', self._h2o_session.session_id(), self._labels["ar_error"])
            return ('Necesario cargar un modelo valid o ar.json valido')
        try:
            return struct_ar['metrics'][source][metric]
        except KeyError:
            return 'Not Found'

    ## Auxiliary Method to convert numerical and string columns on H2OFrame to enum (factor)
    # for classification analysis
    # Returns H2OFrames modified if apply for classification objectives
    # @param atype String in ATypeMetadata().get_artypes(cls)
    # @param objective_column String Column Objective
    # @param training_frame H2OFrame for training (optional)
    # @param valid_frame H2OFrame for validation (optional)
    # @param predict_frame H2OFrame for prediction (optional)
    def need_factor(self, atype, objective_column, training_frame=None, valid_frame=None, predict_frame=None):
        if atype in ['binomial', 'multinomial']:
            if training_frame is not None:
                if isinstance(training_frame[objective_column], (int, float)):
                    training_frame[objective_column] = training_frame[objective_column].asfactor()
                    self._logging.log_exec(self.analysis_id, self._h2o_session.session_id, self._labels["factoring"],
                                           ' train : ' + objective_column)
                else:
                    training_frame[objective_column] = training_frame[objective_column].ascharacter().asfactor()
                    self._logging.log_exec(self.analysis_id, self._h2o_session.session_id, self._labels["factoring"],
                                           ' train : ' + objective_column)
            if valid_frame is not None:
                if isinstance(valid_frame[objective_column], (int, float)):
                    valid_frame[objective_column] = valid_frame[objective_column].asfactor()
                    self._logging.log_exec(self.analysis_id, self._h2o_session.session_id, self._labels["factoring"],
                                           ' validation : ' + objective_column)
                else:
                    valid_frame[objective_column] = valid_frame[objective_column].ascharacter().asfactor()
                    self._logging.log_exec(self.analysis_id, self._h2o_session.session_id, self._labels["factoring"],
                                           ' validation : ' + objective_column)
            if predict_frame is not None and objective_column in predict_frame.columns:
                if isinstance(predict_frame[objective_column], (int, float)):
                    predict_frame[objective_column] = predict_frame[objective_column].asfactor()
                    self._logging.log_exec(self.analysis_id, self._h2o_session.session_id, self._labels["factoring"],
                                           ' predict : ' + objective_column)
                else:
                    predict_frame[objective_column] = predict_frame[objective_column].ascharacter().asfactor()
                    self._logging.log_exec(self.analysis_id, self._h2o_session.session_id, self._labels["factoring"],
                                           ' predict : ' + objective_column)

    ## Method to execute normalizations base on params
    # @param self object pointer
    # @param dataframe  pandas dataframe
    # @param base_ns NormalizationMetadata orderedDict() compatible
    # @return (Dataframe, DFMetadata, Hash_value, True/False)
    def execute_normalization(self, dataframe, base_ns):
        if base_ns is not None:
            data_norm = dataframe.copy(deep=True)
            self._logging.log_exec(self.analysis_id,
                                   self._h2o_session.session_id, self._labels["exec_norm"], str(base_ns))
            normalizer = Normalizer()
            data_norm = normalizer.normalizeDataFrame(data_norm, base_ns)
            df_metadata = DFMetada()
            df_metadata.getDataFrameMetadata(dataframe=data_norm, typedf='pandas')
            df_metadata_hash_value = md5(json.dumps(df_metadata).encode('utf-8')).hexdigest()
            return data_norm, df_metadata, df_metadata_hash_value, True
        else:
            df_metadata = DFMetada()
            df_metadata.getDataFrameMetadata(dataframe=dataframe, typedf='pandas')
            df_metadata_hash_value = md5(json.dumps(df_metadata).encode('utf-8')).hexdigest()
            return dataframe, df_metadata, df_metadata_hash_value, False

            #base_ns = json.load(normalization, object_pairs_hook=NormalizationSet)

    ## Main method to execute sets of analysis and normalizations base on params
    # @param self object pointer
    # @param analysis_id String (Analysis identificator)
    # @param training_pframe pandas.DataFrame
    # @param base_ar ar_template.json
    # @param **kwargs extra arguments for test_frame inclusion
    # @return (String, ArMetadata) equivalent to (analysis_id, analysis_results)
    def order_training(self, analysis_id, training_pframe, base_ar, **kwargs):
        assert isinstance(analysis_id, str)
        assert isinstance(training_pframe, DataFrame)
        assert isinstance(base_ar, ArMetadata)

        # python train parameters effective
        self.analysis_id = analysis_id
        supervised = True
        objective_column = base_ar['objective_column']
        if objective_column is None:
           supervised = False

        train_parameters_list = ['max_runtime_secs', 'fold_column',
                                 'weights_column', 'offset_column']


        valid_frame = None
        test_frame = None
        if "test_frame" in kwargs.keys():
            test_frame = kwargs['test_frame']
        else:
            test_frame = None

        base_ns = get_model_ns(base_ar)
        assert isinstance(base_ns, list) or base_ns is None
        # Applying Normalizations
        data_initial = DFMetada()
        data_initial.getDataFrameMetadata(dataframe=training_pframe, typedf='pandas')
        training_pframe, data_normalized, train_hash_value, norm_executed = \
            self.execute_normalization(dataframe=training_pframe, base_ns=base_ns)
        df_metadata = data_initial
        if not norm_executed:
            data_normalized = None
            try:
                self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["cor_struct"],
                                          str(data_initial['correlation'][objective_column]))
            except KeyError:
                self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["cor_struct"],
                                       str(data_initial['correlation']))

        else:
            df_metadata = data_normalized
            try:
                self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["cor_struct"],
                                       str(data_normalized['correlation'][objective_column]))
            except KeyError:
                self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["cor_struct"],
                                       str(data_initial['correlation']))
            if test_frame is not None:
                test_frame, _, _, _ = self.execute_normalization(dataframe=test_frame, base_ns=base_ns)

        h2o_elements = H2Olist()
        if len(h2o_elements[h2o_elements['key'] == 'train_' + analysis_id + '_' + str(train_hash_value)]):
            if training_pframe.count(axis=0).all() > \
                    self._config['frameworks']['h2o']['conf']['validation_frame_threshold']:
                training_frame = get_frame('train_' + analysis_id + '_' + str(train_hash_value))
                valid_frame = get_frame('valid_' + analysis_id + '_' + str(train_hash_value))
                self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["getting_from_h2o"],
                                       'training_frame(' + str(training_frame.nrows) +
                                       ') validating_frame(' + str(valid_frame.nrows) + ')')
            else:
                training_frame = get_frame('train_' + analysis_id + '_' + str(train_hash_value))
                self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["getting_from_h2o"],
                                       'training_frame(' + str(training_frame.nrows) + ')')
            if "test_frame" in kwargs.keys():
                test_frame = get_frame('test_' + analysis_id + '_' + str(train_hash_value))
                self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["getting_from_h2o"],
                                       'test_frame (' + str(test_frame.nrows) + ')')
        else:
            if training_pframe.count(axis=0).all() >  \
                    self._config['frameworks']['h2o']['conf']['validation_frame_threshold']:
                training_frame, valid_frame = \
                    H2OFrame(python_obj=training_pframe).\
                        split_frame(ratios=[self._config['frameworks']['h2o']['conf']['training_frame_ratio']],
                                    destination_frames=['train_' + analysis_id + '_' + str(train_hash_value),
                                                        'valid_' + analysis_id + '_' + str(train_hash_value)])
                self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["parsing_to_h2o"],
                                       'training_frame(' + str(training_frame.nrows) +
                                       ') validating_frame(' + str(valid_frame.nrows) + ')')
                self._frame_list.append(training_frame.frame_id)
                self._frame_list.append(valid_frame.frame_id)
            else:
                training_frame = \
                    H2OFrame(python_obj=training_pframe,
                             destination_frame='train_' + analysis_id + '_' + str(train_hash_value))
                self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["parsing_to_h2o"],
                                       'training_frame(' + str(training_frame.nrows) + ')')
                self._frame_list.append(training_frame.frame_id)

            if "test_frame" in kwargs.keys():
                test_frame = H2OFrame(python_obj=test_frame,
                                      destination_frame='test_' + analysis_id + '_' + str(train_hash_value))
                self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["parsing_to_h2o"],
                                       'test_frame (' + str(test_frame.nrows) + ')')
                self._frame_list.append(test_frame.frame_id)

        if supervised:
            self.need_factor(atype=base_ar['model_parameters']['h2o']['types'][0]['type'],
                             training_frame=training_frame,
                             valid_frame=valid_frame,
                             predict_frame=test_frame,
                             objective_column=objective_column)

            # Initializing base structures
            self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["objective"],
                                   objective_column + ' - ' + training_frame.type(objective_column))

            tolerance = get_tolerance(df_metadata['columns'], objective_column, self._tolerance)

        # Generating base_path
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["action_type"], base_ar['type'])
        base_path = self.generate_base_path(base_ar, base_ar['type'])

        final_ar_model = copy.deepcopy(base_ar)
        final_ar_model['status'] = self._labels['failed_op']
        final_ar_model['version'] = cluster().version
        model_timestamp = str(time.time())
        final_ar_model['data_initial'] = data_initial
        final_ar_model['data_normalized'] = data_normalized

        model_id = base_ar['model_parameters']['h2o']['model'] + '_' + model_timestamp
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["model_id"], model_id)

        analysis_type = base_ar['model_parameters']['h2o']['types'][0]['type']
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["amode"],
                               base_ar['model_parameters']['h2o']['types'][0]['type'])

        ''' Generating and executing Models '''
        # 06/06/2017: Use X less ignored_columns on train
        x = training_frame.col_names
        if supervised:
            x.remove(objective_column)
        try:
            #for ignore_col in base_ar['model_parameters']['h2o']['parameters']['ignored_columns']['value']:
            norm = Normalizer()
            for ignore_col in norm.ignored_columns(base_ns):
                x.remove(ignore_col)
        except KeyError:
            pass
        except TypeError:
            pass

        '''Generate commands: model and model.train()'''
        model_command = list()
        model_command.append(base_ar['model_parameters']['h2o']['model'])
        model_command.append("(")
        model_command.append("training_frame=training_frame")
        train_command = list()
        # 06/06/2017: Use ignore_columns instead X on train
        if supervised:
            train_command.append("self._model_base.train(x=%s, y=\'%s\', " % (x, objective_column))
        else:
            train_command.append("self._model_base.train(x=%s, " % x)

        train_command.append("training_frame=training_frame")
        if valid_frame is not None:
            model_command.append(", validation_frame=valid_frame")
            train_command.append(", validation_frame=valid_frame")
        model_command.append(", model_id=\'%s%s\'" % (model_id, self._get_ext()))
        generate_commands_parameters(base_ar['model_parameters']['h2o'], model_command, train_command,
                                     train_parameters_list)
        model_command.append(")")
        model_command = ''.join(model_command)
        train_command.append(")")
        train_command = ''.join(train_command)

        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["gmodel"], model_command)

        # Generating model
        if self._debug:
            final_ar_model['log_path'] = StorageMetadata()
            for each_storage_type in final_ar_model['log_path'].get_log_path():
                log_path = base_path + each_storage_type['value'] + '/' + model_id + '.log'
                final_ar_model['log_path'].append(value=log_path, fstype=each_storage_type['type'],
                                                  hash_type=each_storage_type['hash_type'])
            self._persistence.mkdir(type=final_ar_model['log_path'][0]['type'], grants=0o0777,
                                    path=dirname(final_ar_model['log_path'][0]['value']))
            connection().start_logging(final_ar_model['log_path'][0]['value'])

        self._model_base = eval(model_command)
        start = time.time()
        try:
            eval(train_command)
            final_ar_model['status'] = 'Executed'
            # Generating aditional model parameters Model_ID
            final_ar_model['model_parameters']['h2o']['parameters']['model_id'] = ParameterMetadata()
            final_ar_model['model_parameters']['h2o']['parameters']['model_id'].set_value(value=model_id,
                                                                                          seleccionable=False,
                                                                                          type="String")

        except OSError as execution_error:
            self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["abort"],
                                   repr(execution_error))
            return analysis_id, None

        final_ar_model['execution_seconds'] = time.time() - start
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["tmodel"], model_id)
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["exec_time"],
                               str(final_ar_model['execution_seconds']))

        if self._debug:
            connection().stop_logging()
            self._persistence.store_file(filename=final_ar_model['log_path'][0]['value'],
                                         storage_json=final_ar_model['log_path'])

        # Filling whole json ar.json
        final_ar_model['ignored_parameters'], \
        final_ar_model['full_parameters_stack'] = self._generate_params()


        # Generating execution metrics
        final_ar_model['metrics']['execution'] = ExecutionMetricCollection()

        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["gexec_metric"], model_id)


        final_ar_model['metrics']['execution']['train'] = self._generate_execution_metrics(dataframe=None,
                                                                                           source='train',
                                                                                           antype=analysis_type)
        final_ar_model['metrics']['accuracy'] = OrderedDict()

        if supervised:

            final_ar_model['metrics']['accuracy']['train'] = \
            self._accuracy(objective_column, training_frame, antype=analysis_type, tolerance=tolerance,
                               base_type=training_frame.type(objective_column))
            self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["model_tacc"],
                                   model_id + ' - ' + str(final_ar_model['metrics']['accuracy']['train']))
            final_ar_model['tolerance'] = tolerance
        else:

            final_ar_model['metrics']['accuracy']['train'] = 0.0

        final_ar_model['metrics']['execution']['xval'] = \
            self._generate_execution_metrics(dataframe=None, source='xval', antype=analysis_type)

        if valid_frame is not None:

            final_ar_model['metrics']['execution']['valid'] = \
                self._generate_execution_metrics(dataframe=None, source='valid', antype=analysis_type)

            if supervised:

                final_ar_model['metrics']['accuracy']['valid'] = \
                    self._accuracy(objective_column, valid_frame,
                                   antype=analysis_type, tolerance=tolerance,
                                   base_type=valid_frame.type(objective_column))
            else:

                final_ar_model['metrics']['accuracy']['valid'] = 0.0

        if test_frame is not None:

            if supervised:

                final_ar_model['metrics']['accuracy']['test'] = \
                    self._accuracy(objective_column, test_frame, antype=analysis_type, tolerance=tolerance,
                                   base_type=test_frame.type(objective_column))

                train_balance = self._config['frameworks']['h2o']['conf']['train_balance_metric']
                test_balance = 1 - train_balance
                final_ar_model['metrics']['accuracy']['combined'] = \
                    (final_ar_model['metrics']['accuracy']['train']*train_balance +
                     final_ar_model['metrics']['accuracy']['test']*test_balance)
                self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["model_pacc"],
                                       model_id + ' - ' + str(final_ar_model['metrics']['accuracy']['test']))

                self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["model_cacc"],
                                       model_id + ' - ' + str(final_ar_model['metrics']['accuracy']['combined']))
            else:

                final_ar_model['metrics']['accuracy']['test'] = 0.0
                final_ar_model['metrics']['accuracy']['combined'] = 0.0


        #final_ar_model['tolerance'] = tolerance

        # Generating model metrics
        final_ar_model['metrics']['model'] = self._generate_model_metrics()
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["gmodel_metric"], model_id)

        # Generating anomalies metrics
        if analysis_type == 'anomalies':
            final_ar_model['metrics']['anomalies'] = self._generate_execution_metrics(dataframe=training_frame[x],
                                                                                      source='train',
                                                                                      antype=analysis_type)
            self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["ganomaly_metric"], model_id)

        # Generating Variable importance
        final_ar_model['metrics']['var_importance'] = self._generate_importance_variables()
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["gvar_metric"], model_id)

        # Generating scoring_history
        final_ar_model['metrics']['scoring'] = self._generate_scoring_history()
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["gsco_metric"], model_id)

        final_ar_model['status'] = self._labels['success_op']

        generate_json_path(final_ar_model)
        self._persistence.store_json(storage_json=final_ar_model['json_path'], ar_json=final_ar_model)

        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["model_stored"], model_id)
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["end"], model_id)

        for handler in self._logging.logger.handlers:
            handler.flush()
        # Cleaning H2OCluster
        try:
            if self._model_base is not None:
                try:
                    H2Oremove(self._get_cv_ids(self._model_base.model_id,
                                           final_ar_model['model_parameters']['h2o']['parameters']['nfolds']['value']))
                except KeyError:
                    pass
                #H2Oremove(self._model_base.model_id)
        except H2OError:
            self._logging.log_exec(analysis_id,
                                   self._h2o_session.session_id, self._labels["delete_objects"],
                                   self._model_base.model_id)
        H2Oapi("POST /3/GarbageCollect")
        return analysis_id, final_ar_model

    ## Method to save model to persistence layer from armetadata
    # @param armetadata structure to be stored
    # return saved_model (True/False)
    def store_model(self, armetadata):
        saved_model = False

        fw = get_model_fw(armetadata)
        model_id = armetadata['model_parameters'][fw]['parameters']['model_id']['value']
        analysis_id = armetadata['model_id']

        if model_id + self._get_ext() not in H2Olist()['key'].tolist():

            return saved_model
        else:
            model_base = get_model(model_id + self._get_ext())

        source_data = list()
        source_data.append(self.primary_path)
        source_data.append('/')
        source_data.append(armetadata['model_id'])
        source_data.append('/')
        source_data.append(fw)
        source_data.append('/')
        source_data.append(armetadata['type'])
        source_data.append('/')
        source_data.append(str(armetadata['timestamp']))
        source_data.append('/')

        #Updating status
        armetadata['status'] = self._labels["success_st"]
        # Generating load_path
        load_storage = StorageMetadata()
        for each_storage_type in load_storage.get_load_path():
            load_path = ''.join(source_data) + each_storage_type['value']+'/'
            self._persistence.mkdir(type=each_storage_type['type'], path=load_path, grants=0o0777)
            if self._get_ext() == '.pojo':
                download_pojo(model=model_base, path=load_path, get_jar=True)
            elif self._get_ext() == '.mojo':
                self._model_base.download_mojo(path=load_path, get_genmodel_jar=True)
            else:
                save_model(model=model_base, path=load_path, force=True)
            load_storage.append(value=load_path + model_id + self._get_ext(),
                                fstype=each_storage_type['type'], hash_type=each_storage_type['hash_type'])
            saved_model = True
        armetadata['load_path'] = load_storage
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["msaved"], model_id)

        generate_json_path(armetadata)
        self._persistence.store_json(storage_json=armetadata['json_path'], ar_json=armetadata)
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["model_stored"], model_id)


        return saved_model

    ## Main method to execute predictions over traning models
    # Take the ar.json for and execute predictions including its metrics a storage paths
    # @param self object pointer
    # @param predict_frame pandas.DataFrame
    # @param base_ar ArMetadata
    # or compatible tuple (OrderedDict(), OrderedDict())
    # @return (String, [ArMetadata]) equivalent to (analysis_id, List[analysis_results])
    def predict(self, predict_frame, base_ar):
        remove_model = False
        model_timestamp = str(time.time())
        self.analysis_id = base_ar['model_id']
        analysis_id = self.analysis_id
        base_model_id = base_ar['model_parameters']['h2o']['parameters']['model_id']['value'] + '.model'
        model_id = base_model_id + '_' + model_timestamp

        antype = base_ar['model_parameters']['h2o']['types'][0]['type']

        base_ns = get_model_ns(base_ar)

        #Checking file source versus hash_value
        load_fails, remove_model = self._get_model(base_ar, base_model_id, remove_model)

        if load_fails or self._model_base is None:
            self._logging.log_exec(self.analysis_id, self._h2o_session.session_id,
                                   self._labels["no_models"], base_model_id)
            base_ar['status'] = self._labels['failed_op']  # Default Failed Operation Code
            return None

        objective_column = base_ar['objective_column']

        supervised = True
        if objective_column is None:
            supervised = False
        if supervised:
            self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["objective"],
                                   objective_column)
            # Recovering tolerance
            tolerance = base_ar['tolerance']

        data_initial = DFMetada()
        data_initial.getDataFrameMetadata(dataframe=predict_frame, typedf='pandas')
        if objective_column in list(predict_frame.columns.values):
            try:
                self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                   self._labels["cor_struct"],  str(data_initial['correlation'][objective_column]))
            except KeyError:
                self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                   self._labels["cor_struct"],  str(data_initial['correlation']))

        base_ar['data_initial'] = data_initial
        npredict_frame, data_normalized, _, norm_executed = self.execute_normalization(dataframe=predict_frame,
                                                                                      base_ns=base_ns)
        if not norm_executed:
            self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["exec_norm"],
                                   'No Normalizations Required')
        else:
            # Transforming original dataframe to H2OFrame
            predict_frame = H2OFrame(python_obj=predict_frame,
                                     destination_frame='predict_frame_' + base_ar['model_id'])
            self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["parsing_to_h2o"],
                                   'test_frame (' + str(predict_frame.nrows) + ')')
            self._frame_list.append(predict_frame.frame_id)

            base_ar['data_normalized'] = data_normalized
            if objective_column in list(npredict_frame.columns.values):
                try:
                    self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["cor_struct"],
                                       str(data_normalized['correlation'][objective_column]))
                except KeyError:
                    self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["no_cor_struct"],
                                           str(data_normalized['correlation']))

        #Transforming to H2OFrame
        npredict_frame = H2OFrame(python_obj=npredict_frame,
                                  destination_frame='npredict_frame_' + base_ar['model_id'])
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["parsing_to_h2o"],
                               'test_frame (' + str(npredict_frame.nrows) + ')')
        self._frame_list.append(npredict_frame.frame_id)

        if supervised:
            self.need_factor(atype=base_ar['model_parameters']['h2o']['types'][0]['type'],
                             objective_column=objective_column, predict_frame=npredict_frame)

        base_ar['type'] = 'predict'
        self._logging.log_exec(self.analysis_id, self._h2o_session.session_id,
                               self._labels["action_type"], base_ar['type'])

        base_ar['timestamp'] = model_timestamp
        if self._debug:
            for each_storage_type in base_ar['log_path']:
                each_storage_type['value'] = each_storage_type['value'].replace('train', 'predict') \
                    .replace('.log', '_' + model_timestamp + '.log')

            self._persistence.mkdir(type=base_ar['log_path'][0]['type'], grants=0o0777,
                                    path=dirname(base_ar['log_path'][0]['value']))
            connection().start_logging(base_ar['log_path'][0]['value'])

        self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                               self._labels['st_predict_model'],
                               base_model_id)

        if objective_column in npredict_frame.columns:
            objective_type = npredict_frame.type(objective_column)
        else:
            objective_type = None

        start = time.time()
        if supervised:
            accuracy, prediction_dataframe = self._predict_accuracy(objective_column, npredict_frame,
                                                                    antype=antype,
                                                                    tolerance=tolerance, base_type=objective_type)
            self._frame_list.append(prediction_dataframe.frame_id)

            base_ar['execution_seconds'] = time.time() - start
            base_ar['tolerance'] = tolerance

            prediction_dataframe = prediction_dataframe.as_data_frame(use_pandas=True)
        else:
            if antype == 'anomalies':
                if norm_executed:
                    predict_anomalies = self._predict_anomalies(predict_frame, npredict_frame,
                                                                base_ar['metrics']['anomalies'])
                else:
                    predict_anomalies = self._predict_anomalies(npredict_frame, npredict_frame,
                                                                base_ar['metrics']['anomalies'])

            base_ar['execution_seconds'] = time.time() - start


        if self._debug:
            connection().stop_logging()
            self._persistence.store_file(filename=base_ar['log_path'][0]['value'],
                                         storage_json=base_ar['log_path'])

        if objective_column in npredict_frame.columns:
            self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["gexec_metric"], model_id)
            base_ar['metrics']['execution'][base_ar['type']] = self._generate_execution_metrics(dataframe=npredict_frame,
                                                                                            source=None, antype=antype)

            base_ar['metrics']['accuracy']['predict'] = accuracy
            self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["model_pacc"],
                               base_model_id + ' - ' + str(base_ar['metrics']['accuracy']['predict']))

        base_ar['status'] = self._labels['success_op']

        # writing ar.json file
        generate_json_path(base_ar)
        self._persistence.store_json(storage_json=base_ar['json_path'], ar_json=base_ar)
        self._logging.log_exec(self.analysis_id, self._h2o_session.session_id, self._labels["model_stored"], model_id)
        self._logging.log_exec(self.analysis_id, self._h2o_session.session_id, self._labels["end"], model_id)
        for handler in self._logging.logger.handlers:
            handler.flush()

        # Cleaning H2OCluster
        try:
            H2Oremove(npredict_frame)
            if norm_executed:
                H2Oremove(predict_frame)
        except H2OError:
            self._logging.log_exec(analysis_id,
                                   self._h2o_session.session_id, self._labels["delete_frame"],
                                   self._model_base.model_id)
        try:
            if self._model_base is not None and remove_model:
                H2Oremove(self._model_base.model_id)
        except H2OError:
            self._logging.log_exec(analysis_id,
                                   self._h2o_session.session_id, self._labels["delete_objects"],
                                   self._model_base.model_id)
        H2Oapi("POST /3/GarbageCollect")

        if not supervised:
            if antype == 'anomalies':
                prediction = predict_anomalies
        else:
            prediction = prediction_dataframe

        return prediction, base_ar


    ## Internal method to get an H2Omodel from server or file trapasparent to user
    # @param self Object pointer
    # @param base_ar armetadata to load from fs
    # @param base_model_id fro searching on server memory objects
    # @param remove_model to indicate if has been load from memory o need to remove at last
    # @return load_fails, remove_model operation status True/False, needs to remove True/False
    def _get_model(self, base_ar, base_model_id, remove_model):
        if base_model_id in H2Olist()['key'].tolist():
            self._model_base = get_model(base_model_id)
            if self._model_base is None:
                load_fails = True
            else:
                load_fails = False
        else:
            load_fails = self.get_model_from_load_path(base_ar)
            remove_model = True
        return load_fails, remove_model

    ## Method to remove list of model from server
    # @param arlist List of ArMetadata
    # @return remove_fails True/False
    def remove_models(self, arlist):
        remove_fails = True
        try:
            assert isinstance(arlist, list)
        except AssertionError:
            return remove_fails
        for armetadata in arlist:
            fw = get_model_fw(armetadata)
            model_id = armetadata['model_parameters'][fw]['parameters']['model_id']['value']+'.model'
            if model_id in H2Olist()['key'].tolist():
                H2Oremove(model_id)
        return False

## auxiliary function (procedure) to generate model and train chain paramters to execute models
# Modify model_command and train_command String to complete for eval()
# @param each_model object pointer
# @param model_command String with model command definition base structure
# @param train_command String with train command base structure
# @param train_parameters_list list(ATypesMetadata) or compatible OrderedDict()
def generate_commands_parameters(each_model, model_command, train_command, train_parameters_list):
    for key, value in each_model['parameters'].items():
        if value['seleccionable']:
            if isinstance(value['value'], str):
                if key in train_parameters_list and value is not None:
                    train_command.append(", %s=\'%s\'" % (key, value['value']))
                else:
                    model_command.append(", %s=\'%s\'" % (key, value['value']))
            else:
                if key in train_parameters_list and value is not None:
                    train_command.append(", %s=%s" % (key, value['value']))
                else:
                    model_command.append(", %s=%s" % (key, value['value']))



## Auxiliary function to get the level of tolerance for regression analysis
# @param columns list() of OrderedDict() [{Column description}]
# @param objective_column String Column Objective
# @param tolerance  float [0.0, 1.0]  (optional)
# @return float value for tolerance
def get_tolerance(columns, objective_column, tolerance=0.0):
    min_val = None
    max_val = None
    for each_column in columns:
        if each_column["name"] == objective_column:
            min_val = float(each_column["min"])
            max_val = float(each_column["max"])
    if min_val is None or max_val is None:
        threshold = 0
    else:
        threshold = (max_val - min_val) * tolerance
    return threshold