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
from copy import deepcopy

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
from h2o import remove as H2Oremove
from h2o import api as H2Oapi
from h2o import get_frame
from h2o import download_pojo

from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator

from gdayf.common.normalizationset import NormalizationSet
from gdayf.common.storagemetadata import StorageMetadata
from gdayf.common.utils import hash_key
from gdayf.logs.logshandler import LogsHandler
from gdayf.metrics.binomialmetricmetadata import BinomialMetricMetadata
from gdayf.metrics.metricmetadata import MetricMetadata
from gdayf.metrics.executionmetriccollection import ExecutionMetricCollection
from gdayf.metrics.regressionmetricmetadata import RegressionMetricMetadata
from gdayf.metrics.multinomialmetricmetadata import MultinomialMetricMetadata
from gdayf.persistence.persistencehandler import PersistenceHandler
from gdayf.conf.loadconfig import LoadConfig
from gdayf.conf.loadconfig import LoadLabels
from gdayf.common.dfmetada import DFMetada
from gdayf.common.utils import get_model_ns
from gdayf.common.armetadata import ArMetadata
from gdayf.models.parametersmetadata import ParameterMetadata
from gdayf.normalizer.normalizer import Normalizer
from gdayf.common.utils import  get_model_fw


class H2OHandler(object):
    """
    H2OHandler

    Train: Get some analysis list of tuplas (analysis_results.json +  (algorithm + normalzations over a Dataframe) and launch de analysis on H2O platform
    predict: Get some list of [analysis_results.json] and load and execute algorithm
    
    Model's Catalog:

    "H2OGradientBoostingEstimator" :{
      "id" : "number",
      "module" : "h2o.estimators.gbm",
      "types": {
        "binomial": {"active" : true, "valued": "enum"},
        "multinomial": {"active" : true, "valued": "enum"},
        "regression" : {"active" : true, "valued": "float64"},
        "topology" : {"active" : false, "valued": "float64"}
      }

    "H2OGeneralizedLinearEstimator" :{
      "id" : "number",
      "module" : "h2o.estimators.glm",
      "types": {
        "binomial": {"active" : true, "valued": "enum"},
        "multinomial": {"active" : true, "valued": "enum"},
        "regression" : {"active" : true, "valued": "float64"},
        "topology" : {"active" : false, "valued": "float64"}
      }

    "H2ODeepLearningEstimator" :{
      "id" : "number",
      "module" : "h2o.estimators.deeplearning",
      "types": {
        "binomial": {"active" : true, "valued": "enum"},
        "multinomial": {"active" : true, "valued": "enum"},
        "regression" : {"active" : true, "valued": "float64"},
        "topology" : {"active" : false, "valued": "float64"}
      }

    "H2ORandomForestEstimator" :{
      "id" : "number",
      "module" : "h2o.estimators.random_forest",
      "types": {
        "types": {
          "binomial": {"active" : true, "valued": "enum"},
          "multinomial": {"active" : true, "valued": "enum"},
          "regression" : {"active" : true, "valued": "float64"},
          "topology" : {"active" : false, "valued": "float64"}
        }

    Status Codes:
        -1 : Uninitialized
        0: Success
        1: Error

    """
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
        fw = get_model_fw(ar_metadata)
        model_id = ar_metadata['model_parameters'][fw]['parameters']['model_id']['value']
        self.analysis_id = analysis_id = ar_metadata['model_id']
        config = LoadConfig().get_config()['frameworks'][fw]['conf']

        load_fails = self.get_model_from_load_path(ar_metadata)
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
                self._logging.log_exec(self.analysis_id, self._h2o_session.session_id,
                                       self._labels["failed_op"], download_path)
        else:
            try:
                file_path = download_pojo(self._model_base, path=str(download_path), get_jar=True)
            except H2OError:
                self._logging.log_exec(self.analysis_id, self._h2o_session.session_id,
                                       self._labels["failed_op"], download_path)
        try:
            if self._model_base is not None:
                H2Oremove(self._model_base.model_id)
        except H2OError:
            self._logging.log_exec(analysis_id,
                                   self._h2o_session.session_id, self._labels["delete_objects"],
                                   self._model_base.model_id)
        return download_path, hash_key('MD5', filename=file_path)

    ## Generate list of models_id for internal crossvalidation objects_
    # @param self object pointer
    # @param ar_metadata ArMetadata stored model
    # @return implicit self._model_base / None on Error
    def get_model_from_load_path(self, ar_metadata):
        load_fails = True
        counter_storage = 0
        # Checking file source versus hash_value

        assert isinstance(ar_metadata['load_path'], list)
        while counter_storage < len(ar_metadata['load_path']) and load_fails:
            self._logging.log_exec(self.analysis_id, self._h2o_session.session_id, self._labels["hk_check"],
                                   ar_metadata['load_path'][counter_storage]['hash_value'] + ' - ' +
                                   hash_key(ar_metadata['load_path'][counter_storage]['hash_type'],
                                            ar_metadata['load_path'][counter_storage]['value'])
                                   )

            if hash_key(ar_metadata['load_path'][counter_storage]['hash_type'],
                        ar_metadata['load_path'][counter_storage]['value']) == \
                    ar_metadata['load_path'][counter_storage]['hash_value']:
                try:
                    self._model_base = load_model(ar_metadata['load_path'][counter_storage]['value'])
                    load_fails = False
                except H2OError:
                    self._logging.log_exec(self.analysis_id, self._h2o_session.session_id,
                                           self._labels["abort"], ar_metadata['load_path'][counter_storage]['value'])
            counter_storage += 1
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
        else:
            model_metrics = MetricMetadata()

        if dataframe is not None:
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
        return self._model_base.scoring_history().drop("", axis=1).to_json(orient='split')

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
    # @param dataframe H2OFrame
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
    def execute_normalization (self, dataframe, base_ns):
        data_norm = dataframe.copy(deep=True)
        if base_ns is not None:
            data_norm = dataframe.copy(deep=True)
            self._logging.log_exec(self.analysis_id,
                                   self._h2o_session.session_id, self._labels["exec_norm"], str(base_ns))
            normalizer = Normalizer()
            normalizer.normalizeDataFrame(data_norm, base_ns)
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
    # @return (String, ArMetadata) equivalent to (analysis_id, analysis_results)
    def order_training(self, analysis_id, training_pframe, base_ar, **kwargs):
        assert isinstance(analysis_id, str)
        assert isinstance(training_pframe, DataFrame)
        assert isinstance(base_ar, ArMetadata)

        # python train parameters effective
        self.analysis_id = analysis_id
        objective_column = base_ar['objective_column']
        train_parameters_list = ['max_runtime_secs', 'fold_column',
                                 'weights_column', 'offset_column']


        valid_frame = None
        test_frame = None
        if "test_frame" in kwargs.keys():
            test_frame = kwargs['test_frame']
        else:
            test_frame = None

        base_ns = get_model_ns(base_ar)
        assert isinstance(base_ns, NormalizationSet) or base_ns is None
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
            if training_pframe.count(axis=0).all() > 100000:
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
            if training_pframe.count(axis=0).all() > 100000:
                training_frame, valid_frame = \
                    H2OFrame(python_obj=training_pframe).split_frame(ratios=[.85],
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
        x.remove(objective_column)
        try:
            for ignore_col in base_ar['model_parameters']['h2o']['parameters']['ignored_columns']['value']:
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
        train_command.append("self._model_base.train(x=%s, y=\'%s\', " % (x, objective_column))
        '''train_command.append("self._model_base.train(y=\'%s\', " % objective_column)'''
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
            for each_storage_type in base_ar['log_path']:
                log_path = base_path + each_storage_type['value'] + '/' + model_id + '.log'
                final_ar_model['log_path'].append(value=log_path, fstype=each_storage_type['type'],
                                                  hash_type=each_storage_type['hash_type'])
            self._persistence.mkdir(type=final_ar_model['log_path'][0]['type'], grants=0o0777,
                                    path=dirname(final_ar_model['log_path'][0]['value']))
            connection().start_logging(final_ar_model['log_path'][0]['value'])
        self._model_base = eval(model_command)
        start = time.time()
        try:
            model_trained = eval(train_command)
            final_ar_model['status'] = 'Executed'
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

        # Generating load_path
        final_ar_model['load_path'] = StorageMetadata()
        for each_storage_type in base_ar['load_path']:
            load_path = base_path + each_storage_type['value'] + '/'
            self._persistence.mkdir(type=each_storage_type['type'], path=load_path, grants=0o0777)
            if self._get_ext() == '.pojo':
                download_pojo(model=self._model_base, path=load_path, get_jar=True)
            elif self._get_ext() == '.mojo':
                '''MOJOs are currently supported for Distributed Random Forest, 
                Gradient Boosting Machine, 
                Deep Water, GLM, GLRM and word2vec models only.'''
                self._model_base.download_mojo(path=load_path, get_genmodel_jar=True)
            else:
                save_model(model=self._model_base, path=load_path, force=True)

            final_ar_model['load_path'].append(value=load_path + model_id + self._get_ext(),
                                               fstype=each_storage_type['type'],
                                               hash_type=each_storage_type['hash_type'])

        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["msaved"], model_id)

        # Filling whole json ar.json
        final_ar_model['ignored_parameters'], \
        final_ar_model['full_parameters_stack'] = self._generate_params()

        # Generating aditional model parameters
        final_ar_model['model_parameters']['h2o']['parameters']['model_id'] = \
            OrderedDict(ParameterMetadata(value=model_id, seleccionable=False, type="String"))

        # Generating execution metrics
        final_ar_model['metrics']['execution'] = ExecutionMetricCollection()

        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["gexec_metric"], model_id)

        final_ar_model['metrics']['accuracy'] = OrderedDict()

        final_ar_model['metrics']['execution']['train'] = self._generate_execution_metrics(dataframe=None,
                                                                                           source='train',
                                                                                           antype=analysis_type)
        final_ar_model['metrics']['accuracy']['train'] = \
        self._accuracy(objective_column, training_frame, antype=analysis_type, tolerance=tolerance,
                           base_type=training_frame.type(objective_column))
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["model_tacc"],
                               model_id + ' - ' + str(final_ar_model['metrics']['accuracy']['train']))

        final_ar_model['tolerance'] = tolerance

        final_ar_model['metrics']['execution']['xval'] = \
            self._generate_execution_metrics(dataframe=None, source='xval', antype=analysis_type)

        if valid_frame is not None:
            final_ar_model['metrics']['execution']['valid'] = \
                self._generate_execution_metrics(dataframe=None, source='valid', antype=analysis_type)
            final_ar_model['metrics']['accuracy']['valid'] = \
                self._accuracy(objective_column, valid_frame,
                               antype=analysis_type, tolerance=tolerance,
                               base_type=valid_frame.type(objective_column))

        if test_frame is not None:
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

        final_ar_model['metrics']['accuracy']['train'] = \
        self._accuracy(objective_column, training_frame, antype=analysis_type, tolerance=tolerance,
                           base_type=training_frame.type(objective_column))
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["model_tacc"],
                               model_id + ' - ' + str(final_ar_model['metrics']['accuracy']['train']))

        final_ar_model['tolerance'] = tolerance

        # Generating model metrics
        final_ar_model['metrics']['model'] = self._generate_model_metrics()
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["gmodel_metric"], model_id)

        # Generating Variable importance
        final_ar_model['metrics']['var_importance'] = self._generate_importance_variables()
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["gvar_metric"], model_id)

        # Generating scoring_history
        final_ar_model['metrics']['scoring'] = self._generate_scoring_history()
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["gsco_metric"], model_id)

        final_ar_model['status'] = self._labels['success_op']

        # writing ar.json file
        final_ar_model['json_path'] = StorageMetadata()
        for each_storage_type in base_ar['json_path']:
            json_path = base_path + each_storage_type['value'] + '/' + model_id + '.json'
            final_ar_model['json_path'].append(value=json_path, fstype=each_storage_type['type'],
                                               hash_type=each_storage_type['hash_type'])
        self._persistence.store_json(storage_json=final_ar_model['json_path'], ar_json=final_ar_model)

        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["model_stored"], model_id)
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["end"], model_id)

        for handler in self._logging.logger.handlers:
            handler.flush()
        # Cleaning H2OCluster
        try:
            if self._model_base is not None:
                H2Oremove(self._get_cv_ids(self._model_base.model_id,
                                           final_ar_model['model_parameters']['h2o']['parameters']['nfolds']['value']))
                H2Oremove(self._model_base.model_id)
        except H2OError:
            self._logging.log_exec(analysis_id,
                                   self._h2o_session.session_id, self._labels["delete_objects"],
                                   self._model_base.model_id)
        H2Oapi("POST /3/GarbageCollect")
        return analysis_id, final_ar_model

    ## Main method to execute predictions over traning models
    # Take the ar.json for and execute predictions including its metrics a storage paths
    # @param self object pointer
    # @param predict_frame pandas.DataFrame
    # @param base_ar ArMetadata
    # or compatible tuple (OrderedDict(), OrderedDict())
    # @return (String, [ArMetadata]) equivalent to (analysis_id, List[analysis_results])
    def predict(self, predict_frame, base_ar):
        model_timestamp = str(time.time())
        self.analysis_id = base_ar['model_id']
        analysis_id = self.analysis_id
        base_model_id = base_ar['model_parameters']['h2o']['parameters']['model_id']['value']
        model_id = base_model_id + '_' + model_timestamp

        antype = base_ar['model_parameters']['h2o']['types'][0]['type']

        base_ns = get_model_ns(base_ar)

        #Checking file source versus hash_value
        load_fails = self.get_model_from_load_path(base_ar)

        if load_fails:
            self._logging.log_exec(self.analysis_id, self._h2o_session.session_id,
                                   self._labels["no_models"], str(base_ar))
            base_ar['status'] = self._labels['failed_op']  # Default Failed Operation Code
            return None

        objective_column = base_ar['objective_column']
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["objective"], objective_column)

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
        predict_frame, data_normalized, _, norm_executed = self.execute_normalization(dataframe=predict_frame,
                                                                                      base_ns=base_ns)
        if not norm_executed:
            self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["exec_norm"],
                                   'No Normalizations Required')
        else:
            base_ar['data_normalized'] = data_normalized
            if objective_column in list(predict_frame.columns.values):
                self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["cor_struct"],
                                       str(data_normalized['correlation'][objective_column]))

        #Transforming to H2OFrame
        predict_frame = H2OFrame(python_obj=predict_frame,
                                 destination_frame='predict_frame_' + base_ar['model_id'])
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["parsing_to_h2o"],
                               'test_frame (' + str(predict_frame.nrows) + ')')
        self._frame_list.append(predict_frame.frame_id)

        self.need_factor(atype=base_ar['model_parameters']['h2o']['types'][0]['type'],
                         objective_column=objective_column, predict_frame=predict_frame)

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
        start = time.time()
        if objective_column in predict_frame.columns:
            objective_type = predict_frame.type(objective_column)
        else:
            objective_type = None
        accuracy, prediction_dataframe = self._predict_accuracy(objective_column, predict_frame, antype=antype,
                                                                tolerance=tolerance, base_type=objective_type)
        base_ar['tolerance'] = tolerance
        base_ar['execution_seconds'] = time.time() - start
        self._frame_list.append(prediction_dataframe.frame_id)

        if self._debug:
            connection().stop_logging()
            self._persistence.store_file(filename=base_ar['log_path'][0]['value'],
                                         storage_json=base_ar['log_path'])

        if objective_column in predict_frame.columns:
            self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["gexec_metric"], model_id)
            base_ar['metrics']['execution'][base_ar['type']] = self._generate_execution_metrics(dataframe=predict_frame,
                                                                                            source=None, antype=antype)

            base_ar['metrics']['accuracy']['predict'] = accuracy
            self._logging.log_exec(analysis_id, self._h2o_session.session_id, self._labels["model_pacc"],
                               base_model_id + ' - ' + str(base_ar['metrics']['accuracy']['predict']))

        base_ar['status'] = self._labels['success_op']

        # writing ar.json file
        json_files = StorageMetadata()
        for each_storage_type in base_ar['json_path']:
            each_storage_type['value'] = each_storage_type['value'].replace('train', 'predict') \
                .replace('.json', '_' + model_timestamp + '.json')
            json_files.append(value=each_storage_type['value'], fstype=each_storage_type['type'],
                              hash_type=each_storage_type['hash_type'])

        self._persistence.store_json(json_files, base_ar)
        self._logging.log_exec(self.analysis_id, self._h2o_session.session_id, self._labels["model_stored"], model_id)
        self._logging.log_exec(self.analysis_id, self._h2o_session.session_id, self._labels["end"], model_id)
        for handler in self._logging.logger.handlers:
            handler.flush()

        prediction_dataframe = prediction_dataframe.as_data_frame(use_pandas=True)

        # Cleaning H2OCluster
        try:
            H2Oremove(predict_frame)
        except H2OError:
            self._logging.log_exec(analysis_id,
                                   self._h2o_session.session_id, self._labels["delete_frame"],
                                   self._model_base.model_id)
        try:
            if self._model_base is not None:
                H2Oremove(self._model_base.model_id)
        except H2OError:
            self._logging.log_exec(analysis_id,
                                   self._h2o_session.session_id, self._labels["delete_objects"],
                                   self._model_base.model_id)
        H2Oapi("POST /3/GarbageCollect")

        return prediction_dataframe, base_ar


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