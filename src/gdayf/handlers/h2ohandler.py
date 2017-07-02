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

from h2o import H2OFrame as H2OFrame
from h2o import cluster as cluster
from h2o import connect as connect
from h2o import connection as connection
from h2o import init as init
from h2o import load_model as load_model
from h2o import save_model as save_model
from h2o.exceptions import H2OError
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
from gdayf.common.dfmetada import DFMetada
from gdayf.common.utils import get_model_ns
from gdayf.common.armetadata import ArMetadata
from gdayf.models.parametersmetadata import ParameterMetadata

__name__ = 'engines.h2o'
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
        print(__name__)
        self._framework = 'h2o'
        self._config = LoadConfig().get_config()
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
        self._persistence = PersistenceHandler()
        try:
            self._h2o_session = connect(url=self.url)
        except H2OError:
            init(url=self.url, nthreads=self.nthreads, ice_root=self.ice_root, max_mem_size=self.max_mem_size)
            self._h2o_session = connection()
        self._logging = LogsHandler(__name__)
        self._logging.log_exec('gDayF', self._h2o_session.session_id(), 'Connected to active cluster and ready')

    ## Destructor
    def __del__(self):
        self._h2o_session.close()
        #del self._logging

    ## Class Method for cluster shutdown
    # @param cls class pointer
    @classmethod
    def shutdown_cluster(cls):
        try:
            cluster().shutdown()
        except:
            print('H20-cluster not working')

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
        """
        Generate model execution metrics for this model on test_data, train, valid frame or crossvalidation.

        :param H2OFrame test_data: Data set for which model metrics shall be computed against. All three of train,
            valid and xval arguments are ignored if test_data is not None.
        :param source 'train': Report the training metrics for the model.
        :param source 'valid': Report the validation metrics for the model.
        :param source 'xval': Report the cross-validation metrics for the model. If train and valid are True, then it
            defaults to True.
        """
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
    # @param tolerance (optional) default value 0.0. Only for regression
    # @return float accuracy of model
    def _accuracy(self, objective, dataframe, antype, tolerance=0.0):
        if antype == 'regression':
            fmin = eval("lambda x: x - " + str(tolerance))
            fmax = eval("lambda x: x + " + str(tolerance))
            accuracy = dataframe[objective].apply(fmin) <= \
                       self._model_base.predict(dataframe)[0] <= \
                       dataframe[objective].apply(fmax)
        else:
            accuracy = self._model_base.predict(dataframe)[0] == dataframe[objective]
        return accuracy.sum() / accuracy.nrows

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
            return ('Necesario cargar un modelo valid o ar.json valido')
        try:
            return struct_ar['metrics'][source][metric]
        except KeyError:
            return 'Not Found'

    ## Main method to execute sets of analysis and normalizations base on params
    # @param self object pointer
    # @param analysis_id String (Analysis identificator)
    # @param training_frame pandas.DataFrame
    # @param base_ar ar_template.json
    # @return (String, ArMetadata) equivalent to (analysis_id, analysis_results)
    def order_training(self, analysis_id, training_frame, base_ar, **kwargs):
        assert isinstance(analysis_id, str)
        assert isinstance(training_frame, DataFrame)
        # Not used now assert isinstance(valid_frame, DataFrame) or valid_frame is None
        assert isinstance(base_ar, OrderedDict)
        # python train parameters effective

        train_parameters_list = ['max_runtime_secs', 'fold_column',
                                 'weights_column', 'offset_column']

        status = -1  # Operation Code
        valid_frame = None
        test_frame = None

        analysis_timestamp = str(time.time())
        # Loading_data structure
        df_metadata = DFMetada()
        df_metadata.getDataFrameMetadata(dataframe=training_frame, typedf='pandas')

        self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                               'Dataframe_structure: %s' % df_metadata)
        self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                               'Starting analysis')

        if training_frame.count(axis=0).all() > 100000:
            training_frame, valid_frame = \
                H2OFrame(python_obj=training_frame).split_frame(ratios=[.85],
                                                                destination_frames=['training_frame_' + analysis_id,
                                                                                    'valid_frame_' + analysis_id])
            self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                   'Parsing from pandas to H2OFrames: training_frame (' + str(training_frame.nrows) +
                                   ') validating_frame(' + str(valid_frame.nrows) + ')')
        else:
            training_frame = \
                H2OFrame(python_obj=training_frame, destination_frame='training_frame_' + analysis_id)
            self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                   'Parsing from pandas to H2OFrames: training_frame (' + str(training_frame.nrows) +
                                   ')')

        if "test_frame" in kwargs.keys():
            test_frame = H2OFrame(python_obj=kwargs['test_frame'], destination_frame='test_frame_' + analysis_id)
            self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                   'Parsing from pandas to H2OFrames: test_frame (' + str(test_frame.nrows) +
                                   ')')

        # Initializing base structures
        normalization = get_model_ns(base_ar)
        objective_column = base_ar['objective_column']

        tolerance = get_tolerance(df_metadata['columns'], objective_column, self._tolerance)
        print(tolerance)

        base_ar['data_initial'] = df_metadata

        # Generating base_path
        print(base_ar['type'])
        base_path = self.generate_base_path(base_ar, base_ar['type'])

        # Applying Normalizations
        if normalization is not None:
            base_ns = json.load(normalization, object_pairs_hook=NormalizationSet)
            self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                  'Executing Normalizations: ' + base_ns)
            '''Include normalizations a registry activities'''
            '''Assign data_normalized  base_ar['data_normalized']'''
        else:
            base_ns = None
            self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                  'No Normalizations Required')

        assert isinstance(base_ar, ArMetadata)
        assert isinstance(base_ns, NormalizationSet) or normalization is None

        final_ar_model = copy.deepcopy(base_ar)
        model_timestamp = str(time.time())

        model_id = base_ar['model_parameters']['h2o']['model'] + '_' + model_timestamp
        print(base_ar['model_parameters']['h2o']['types'][0])
        analysis_type = base_ar['model_parameters']['h2o']['types'][0]['type']

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

        need_factor(atype=base_ar['model_parameters']['h2o']['types'][0]['type'], training_frame=training_frame,
                    valid_frame=valid_frame, objective_column=objective_column)

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

        self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                               "Generating Model: " + model_command)
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
        try:
            eval(train_command)
            final_ar_model['status'] = 'Executed'
        except OSError as execution_error:
            self._logging.log_error(analysis_id, self._h2o_session.session_id,
                                   "Aborting Execution Model: " + repr(execution_error))
            return analysis_id, None



        self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                               ("Trained Model: %s :" % model_id) + train_command)
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

        self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                               model_id + " :Saved Model ")

        # Filling whole json ar.json
        final_ar_model['ignored_parameters'], \
        final_ar_model['full_parameters_stack'] = self._generate_params()

        # Generating aditional model parameters
        final_ar_model['model_parameters']['h2o']['parameters']['model_id'] = \
            OrderedDict(ParameterMetadata(value=model_id, seleccionable=False, type="String"))

        # Generating execution metrics
        final_ar_model['metrics']['execution'] = ExecutionMetricCollection()
        print(analysis_type)

        self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                               "Generating Model %s Execution Metrics" % model_id)

        final_ar_model['metrics']['execution']['train'] = self._generate_execution_metrics(dataframe=None,
                                                                                           source='train',
                                                                                           antype=analysis_type)
        final_ar_model['metrics']['accuracy'] = OrderedDict()
        final_ar_model['metrics']['execution']['xval'] = \
            self._generate_execution_metrics(dataframe=None, source='xval', antype=analysis_type)

        if valid_frame is not None:
            final_ar_model['metrics']['execution']['valid'] = \
                self._generate_execution_metrics(dataframe=None, source='valid', antype=analysis_type)
            final_ar_model['metrics']['accuracy']['train'] = \
                self._accuracy(objective_column, training_frame.rbind(valid_frame),
                               antype=analysis_type, tolerance=tolerance)
        else:
            final_ar_model['metrics']['accuracy']['train'] = \
                self._accuracy(objective_column, training_frame, antype=analysis_type, tolerance=tolerance)
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, "Model %s Accuracy: %s " %
                               (model_id, final_ar_model['metrics']['accuracy']['train']))

        if test_frame is not None:
            final_ar_model['metrics']['accuracy']['test'] = \
                self._accuracy(objective_column, test_frame, antype=analysis_type, tolerance=tolerance)
            final_ar_model['metrics']['accuracy']['combined'] = \
                (final_ar_model['metrics']['accuracy']['train']*0.4 + final_ar_model['metrics']['accuracy']['test']*0.6)
            self._logging.log_exec(analysis_id, self._h2o_session.session_id, "Model %s Test Accuracy: %s " %
                                   (model_id, final_ar_model['metrics']['accuracy']['test']))
            self._logging.log_exec(analysis_id, self._h2o_session.session_id, "Model %s CombinedAccuracy: %s " %
                                   (model_id, final_ar_model['metrics']['accuracy']['combined']))




        # Generating model metrics
        final_ar_model['metrics']['model'] = self._generate_model_metrics()
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, "Model %s Metrics: %s " %
                               (model_id, final_ar_model['metrics']['model']))

        # Generating Variable importance
        final_ar_model['metrics']['var_importance'] = self._generate_importance_variables()
        self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                               "Model %s Variable Importance: %s " %
                               (model_id, final_ar_model['metrics']['var_importance']))

        # Generating scoring_history
        final_ar_model['metrics']['scoring'] = self._generate_scoring_history()
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, "Model %s Scoring: %s " %
                               (model_id, final_ar_model['metrics']['scoring']))

        # writing ar.json file
        final_ar_model['json_path'] = StorageMetadata()
        for each_storage_type in base_ar['json_path']:
            json_path = base_path + each_storage_type['value'] + '/' + model_id + '.json'
            final_ar_model['json_path'].append(value=json_path, fstype=each_storage_type['type'],
                                               hash_type=each_storage_type['hash_type'])
        self._persistence.store_json(storage_json=final_ar_model['json_path'], ar_json=final_ar_model)
        self._logging.log_exec(analysis_id, self._h2o_session.session_id, "Model %s Generated" % model_id)
        for handler in self._logging.logger.handlers:
            handler.flush()

        return analysis_id, final_ar_model

    ## Main method to execute predictions over traning models
    # Take the ar.json for and execute predictions including its metrics a storage paths
    # @param self object pointer
    # @param predict_frame pandas.DataFrame
    # @param algorithm_description ArMetadata.json path
    # or compatible tuple (OrderedDict(), OrderedDict())
    # @return (String, [ArMetadata]) equivalent to (analysis_id, List[analysis_results])
    def predict(self, predict_frame, algorithm_description):
        model_timestamp = str(time.time())

        base_ar = json.load(algorithm_description, object_pairs_hook=OrderedDict)
        antype = base_ar['model_parameters']['h2o']['types'][0]['type']
        load_fails = True
        counter_storage = 0

        #Checking file source versus hash_value
        assert isinstance(base_ar['load_path'], list)
        while counter_storage < len(base_ar['load_path']) and load_fails:
            self._logging.log_exec('Predict', self._h2o_session.session_id,
                                   "Model hash keys (stored, %s) (generated, %s)" %
                                   (base_ar['load_path'][counter_storage]['hash_value'],
                                    hash_key(base_ar['load_path'][counter_storage]['hash_type'],
                                                    base_ar['load_path'][counter_storage]['value'])
                                    ))

            if hash_key(base_ar['load_path'][counter_storage]['hash_type'],
                        base_ar['load_path'][counter_storage]['value']) == \
                    base_ar['load_path'][counter_storage]['hash_value']:
                load_fails = False
                try:
                    if self._get_ext() == '.pojo':
                        '''Not implemented yet'''
                        pass
                    else:
                        self._model_base = load_model(base_ar['load_path'][counter_storage]['value'])
                except H2OError:
                    self._logging.log_error('root', self._h2o_session.session_id,
                                            "Invalid model on:  %s" %
                                            base_ar['load_path'][counter_storage]['value'])
            counter_storage += 1

        if load_fails:
            self._logging.log_error('root', self._h2o_session.session_id,
                                    "Invalid models on:  %s" % base_ar['load_path'])
            return 1

        objective_column = base_ar['objective_column']

        # Recovering tolerance
        tolerance = get_tolerance(base_ar["data_initial"]['columns'], objective_column, tolerance=0.005)

        df_metadata = DFMetada()
        df_metadata.getDataFrameMetadata(dataframe=predict_frame, typedf='pandas')
        self._logging.log_exec(base_ar['model_id']['value'], self._h2o_session.session_id,
                              'Predict dataframe_structure: %s'% df_metadata)

        base_ar['data_initial'] = df_metadata

        if base_ar['normalizations_set'] is not None:
            base_ns = json.load(base_ar['normalizations_set'], object_pairs_hook=NormalizationSet)
            self._logging.log_exec(base_ar['model_id']['value'], self._h2o_session.session_id,
                                   'Executing Normalizations: ' + base_ns)
            '''Include normalizations a registry activities'''
            '''Assign data_normalized  base_ar['data_normalized']'''
        else:
            self._logging.log_exec(base_ar['model_id']['value'], self._h2o_session.session_id,
                                   'No Normalizations Required')

        #Transforming to H2OFrame
        predict_frame = H2OFrame(python_obj=predict_frame,
                                 destination_frame='predict_frame' + base_ar['model_id']['value'])

        need_factor(atype=base_ar['model_parameters']['h2o']['types'][0]['type'],
                    objective_column=objective_column, predict_frame=predict_frame)

        base_ar['type'] = 'predict'
        base_ar['timestamp'] = model_timestamp
        self._logging.log_exec(base_ar['model_id']['value'], self._h2o_session.session_id,
                               "Generating model performance metrics %s "
                               % base_ar['model_parameters']['h2o']['parameters']['model_id'])

        base_ar['metrics']['execution'][base_ar['type']] = self._generate_execution_metrics(dataframe=predict_frame,
                                                                               source=None, antype=antype)
        print(self._generate_execution_metrics(dataframe=predict_frame,
                                               source=None, antype=antype))

        base_ar['metrics']['accuracy']['predict'] = self._accuracy(objective_column, predict_frame,
                                                        antype=antype, tolerance=tolerance)
        self._logging.log_exec(base_ar['model_id']['value'], self._h2o_session.session_id, "Prediction Accuracy: %s " %
                               (base_ar['metrics']['accuracy']['predict']))

        if self._debug:
            for each_storage_type in base_ar['log_path']:
                each_storage_type['value'] = each_storage_type['value'].replace('train', 'predict') \
                    .replace('.log', '_' + model_timestamp + '.log')

            self._persistence.mkdir(type=base_ar['log_path'][0]['type'], grants=0o0777,
                                    path=dirname(base_ar['log_path'][0]['value']))
            connection().start_logging(base_ar['log_path'][0]['value'])
        self._logging.log_exec(base_ar['model_id']['value'], self._h2o_session.session_id,
                               "starting Prediction over Model %s "
                               % (base_ar['model_parameters']['h2o']['parameters']['model_id']))
        prediction_dataframe = self._model_base.predict(predict_frame).as_data_frame(use_pandas=True)

        if self._debug:
            connection().stop_logging()
            self._persistence.store_file(filename=base_ar['log_path'][0]['value'],
                                         storage_json=base_ar['log_path'])

        # writing ar.json file
        json_files = StorageMetadata()
        for each_storage_type in base_ar['json_path']:
            each_storage_type['value'] = each_storage_type['value'].replace('train', 'predict') \
                .replace('.json', '_' + model_timestamp + '.json')
            json_files.append(value=each_storage_type['value'], fstype=each_storage_type['type'],
                              hash_type=each_storage_type['hash_type'])

        self._persistence.store_json(json_files, base_ar)
        for handler in self._logging.logger.handlers:
            handler.flush()

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


## Auxiliary function to convert numerical and string columns on H2OFrame to enum (factor)
# for classification analysis
# Returns H2OFrames modified if apply for classification objectives
# @param atype String in ATypeMetadata().get_artypes(cls)
# @param objective_column String Column Objective
# @param training_frame H2OFrame for training (optional)
# @param valid_frame H2OFrame for validation (optional)
# @param predict_frame H2OFrame for prediction (optional)
def need_factor(atype, objective_column, training_frame=None, valid_frame=None, predict_frame=None):
    if atype in ['binomial', 'multinomial']:
        if training_frame is not None:
            if isinstance(training_frame[objective_column], (int, float)):
                training_frame[objective_column] = training_frame[objective_column].asfactor()
            else:
                training_frame[objective_column] = training_frame[objective_column].ascharacter().asfactor()
        if valid_frame is not None:
            if isinstance(valid_frame[objective_column], (int, float)):
                valid_frame[objective_column] = valid_frame[objective_column].asfactor()
            else:
                valid_frame[objective_column] = valid_frame[objective_column].ascharacter().asfactor()
        if predict_frame is not None:
            if isinstance(predict_frame[objective_column], (int, float)):
                predict_frame[objective_column] = predict_frame[objective_column].asfactor()
            else:
                predict_frame[objective_column] = predict_frame[objective_column].ascharacter().asfactor()


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
    print('min: %s' % min_val)
    print('max: %s' % max_val)
    print('tolerance: %s' % tolerance)
    return threshold