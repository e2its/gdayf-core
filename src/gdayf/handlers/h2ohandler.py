from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.model.metrics_base import H2OBinomialModelMetrics as H2OBinomialModelMetrics
from h2o.model.metrics_base import H2OMultinomialModelMetrics as H2OMultinomialModelMetrics
from h2o.model.metrics_base import H2ORegressionModelMetrics as H2ORegressionModelMetrics
from h2o import save_model as save_model
from h2o import load_model as load_model
from h2o import init as init
from h2o import connect as connect
from h2o import connection as connection
from h2o import cluster as cluster
from h2o import H2OFrame as H2OFrame
from pandas import DataFrame as DataFrame
import json
import time
from collections import OrderedDict as OrderedDict
import copy
from gdayf.logs.logshandler import LogsHandler
from gdayf.common.armetadata import ArMetadata
from gdayf.common.normalizationset import NormalizationSet
from gdayf.common.utils import mkdir
from gdayf.common.binomialmetricmetadata import BinomialMetricMetadata
from gdayf.common.multinomialmetricmetadata import MultinomialMetricMetadata
from gdayf.common.regressionmetricmetadata import RegressionMetricMetadata
from gdayf.common.metricmetadata import MetricMetadata
from gdayf.common.metricscollection import MetricCollection
from gdayf.persistence.persistencehandler import PersistenceHandler
from gdayf.common.storagemetadata import StorageMetadata
from os.path import dirname


__name__ = 'engines.h2o'

class H2OHandler(object):
    """
    H2OHandler

    Train: Get some analysis list of tuplas (analysis_results.json +  (algorithm + normalzations over a Dataframe) and launch de analysis on H2O platform
    predict: Get some list of [analysis_results.json] and load and execute algorithm
    Algorithms and model operation:

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

    def __init__(self):
        self._model_base = None
        self.path_localfs = r'D:/Data/models'
        self.path_hdfs = None
        self.primary_path = self.path_localfs
        self.url = 'http://127.0.0.1:54321'
        self.nthreads = 4
        self.ice_root = r'D:/Data/logs'
        self.max_mem_size = '8G'
        self.start_h2o = True
        self._debug = False
        self._framework = 'h2o'
        self._persistence = PersistenceHandler()

        try:
            self._h2o_session = connect(url=self.url)
        except:
            init(url=self.url, nthreads=self.nthreads, ice_root=self.ice_root, max_mem_size=self.max_mem_size)
            self._h2o_session = connection()
        self._logging = LogsHandler(__name__)
        self._logging.log_exec('DayF', self._h2o_session.session_id(), 'Connected to active cluster and ready')

    def __del__(self):
        self._h2o_session.close()
        del self._logging

    @classmethod
    def shutdown_cluster(cls):
        try:
            cluster().shutdown()

        except:
            print('H20-cluster not working')

    def generate_base_path(self, base_ar):
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
            load_path.append('train')
            load_path.append('/')
            return ''.join(load_path)

    @staticmethod
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

    @staticmethod
    def need_factor(each_model, training_frame, valid_frame, y):
        if each_model['types'][0]['type'] in ['binomial', 'multinomial']:
            if isinstance(training_frame[y], int):
                training_frame[y] = training_frame[y].asfactor()
                if valid_frame is not None:
                    valid_frame[y] = valid_frame[y].asfactor()
            else:
                training_frame[y] = training_frame[y].ascharacter().asfactor()
                if valid_frame is not None:
                    valid_frame[y] = valid_frame[y].asfactor()

    def predict(self, dataframe, algorithm_description):
        model_timestamp = str(time.time())

        struct_ar = json.load(algorithm_description, object_pairs_hook=OrderedDict)
        load_fails = True
        hash_fails = True
        counter_storage = 0
        counter_hash = 0

        assert isinstance(struct_ar['load_path'], list)
        while counter_storage < len(struct_ar['load_path']) and load_fails:
            while counter_hash < len(struct_ar['load_path'][counter_storage]['hash_list']) and \
                    hash_fails:

                self._logging.log_exec('Predict', self._h2o_session.session_id,
                                       "Model hash keys (stored, %s) (generated, %s)" %
                                       (struct_ar['load_path'][counter_storage]['hash_list'][counter_hash]['value'],
                                        self._persistence.hash_keys(struct_ar['load_path'][counter_storage]['hash_list']
                                                        [counter_hash]['type'],
                                                        struct_ar['load_path'][counter_storage]['value'])
                                        ))

                if self._hash_keys(struct_ar['load_path'][counter_storage]['hash_list'][counter_hash]['type'],
                                   struct_ar['load_path'][counter_storage]['value']) == \
                        struct_ar['load_path'][counter_storage]['hash_list'][counter_hash]['value']:
                    load_fails = False
                    hash_fails = False
                    try:
                        self._model_base = load_model(struct_ar['load_path'][counter_storage]['value'])
                    except:
                        self._logging.log_error('root', self._h2o_session.session_id,
                                                "Invalid model on:  %s" %
                                                struct_ar['load_path'][counter_storage]['value'])
                else:
                    counter_hash += 1
            counter_hash = 0
            counter_storage += 1

        if load_fails:
            self._logging.log_error('root', self._h2o_session.session_id,
                                    "Invalid models on:  %s" % struct_ar['load_path'])
            return 1

        predict_frame = H2OFrame(python_obj=dataframe, destination_frame='predict_frame' + struct_ar['model_id'])

        y = struct_ar['model_parameters']['h2o'][0]['parameters']['response_column']['value']
        if struct_ar['model_parameters']['h2o'][0]['types'][0]['type'] in ['binomial', 'multinomial']:
            predict_frame[y].asfactor()

        struct_ar['type'] = 'predict'
        struct_ar['timestamp'] = model_timestamp
        struct_ar['metrics'] = OrderedDict()
        self._logging.log_exec(struct_ar['model_id'], self._h2o_session.session_id,
                               "Generating model performance metrics %s "
                               % struct_ar['model_parameters']['h2o'][0]['parameters']['model_id'])
        struct_ar['metrics']['predict'] = self._generate_metrics(predict_frame, source=None)

        # writing ar.json file
        json_files = list()
        for each_storage_type in struct_ar['json_path']:
            if each_storage_type['type'] == 'localfs':
                mkdir(dirname(each_storage_type['value'].replace('train', 'predict')), 0o0777)
                each_storage_type['value'] = each_storage_type['value'].replace('train', 'predict') \
                    .replace('.json', '_' + model_timestamp + '.json')
                json_files.append(each_storage_type)
            elif each_storage_type['type'] == 'hdfs':
                None
            elif each_storage_type['type'] == 'mongoDB':
                None
        self._persistence.store_json(json_files, struct_ar)

        self._logging.log_exec(struct_ar['model_id'], self._h2o_session.session_id,
                               "starting Prediction over Model %s "
                               % (struct_ar['model_parameters']['h2o'][0]['parameters']['model_id']))

        for handler in self._logging.logger.handlers:
            handler.flush()

        return (self._model_base.predict(predict_frame).as_data_frame(use_pandas=True), struct_ar)

    def _generate_metrics(self, dataframe, source, antype):
        """
        Generate model metrics for this model on test_data.

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

        for parameter, _ in model_metrics.items():
            if parameter in ['hit_ratio_table', 'gains_lift_table', 'max_criteria_and_metric_scores']:
                model_metrics[parameter] = perf_metrics._metric_json[parameter].as_data_frame().to_json(orient='split')
            elif parameter in ['cm']:
                model_metrics[parameter] = \
                    perf_metrics._metric_json[parameter]['table'].as_data_frame().to_json(orient='split')
            elif parameter in ['thresholds_and_metric_scores']:
                model_metrics['cm'] = OrderedDict()
                for each_parameter in ['min_per_class_accuracy', 'absolute_mcc', 'precision', 'accuracy',
                                       'f0point5', 'f2', 'f1', 'mean_per_class_accuracy']:
                    model_metrics['cm'][each_parameter] = \
                        perf_metrics.confusion_matrix(
                            metrics=each_parameter).table.as_data_frame().to_json(orient='split')
                model_metrics[parameter] = perf_metrics._metric_json[parameter].as_data_frame().to_json(orient='split')
            else:
                model_metrics[parameter] = perf_metrics._metric_json[parameter]

        return model_metrics

    def _generate_params(self):
        """
        Generate model params for this model.

        :param base parmas_struct:
        :return dict(full_stack_parameters)
                """
        params = self._model_base.get_params()
        full_stack_params = OrderedDict()
        for key, values in params.items():
            if key not in ['model_id', 'training_frame', 'validation_frame', 'response_column']:
                full_stack_params[key] = values['actual_value']
        return ('Not implemented yet', full_stack_params)

    def get_metric(self, algorithm_description, metric, source):  # not tested
        try:
            struct_ar = OrderedDict(json.load(algorithm_description))
            model_metrics = struct_ar['metrics']['source']
        except:
            return ('Necesario cargar un modelo valid o ar.json valido')
        try:
            return struct_ar['metrics']['source'][metric]
        except:
            return 'Not Found'

    def order_training(self, analysis_id, training_frame, analysis_list):
        assert isinstance(analysis_id, str)
        assert isinstance(training_frame, DataFrame)
        # Not used now assert isinstance(valid_frame, DataFrame) or valid_frame is None
        assert isinstance(analysis_list, list)
        # python train parameters effective
        train_parameters_list = ['max_runtime_secs', 'fold_column',
                                 'weights_column', 'offset_column',
                                 'ignore_columns']


        status = -1  # Operation Code
        valid_frame = None
        model_list = list()
        analysis_timestamp = str(time.time())
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

        for algorithm_description, normalization in analysis_list:

            # Initializing base structures
            base_ar = json.load(algorithm_description, object_pairs_hook=ArMetadata)
            objective_column = base_ar['objective_column']

            # Generating base_path
            base_path = self.generate_base_path(base_ar)
            '''A eliminar con persistenciaHandler'''
            mkdir(base_path, 0o0777)

            if normalization is not None:
                base_ns = json.load(normalization, object_pairs_hook=NormalizationSet)
                self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                      'Executing Normalizations: ' + base_ns)
            else:
                base_ns = None
                self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                      'No Normalizations Required')

            assert isinstance(base_ar, ArMetadata)
            assert isinstance(base_ns, NormalizationSet) or normalization is None

            for each_model in base_ar['model_parameters']['h2o']:

                final_ar_model = copy.deepcopy(base_ar)
                final_ar_model['type'] = 'train'
                final_ar_model['timestamp'] = analysis_timestamp
                model_timestamp = str(time.time())
                model_path = base_path + final_ar_model['load_path'][0]['value']

                #Generating Paths
                # Generating log_path
                final_ar_model['log_path'] = StorageMetadata()
                for each_storage_type in base_ar['log_path']:
                    final_ar_model['log_path'].append(model_path + each_storage_type['value'] +
                                                      '/' + model_id + '.log', each_storage_type['type'])
                    mkdir(model_path + '/' + model_id + '/' + each_storage_type['value'] + '/' + model_id + '.log',
                          0o0777)

                # Generating load_path
                final_ar_model['load_path'] = StorageMetadata()
                for each_storage_type in base_ar['load_path']:
                    final_ar_model['load_path'].append(model_path + '/' + model_id, each_storage_type['type'])
                    mkdir(model_path + '/' + model_id, 0o0777)

                # Generating json_path

                # writing ar.json file
                final_ar_model['json_path'] = StorageMetadata
                for each_storage_type in base_ar['load_path']:
                    final_ar_model['json_path'].append(model_path + each_storage_type['value'] +
                                                       '/' + model_id + '.log', each_storage_type['type'])
                    mkdir(model_path + '/' + model_id + '/' + each_storage_type['value'] + '/' + model_id + '.log',
                          0o0777)
                    self._persistence.store_json(storage_json=final_ar_model['json_path'], ar_json=final_ar_model)

                ''' Generating and executing Models '''
                # 06/06/2017: Use ignore_columns instead X on train

                if each_model['types'][0]['active']:
                    self.need_factor(each_model, training_frame, valid_frame, objective_column)

                '''Generate commands: model and model.train()'''
                model_command = list()
                model_command.append(each_model['model'])
                model_command.append("(")
                model_command.append("training_frame=training_frame")
                train_command = list()
                # 06/06/2017: Use ignore_columns instead X on train
                '''train_command.append("self._model_base.train(x=%s, y=\'%s\', " % (x, y))'''
                train_command.append("self._model_base.train(y=\'%s\', " % objective_column)
                train_command.append("training_frame=training_frame")
                if valid_frame is not None:
                    model_command.append(", validation_frame=valid_frame")
                    train_command.append(", validation_frame=valid_frame")
                model_id = each_model['model'] + '_' + model_timestamp
                model_command.append(", model_id='%s'" % model_id)
                self.generate_commands_parameters(each_model, model_command, train_command, train_parameters_list)
                model_command.append(")")
                model_command = ''.join(model_command)
                train_command.append(")")
                train_command = ''.join(train_command)

                self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                      "Generating Model: " + model_command)
                # Generating model
                if self._debug:
                            connection().start_logging('DEBUG_' + final_ar_model['log_path'][0]['value'])
                self._model_base = eval(model_command)
                eval(train_command)
                self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                       ("Trained Model: %s :" % model_id) + train_command)
                if self._debug:
                    connection().stop_logging()

                mkdir(model_path, 0o0777)
                save_model(model=self._model_base, path=model_path, force=True)

                self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                       model_id + " :Saved Model ")

                # Filling whole json ar.json
                final_ar_model['ignored_parameters'], \
                final_ar_model['full_parameters_stack'] = self._generate_params()

                # Generating model parameters
                final_ar_model['model_parameters']['h2o'] = list()
                final_ar_model['model_parameters']['h2o'].append(each_model.copy())
                final_ar_model['model_parameters']['h2o'][0]['parameters']['model_id'] = model_id

                # Generating metrics
                final_ar_model['metrics'] = MetricCollection()
                final_ar_model['metrics']['train'] = self._generate_metrics(dataframe=None, source='train')
                if valid_frame is not None:
                    final_ar_model['metrics']['valid'] = self._generate_metrics(dataframe=None, source='valid')
                final_ar_model['metrics']['xval'] = self._generate_metrics(dataframe=None, source='xval')


                self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                      "Model %s Generated" % model_id)
                for handler in self._logging.logger.handlers:
                    handler.flush()
                model_list.append(final_ar_model)

            return analysis_id, model_list
