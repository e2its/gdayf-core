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
from gdayf.metrics.metricscollection import MetricCollection
from gdayf.metrics.regressionmetricmetadata import RegressionMetricMetadata
from gdayf.metrics.multinomialmetricmetadata import MultinomialMetricMetadata
from gdayf.persistence.persistencehandler import PersistenceHandler
from gdayf.conf.loadconfig import LoadConfig

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

        self._model_base = None
        self._persistence = PersistenceHandler()
        try:
            self._h2o_session = connect(url=self.url)
        except H2OError:
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
    def need_factor(atype, y, training_frame=None, valid_frame=None, predict_frame=None):
        if atype in ['binomial', 'multinomial']:
            if training_frame is not None:
                if isinstance(training_frame[y], (int, float)):
                    training_frame[y] = training_frame[y].asfactor()
                else:
                    training_frame[y] = training_frame[y].ascharacter().asfactor()
            if valid_frame is not None:
                if isinstance(valid_frame[y], (int, float)):
                    valid_frame[y] = valid_frame[y].asfactor()
                else:
                    valid_frame[y] = valid_frame[y].ascharacter().asfactor()
            if predict_frame is not None:
                if isinstance(predict_frame[y], (int, float)):
                    predict_frame[y] = predict_frame[y].asfactor()
                else:
                    predict_frame[y] = predict_frame[y].ascharacter().asfactor()

    def predict(self, dataframe, algorithm_description):
        model_timestamp = str(time.time())

        base_ar = json.load(algorithm_description, object_pairs_hook=OrderedDict)
        antype = base_ar['model_parameters']['h2o'][0]['types'][0]['type']
        load_fails = True
        counter_storage = 0

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
                        None
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

        predict_frame = H2OFrame(python_obj=dataframe, destination_frame='predict_frame' + base_ar['model_id'])

        objective_column = base_ar['objective_column']
        self.need_factor(atype=base_ar['model_parameters']['h2o'][0]['types'][0]['type'],
                         y=objective_column, predict_frame=predict_frame)

        base_ar['type'] = 'predict'
        base_ar['timestamp'] = model_timestamp
        self._logging.log_exec(base_ar['model_id'], self._h2o_session.session_id,
                               "Generating model performance metrics %s "
                               % base_ar['model_parameters']['h2o'][0]['parameters']['model_id'])

        base_ar['metrics'][base_ar['type']] = self._generate_metrics(dataframe=predict_frame,
                                                                     source=None, antype=antype)
        print(self._generate_metrics(dataframe=predict_frame,
                               source=None, antype=antype))

        if self._debug:
            for each_storage_type in base_ar['log_path']:
                each_storage_type['value'] = each_storage_type['value'].replace('train', 'predict') \
                    .replace('.log', '_' + model_timestamp + '.log')

            self._persistence.mkdir(type=base_ar['log_path'][0]['type'], grants=0o0777,
                                    path=dirname(base_ar['log_path'][0]['value']))
            connection().start_logging(base_ar['log_path'][0]['value'])
        self._logging.log_exec(base_ar['model_id'], self._h2o_session.session_id,
                               "starting Prediction over Model %s "
                               % (base_ar['model_parameters']['h2o'][0]['parameters']['model_id']))
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

    def _get_ext(self):
        if self._save_model == 'POJO':
            return '.pojo'
        else:
            return '.model'

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
        model_metrics.set_metrics(perf_metrics)
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
                                 'weights_column', 'offset_column']


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
            base_ar = json.load(algorithm_description, object_pairs_hook=OrderedDict)
            objective_column = base_ar['objective_column']

            # Generating base_path
            base_path = self.generate_base_path(base_ar, base_ar['type'])
            if normalization is not None:
                base_ns = json.load(normalization, object_pairs_hook=NormalizationSet)
                self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                      'Executing Normalizations: ' + base_ns)
            else:
                base_ns = None
                self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                      'No Normalizations Required')

            assert isinstance(base_ar, OrderedDict)
            assert isinstance(base_ns, NormalizationSet) or normalization is None

            for each_model in base_ar['model_parameters']['h2o']:

                final_ar_model = copy.deepcopy(base_ar)
                final_ar_model['type'] = 'train'
                final_ar_model['timestamp'] = analysis_timestamp
                model_timestamp = str(time.time())
                model_id = each_model['model'] + '_' + model_timestamp

                #model_path = base_path + base_ar['load_path'][0]['value'] + '/'
                analysis_type = each_model['types'][0]['type']

                ''' Generating and executing Models '''
                # 06/06/2017: Use X less ignored_columns on train
                x = training_frame.col_names
                x.remove(objective_column)
                try:
                    for ignore_col in each_model['parameters']['ignored_columns']['value']:
                        x.remove(ignore_col)
                except KeyError:
                    None

                self.need_factor(atype=each_model['types'][0]['type'], training_frame=training_frame,
                                 valid_frame=valid_frame, y=objective_column)

                '''Generate commands: model and model.train()'''
                model_command = list()
                model_command.append(each_model['model'])
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
                self.generate_commands_parameters(each_model, model_command, train_command, train_parameters_list)
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
                eval(train_command)
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
                        '''Not implemented yet'''
                        None
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

                # Generating model parameters
                final_ar_model['model_parameters']['h2o'] = list()
                final_ar_model['model_parameters']['h2o'].append(each_model.copy())
                final_ar_model['model_parameters']['h2o'][0]['parameters']['model_id'] = model_id + self._get_ext()

                # Generating metrics
                final_ar_model['metrics'] = MetricCollection()
                print(analysis_type)
                final_ar_model['metrics']['train'] = self._generate_metrics(dataframe=None, source='train',
                                                                            antype=analysis_type)
                if valid_frame is not None:
                    final_ar_model['metrics']['valid'] = self._generate_metrics(dataframe=None, source='valid',
                                                                                antype=analysis_type)
                final_ar_model['metrics']['xval'] = self._generate_metrics(dataframe=None, source='xval',
                                                                           antype=analysis_type)

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
                model_list.append(final_ar_model)
            return analysis_id, model_list
