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
from hashlib import md5 as md5
from hashlib import sha256 as sha256
from os import makedirs as mkdir
from os import path as path
from shutil import copyfile
import copy
from gdayf.logs.logshandler import LogsHandler

__name__ = 'engines'

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
        self.url = 'http://127.0.0.1:54321'
        self.nthreads = 6
        self.ice_root = r'D:/Data/logs'
        self.max_mem_size = '8G'
        self.start_h2o = True
        self._debug = False
        self._framework = 'h2o'

        try:
            self._h2o_session = connect(url=self.url)
        except:
            init(url=self.url, nthreads=self.nthreads, ice_root=self.ice_root, max_mem_size=self.max_mem_size)
            self._h2o_session = connection()
        self._logging = LogsHandler()
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

    @staticmethod
    def _hash_keys(hash_type, filename):

            if hash_type == 'MD5':
                return md5(open(filename, 'rb').read()).hexdigest()
            elif hash_type == 'SHA256':
                return sha256(open(filename, 'rb').read()).hexdigest()

    def order_training(self, analysis_id, training_frame, valid_frame, analysis_list):
        assert isinstance(analysis_id, str)
        assert isinstance(training_frame, DataFrame)
        assert isinstance(valid_frame, DataFrame) or valid_frame is None
        assert isinstance(analysis_list, list)


        status = -1  # Operation Code
        model_list = list()
        analysis_timestamp = str(time.time())
        self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                              'Starting analysis')

        self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                              'Parsing from pandas to H2OFrame ' + 'training_frame')
        training_frame = H2OFrame(python_obj=training_frame)
        if valid_frame is not None:
            self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                  'Parsing from pandas to H2OFrame ' + 'validation_frame')
            valid_frame = H2OFrame(python_obj=valid_frame)

        for algorithm_description, normalization in analysis_list:

            # Initializing base structures
            struct_ar = json.load(algorithm_description, object_pairs_hook=OrderedDict)
            #print(struct_ar)

            # Generating base_path
            load_path = list()
            load_path.append(self.path_localfs)
            load_path.append('/')
            load_path.append(self._framework)
            load_path.append('/')
            load_path.append(struct_ar['model_id'])
            load_path.append('/')
            load_path.append('train')
            load_path.append('/')
            base_path = ''.join(load_path)
            if not path.exists(base_path):
                mkdir(base_path, 0o0777)

            if normalization is not None:
                struct_ns = json.load(normalization, object_pairs_hook=OrderedDict)
                self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                      'Executing Normalizations: ' + struct_ns)
            else:
                struct_ns = None
                self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                      'No Normalizations Required')

            assert isinstance(struct_ar, OrderedDict)
            assert isinstance(struct_ns, OrderedDict) or normalization is None

            for each_model in struct_ar['model_parameters']['h2o']:

                final_ar_model = copy.deepcopy(struct_ar)
                final_ar_model['type'] = 'train'
                final_ar_model['timestamp'] = analysis_timestamp
                model_timestamp = str(time.time())

                y = each_model['parameters']['response_column']['value']
                x = training_frame.col_names
                x.remove(y)

                if each_model['types'][0]['active']:
                    if each_model['types'][0]['type'] in ['binomial', 'multinomial']:
                        training_frame[y] = training_frame[y].asfactor()
                        if valid_frame is not None:
                            valid_frame[y] = valid_frame[y].asfactor()

                model_command = list()
                model_command.append(each_model['model'])
                model_command.append("(")
                model_command.append("training_frame=training_frame")

                if valid_frame is not None:
                    model_command.append(", validation_frame=valid_frame")

                model_id = each_model['model'] + '_' + model_timestamp
                model_command.append(", model_id='%s'" % model_id)

                for key, value in each_model['parameters'].items():
                    if value['seleccionable']:
                        if isinstance(value['value'], str):
                            model_command.append(", %s=\'%s\'" % (key, value['value']))
                        else:
                            model_command.append(", %s=%s" % (key, value['value']))

                model_command.append(")")
                model_command = ''.join(model_command)
                self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                      "Generating Model: " + model_command)

                #Modify when hdfs method will be implemented actually first value must be localfs

                # Generating model
                if self._debug:
                            connection().start_logging('DEBUG_' + final_ar_model['log_path'][0]['value'])
                self._model_base = eval(model_command)

                if valid_frame is not None:
                    self._model_base.train(x=x, y=y, training_frame=training_frame, validation_frame=valid_frame)
                else:
                    self._model_base.train(x=x, y=y, training_frame=training_frame)

                self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                      "Trained Model: " + model_command)
                if self._debug:
                    connection().stop_logging()

                model_path = base_path + final_ar_model['load_path'][0]['value']
                if not path.exists(model_path):
                    mkdir(model_path, 0o0777)
                save_model(model=self._model_base, path=model_path, force=True)

                self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                       model_id + " :Generating Execution log ")
                final_ar_model['ignored_parameters'], \
                final_ar_model['full_parameters_stack'] = self._generate_params()

                # Generating json ar.json
                # setting load_path
                counter_loop = True  #define First element
                for each_storage_type in final_ar_model['load_path']:
                    if counter_loop:
                        if each_storage_type['type'] == 'localfs':
                            if not path.exists(model_path):
                                mkdir(model_path, 0o0777)

                            each_storage_type['value'] = model_path + '/' + model_id
                            # Generating hash tags
                            for each_hash_type in each_storage_type['hash_list']:
                                if each_hash_type['type'] == 'MD5':
                                    each_hash_type['value'] = self._hash_keys(each_hash_type['type'],
                                                                              each_storage_type['value'])
                                elif each_hash_type['type'] == 'SHA256':
                                    each_hash_type['value'] = self._hash_keys(each_hash_type['type'],
                                                                              each_storage_type['value'])
                        elif each_storage_type['type'] == 'hdfs':
                            None
                        counter_loop = False
                    else:
                        if each_storage_type['type'] == 'localfs':
                            if not path.exists(base_path + each_storage_type['value']):
                                mkdir(base_path + each_storage_type['value'], 0o0777)
                            each_storage_type['value'] = base_path + each_storage_type['value'] + '/' + model_id
                            self._replicate_file(each_storage_type['type'],
                                                 each_storage_type['value'],
                                                 final_ar_model['load_path'][0]['type'],
                                                 final_ar_model['load_path'][0]['value'])
                            # Generating hash tags
                            for each_hash_type in each_storage_type['hash_list']:
                                if each_hash_type['type'] == 'MD5':
                                    each_hash_type['value'] = self._hash_keys(each_hash_type['type'],
                                                                              each_storage_type['value'])
                                elif each_hash_type['type'] == 'SHA256':
                                    each_hash_type['value'] = self._hash_keys(each_hash_type['type'],
                                                                              each_storage_type['value'])
                        elif each_storage_type['type'] == 'hdfs':
                            None

                # Generating model parameters
                final_ar_model['model_parameters']['h2o'] = list()
                final_ar_model['model_parameters']['h2o'].append(each_model.copy())
                final_ar_model['model_parameters']['h2o'][0]['parameters']['model_id'] = model_id

                # Generating metrics
                final_ar_model['metrics'] = OrderedDict()
                final_ar_model['metrics']['train'] = self._generate_metrics(dataframe=None, source='train')
                if valid_frame is not None:
                    final_ar_model['metrics']['valid'] = self._generate_metrics(dataframe=None, source='valid')
                final_ar_model['metrics']['xval'] = self._generate_metrics(dataframe=None, source='xval')

                # Generating log_path
                for each_storage_type in final_ar_model['log_path']:
                    if each_storage_type['type'] == 'localfs':
                        if not path.exists(base_path + each_storage_type['value']):
                            mkdir(base_path + each_storage_type['value'], 0o0777)
                        each_storage_type['value'] = base_path + each_storage_type['value'] + '/' + model_id + '.log'
                    elif each_storage_type['type'] == 'hdfs':
                        None

                # writing ar.json file
                json_files = list()
                for each_storage_type in final_ar_model['json_path']:
                    if each_storage_type['type'] == 'localfs':
                        if not path.exists(base_path + each_storage_type['value']):
                            mkdir(base_path + each_storage_type['value'], 0o0777)
                        each_storage_type['value'] = base_path + each_storage_type['value'] + '/' + model_id + '.json'
                        json_files.append(each_storage_type)
                    elif each_storage_type['type'] == 'hdfs':
                        None
                    elif each_storage_type['type'] == 'mongoDB':
                        None
                self._store_files(json_files, final_ar_model)

                self._logging.log_exec(analysis_id, self._h2o_session.session_id,
                                      "Model %s Generated" % model_id)
                for handler in self._logging.logger.handlers:
                    handler.flush()


                model_list.append(final_ar_model)

            return analysis_id, model_list

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
                                        self._hash_keys(struct_ar['load_path'][counter_storage]['hash_list']
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


        predict_frame = H2OFrame(python_obj=dataframe)

        y = struct_ar['model_parameters']['h2o'][0]['parameters']['response_column']['value']
        if struct_ar['model_parameters']['h2o'][0]['types'][0]['type'] in ['binomial', 'multinomial']:
            predict_frame[y].asfactor()

        struct_ar['type'] = 'predict'
        struct_ar['timestamp'] = model_timestamp
        struct_ar['metrics'] = OrderedDict()
        self._logging.log_exec(struct_ar['model_id'], self._h2o_session.session_id,
                              "Generating model performance metrics %s "
                               % struct_ar['model_parameters']['h2o'][0]['model'])
        struct_ar['metrics']['predict'] = self._generate_metrics(predict_frame, source=None)

        # writing ar.json file
        json_files = list()
        for each_storage_type in struct_ar['json_path']:
            if each_storage_type['type'] == 'localfs':
                if not path.exists(path.dirname(each_storage_type['value'].replace('train', 'predict'))):
                    mkdir(path.dirname(each_storage_type['value'].replace('train', 'predict')), 0o0777)
                each_storage_type['value'] = each_storage_type['value'].replace('train', 'predict')\
                    .replace('.json', '_' + model_timestamp + '.json')
                json_files.append(each_storage_type)
            elif each_storage_type['type'] == 'hdfs':
                None
            elif each_storage_type['type'] == 'mongoDB':
                None
        self._store_files(json_files, struct_ar)

        self._logging.log_exec(struct_ar['model_id'], self._h2o_session.session_id,
                              "starting Prediction over Model %s "
                               % (struct_ar['model_parameters']['h2o'][0]['model']))

        for handler in self._logging.logger.handlers:
            handler.flush()

        return (self._model_base.predict(predict_frame).as_data_frame(use_pandas=True), struct_ar)

    def _generate_metrics(self, dataframe, source):
        """
        Generate model metrics for this model on test_data.

        :param H2OFrame test_data: Data set for which model metrics shall be computed against. All three of train,
            valid and xval arguments are ignored if test_data is not None.
        :param source 'train': Report the training metrics for the model.
        :param source 'valid': Report the validation metrics for the model.
        :param source 'xval': Report the cross-validation metrics for the model. If train and valid are True, then it
            defaults to True.
        """

        model_metrics = OrderedDict()

        if dataframe is not None:
            perf_metrics = self._model_base.model_performance(dataframe)
        else:
            if source == 'valid':
                perf_metrics = self._model_base.model_performance(valid=True)
            elif source == 'xval':
                perf_metrics = self._model_base.model_performance(xval=True)
            else:
                perf_metrics = self._model_base.model_performance(train=True)

        for parameter, value in perf_metrics._metric_json.items():
            if parameter in ['hit_ratio_table', 'gains_lift_table', 'max_criteria_and_metric_scores']:
                model_metrics[parameter] = value.as_data_frame().to_json(orient='split')
            elif parameter in ['cm']:
                model_metrics[parameter] = value['table'].as_data_frame().to_json(orient='split')
            elif parameter in ['thresholds_and_metric_scores']:
                model_metrics['cm'] = OrderedDict()
                for each_parameter in ['min_per_class_accuracy', 'absolute_mcc', 'precision', 'accuracy',
                                       'f0point5', 'f2', 'f1', 'mean_per_class_accuracy']:
                    model_metrics['cm'][each_parameter] = \
                        perf_metrics.confusion_matrix(
                            metrics=each_parameter).table.as_data_frame().to_json(orient='split')
                model_metrics[parameter] = value.as_data_frame().to_json(orient='split')
            elif not isinstance(value, dict) and parameter not in ['model_checksum', 'frame_checksum', 'description']:
                model_metrics[parameter] = value

        return model_metrics

    def _generate_params(self):
        """
        Generate model params for this model.

        :param base parmas_struct:
        :return dict(full_stack_parameters)
                """
        params = self._model_base.get_params()
        full_stack_params = OrderedDict()
        print(type(params))
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

    @staticmethod
    def _replicate_file(type_dest, path_dest, type_source, path_source):
        if type_source == 'localfs':
            if type_dest == 'localfs':
                if not path.exists(path.dirname(path_dest)):
                    mkdir(path.dirname(path_dest), 0o0777)
                copyfile(path_source, path_dest)
            elif type_dest == 'hdfs':
                None
            elif type_dest == 'mongoDB':
                None
        elif type_source == 'hdfs':
            if type_dest == 'localfs':
                None
            elif type_dest == 'hdfs':
                None
            elif type_dest == 'mongoDB':
                None
        elif type_source == 'mongoDB':
            if type_dest == 'localfs':
                None
            elif type_dest == 'hdfs':
                None
            elif type_dest == 'mongoDB':
                None

    @staticmethod
    def _store_files(json_files, struct_ar):
        for each_storage_type in json_files:
            if each_storage_type['type'] == 'localfs':
                file = open(each_storage_type['value'], 'w')
                json.dump(struct_ar, file, indent=4)
                file.close()
            elif each_storage_type['type'] == 'hdfs':
                None
            elif each_storage_type['type'] == 'mongoDB':
                None

    @staticmethod
    def _store_files(json_files, struct_ar):
        for each_storage_type in json_files:
            if each_storage_type['type'] == 'localfs':
                file = open(each_storage_type['value'], 'w')
                json.dump(struct_ar, file, indent=4)
                file.close()
            elif each_storage_type['type'] == 'hdfs':
                None
            elif each_storage_type['type'] == 'mongoDB':
                None
