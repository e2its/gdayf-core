## @package gdayf.handlers.sparkhandler
# Define all objects, functions and structures related to executing actions or activities over spark.ai framework
#
# Main class sparkHandler. Lets us execute analysis, make prediction and execute multi-packet operations structures on
# format [(Analysis_results.json, normalization_sets.json) ]
# Analysis_results.json could contain executions models for various different model or parameters


import copy
import json
import time
import sys
from collections import OrderedDict as OrderedDict
from os.path import dirname
from pandas import DataFrame as DataFrame
from hashlib import md5 as md5
from py4j.protocol import Py4JJavaError


try:
    # Now we are ready to import Spark Modules
    from pyspark.sql import SparkSession
    from pyspark.ml import Pipeline
    from pyspark.ml import PipelineModel
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.feature import VectorIndexer
    from pyspark.ml.classification import *
    from pyspark.ml.regression import *
    from pyspark.ml.clustering import *
    from pyspark.ml.evaluation import *
    from pyspark.ml.tuning import *
    from pyspark.ml.feature import StringIndexer
    from pyspark.ml.feature import IndexToString
    from pyspark.ml.feature import OneHotEncoder
    from pyspark.sql.utils import IllegalArgumentException

    print("Successfully imported all Spark modules")
except ImportError as e:
    print("Error importing Spark Modules", e)
    sys.exit(1)


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


class sparkHandler(object):

    ## Constructor
    # Initialize all framework variables and starts or connect to spark cluster
    # Aditionally starts PersistenceHandler and logsHandler
    def __init__(self):
        self._framework = 'spark'
        self._config = LoadConfig().get_config()
        self._labels = LoadLabels().get_config()['messages']['corehandler']
        self.localfs = self._config['storage']['localfs']['value']
        self.hdfs =self._config['storage']['hdfs']['value']
        self.mongoDB = self._config['storage']['mongoDB']['value']
        self.primary_path = self._config['storage'][self._config['storage']['primary_path']]['value']
        self.url = self._config['frameworks'][self._framework]['conf']['master']
        self.nthreads = self._config['frameworks'][self._framework]['conf']['nthreads']
        self.spark_warehouse_dir = self._config['frameworks'][self._framework]['conf']['spark_warehouse_dir']
        self.spark_executor_mem = self._config['frameworks'][self._framework]['conf']['spark.executor.memory']
        self.spark_driver_mem = self._config['frameworks'][self._framework]['conf']['spark.driver.memory']
        self.start_spark = self._config['frameworks'][self._framework]['conf']['start_spark']
        self._save_model = self._config['frameworks'][self._framework]['conf']['save_model']
        self._tolerance = self._config['frameworks'][self._framework]['conf']['tolerance']
        self._model_base = None
        self.analysis_id = None
        self._spark_session = None
        self._persistence = PersistenceHandler()
        self._logging = LogsHandler(__name__)
        self._frame_list = list()

    ## Destructor
    def __del__(self):
        if self._spark_session is not None and self.is_alive():
            self._spark_session.stop()

    ## Class Method for cluster shutdown
    # @param cls class pointer
    # Not implemented
    @classmethod
    def shutdown_cluster(cls):
        try:
            pass
        except Py4JJavaError:
            print('Apache Spark Cluster not working')

    ''''@staticmethod
    def addColumnIndex(dataframe):
        # Create new column names
        oldColumns = dataframe.schema.names
        newColumns = oldColumns + ["columnindex"]

        # Add Column index
        df_indexed = df.rdd.zipWithIndex().map(lambda row, columnindex: \
                                                      row + (columnindex,)).toDF()

        # Rename all the columns
        new_df = reduce(lambda data, idx: data.withColumnRenamed(oldColumns[idx],
                                                                 newColumns[idx]), range(len(oldColumns)),
                        df_indexed)
        return new_df'''

    ## Connexion_method to cluster
    #If cluster is up connect to cluster on another case start a cluster
    # @return initiatiated True/False
    def connect(self):
        initiated = False
        try:
            spark = SparkSession.builder.master(self.url + '[' + str(self.nthreads) + ']')\
                .appName('job_gdayf_'+self.url+'_' + time.strftime("%b-%d-%Y_%H:%M:%S-%z", time.localtime())) \
                .config("spark.executor.memory", self.spark_executor_mem) \
                .config("spark.driver.memory", self.spark_driver_mem) \
                .config("spark_warehouse_dir", self.spark_warehouse_dir)

            self._spark_session = spark.getOrCreate()
            self._spark_session.spark.sparkContext.setLogLevel("INFO")

        except Py4JJavaError:
            self._logging.log_critical('gDayF', "sparkHandler", self._labels["failed_conn"])
            raise Exception
        finally:
            self._logging.log_info('gDayF', "sparkHandler", self._labels["start"])
            self._logging.log_info('gDayF', "sparkHandler", self._labels["framework"], self._framework)
            self._logging.log_info('gDayF', "sparkHandler", self._labels["sess"],
                                   self._spark_session.sparkContext.applicationId)
            return initiated

    ## Get Spark dtype for column
    # @param list_dtypes Spark Dataframe.dtypes structure
    # @param column string
    # @return dtype or None if not exist
    @ staticmethod
    def _get_dtype(list_dtypes, column):
        for element in list_dtypes:
            if element[0] == column:
                return element[1]
        return None

    ## Is alive_connection method
    def is_alive(self):
        if self._spark_session is None:
            return False
        elif self._spark_session._instantiatedSession is None:
            return False
        else:
            return True


    ## Generate list of models_id for internal crossvalidation objects_
    # @param self object pointer
    # @param model_id base id_model
    # @param nfols number of cv buckets
    # @return models_ids lst of models_ids
    def _get_temporal_objects_ids(self, model_id, nfolds):
        pass

    ## Generate pdml model class_
    # @param self object pointer
    # @param ar_metadata ArMetadata stored model
    # @param type ['pojo', 'mojo']
    # @param user user_id
    # @return download_path, MD5 hash_key
    # Not implemented
    def get_external_model(self, ar_metadata, type, user):
        return False

    ## Load a model in sparkCluster from disk
    # @param self object pointer
    # @param ar_metadata ArMetadata stored model
    # @return implicit self._model_base / None on Error
    def _get_model_from_load_path(self, ar_metadata):
        load_fails = True
        counter_storage = 0
        # Checking file source versus hash_value

        try:
            assert isinstance(ar_metadata['load_path'], list)
        except AssertionError:
            return load_fails

        while ar_metadata['load_path'] is not None and counter_storage < len(ar_metadata['load_path']) and load_fails:

            if ar_metadata['load_path'][counter_storage]['hash_value'] is None or \
                            hash_key(ar_metadata['load_path'][counter_storage]['hash_type'],
                            ar_metadata['load_path'][counter_storage]['value']) == \
                            ar_metadata['load_path'][counter_storage]['hash_value']:
                try:
                    self._model_base = PipelineModel.load(ar_metadata['load_path'][counter_storage]['value'])
                    if self._model_base is not None:
                        load_fails = False

                        if ar_metadata['load_path'][counter_storage]['hash_value'] is not None:
                            self._logging.log_info(self.analysis_id, self._spark_session.sparkContext.applicationId,
                                                   self._labels["hk_check"],
                                                   ar_metadata['load_path'][counter_storage]['hash_value'] + ' - ' +
                                                   hash_key(ar_metadata['load_path'][counter_storage]['hash_type'],
                                                            ar_metadata['load_path'][counter_storage]['value'])
                                                   )
                except Py4JJavaError:
                    self._logging.log_error(self.analysis_id, self._spark_session.sparkContext.applicationId,
                                            self._labels["abort"], ar_metadata['load_path'][counter_storage]['value'])

            counter_storage += 1
        return load_fails

    ## Remove used dataframes during analysis execution_
    # @param self object pointer
    # Not implemented
    def delete_frames(self):
        pass

    ## Generate base path to store all files [models, logs, json] relative to it
    # @param self object pointer
    # @param base_ar initial ar.json template pass to object instance
    # @param type_ type of analysis to execute
    # @return base path string
    def generate_base_path(self, base_ar, type_):
        assert type_ in ['PoC', 'train', 'predict']
        if self.primary_path == self.mongoDB:
            return None
        elif self.primary_path == self.hdfs:
            # Generating base_path
            load_path = list()
            load_path.append(self.hdfs)
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
        else:
            # Generating base_path
            load_path = list()
            load_path.append(self.localfs)
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
    # @return ['.spark']
    def _get_ext(self):
        return '.spark'

    ## Get Evaluator for model
    # @param analysis_type ['binomial', 'multinomial', regression, ..]
    # @param objective_column
    # @return pyspark.ml.evaluation.Evaluator children or None if not exist
    @ staticmethod
    def _get_evaluator(analysis_type, objective_column=None):
        if objective_column is None:
            if analysis_type == 'clustering':
                return None
        else:
            if analysis_type == 'binomial':
                #print('TRC' + objective_column)
                return BinaryClassificationEvaluator(labelCol=objective_column)
            elif analysis_type == 'multinomial':
                return MulticlassClassificationEvaluator(labelCol=objective_column)
            elif analysis_type == 'regression':
                return RegressionEvaluator(labelCol=objective_column)
        return None

    ## Generate execution metrics for the correct model
    # @param self object pointer
    # @param dataframe sparkFrame for prediction metrics
    # @param source [train, val, xval]
    # @param  antype Atypemetadata().get_artypes() values allowed
    # @return model_metrics Subclass Metrics Metadata
    def _generate_execution_metrics(self, dataframe, antype, objective_column):
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

        evaluator = self._get_evaluator(analysis_type=antype, objective_column=objective_column)

        if isinstance(model_metrics, ClusteringMetricMetadata):
            model_metrics.set_sparkmetrics(model=self._model_base.stages[-1], data=dataframe)
        else:
            model_metrics.set_sparkmetrics(evaluator=evaluator, data=dataframe, objective_column=objective_column)
        return model_metrics

    ## Generate model scoring_history metrics
    # @param self object pointer
    # @result_dataframe = json_pandas_dataframe structure orient=split
    def _generate_scoring_history(self):
        result_dataframe = None
        if isinstance(self._model_base.stages[-1], GBTRegressionModel) or \
                isinstance(self._model_base.stages[-1], RandomForestRegressionModel) or \
                isinstance(self._model_base.stages[-1], GBTClassificationModel):
            maximo = 0
            for itera in range(0, len(self._model_base.stages[-1].trees)):
                maximo = max(maximo, self._model_base.stages[-1].trees[itera].depth)

            result_dataframe = DataFrame(data={'trees': self._model_base.stages[-1].getNumTrees,
                                   'max_depth': maximo,
                                   'total_nodes': self._model_base.stages[-1].totalNumNodes,
                                   'numFeatures': self._model_base.stages[-1].numFeatures},
                             index=[0]).to_json(orient='split')
        elif isinstance(self._model_base.stages[-1], RandomForestClassificationModel):
            maximo = 0
            for itera in range(0, len(self._model_base.stages[-1].trees)):
                maximo = max(maximo, self._model_base.stages[-1].trees[itera].depth)

            result_dataframe = DataFrame(data={'trees': self._model_base.stages[-1].getNumTrees,
                                   'max_depth': maximo,
                                   'total_nodes': self._model_base.stages[-1].totalNumNodes,
                                   'numFeatures': self._model_base.stages[-1].numFeatures,
                                   'numClasses': self._model_base.stages[-1].numClasses},
                             index=[0]).to_json(orient='split')
        elif isinstance(self._model_base.stages[-1], DecisionTreeRegressionModel):
            result_dataframe = DataFrame(data={'trees': 1,
                                   'max_depth': self._model_base.stages[-1].depth,
                                   'total_nodes': self._model_base.stages[-1].numNodes,
                                   'numFeatures': self._model_base.stages[-1].numFeatures},
                             index=[0]).to_json(orient='split')
        elif isinstance(self._model_base.stages[-1], NaiveBayesModel):
            result_dataframe = DataFrame(data={'numFeatures': self._model_base.stages[-1].numFeatures,
                                   'numClasses': self._model_base.stages[-1].numClasses},
                             index=[0]).to_json(orient='split')
        elif isinstance(self._model_base.stages[-1], DecisionTreeClassificationModel):
            result_dataframe = DataFrame(data={'trees': 1,
                                   'max_depth': self._model_base.stages[-1].depth,
                                   'total_nodes': self._model_base.stages[-1].numNodes,
                                   'numFeatures': self._model_base.stages[-1].numFeatures,
                                   'numClasses': self._model_base.stages[-1].numClasses},
                             index=[0]).to_json(orient='split')
        elif isinstance(self._model_base.stages[-1], GeneralizedLinearRegressionModel):
            summary = self._model_base.stages[-1].summary
            result_dataframe = DataFrame(data={'aic': summary.aic,
                                   'intercept': str(self._model_base.stages[-1].intercept),
                                   'degreesOfFreedom': summary.degreesOfFreedom,
                                   'numInstances': summary.numInstances,
                                   'rank': summary.rank,
                                   'dispersion': summary.dispersion,
                                   'nullDeviance': summary.nullDeviance,
                                   'residuals': summary.residuals,
                                   'numFeatures': self._model_base.stages[-1].numFeatures},
                             index=[0]).to_json(orient='split')
        elif isinstance(self._model_base.stages[-1], LinearRegressionModel):
            summary = self._model_base.stages[-1].summary
            result_dataframe = DataFrame(data={'coefifients': str(self._model_base.stages[-1].coefficients),
                                   'degreesOfFreedom': summary.degreesOfFreedom,
                                   'numInstances': summary.numInstances,
                                   'totalIterations': summary.totalIterations,
                                   'devianceResiduals': str(summary.devianceResiduals),
                                   'explainedVariance': summary.explainedVariance,
                                   'numFeatures': self._model_base.stages[-1].numFeatures},
                             index=[0]).to_json(orient='split')
        elif isinstance(self._model_base.stages[-1], LinearSVCModel):
            result_dataframe = DataFrame(data={'coefifients': str(self._model_base.stages[-1].coefficients),
                                   'intercept': self._model_base.stages[-1].intercept,
                                   'numClasses': self._model_base.stages[-1].numClasses,
                                   'numFeatures': self._model_base.stages[-1].numFeatures},
                             index=[0]).to_json(orient='split')
        elif isinstance(self._model_base.stages[-1], LogisticRegressionModel):
            try:
                summary = self._model_base.stages[-1].summary
                result_dataframe = DataFrame(data={'coefifients': str(self._model_base.stages[-1].coefficients),
                                       'intercept': self._model_base.stages[-1].intercept,
                                       'totalIterations': summary.totalIterations,
                                       'roc': summary.roc.toPandas().to_json(orient='split'),
                                       'pr': summary.pr.toPandas().to_json(orient='split')},
                                 index=[0]).to_json(orient='split')
            except RuntimeError:
                result_dataframe = DataFrame(data={'coefifientsMatrix': str(self._model_base.stages[-1].coefficientMatrix),
                                       'interceptVector': str(self._model_base.stages[-1].interceptVector)},
                                 index=[0]).to_json(orient='split')
        elif isinstance(self._model_base.stages[-1], BisectingKMeansModel) or \
                isinstance(self._model_base.stages[-1], KMeansModel):
            summary = self._model_base.stages[-1].summary
            result_dataframe = DataFrame(data={'clusterCenters': str(self._model_base.stages[-1].clusterCenters()),
                                   'clusterSizes': str(summary.clusterSizes),
                                   'k': summary.k},
                             index=[0]).to_json(orient='split')

        # Change 27/01/2018 sprint 6
        if result_dataframe is not None:
            return json.loads(result_dataframe, object_pairs_hook=OrderedDict)
        else:
            return None

    ## Generate variable importance metrics
    # @param self object pointer
    # @param column_chain list of columns mapping features col
    # @return OrderedDict() for variable importance Key=column name
    def _generate_importance_variables(self, column_chain):
        var_importance = OrderedDict()
        for columns in column_chain:
            try:
                var_importance[columns] = self._model_base.stages[-1].featureImportances[column_chain.index(columns)]
            except AttributeError:
                var_importance[columns] = None
        return var_importance

    ## Generate model summary metrics
    # @param self object pointer
    # @return json_pandas_dataframe structure orient=split
    def _generate_model_metrics(self):
        if isinstance(self._model_base.stages[-1], LogisticRegressionModel) or \
                isinstance(self._model_base.stages[-1], LinearRegressionModel):
            try:
                summary = self._model_base.stages[-1].summary
                return json.loads(DataFrame(summary.objectiveHistory, columns=['Metrics']).to_json(orient='split'),
                                  object_pairs_hook=OrderedDict)
            except RuntimeError:
                return None
        elif isinstance(self._model_base.stages[-1], NaiveBayesModel):
            metrics = OrderedDict()
            metrics['pi'] = json.loads(DataFrame(self._model_base.stages[-1].pi.values).to_json(orient='split'),
                                       object_pairs_hook=OrderedDict)
            metrics['theta'] = json.loads(DataFrame(self._model_base.stages[-1].theta.values).to_json(orient='split'),
                                          object_pairs_hook=OrderedDict)

            return metrics
        return None

    ## Generate accuracy metrics for model
    #for regression assume tolerance on results equivalent to 2*tolerance % over (max - min) values
    # on dataframe objective's column
    # @param self object pointer
    # @param dataframe prediction sparkFrame
    # @param tolerance (optional) default value 0.0. Only for regression
    # @return float accuracy of model
    def _accuracy(self, objective, dataframe, tolerance=0.0):

        fmin = eval("lambda x: x - " + str(tolerance / 2))
        fmax = eval("lambda x: x + " + str(tolerance / 2))

        resultado_train = dataframe.select("prediction", objective)

        accuracy = resultado_train.filter(resultado_train.prediction >= fmin(resultado_train[objective])) \
                                  .filter(resultado_train.prediction <= fmax(resultado_train[objective])).count() \
                                  / float(resultado_train.count())

        self._logging.log_exec(self.analysis_id, self._spark_session.sparkContext.applicationId, self._labels["tolerance"],
                               str(tolerance))
        return accuracy

    ## Generate accuracy metrics for model
    #for regression assume tolerance on results equivalent to 2*tolerance % over (max - min) values
    # on dataframe objective's column
    # @param self object pointer
    # @param objective objective column if apply
    # @param dataframe normalized sparkFrame
    # @param tolerance (optional) default value 0.0. Only for regression
    # @return float accuracy of model, prediction_dataframe
    def _predict_accuracy(self, objective, dataframe, tolerance=0.0):
        accuracy = -1.0
        #bug SPARK-14948
        #prediccion = dataframe.withColumn('prediction', self._model_base.transform(dataframe).prediction)
        prediccion = self._model_base.transform(dataframe)
        columns = prediccion.columns
        if objective in columns:
            accuracy = self._accuracy(objective=objective, dataframe=prediccion, tolerance=tolerance)
        return accuracy, prediccion

    ## Generate detected anomalies on dataframe
    # @param self object pointer
    # @param dataframe normalized sparkFrame
    # @param objective objective column if apply
    # @return OrderedDict with anomalies
    def _predict_clustering(self, dataframe, objective=None):
        return self._predict_accuracy(objective=objective, dataframe=dataframe)

    ## Generate model full values parameters for execution analysis
    # @param self object pointer
    # @para, modeldef modeldef instance
    # @return (status (success 0, error 1) ,OrderedDict())
    def _generate_params(self, modeldef):
        """
        Generate model params for this model.
        :return (status (success 0, error 1) , OrderedDict(full_stack_parameters))
        """
        full_stack_params = OrderedDict()
        for key, item in modeldef.extractParamMap().items():
            full_stack_params[str(key)[str(key).find('__') + 2:]] = item
        return 0, full_stack_params

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
            self._logging.log_error('gDayF', self._spark_session.sparkContext.applicationId(), self._labels["ar_error"])
            return ('Necesario cargar un modelo valid o ar.json valido')
        try:
            return struct_ar['metrics'][source][metric]
        except KeyError:
            return 'Not Found'

    ## Method to execute normalizations base on params
    # @param self object pointer
    # @param dataframe  pandas dataframe
    # @param base_ns NormalizationMetadata orderedDict() compatible
    # @param model_id base model identificator
    # @param filtering STANDARDIZE if standardize filtering rules need to be applied
    # or DROP drop_columns filtering rules need to be applied
    # @param exist_objective True if exist False if not
    # @return (Dataframe, DFMetadata, Hash_value, True/False, base_ns)
    def execute_normalization(self, dataframe, base_ns, model_id, filtering='NONE', exist_objective=True):
        if base_ns is not None:
            data_norm = dataframe.copy(deep=True)
            self._logging.log_exec(self.analysis_id,
                                   self._spark_session.sparkContext.applicationId, self._labels["exec_norm"], str(base_ns))
            normalizer = Normalizer()
            if not exist_objective:
                base_ns = normalizer.filter_objective_base(normalizemd=base_ns)
            if filtering == 'STANDARDIZE':
                base_ns = normalizer.filter_standardize(normalizemd=base_ns, model_id=model_id)
            elif filtering == 'DROP':
                base_ns = normalizer.filter_drop_missing(normalizemd=base_ns)
            data_norm = normalizer.normalizeDataFrame(data_norm, base_ns)
            del normalizer
            df_metadata = DFMetada()
            df_metadata.getDataFrameMetadata(dataframe=data_norm, typedf='pandas')
            df_metadata_hash_value = md5(json.dumps(df_metadata).encode('utf-8')).hexdigest()
            return data_norm, df_metadata, df_metadata_hash_value, True, base_ns
        else:
            df_metadata = DFMetada()
            df_metadata.getDataFrameMetadata(dataframe=dataframe, typedf='pandas')
            df_metadata_hash_value = md5(json.dumps(df_metadata).encode('utf-8')).hexdigest()
            return dataframe, df_metadata, df_metadata_hash_value, False, base_ns

            #base_ns = json.load(normalization, object_pairs_hook=NormalizationSet)

    ## Method to generate special normalizations for Naive non negative work restrictions
    # @param self object pointer
    # @param dataframe  pandas dataframe
    # @return (base_ns)
    def define_special_spark_naive_norm(self, df_metadata):
        self._logging.log_exec(self.analysis_id,
                               self._spark_session.sparkContext.applicationId, self._labels["def_naive_norm"])
        normalizer = Normalizer()
        aux_ns = normalizer.define_special_spark_naive_norm(dataframe_metadata=df_metadata)
        del normalizer
        return aux_ns


    ## Main method to execute sets of analysis and normalizations base on params
    # @param self object pointer
    # @param analysis_id String (Analysis identificator)
    # @param training_pframe pandas.DataFrame
    # @param base_ar ar_template.json
    # @param **kwargs extra arguments
    # @return (String, ArMetadata) equivalent to (analysis_id, analysis_results)
    def order_training(self, analysis_id, training_pframe, base_ar, **kwargs):
        assert isinstance(analysis_id, str)
        assert isinstance(training_pframe, DataFrame)
        assert isinstance(base_ar, ArMetadata)

        filtering = 'NONE'
        user = 'guest'

        for pname, pvalue in kwargs.items():
            if pname == 'filtering':
                assert isinstance(pvalue, str)
                filtering = pvalue
            if pname == 'user':
                user = str(pvalue)



        # python train parameters effective
        self.analysis_id = analysis_id
        supervised = True
        tolerance = 0.0
        objective_column = base_ar['objective_column']
        if objective_column is None:
           supervised = False

        valid_frame = None
        test_frame = None


        if "test_frame" in kwargs.keys():
            test_frame = kwargs['test_frame']
        else:
            test_frame = None

        base_ns = get_model_ns(base_ar)
        modelid = base_ar['model_parameters']['spark']['model']
        artype = base_ar['model_parameters']['spark']['types'][0]["type"]
        self._logging.log_info(analysis_id,
                               self._spark_session.sparkContext.applicationId,
                               self._labels["st_analysis"], modelid)

        assert isinstance(base_ns, list) or base_ns is None
        # Applying Normalizations
        data_initial = DFMetada()
        data_initial.getDataFrameMetadata(dataframe=training_pframe, typedf='pandas')
        training_pframe, data_normalized, train_hash_value, norm_executed, base_ns = \
            self.execute_normalization(dataframe=training_pframe, base_ns=base_ns, model_id=modelid,
                                       filtering=filtering, exist_objective=True)

        if modelid == 'NaiveBayes' and artype == 'multinomial':
            training_pframe, data_normalized, train_hash_value, aux_norm_executed, aux_norm = \
                self.execute_normalization(dataframe=training_pframe,
                                           base_ns=self.define_special_spark_naive_norm(data_normalized),
                                           model_id=modelid)
            if aux_norm is not None:
                base_ns.extend(aux_norm)
                norm_executed = norm_executed | aux_norm_executed

        if base_ar['round'] == 1:
            aux_ns = Normalizer().define_ignored_columns(data_normalized, objective_column)
            if aux_ns is not None:
                base_ns.extend(aux_ns)

        df_metadata = data_initial
        if not norm_executed:
            data_normalized = None
            try:
                self._logging.log_info(analysis_id,
                                       self._spark_session.sparkContext.applicationId,
                                       self._labels["cor_struct"],
                                       str(data_initial['correlation'][objective_column]))
            except KeyError:
                self._logging.log_exec(analysis_id,
                                       self._spark_session.sparkContext.applicationId,
                                       self._labels["cor_struct"],
                                       str(data_initial['correlation']))
        else:
            df_metadata = data_normalized
            base_ar['normalizations_set'] = base_ns
            try:
                self._logging.log_info(analysis_id,
                                       self._spark_session.sparkContext.applicationId,
                                       self._labels["cor_struct"],
                                       str(data_normalized['correlation'][objective_column]))
            except KeyError:
                self._logging.log_exec(analysis_id,
                                       self._spark_session.sparkContext.applicationId,
                                       self._labels["cor_struct"],
                                       str(data_initial['correlation']))
            if test_frame is not None:
                test_frame, _, test_hash_value, _, _ = self.execute_normalization(dataframe=test_frame, base_ns=base_ns,
                                                                    model_id=modelid, filtering=filtering,
                                                                    exist_objective=True)

        training_frame = self._spark_session.createDataFrame(training_pframe).cache()
        self._logging.log_info(analysis_id,
                               self._spark_session.sparkContext.applicationId,
                               self._labels["parsing_to_spark"],
                               'training_frame(' + str(training_frame.count()) + ')')

        if "test_frame" in kwargs.keys():
            test_frame = self._spark_session.createDataFrame(test_frame).cache()
            self._logging.log_info(analysis_id, self._spark_session.sparkContext.applicationId,
                                   self._labels["parsing_to_spark"],
                                   'test_frame (' + str(test_frame.count()) + ')')

        if supervised and artype == 'regression':
            # Initializing base structures
            self._logging.log_info(analysis_id,
                                   self._spark_session.sparkContext.applicationId, self._labels["objective"],
                                   objective_column + ' - ' + self._get_dtype(training_frame.dtypes, objective_column))

            tolerance = get_tolerance(df_metadata['columns'], objective_column, self._tolerance)

        # Generating base_path
        self._logging.log_info(analysis_id,
                               self._spark_session.sparkContext.applicationId,
                               self._labels["action_type"], base_ar['type'])
        base_path = self.generate_base_path(base_ar, base_ar['type'])

        final_ar_model = copy.deepcopy(base_ar)
        final_ar_model['status'] = self._labels['failed_op']
        final_ar_model['model_parameters']['spark']['id'] = self._spark_session.version
        model_timestamp = str(time.time())
        final_ar_model['data_initial'] = data_initial
        final_ar_model['data_normalized'] = data_normalized

        model_id = modelid + '_' + model_timestamp
        self._logging.log_info(analysis_id, self._spark_session.sparkContext.applicationId,
                               self._labels["model_id"],
                               model_id)

        analysis_type = base_ar['model_parameters']['spark']['types'][0]['type']
        self._logging.log_info(analysis_id, self._spark_session.sparkContext.applicationId, self._labels["amode"],
                               base_ar['model_parameters']['spark']['types'][0]['type'])

        '''Generate commands pipeline : model and model.train()'''
        invalid_types = ['string']
        transformation_chain = list()
        column_chain = list()
        norm = Normalizer()
        ignored_columns = norm.ignored_columns(base_ns)
        decoder = None
        for element in training_frame.dtypes:
            if element[0] not in ignored_columns:
                if element[1] in invalid_types or (modelid == 'NaiveBayes' and artype == 'binomial'):
                    transformation_chain.append(StringIndexer() \
                                                .setInputCol(element[0]) \
                                                .setOutputCol(element[0] + '_to_index')
                                                .setHandleInvalid("keep"))
                    column_rename = element[0] + '_to_index'
                    if element[0] != objective_column:
                        transformation_chain.append(OneHotEncoder() \
                                                    .setInputCol(element[0] + '_to_index') \
                                                    .setOutputCol(element[0] + '_to_onehot'))
                        column_rename = element[0] + '_to_onehot'
                    else:
                        objective_column = column_rename
                        decoder = len(transformation_chain) - 1
                    column_chain.append(column_rename)
                else:
                    column_chain.append(element[0])
        del norm
        ''' Packaging Features '''

        try:
            column_chain.remove(objective_column)
            # Remove ignored_columns
            for column in ignored_columns:
                column_chain.remove(column)

        except ValueError:
            pass
        transformation_chain.append(VectorAssembler().setInputCols(column_chain).setOutputCol('features'))

        #Only for trace issues
        trc_pipeline = Pipeline(stages=transformation_chain.copy())
        ''' Compose Model'''
        model_command = list()
        model_command.append(modelid)
        model_command.append("(")
        model_command.append("featuresCol=\'features\'")

        if supervised:
            model_command.append(", labelCol=\'%s\'" % objective_column)

        generate_commands_parameters(base_ar['model_parameters']['spark'], model_command)

        model_command.append(")")
        model_command = ''.join(model_command)
        #print('TRC:' +  model_command)

        modeldef = eval(model_command)
        self._logging.log_exec(analysis_id, self._spark_session.sparkContext.applicationId,
                               self._labels["gmodel"], model_command)

        transformation_chain.append(modeldef)
        pipeline = Pipeline(stages=transformation_chain)
        grid = ParamGridBuilder().build()
        antype = base_ar['model_parameters']['spark']['types'][0]['type']
        aborted = False
        try:
            if supervised:
                if training_pframe.count(axis=0).all() <=  \
                        self._config['frameworks']['spark']['conf']['validation_frame_threshold']:

                    model = CrossValidator(estimator=pipeline,
                                           estimatorParamMaps=grid,
                                           evaluator=self._get_evaluator(analysis_type=antype,
                                                                         objective_column=objective_column),
                                           numFolds=self._config['frameworks']['spark']['conf']['nfolds'],
                                           seed=int(base_ar['timestamp']))
                else:
                    model = TrainValidationSplit(estimator=pipeline,
                                                 estimatorParamMaps=grid,
                                                 evaluator=self._get_evaluator(analysis_type=antype,
                                                                               objective_column=objective_column),
                                                 tranRation=self._config['frameworks']['spark']['conf']['validation_frame_ratio'],
                                                 seed=int(base_ar['timestamp']))
                start = time.time()

                trc_dataframe = trc_pipeline.fit(training_frame).transform(training_frame)
                self._logging.log_info(analysis_id,
                                       self._spark_session.sparkContext.applicationId,
                                       self._labels["trc:label_cardinality"],
                                       "( " + objective_column + "," +
                                       str(trc_dataframe.select(objective_column).distinct().count()) +
                                       " )")
                self._model_base = model.fit(training_frame).bestModel
            else:
                model = pipeline
                start = time.time()
                self._model_base = model.fit(training_frame)
                final_ar_model['status'] = self._labels["success_op"]

            # Generating aditional model parameters Model_ID
            final_ar_model['execution_seconds'] = time.time() - start
            final_ar_model['model_parameters']['spark']['parameters']['model_id'] = ParameterMetadata()
            final_ar_model['model_parameters']['spark']['parameters']['model_id'].set_value(value=model_id,
                                                                                            seleccionable=False,
                                                                                            type="str")
            # Filling whole json ar.json
            final_ar_model['ignored_parameters'], \
                final_ar_model['full_parameters_stack'] = self._generate_params(modeldef=modeldef)


            self._logging.log_info(analysis_id,
                                   self._spark_session.sparkContext.applicationId, self._labels["tmodel"],
                                   model_id)
            self._logging.log_info(analysis_id,
                                   self._spark_session.sparkContext.applicationId,
                                   self._labels["exec_time"],
                                   str(final_ar_model['execution_seconds']))


            # Generating execution metrics
            final_ar_model['metrics']['execution'] = ExecutionMetricCollection()

            self._logging.log_info(analysis_id,
                                   self._spark_session.sparkContext.applicationId, self._labels["gexec_metric"],
                                   model_id)

            prediction_train = self._model_base.transform(training_frame)

            final_ar_model['metrics']['execution']['train'] = \
                self._generate_execution_metrics(dataframe=prediction_train,
                                                 antype=analysis_type,
                                                 objective_column=objective_column)
            if test_frame is not None:
                prediction_test = self._model_base.transform(test_frame)

                final_ar_model['metrics']['execution']['test'] = \
                    self._generate_execution_metrics(dataframe=prediction_test,
                                                     antype=analysis_type,
                                                     objective_column=objective_column)

            final_ar_model['metrics']['execution']['predict'] = OrderedDict()
            final_ar_model['metrics']['execution']['predict']['decoder'] = decoder

            final_ar_model['metrics']['accuracy'] = OrderedDict()
            final_ar_model['metrics']['accuracy'] = OrderedDict()

            if supervised:
                final_ar_model['metrics']['accuracy']['train'] = \
                    self._accuracy(objective=objective_column, dataframe=prediction_train, tolerance=tolerance)
                self._logging.log_exec(analysis_id,
                                       self._spark_session.sparkContext.applicationId, self._labels["model_tacc"],
                                       model_id + ' - ' + str(final_ar_model['metrics']['accuracy']['train']))
                final_ar_model['tolerance'] = tolerance
            else:
                final_ar_model['metrics']['accuracy']['train'] = 0.0

            if test_frame is not None:
                prediction_test = self._model_base.transform(test_frame)
                if supervised:
                    final_ar_model['metrics']['accuracy']['test'] = \
                        self._accuracy(objective=objective_column, dataframe=prediction_test,  tolerance=tolerance)

                    train_balance = self._config['frameworks']['spark']['conf']['train_balance_metric']
                    test_balance = 1 - train_balance
                    final_ar_model['metrics']['accuracy']['combined'] = \
                        (final_ar_model['metrics']['accuracy']['train']*train_balance +
                         final_ar_model['metrics']['accuracy']['test']*test_balance)

                    self._logging.log_exec(analysis_id, self._spark_session.sparkContext.applicationId,
                                           self._labels["model_pacc"],
                                           model_id + ' - ' + str(final_ar_model['metrics']['accuracy']['test']))

                    self._logging.log_exec(analysis_id, self._spark_session.sparkContext.applicationId,
                                           self._labels["model_cacc"],
                                           model_id + ' - ' + str(final_ar_model['metrics']['accuracy']['combined']))
                else:

                    final_ar_model['metrics']['accuracy']['test'] = 0.0
                    final_ar_model['metrics']['accuracy']['combined'] = 0.0

            # Generating model metrics
            final_ar_model['metrics']['model'] = self._generate_model_metrics()
            self._logging.log_info(analysis_id,
                                   self._spark_session.sparkContext.applicationId,
                                   self._labels["gmodel_metric"], model_id)

            # Generating Variable importance
            final_ar_model['metrics']['var_importance'] = self._generate_importance_variables(column_chain=column_chain)
            self._logging.log_info(analysis_id,
                                   self._spark_session.sparkContext.applicationId,
                                   self._labels["gvar_metric"], model_id)

            # Generating scoring_history
            final_ar_model['metrics']['scoring'] = self._generate_scoring_history()
            self._logging.log_info(analysis_id,
                                   self._spark_session.sparkContext.applicationId,
                                   self._labels["gsco_metric"], model_id)

            final_ar_model['status'] = self._labels['success_op']

        except Exception as execution_error:
            for handler in self._logging.logger.handlers:
                handler.flush()
            # Generating aditional model parameters Model_ID
            final_ar_model['execution_seconds'] = time.time() - start
            aborted = True
            final_ar_model['model_parameters']['spark']['parameters']['model_id'] = ParameterMetadata()
            final_ar_model['model_parameters']['spark']['parameters']['model_id'].set_value(value=model_id,
                                                                                            seleccionable=False,
                                                                                            type="str")
            self._logging.log_info(analysis_id,
                                   self._spark_session.sparkContext.applicationId, self._labels["abort_data_nc"],
                                   model_id)
            # Filling whole json ar.json
            final_ar_model['ignored_parameters'], \
                final_ar_model['full_parameters_stack'] = self._generate_params(modeldef=modeldef)
            # Filling whole json ar.json
            final_ar_model['ignored_parameters'], \
                final_ar_model['full_parameters_stack'] = self._generate_params(modeldef=modeldef)

            final_ar_model['status'] = self._labels["failed_op"]
            self._logging.log_critical(analysis_id,
                                       self._spark_session.sparkContext.applicationId,
                                       self._labels["abort"],
                                       repr(execution_error))
            final_ar_model['metrics'] = OrderedDict()
            final_ar_model['metrics']['accuracy'] = OrderedDict()
            final_ar_model['metrics']['accuracy']['train'] = 0.0
            final_ar_model['metrics']['accuracy']['test'] = 0.0
            final_ar_model['metrics']['accuracy']['combined'] = 0.0
            final_ar_model['metrics']['execution'] = OrderedDict()
            final_ar_model['metrics']['execution']['train'] = OrderedDict()
            final_ar_model['metrics']['execution']['train']['RMSE'] = 1e+16
            final_ar_model['metrics']['execution']['train']['tot_withinss'] = 1e+16
            final_ar_model['metrics']['execution']['train']['betweenss'] = 1e+16
            final_ar_model['metrics']['execution']['test'] = OrderedDict()
            final_ar_model['metrics']['execution']['test']['RMSE'] = 1e+16
            final_ar_model['metrics']['execution']['test']['tot_withinss'] = 1e+16
            final_ar_model['metrics']['execution']['test']['betweenss'] = 1e+16

        finally:
            generate_json_path(final_ar_model, user)
            self._persistence.store_json(storage_json=final_ar_model['json_path'], ar_json=final_ar_model)

            self._logging.log_info(analysis_id,
                                   self._spark_session.sparkContext.applicationId,
                                   self._labels["model_stored"], model_id)
            self._logging.log_info(analysis_id,
                                   self._spark_session.sparkContext.applicationId,
                                   self._labels["end"], model_id)
            if not aborted:
                self.store_model(final_ar_model, user=user)

            for handler in self._logging.logger.handlers:
                handler.flush()
        return analysis_id, final_ar_model

    ## Method to save model to persistence layer from armetadata
    # @param armetadata structure to be stored
    # @param user user_id
    # return saved_model (True/False)
    def store_model(self, armetadata, user='guest'):
        saved_model = False

        fw = get_model_fw(armetadata)
        model_id = armetadata['model_parameters'][fw]['parameters']['model_id']['value']
        analysis_id = armetadata['model_id']

        #Updating status
        armetadata['status'] = self._labels["success_op"]
        # Generating load_path
        load_storage = StorageMetadata()
        for each_storage_type in load_storage.get_load_path():
            source_data = list()
            primary_path = self._config['storage'][each_storage_type['type']]['value']
            source_data.append(primary_path)
            source_data.append('/')
            source_data.append(user)
            source_data.append('/')
            source_data.append(armetadata['model_id'])
            source_data.append('/')
            source_data.append(fw)
            source_data.append('/')
            source_data.append(armetadata['type'])
            source_data.append('/')
            source_data.append(str(armetadata['timestamp']))
            source_data.append('/')

            load_path = ''.join(source_data) + each_storage_type['value']+'/'
            self._persistence.mkdir(type=each_storage_type['type'], path=load_path,
                                    grants=self._config['storage']['grants'])
            if each_storage_type['type'] == 'hdfs':
                load_path = self._config['storage'][each_storage_type['type']]['uri'] + load_path

            self._model_base.write().overwrite().save(path=load_path + model_id + self._get_ext())

            load_storage.append(value=load_path + model_id + self._get_ext(),
                                fstype=each_storage_type['type'], hash_type=each_storage_type['hash_type'])
            saved_model = True

        armetadata['load_path'] = load_storage

        self._logging.log_exec(analysis_id,
                               self._spark_session.sparkContext.applicationId, self._labels["msaved"],
                               model_id)

        self._persistence.store_json(storage_json=armetadata['json_path'], ar_json=armetadata)
        self._logging.log_info(analysis_id,
                               self._spark_session.sparkContext.applicationId,
                               self._labels["model_stored"], model_id)

        return saved_model

    ## Method to load  model from persistence layer by armetadata
    # @param armetadata structure to be stored
    # return armetadata if model loaded successfully or None if not loaded
    def load_model(self, armetadata):
        from_disk = False

        fw = get_model_fw(armetadata)
        model_id = armetadata['model_parameters'][fw]['parameters']['model_id']['value']
        analysis_id = armetadata['model_id']

        load_fail, from_disk = self._get_model(base_ar=armetadata, base_model_id=model_id, remove_model=from_disk)
        if load_fail:
            return None
        else:
            return armetadata

    ## Main method to execute predictions over traning models
    # Take the ar.json for and execute predictions including its metrics a storage paths
    # @param self object pointer
    # @param predict_frame pandas.DataFrame
    # @param base_ar ArMetadata
    # or compatible tuple (OrderedDict(), OrderedDict())
    # @param **kwargs extra arguments
    # @return (String, [ArMetadata]) equivalent to (analysis_id, List[analysis_results])
    def predict(self, predict_frame, base_ar, **kwargs):

        user = 'guest'
        for pname, pvalue in kwargs.items():
            if pname == 'user':
                user = str(pvalue)

        remove_model = False
        model_timestamp = str(time.time())
        self.analysis_id = base_ar['model_id']
        analysis_id = self.analysis_id
        base_model_id = base_ar['model_parameters']['spark']['parameters']['model_id']['value'] + self._get_ext()
        model_id = base_model_id + '_' + model_timestamp
        antype = base_ar['model_parameters']['spark']['types'][0]['type']

        modelid = base_ar['model_parameters']['spark']['model']
        base_ns = get_model_ns(base_ar)

        #Checking file source versus hash_value
        load_fails, remove_model = self._get_model(base_ar, base_model_id, remove_model)

        if load_fails or self._model_base is None:
            self._logging.log_critical(self.analysis_id, self._spark_session.sparkContext.applicationId,
                                       self._labels["no_models"], base_model_id)
            base_ar['status'] = self._labels['failed_op']  # Default Failed Operation Code
            return None

        objective_column = base_ar['objective_column']

        exist_objective = True
        if objective_column is None:
            exist_objective = False
        if exist_objective:
            self._logging.log_info(analysis_id, self._spark_session.sparkContext.applicationId, self._labels["objective"],
                                   objective_column)
            # Recovering tolerance
            tolerance = base_ar['tolerance']

        data_initial = DFMetada()
        data_initial.getDataFrameMetadata(dataframe=predict_frame, typedf='pandas')
        base_ar['data_initial'] = data_initial

        if objective_column in list(predict_frame.columns.values):
            try:
                self._logging.log_info(analysis_id, self._spark_session.sparkContext.applicationId,
                                       self._labels["cor_struct"], str(data_initial['correlation'][objective_column]))
            except KeyError:
                self._logging.log_exec(analysis_id, self._spark_session.sparkContext.applicationId,
                                       self._labels["cor_struct"], str(data_initial['correlation']))
            npredict_frame, data_normalized, _, norm_executed, _ = self.execute_normalization(dataframe=predict_frame,
                                                                                              base_ns=base_ns,
                                                                                              model_id=modelid,
                                                                                              filtering='NONE',
                                                                                              exist_objective=True)

        else:
            npredict_frame, data_normalized, _, norm_executed, _ = self.execute_normalization(dataframe=predict_frame,
                                                                                              base_ns=base_ns,
                                                                                              model_id=modelid,
                                                                                              filtering='DROP',
                                                                                              exist_objective=False)

        if not norm_executed:
            self._logging.log_exec(analysis_id, self._spark_session.sparkContext.applicationId, self._labels["exec_norm"],
                                   'No Normalizations Required')
        else:
            # Transforming original dataframe to sparkFrame
            '''predict_frame = self._spark_session.createDataFrame(predict_frame).cache()
            self._logging.log_info(analysis_id, self._spark_session.sparkContext.applicationId, self._labels["parsing_to_spark"],
                                   'test_frame (' + str(predict_frame.count()) + ')')'''

            base_ar['data_normalized'] = data_normalized
            if objective_column in list(npredict_frame.columns.values):
                try:
                    self._logging.log_exec(analysis_id, self._spark_session.sparkContext.applicationId, self._labels["cor_struct"],
                                           str(data_normalized['correlation'][objective_column]))
                except KeyError:
                    self._logging.log_exec(analysis_id, self._spark_session.sparkContext.applicationId, self._labels["no_cor_struct"],
                                           str(data_normalized['correlation']))

        #Transforming to sparkFrame
        npredict_frame = self._spark_session.createDataFrame(npredict_frame).cache()
        self._logging.log_info(analysis_id, self._spark_session.sparkContext.applicationId, self._labels["parsing_to_spark"],
                               'test_frame (' + str(npredict_frame.count()) + ')')

        base_ar['type'] = 'predict'
        self._logging.log_info(self.analysis_id, self._spark_session.sparkContext.applicationId,
                               self._labels["action_type"], base_ar['type'])

        base_ar['timestamp'] = model_timestamp

        self._logging.log_info(analysis_id, self._spark_session.sparkContext.applicationId,
                               self._labels['st_predict_model'],
                               base_model_id)
        if base_ar['metrics']['execution']['predict']['decoder'] is not None:
            decoder = self._model_base.stages[base_ar['metrics']['execution']['predict']['decoder']]
        else:
            decoder = None

        if objective_column in npredict_frame.columns:
            for element in npredict_frame.dtypes:
                if element[0] == objective_column:
                    if element[1] == 'string':
                        objective_column = objective_column + '_to_index'
                        objective_type = 'double'
                    else:
                        objective_type = element[1]
        else:
            objective_type = None

        start = time.time()

        if exist_objective:
            accuracy, prediction_dataframe = self._predict_accuracy(objective_column, npredict_frame,
                                                                        tolerance=tolerance)

            base_ar['execution_seconds'] = time.time() - start
            base_ar['tolerance'] = tolerance

            #prediction_dataframe = prediction_dataframe.toPandas()
        else:
            if antype == 'clustering':
                if norm_executed:
                    accuracy, prediction_dataframe = self._predict_clustering(npredict_frame)
                else:
                    accuracy, prediction_dataframe = self._predict_clustering(npredict_frame)

                base_ar['execution_seconds'] = time.time() - start
                #prediction_dataframe = prediction_dataframe.toPandas()


        if not exist_objective or objective_type is not None:
            self._logging.log_info(analysis_id, self._spark_session.sparkContext.applicationId,
                                   self._labels["gexec_metric"], model_id)

            base_ar['metrics']['execution'][base_ar['type']] = self._generate_execution_metrics( \
                                                                                        dataframe=prediction_dataframe,
                                                                                        objective_column=objective_column,
                                                                                        antype=antype)
        if objective_column in prediction_dataframe.columns:
            base_ar['metrics']['accuracy']['predict'] = accuracy
            self._logging.log_info(analysis_id,
                                   self._spark_session.sparkContext.applicationId, self._labels["model_pacc"],
                                   base_model_id + ' - ' + str(base_ar['metrics']['accuracy']['predict']))

        base_ar['status'] = self._labels['success_op']

        # writing ar.json file
        generate_json_path(base_ar, user)
        self._persistence.store_json(storage_json=base_ar['json_path'], ar_json=base_ar)
        self._logging.log_exec(self.analysis_id,
                               self._spark_session.sparkContext.applicationId,
                               self._labels["model_stored"], model_id)
        self._logging.log_info(self.analysis_id,
                               self._spark_session.sparkContext.applicationId,
                               self._labels["end"], model_id)
        for handler in self._logging.logger.handlers:
            handler.flush()

        if decoder is not None:
            labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                           labels=decoder.labels)
            prediction_dataframe = labelConverter.transform(prediction_dataframe).drop("prediction")
            prediction_dataframe = prediction_dataframe.withColumnRenamed("predictedLabel", "prediction")

        command = list()
        col_sep = False
        '''command.append("prediction_dataframe.select(")
        for element in columns:
            if not col_sep:
                command.append('\"' + element + '\"')
                col_sep = True
            else:
                command.append(',')
                command.append('\"' + element + '\"')
        command.append(', \"prediction\"')
        if antype in ['binomial', 'multinomial']:
            command.append(', \"probability\"')
        command.append(").toPandas()")'''

        command.append("prediction_dataframe.select(")
        command.append('\"prediction\"')
        if antype in ['binomial', 'multinomial']:
            command.append(', \"probability\"')
        command.append(").toPandas()")

        presults = eval("".join(command))
        prediction =predict_frame.copy()
        prediction['prediction'] = presults.loc[:,'prediction']
        if antype in ['binomial', 'multinomial']:
            prediction['probability'] = presults.loc[:,'probability']


        return prediction, base_ar

    ## Internal method to get an sparkmodel from server or file transparent to user
    # @param self Object pointer
    # @param base_ar armetadata to load from fs
    # @param base_model_id from searching on server memory objects
    # @param remove_model to indicate if has been loaded from memory or need to be removed at last
    # @return load_fails, remove_model operation status True/False, removed True/False
    def _get_model(self, base_ar, base_model_id, remove_model):
        load_fails = self._get_model_from_load_path(base_ar)
        remove_model = True
        return load_fails, remove_model

    ## Method to remove list of model from disk
    # @param arlist List of ArMetadata
    # @return remove_fails True/False
    @staticmethod
    def remove_models(arlist):
        remove_fails = False
        try:
            assert isinstance(arlist, list)
        except AssertionError:
            return remove_fails

        persistence = PersistenceHandler()
        for ar_metadata in arlist:
            try:
                assert isinstance(ar_metadata['load_path'], list)
            except AssertionError:
                return True

            _, ar_metadata['load_path'] = persistence.remove_file(load_path=ar_metadata['load_path'])

            if len(ar_metadata['load_path']) == 0:
                ar_metadata['load_path'] = None
            else:
                remove_fails = True

            persistence.store_json(storage_json=ar_metadata["json_path"], ar_json=ar_metadata)

        del persistence
        return remove_fails


## auxiliary function (procedure) to generate model and train chain paramters to execute models
# Modify model_command and train_command String to complete for eval()
# @param each_model object pointer
# @param model_command String with model command definition base structure
def generate_commands_parameters(each_model, model_command):
    for key, value in each_model['parameters'].items():
        if value['seleccionable']:
            if isinstance(value['value'], str):
                model_command.append(", %s=\'%s\'" % (key, value['value']))
            else:
                if value is not None:
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