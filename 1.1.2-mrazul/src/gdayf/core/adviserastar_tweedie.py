## @package gdayf.core.adviserastar_tweedie
# Define all objects, functions and structured related to manage and execute Smart analysis based on A* Algorithm
# and defined heuristic
# Main class AdviserAStarAvg. Lets us execute analysis, make recommendations over optimizing on selected algoritms

'''
Copyright (C) e2its - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 *
 * This file is part of gDayF project.
 *
 * Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2018
'''

from gdayf.conf.loadconfig import LoadConfig
from gdayf.common.utils import decode_ordered_dict_to_dataframe
from gdayf.core.adviserbase import Adviser
from gdayf.common.utils import get_model_fw
from gdayf.models.parametersmetadata import ParameterMetadata

## Class focused on execute A* based analysis on three modalities of working
# Fast: 1 level analysis over default parameters
# Normal: One A* analysis for all models based until max_deep with early_stopping
# Paranoiac: One A* algorithm per model analysis until max_deep without early stoping
class AdviserAStar(Adviser):

    ## Constructor
    # @param self object pointer
    # @param analysis_id main id traceability code
    # @param deep_impact A* max_deep
    # @param metric metrict for priorizing models ['train_accuracy', 'test_rmse', 'train_rmse', 'test_accuracy', 'combined_accuracy'] on train
    # @param dataframe_name dataframe_name or id
    # @param hash_dataframe MD5 hash value
    # @param workflow_id Workflow identifier
    # @param user_id user identifier
    def __init__(self, analysis_id, deep_impact=3, metric='train_accuracy', dataframe_name='', hash_dataframe='',
                 workflow_id='', user_id='guest'):
        super(AdviserAStar, self).__init__(analysis_id, deep_impact=deep_impact, metric=metric,
                                           dataframe_name=dataframe_name, hash_dataframe=hash_dataframe,
                                           workflow_id=workflow_id, user_id=user_id)

    ## Method mangaing the generation of possible optimized models
    # @param armetadata ArMetadata Model
    # @return list of possible optimized models to execute return None if nothing to do
    def optimize_models(self, armetadata):
        if get_model_fw(armetadata) == 'h2o':
            return self.optimize_models_h2o(armetadata)
        elif get_model_fw(armetadata) == 'spark':
            return self.optimize_models_spark(armetadata)
        else:
            return None

    ## Method manging generation of possible optimized H2O models
    # params: results for Handlers (gdayf.handlers)
    # @param armetadata ArMetadata Model
    # @return list of possible optimized models to execute return None if nothing to do
    def optimize_models_h2o(self, armetadata):
        model_list = list()
        model = armetadata['model_parameters'][get_model_fw(armetadata)]
        metric_value, _, objective = eval('self.get_' + self.metric + '(armetadata)')
        if get_model_fw(armetadata) == 'h2o' and metric_value != objective \
                and armetadata['status'] != self._labels['failed_op']:
            try:
                model_metric = decode_ordered_dict_to_dataframe(armetadata['metrics']['model'])
                if model['model'] not in ['H2ONaiveBayesEstimator']:
                    scoring_metric = decode_ordered_dict_to_dataframe(armetadata['metrics']['scoring'])
                config = LoadConfig().get_config()['optimizer']['AdviserStart_rules']['h2o']
                nfold_limit = config['nfold_limit']
                min_rows_limit = config['min_rows_limit']
                cols_breakdown = config['cols_breakdown']
                nfold_increment = config['nfold_increment']
                min_rows_increment = config['min_rows_increment']
                max_interactions_rows_breakdown = config['max_interactions_rows_breakdown']
                max_interactions_increment = config['max_interactions_increment']
                max_depth_increment = config['max_depth_increment']
                ntrees_increment = config['ntrees_increment']
                dpl_rcount_limit = config['dpl_rcount_limit']
                dpl_divisor = config['dpl_divisor']
                h_dropout_ratio = config['h_dropout_ratio']
                epochs_increment = config['epochs_increment']
                dpl_min_batch_size = config['dpl_min_batch_size']
                dpl_batch_reduced_divisor = config['dpl_batch_reduced_divisor']
                deeper_increment = config['deeper_increment']
                wider_increment = config['wider_increment']
                learning_conf = config['learning_conf']
                rho_conf = config['rho_conf']
                nv_laplace = config['nv_laplace']
                nv_min_prob = config['nv_min_prob']
                nv_min_sdev = config['nv_min_sdev']
                nv_improvement = config['nv_improvement']
                nv_divisor = config['nv_divisor']
                clustering_increment = config['clustering_increment']

                if model['model'] == 'H2OGradientBoostingEstimator':
                    if (self.deepness == 2) and model['types'][0]['type'] == 'regression':
                        for tweedie_power in [1.1, 1.5, 1.9]:
                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['h2o']
                            model_aux['parameters']['distribution']['value'] = 'tweedie'
                            model_aux['parameters']['tweedie_power'] = ParameterMetadata()
                            model_aux['parameters']['tweedie_power'].set_value(tweedie_power)
                            self.safe_append(model_list, new_armetadata)
                    if self.deepness == 2:
                        for learning in learning_conf:
                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['h2o']
                            model_aux['parameters']['learn_rate']['value'] = learning['learn']
                            model_aux['parameters']['learn_rate_annealing']['value'] = learning['improvement']
                            self.safe_append(model_list, new_armetadata)
                    if model_metric['number_of_trees'][0] >= model['parameters']['ntrees']['value']:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['ntrees']['value'] *= ntrees_increment
                        self.safe_append(model_list, new_armetadata)
                    if model_metric['max_depth'][0] >= model['parameters']['max_depth']['value']:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['max_depth']['value'] *= max_depth_increment
                        self.safe_append(model_list, new_armetadata)
                    if model['parameters']['nfolds']['value'] < nfold_limit:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['nfolds']['value'] += nfold_increment
                        self.safe_append(model_list, new_armetadata)
                    if model['parameters']['min_rows']['value'] > min_rows_limit:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['min_rows']['value'] = round(model_aux['parameters']['min_rows']['value']
                                                                             / min_rows_increment, 0)
                        self.safe_append(model_list, new_armetadata)

                elif model['model'] == 'H2OGeneralizedLinearEstimator':

                    if model_metric['number_of_iterations'][0] >= model['parameters']['max_iterations']['value']:

                        if self.deepness == 2:
                            max_iterations = model['parameters']['max_iterations']['value'] * \
                                             max(round(
                                                 armetadata['data_initial']['rowcount'] / max_interactions_rows_breakdown),
                                                 1)
                        else:
                            max_iterations = model['parameters']['max_iterations']['value'] * max_interactions_increment
                    else:
                        max_iterations = model['parameters']['max_iterations']['value']

                    if (self.deepness == 2) and model['types'][0]['type'] == 'regression':
                        for tweedie_power in [1.0, 1.5, 2.0, 2.5, 3.0]:
                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['h2o']
                            model_aux['parameters']['tweedie_variance_power']['value'] = tweedie_power
                            model_aux['parameters']['max_iterations']['value'] = max_iterations
                            self.safe_append(model_list, new_armetadata)
                    if self.deepness == 2:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['alpha']['value'] = 0.0
                        model_aux['parameters']['max_iterations']['value'] = max_iterations
                        self.safe_append(model_list, new_armetadata)
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['alpha']['value'] = 1.0
                        model_aux['parameters']['max_iterations']['value'] = max_iterations
                        self.safe_append(model_list, new_armetadata)
                        if armetadata['data_initial']['cols'] > cols_breakdown:
                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['h2o']
                            model_aux['parameters']['solver']['value'] = 'L_BFGS'
                            model_aux['parameters']['max_iterations']['value'] = max_iterations
                            self.safe_append(model_list, new_armetadata)
                    if self.deepness == 2:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['balance_classes']['value'] = \
                            not model_aux['parameters']['balance_classes']['value']
                        model_aux['parameters']['max_iterations']['value'] = max_iterations
                        self.safe_append(model_list, new_armetadata)
                    if model['parameters']['nfolds']['value'] < nfold_limit:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['nfolds']['value'] += nfold_increment
                        model_aux['parameters']['max_iterations']['value'] = max_iterations
                        self.safe_append(model_list, new_armetadata)

                elif model['model'] == 'H2ODeepLearningEstimator':

                    if scoring_metric.shape[0] == 0 or \
                            (scoring_metric['epochs'].max() >=
                                     model['parameters']['epochs']['value']):
                        epochs = model['parameters']['epochs']['value'] * epochs_increment
                    else:
                        epochs = model['parameters']['epochs']['value']

                    if self.deepness == 2:
                        if armetadata['data_initial']['rowcount'] > dpl_rcount_limit:
                            hidden = [round(armetadata['data_initial']['rowcount'] / (dpl_divisor * 0.5)),
                                      round(armetadata['data_initial']['rowcount'] / (dpl_divisor * self.deep_impact))]
                        else:
                            hidden = model['parameters']['hidden']['value']
                        drop_out = model['parameters']['hidden_dropout_ratios']['value'] = [h_dropout_ratio,
                                                                                            h_dropout_ratio]
                        for learning in rho_conf:
                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['h2o']
                            model_aux['parameters']['rho']['value'] = learning['learn']
                            model_aux['parameters']['epsilon']['value'] = learning['improvement']
                            model_aux['parameters']['hidden']['value'] = [hidden[0], hidden[1]]
                            model_aux['parameters']['hidden_dropout_ratios']['value'] = drop_out
                            model_aux['parameters']['epochs']['value'] = epochs
                            self.safe_append(model_list, new_armetadata)

                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['h2o']
                            model_aux['parameters']['rho']['value'] = learning['learn']
                            model_aux['parameters']['epsilon']['value'] = learning['improvement']
                            model_aux['parameters']['hidden']['value'] = [hidden[1], hidden[0]]
                            model_aux['parameters']['hidden_dropout_ratios']['value'] = drop_out
                            model_aux['parameters']['epochs']['value'] = epochs
                            self.safe_append(model_list, new_armetadata)

                    if (self.deepness == 3) and model['types'][0]['type'] == 'regression':
                        for tweedie_power in [1.1, 1.5, 1.9]:
                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['h2o']
                            model_aux['parameters']['distribution']['value'] = 'tweedie'
                            model_aux['parameters']['tweedie_power'] = ParameterMetadata()
                            model_aux['parameters']['tweedie_power'].set_value(tweedie_power)
                            model_aux['parameters']['activation']['value'] = 'tanh_with_dropout'
                            model_aux['parameters']['epochs']['value'] = epochs
                            self.safe_append(model_list, new_armetadata)

                    if self.deepness == 3 and not model['parameters']['sparse']['value']:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['sparse']['value'] = not model_aux['parameters']['sparse']['value']
                        model_aux['parameters']['epochs']['value'] = epochs
                        self.safe_append(model_list, new_armetadata)
                    '''Eliminado 19/09/2017
                    if self.deepness == 3 and model['parameters']['activation']['value'] == "rectifier_with_dropout":
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['activation']['value'] = 'tanh_with_dropout'
                        self.safe_append(model_list, new_armetadata)'''

                    if self.deepness == 3 and model['parameters']['initial_weight_distribution']['value'] == "normal":
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['initial_weight_distribution']['value'] = "uniform"
                        model_aux['parameters']['epochs']['value'] = epochs
                        self.safe_append(model_list, new_armetadata)
                    elif self.deepness == 3:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['initial_weight_distribution']['value'] = "normal"
                        model_aux['parameters']['epochs']['value'] = epochs
                        self.safe_append(model_list, new_armetadata)

                    if self.deepness > 2 and self.deepness <= self.deep_impact:
                        if len(armetadata['model_parameters']['h2o']['parameters']['hidden']['value']) < 4:
                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['h2o']
                            if model_aux['parameters']['hidden']['value'][0] > model_aux['parameters']['hidden']['value'][
                                1]:
                                model_aux['parameters']['hidden']['value'].insert(0,
                                                                                  round(model_aux['parameters']['hidden'][
                                                                                            'value'][0] * deeper_increment))
                                model_aux['parameters']['hidden_dropout_ratios']['value'].insert(0, h_dropout_ratio)
                            else:
                                model_aux['parameters']['hidden']['value'].append(
                                    round(model_aux['parameters']['hidden']['value'][-1] * deeper_increment))
                                model_aux['parameters']['hidden_dropout_ratios']['value'].append(h_dropout_ratio)

                            model_aux['parameters']['epochs']['value'] = epochs
                            self.safe_append(model_list, new_armetadata)

                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['h2o']

                            for iterador in range(0, len(model_aux['parameters']['hidden']['value'])):
                                model_aux['parameters']['hidden']['value'][iterador] = \
                                    int(round(model_aux['parameters']['hidden']['value'][iterador]) * wider_increment)

                            model_aux['parameters']['epochs']['value'] = epochs
                            self.safe_append(model_list, new_armetadata)

                    if model['parameters']['mini_batch_size']['value'] >= dpl_min_batch_size:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['mini_batch_size']['value'] = \
                            round(model_aux['parameters']['mini_batch_size']['value'] / dpl_batch_reduced_divisor)
                        model_aux['parameters']['epochs']['value'] = epochs
                        self.safe_append(model_list, new_armetadata)

                elif model['model'] == 'H2ORandomForestEstimator':
                    if model_metric['number_of_trees'][0] == model['parameters']['ntrees']['value']:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['ntrees']['value'] *= ntrees_increment
                        self.safe_append(model_list, new_armetadata)
                    if model_metric['max_depth'][0] == model['parameters']['max_depth']['value']:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['max_depth']['value'] *= max_depth_increment
                        self.safe_append(model_list, new_armetadata)
                    if model['parameters']['nfolds']['value'] < nfold_limit:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['nfolds']['value'] += nfold_increment
                        self.safe_append(model_list, new_armetadata)
                    if model['parameters']['mtries']['value'] not in [round(armetadata['data_initial']['cols'] / 2),
                                                                      round(armetadata['data_initial']['cols'] * 3 / 4)]:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['mtries']['value'] = round(armetadata['data_initial']['cols'] / 2)
                        self.safe_append(model_list, new_armetadata)
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['mtries']['value'] = round(armetadata['data_initial']['cols'] * 3 / 4)
                        self.safe_append(model_list, new_armetadata)
                    if model['parameters']['min_rows']['value'] > (min_rows_limit / 2):
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['min_rows']['value'] = round(model_aux['parameters']['min_rows']['value']
                                                                             / min_rows_increment, 0)
                        self.safe_append(model_list, new_armetadata)

                elif model['model'] == 'H2ONaiveBayesEstimator':
                    if self.deepness == 2:
                        for laplace in nv_laplace:
                            for min_prob in nv_min_prob:
                                for min_sdev in nv_min_sdev:
                                    new_armetadata = armetadata.copy_template()
                                    model_aux = new_armetadata['model_parameters']['h2o']
                                    model_aux['parameters']['laplace']['value'] = laplace
                                    model_aux['parameters']['min_prob']['value'] = min_prob
                                    model_aux['parameters']['min_sdev']['value'] = min_sdev
                                    self.safe_append(model_list, new_armetadata)
                    elif self.deepness >= 2:
                        if self.deepness == self.deep_impact:
                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['h2o']
                            model_aux['parameters']['balance_classes']['value'] = \
                                not model_aux['parameters']['balance_classes']['value']
                            self.safe_append(model_list, new_armetadata)
                        if model['parameters']['nfolds']['value'] < nfold_limit:
                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['h2o']
                            model_aux['parameters']['nfolds']['value'] += nfold_increment
                            self.safe_append(model_list, new_armetadata)

                        for laplace in ['improvement', 'decrement']:
                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['h2o']
                            if laplace == 'improvement':
                                model_aux['parameters']['laplace']['value'] = model_aux['parameters']['laplace'][
                                                                                  'value'] * (1 + nv_improvement)
                            else:
                                model_aux['parameters']['laplace']['value'] = model_aux['parameters']['laplace'][
                                                                                  'value'] * (1 - nv_divisor)
                            self.safe_append(model_list, new_armetadata)

                elif model['model'] == 'H2OAutoEncoderEstimator':
                    if scoring_metric.shape[0] == 0 or \
                            (scoring_metric['epochs'].max() >=
                                     model['parameters']['epochs']['value']):
                        epochs = model['parameters']['epochs']['value'] * epochs_increment
                    else:
                        epochs = model['parameters']['epochs']['value']

                    if self.deepness == 2:
                        for learning in rho_conf:
                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['h2o']
                            model_aux['parameters']['rho']['value'] = learning['learn']
                            model_aux['parameters']['epsilon']['value'] = learning['improvement']
                            model_aux['parameters']['epochs']['value'] = epochs
                            self.safe_append(model_list, new_armetadata)

                    if self.deepness == 3:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['sparse']['value'] = not model_aux['parameters']['sparse']['value']
                        model_aux['parameters']['epochs']['value'] = epochs
                        self.safe_append(model_list, new_armetadata)
                    if self.deepness > 1 and model['parameters']['activation']['value'] == "rectifier_with_dropout":
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['activation']['value'] = 'tanh_with_dropout'
                        model_aux['parameters']['epochs']['value'] = epochs
                        self.safe_append(model_list, new_armetadata)
                    if self.deepness == 3 and model['parameters']['initial_weight_distribution']['value'] == "normal":
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['initial_weight_distribution']['value'] = "uniform"
                        model_aux['parameters']['epochs']['value'] = epochs
                        self.safe_append(model_list, new_armetadata)
                    elif self.deepness == 3:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['initial_weight_distribution']['value'] = "normal"
                        model_aux['parameters']['epochs']['value'] = epochs
                        self.safe_append(model_list, new_armetadata)

                    if self.deepness <= self.deep_impact:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']

                        for iterador in range(0, len(model_aux['parameters']['hidden']['value'])):
                            if iterador != int((float(len(model_aux['parameters']['hidden']['value'])) / 2) - 0.5):
                                model_aux['parameters']['hidden']['value'][iterador] = \
                                    int(round(model_aux['parameters']['hidden']['value'][iterador] * wider_increment, 0))
                        model_aux['parameters']['epochs']['value'] = epochs
                        self.safe_append(model_list, new_armetadata)
                        if len(model_aux['parameters']['hidden']['value']) < 5:
                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['h2o']
                            next_hidden = int(
                                round(model_aux['parameters']['hidden']['value'][0] * deeper_increment, 0))
                            model_aux['parameters']['hidden']['value'].insert(0, next_hidden)
                            model_aux['parameters']['hidden_dropout_ratios']['value'].insert(0, h_dropout_ratio)
                            model_aux['parameters']['hidden']['value'].append(next_hidden)
                            model_aux['parameters']['hidden_dropout_ratios']['value'].append(h_dropout_ratio)
                            model_aux['parameters']['epochs']['value'] = epochs
                            self.safe_append(model_list, new_armetadata)

                    if model['parameters']['mini_batch_size']['value'] >= dpl_min_batch_size:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['mini_batch_size']['value'] = \
                            round(model_aux['parameters']['mini_batch_size']['value'] / dpl_batch_reduced_divisor)
                        model_aux['parameters']['epochs']['value'] = epochs
                        self.safe_append(model_list, new_armetadata)

                elif model['model'] == 'H2OKMeansEstimator':
                    if scoring_metric.shape[0] == 0 or \
                            (int(scoring_metric['number_of_reassigned_observations'][-1:]) >= 0):
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['h2o']
                        model_aux['parameters']['max_iterations']['value'] = \
                            int(model_aux['parameters']['max_iterations']['value'] * clustering_increment)
                        self.safe_append(model_list, new_armetadata)
            except KeyError:
                return None
        else:
            return None
        if len(model_list) == 0:
            return None
        else:
            return model_list

    ## Method manging generation of possible optimized Apache Spark models
    # params: results for Handlers (gdayf.handlers)
    # @param armetadata ArMetadata Model
    # @return list of possible optimized models to execute return None if nothing to do
    def optimize_models_spark(self, armetadata):
        model_list = list()
        model = armetadata['model_parameters'][get_model_fw(armetadata)]
        metric_value, _, objective = eval('self.get_' + self.metric + '(armetadata)')

        if get_model_fw(armetadata) == 'spark' and metric_value != objective \
                and armetadata['status'] != self._labels['failed_op']:
            try:
                scoring_metric = decode_ordered_dict_to_dataframe(armetadata['metrics']['scoring'])
            except ValueError:
                print("TRACE: Not scoring: " + model)
            config = LoadConfig().get_config()['optimizer']['AdviserStart_rules']['spark']
            min_rows_limit = config['min_rows_limit']
            min_rows_increment = config['min_rows_increment']
            max_interactions_increment = config['max_interactions_increment']
            interactions_increment = config['interactions_increment']
            max_depth_increment = config['max_depth_increment']
            ntrees_increment = config['ntrees_increment']
            stepSize = config['stepSize']
            aggregationDepth_increment = config['aggregationDepth_increment']
            regParam = config['regParam']
            elastic_variation = config['elastic_variation']
            nv_smoothing = config['nv_smoothing']
            nv_improvement = config['nv_improvement']
            nv_divisor = config['nv_divisor']
            clustering_increment = config['clustering_increment']
            initstep_increment = config['initstep_increment']

            if model['model'] == 'LinearSVC':
                if self.deepness == 2 and len(regParam) != 0:
                    for elastic in regParam:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['regParam']['value'] = elastic['value']
                        self.safe_append(model_list, new_armetadata)

                new_armetadata = armetadata.copy_template()
                model_aux = new_armetadata['model_parameters']['spark']
                model_aux['parameters']['maxDepth']['value'] *= max_depth_increment
                self.safe_append(model_list, new_armetadata)
                new_armetadata = armetadata.copy_template()
                model_aux = new_armetadata['model_parameters']['spark']
                model_aux['parameters']['aggregationDepth']['value'] *= aggregationDepth_increment
                self.safe_append(model_list, new_armetadata)

            elif model['model'] == 'LogisticRegression' or model['model'] == 'LinearRegression':
                if self.deepness == 2 and len(regParam) != 0:
                    for elastic in regParam:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['regParam']['value'] = elastic['value']
                        self.safe_append(model_list, new_armetadata)

                if model['parameters']['elasticNetParam']['value'] \
                        * (1 + elastic_variation) <= 1.0:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['elasticNetParam']['value'] = \
                        model_aux['parameters']['elasticNetParam']['value'] * (1 + elastic_variation)
                    self.safe_append(model_list, new_armetadata)
                if model['parameters']['elasticNetParam']['value'] \
                        * (1 - elastic_variation) >= 0.0:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['elasticNetParam']['value'] = \
                        model_aux['parameters']['elasticNetParam']['value'] * (1 + elastic_variation)
                    self.safe_append(model_list, new_armetadata)

                try:
                    if model['parameters']['maxIter']['value'] \
                            >= scoring_metric['totalIterations'][0] and \
                               scoring_metric['totalIterations'][0] <= max_interactions_increment:

                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['maxIter']['value'] *= interactions_increment
                        self.safe_append(model_list, new_armetadata)
                except KeyError:
                    if model['parameters']['maxIter']['value'] \
                            <= max_interactions_increment:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['maxIter']['value'] *= interactions_increment
                        self.safe_append(model_list, new_armetadata)

                new_armetadata = armetadata.copy_template()
                model_aux = new_armetadata['model_parameters']['spark']
                model_aux['parameters']['aggregationDepth']['value'] *= aggregationDepth_increment
                self.safe_append(model_list, new_armetadata)

            elif model['model'] == 'DecisionTreeClassifier' or model['model'] == 'DecisionTreeRegressor':
                '''if self.deepness == 2 and len(eval(model['parameters']['impurity']['type'])) != 0:
                    for element in eval(model['parameters']['impurity']['type']):
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['impurity']['value'] = element
                        self.safe_append(model_list, new_armetadata)'''
                    
                if model['parameters']['minInstancesPerNode']['value'] > (min_rows_limit / 2):
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['minInstancesPerNode']['value'] = round(
                        model_aux['parameters']['minInstancesPerNode']['value']
                        / min_rows_increment, 0)
                    self.safe_append(model_list, new_armetadata)

                if scoring_metric['max_depth'][0] >= model['parameters']['maxDepth']['value']:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['maxDepth']['value'] = \
                        model_aux['parameters']['maxDepth']['value'] * max_depth_increment
                    self.safe_append(model_list, new_armetadata)

            elif model['model'] == 'GBTRegressor':
                if self.deepness == 2 and len(stepSize) != 0 and len(eval(model['parameters']['lossType']['type'])) != 0:
                    for stepsize in stepSize:
                        for element in eval(model['parameters']['lossType']['type']):
                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['spark']
                            model_aux['parameters']['lossType']['value'] = element
                            model_aux['parameters']['stepSize']['value'] = stepsize['learn']
                            self.safe_append(model_list, new_armetadata)
                elif self.deepness == 2 and len(stepSize) != 0:
                    for stepsize in stepSize:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['stepSize']['value'] = stepsize['learn']
                        self.safe_append(model_list, new_armetadata)
                elif self.deepness == 2 and len(eval(model['parameters']['lossType']['type'])) != 0:
                    for element in eval(model['parameters']['lossType']['type']):
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['impurity']['value'] = element
                        self.safe_append(model_list, new_armetadata)

                if model['parameters']['minInstancesPerNode']['value'] > (min_rows_limit / 2):
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['minInstancesPerNode']['value'] = round(
                        model_aux['parameters']['minInstancesPerNode']['value']
                        / min_rows_increment, 0)
                    self.safe_append(model_list, new_armetadata)

                if scoring_metric['max_depth'][0] >= model['parameters']['maxDepth']['value']:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['maxDepth']['value'] *= max_depth_increment
                    self.safe_append(model_list, new_armetadata)

                if scoring_metric['trees'][0] >= model['parameters']['maxIter']['value']:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['maxIter']['value'] *= ntrees_increment
                    self.safe_append(model_list, new_armetadata)

            elif model['model'] == 'GBTClassifier':
                if self.deepness == 2 and len(stepSize) != 0 and len(eval(model['parameters']['lossType']['type'])) != 0:
                    for stepsize in stepSize:
                        for element in eval(model['parameters']['lossType']['type']):
                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['spark']
                            model_aux['parameters']['lossType']['value'] = element
                            model_aux['parameters']['stepSize']['value'] = stepsize['learn']
                            self.safe_append(model_list, new_armetadata)
                elif self.deepness == 2 and len(stepSize) != 0:
                    for stepsize in stepSize:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['stepSize']['value'] = stepsize['learn']
                        self.safe_append(model_list, new_armetadata)
                elif self.deepness == 2 and len(eval(model['parameters']['impurity']['type'])) != 0:
                    for element in eval(model['parameters']['lossType']['type']):
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['lossType']['value'] = element
                        self.safe_append(model_list, new_armetadata)

                if model['parameters']['minInstancesPerNode']['value'] > (min_rows_limit / 2):
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['minInstancesPerNode']['value'] = round(
                        model_aux['parameters']['minInstancesPerNode']['value']
                        / min_rows_increment, 0)
                    self.safe_append(model_list, new_armetadata)

                if scoring_metric['max_depth'][0] >= model['parameters']['maxDepth']['value']:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['maxDepth']['value'] *= max_depth_increment
                    self.safe_append(model_list, new_armetadata)

                if scoring_metric['trees'][0] >= model['parameters']['maxIter']['value']:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['maxIter']['value'] *= ntrees_increment
                    self.safe_append(model_list, new_armetadata)

            elif model['model'] == 'RandomForestClassifier' or model['model'] == 'RandomForestRegressor':

                if self.deepness == 2 and len(eval(model['parameters']['featureSubsetStrategy']['type'])) != 0 \
                        and len(eval(model['parameters']['impurity']['type'])) != 0:
                    for featuresubsetstrategy in eval(model['parameters']['featureSubsetStrategy']['type']):
                        for element in eval(model['parameters']['impurity']['type']):
                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['spark']
                            model_aux['parameters']['impurity']['value'] = element
                            model_aux['parameters']['featureSubsetStrategy']['value'] = featuresubsetstrategy
                            self.safe_append(model_list, new_armetadata)
                elif self.deepness == 2 and len(eval(model['parameters']['featureSubsetStrategy']['type'])) != 0:
                    for featuresubsetstrategy in eval(model['parameters']['featureSubsetStrategy']['type']):
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['featureSubsetStrategy']['value'] = featuresubsetstrategy
                        self.safe_append(model_list, new_armetadata)
                elif self.deepness == 2 and len(eval(model['parameters']['impurity']['type'])) != 0:
                    for element in model['parameters']['impurity']['type']:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['impurity']['value'] = element
                        self.safe_append(model_list, new_armetadata)

                if model['parameters']['minInstancesPerNode']['value'] > (min_rows_limit / 2):
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['minInstancesPerNode']['value'] = round(
                        model_aux['parameters']['minInstancesPerNode']['value']
                        / min_rows_increment, 0)
                    self.safe_append(model_list, new_armetadata)

                if scoring_metric['max_depth'][0] >= model['parameters']['maxDepth']['value']:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['maxDepth']['value'] *= max_depth_increment
                    self.safe_append(model_list, new_armetadata)

                if scoring_metric['trees'][0] >= model['parameters']['numTrees']['value']:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['numTrees']['value'] *= ntrees_increment
                    self.safe_append(model_list, new_armetadata)

            elif model['model'] == 'GeneralizedLinearRegression':
                if self.deepness == 2 and len(regParam) != 0:
                    for elastic in regParam:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['regParam']['value'] = elastic['value']
                        self.safe_append(model_list, new_armetadata)
                if self.deepness == 2:
                    if model['parameters']['family']['value'] in ['gaussian', 'gamma']:
                        linklist = ['log', 'inverse']
                    elif model['parameters']['family']['value'] in ['poisson']:
                        linklist = ['log', 'sqrt']
                    elif model['parameters']['family']['value'] in ['poisson', 'tweedie']:
                        linklist = []
                    for linkin in linklist:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['link']['value'] = linkin
                        self.safe_append(model_list, new_armetadata)

                if model['parameters']['maxIter']['value'] \
                        <= max_interactions_increment:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['maxIter']['value'] *= interactions_increment
                    self.safe_append(model_list, new_armetadata)

            elif model['model'] == 'NaiveBayes':
                if self.deepness == 2 and len(nv_smoothing) != 0:
                    for elastic in nv_smoothing:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['smoothing']['value'] = elastic['value']
                        self.safe_append(model_list, new_armetadata)

                for adjusting in ['improvement', 'decrement']:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    if adjusting == 'improvement':
                        model_aux['parameters']['smoothing']['value'] = model_aux['parameters']['smoothing'][
                                                                          'value'] * (1 + nv_improvement)
                    else:
                        model_aux['parameters']['smoothing']['value'] = model_aux['parameters']['smoothing'][
                                                                          'value'] * (1 - nv_divisor)
                    self.safe_append(model_list, new_armetadata)

            elif model['model'] == 'BisectingKMeans':

                new_armetadata = armetadata.copy_template()
                model_aux = new_armetadata['model_parameters']['spark']
                model_aux['parameters']['maxIter']['value'] = \
                    int(model_aux['parameters']['maxIter']['value'] * clustering_increment)
                self.safe_append(model_list, new_armetadata)

                '''new_armetadata = armetadata.copy_template()
                model_aux = new_armetadata['model_parameters']['spark']
                model_aux['parameters']['minDivisibleClusterSize']['value'] *= clustering_increment
                self.safe_append(model_list, new_armetadata)'''

            elif model['model'] == 'KMeans':

                if self.deepness == 2 and len(eval(model['parameters']['initMode']['type'])) != 0:
                    for element in eval(model['parameters']['initMode']['type']):
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['initMode']['value'] = element
                        self.safe_append(model_list, new_armetadata)

                new_armetadata = armetadata.copy_template()
                model_aux = new_armetadata['model_parameters']['spark']
                model_aux['parameters']['maxIter']['value'] = \
                    int(model_aux['parameters']['maxIter']['value'] * clustering_increment)
                self.safe_append(model_list, new_armetadata)

                '''new_armetadata = armetadata.copy_template()
                model_aux = new_armetadata['model_parameters']['spark']
                model_aux['parameters']['initSteps']['value'] *= initstep_increment
                self.safe_append(model_list, new_armetadata)'''

            else:
                return None

            if len(model_list) == 0:
                return None
            else:
                return model_list