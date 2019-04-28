## @package gdayf.optimizers.spark_optimizer
# Define all objects, functions and structured related to manage and execute Smart analysis based on A* Algorithm
# and defined heuristic for H2O.ai framework
# Main class AdviserAStarAvg. Lets us execute analysis, make recommendations over optimizing on selected algoritms

from gdayf.common.utils import get_model_fw
from gdayf.conf.loadconfig import LoadConfig
from gdayf.common.utils import decode_ordered_dict_to_dataframe

class Optimizer(object):

    ## Constructor
    # Initialize all framework variables and starts or connect to spark cluster
    # Aditionally starts PersistenceHandler and logsHandler
    # @param self object pointer
    # @param e_c context pointer
    def __init__(self, e_c):
        self._ec = e_c
        self._labels = self._ec.labels.get_config()['messages']['adviser']
        self._config = LoadConfig().get_config()['optimizer']['AdviserStart_rules']['spark']

    ## Method manging generation of possible optimized H2O models loadded dinamically on adviserclass
    # params: results for Handlers (gdayf.handlers)
    # @param armetadata ArMetadata Model
    # @param metric_value metrict for priorizing models ['train_accuracy', 'test_rmse', 'train_rmse', 'test_accuracy', 'combined_accuracy', ...]
    # @param objective objective for analysis ['regression, binomial, ...]
    # @param deepness currrent deeepnes of the analysis
    # @param deep_impact max deepness of analysis
    # @return list of possible optimized models to execute return None if nothing to do
    def optimize_models(self, armetadata, metric_value, objective, deepness, deep_impact):
        model_list = list()
        model = armetadata['model_parameters'][get_model_fw(armetadata)]
        config = self._config
    
        if get_model_fw(armetadata) == 'spark' and metric_value != objective \
                and armetadata['status'] != self._labels['failed_op']:
            try:
                scoring_metric = decode_ordered_dict_to_dataframe(armetadata['metrics']['scoring'])
            except ValueError:
                print("TRACE: Not scoring: " + model)
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
                if deepness == 2 and len(regParam) != 0:
                    for elastic in regParam:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['regParam']['value'] = elastic['value']
                        model_list.append(new_armetadata)
    
                try:
                    if model['parameters']['maxIter']['value'] \
                            >= scoring_metric['totalIterations'][0] and \
                            scoring_metric['totalIterations'][0] <= max_interactions_increment:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['maxIter']['value'] *= interactions_increment
                        model_list.append(new_armetadata)
                except KeyError:
                    if model['parameters']['maxIter']['value'] \
                            <= max_interactions_increment:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['maxIter']['value'] *= interactions_increment
                        model_list.append(new_armetadata)
    
                new_armetadata = armetadata.copy_template()
                model_aux = new_armetadata['model_parameters']['spark']
                model_aux['parameters']['aggregationDepth']['value'] *= aggregationDepth_increment
                model_list.append(new_armetadata)
    
            elif model['model'] == 'LogisticRegression' or model['model'] == 'LinearRegression':
                if deepness == 2 and len(regParam) != 0:
                    for elastic in regParam:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['regParam']['value'] = elastic['value']
                        model_list.append(new_armetadata)
    
                if model['parameters']['elasticNetParam']['value'] \
                        * (1 + elastic_variation) <= 1.0:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['elasticNetParam']['value'] = \
                        model_aux['parameters']['elasticNetParam']['value'] * (1 + elastic_variation)
                    model_list.append(new_armetadata)
                if model['parameters']['elasticNetParam']['value'] \
                        * (1 - elastic_variation) >= 0.0:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['elasticNetParam']['value'] = \
                        model_aux['parameters']['elasticNetParam']['value'] * (1 + elastic_variation)
                    model_list.append(new_armetadata)
    
                try:
                    if model['parameters']['maxIter']['value'] \
                            >= scoring_metric['totalIterations'][0] and \
                            scoring_metric['totalIterations'][0] <= max_interactions_increment:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['maxIter']['value'] *= interactions_increment
                        model_list.append(new_armetadata)
                except KeyError:
                    if model['parameters']['maxIter']['value'] \
                            <= max_interactions_increment:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['maxIter']['value'] *= interactions_increment
                        model_list.append(new_armetadata)
    
                new_armetadata = armetadata.copy_template()
                model_aux = new_armetadata['model_parameters']['spark']
                model_aux['parameters']['aggregationDepth']['value'] *= aggregationDepth_increment
                model_list.append(new_armetadata)
    
            elif model['model'] == 'DecisionTreeClassifier' or model['model'] == 'DecisionTreeRegressor':

                if model['parameters']['minInstancesPerNode']['value'] > (min_rows_limit / 2):
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['minInstancesPerNode']['value'] = round(
                        model_aux['parameters']['minInstancesPerNode']['value']
                        / min_rows_increment, 0)
                    model_list.append(new_armetadata)
    
                if scoring_metric['max_depth'][0] >= model['parameters']['maxDepth']['value']:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['maxDepth']['value'] = \
                        model_aux['parameters']['maxDepth']['value'] * max_depth_increment
                    model_list.append(new_armetadata)
    
            elif model['model'] == 'GBTRegressor':
                if deepness == 2 and len(stepSize) != 0 and len(eval(model['parameters']['lossType']['type'])) != 0:
                    for stepsize in stepSize:
                        for element in eval(model['parameters']['lossType']['type']):
                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['spark']
                            model_aux['parameters']['lossType']['value'] = element
                            model_aux['parameters']['stepSize']['value'] = stepsize['learn']
                            model_list.append(new_armetadata)
                elif deepness == 2 and len(stepSize) != 0:
                    for stepsize in stepSize:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['stepSize']['value'] = stepsize['learn']
                        model_list.append(new_armetadata)
                elif deepness == 2 and len(eval(model['parameters']['lossType']['type'])) != 0:
                    for element in eval(model['parameters']['lossType']['type']):
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['impurity']['value'] = element
                        model_list.append(new_armetadata)
    
                if model['parameters']['minInstancesPerNode']['value'] > (min_rows_limit / 2):
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['minInstancesPerNode']['value'] = round(
                        model_aux['parameters']['minInstancesPerNode']['value']
                        / min_rows_increment, 0)
                    model_list.append(new_armetadata)
    
                # 05/07/2018. Included platform base restriction maxDepth <=30
                if scoring_metric['max_depth'][0] >= model['parameters']['maxDepth']['value'] \
                        and model['parameters']['maxDepth']['value'] != 30:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    if model_aux['parameters']['maxDepth']['value'] * max_depth_increment > 30:
                        model_aux['parameters']['maxDepth']['value'] = 30
                    else:
                        model_aux['parameters']['maxDepth']['value'] *= max_depth_increment
    
                    model_list.append(new_armetadata)
    
                if scoring_metric['trees'][0] >= model['parameters']['maxIter']['value']:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['maxIter']['value'] *= ntrees_increment
                    model_list.append(new_armetadata)
    
            elif model['model'] == 'GBTClassifier':
                if deepness == 2 and len(stepSize) != 0 and len(eval(model['parameters']['lossType']['type'])) != 0:
                    for stepsize in stepSize:
                        for element in eval(model['parameters']['lossType']['type']):
                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['spark']
                            model_aux['parameters']['lossType']['value'] = element
                            model_aux['parameters']['stepSize']['value'] = stepsize['learn']
                            model_list.append(new_armetadata)
                elif deepness == 2 and len(stepSize) != 0:
                    for stepsize in stepSize:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['stepSize']['value'] = stepsize['learn']
                        model_list.append(new_armetadata)
                elif deepness == 2 and len(eval(model['parameters']['impurity']['type'])) != 0:
                    for element in eval(model['parameters']['lossType']['type']):
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['lossType']['value'] = element
                        model_list.append(new_armetadata)
    
                if model['parameters']['minInstancesPerNode']['value'] > (min_rows_limit / 2):
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['minInstancesPerNode']['value'] = round(
                        model_aux['parameters']['minInstancesPerNode']['value']
                        / min_rows_increment, 0)
                    model_list.append(new_armetadata)
    
                if scoring_metric['max_depth'][0] >= model['parameters']['maxDepth']['value']:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['maxDepth']['value'] *= max_depth_increment
                    model_list.append(new_armetadata)
    
                if scoring_metric['trees'][0] >= model['parameters']['maxIter']['value']:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['maxIter']['value'] *= ntrees_increment
                    model_list.append(new_armetadata)
    
            elif model['model'] == 'RandomForestClassifier' or model['model'] == 'RandomForestRegressor':
    
                if deepness == 2 and len(eval(model['parameters']['featureSubsetStrategy']['type'])) != 0 \
                        and len(eval(model['parameters']['impurity']['type'])) != 0:
                    for featuresubsetstrategy in eval(model['parameters']['featureSubsetStrategy']['type']):
                        for element in eval(model['parameters']['impurity']['type']):
                            new_armetadata = armetadata.copy_template()
                            model_aux = new_armetadata['model_parameters']['spark']
                            model_aux['parameters']['impurity']['value'] = element
                            model_aux['parameters']['featureSubsetStrategy']['value'] = featuresubsetstrategy
                            model_list.append(new_armetadata)
                elif deepness == 2 and len(eval(model['parameters']['featureSubsetStrategy']['type'])) != 0:
                    for featuresubsetstrategy in eval(model['parameters']['featureSubsetStrategy']['type']):
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['featureSubsetStrategy']['value'] = featuresubsetstrategy
                        model_list.append(new_armetadata)
                elif deepness == 2 and len(eval(model['parameters']['impurity']['type'])) != 0:
                    for element in model['parameters']['impurity']['type']:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['impurity']['value'] = element
                        model_list.append(new_armetadata)
    
                if model['parameters']['minInstancesPerNode']['value'] > (min_rows_limit / 2):
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['minInstancesPerNode']['value'] = round(
                        model_aux['parameters']['minInstancesPerNode']['value']
                        / min_rows_increment, 0)
                    model_list.append(new_armetadata)
    
                if scoring_metric['max_depth'][0] >= model['parameters']['maxDepth']['value']:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['maxDepth']['value'] *= max_depth_increment
                    model_list.append(new_armetadata)
    
                if scoring_metric['trees'][0] >= model['parameters']['numTrees']['value']:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['numTrees']['value'] *= ntrees_increment
                    model_list.append(new_armetadata)
    
            elif model['model'] == 'GeneralizedLinearRegression':
                if deepness == 2 and len(regParam) != 0:
                    for elastic in regParam:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['regParam']['value'] = elastic['value']
                        model_list.append(new_armetadata)
                if deepness == 2:
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
                        model_list.append(new_armetadata)
    
                if model['parameters']['maxIter']['value'] \
                        <= max_interactions_increment:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    model_aux['parameters']['maxIter']['value'] *= interactions_increment
                    model_list.append(new_armetadata)
    
            elif model['model'] == 'NaiveBayes':
                if deepness == 2 and len(nv_smoothing) != 0:
                    for elastic in nv_smoothing:
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['smoothing']['value'] = elastic['value']
                        model_list.append(new_armetadata)
    
                for adjusting in ['improvement', 'decrement']:
                    new_armetadata = armetadata.copy_template()
                    model_aux = new_armetadata['model_parameters']['spark']
                    if adjusting == 'improvement':
                        model_aux['parameters']['smoothing']['value'] = model_aux['parameters']['smoothing'][
                                                                            'value'] * (1 + nv_improvement)
                    else:
                        model_aux['parameters']['smoothing']['value'] = model_aux['parameters']['smoothing'][
                                                                            'value'] * (1 - nv_divisor)
                    model_list.append(new_armetadata)
    
            elif model['model'] == 'BisectingKMeans':
    
                new_armetadata = armetadata.copy_template()
                model_aux = new_armetadata['model_parameters']['spark']
                model_aux['parameters']['maxIter']['value'] = \
                    int(model_aux['parameters']['maxIter']['value'] * clustering_increment)
                model_list.append(new_armetadata)
    
            elif model['model'] == 'KMeans':
    
                if deepness == 2 and len(eval(model['parameters']['initMode']['type'])) != 0:
                    for element in eval(model['parameters']['initMode']['type']):
                        new_armetadata = armetadata.copy_template()
                        model_aux = new_armetadata['model_parameters']['spark']
                        model_aux['parameters']['initMode']['value'] = element
                        model_list.append(new_armetadata)
    
                new_armetadata = armetadata.copy_template()
                model_aux = new_armetadata['model_parameters']['spark']
                model_aux['parameters']['maxIter']['value'] = \
                    int(model_aux['parameters']['maxIter']['value'] * clustering_increment)
                model_list.append(new_armetadata)

            else:
                return None
    
            if len(model_list) == 0:
                return None
            else:
                return model_list