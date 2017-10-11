if __name__ == "__main__":

    from gdayf.core.controller import Controller
    from gdayf.common.constants import *
    from pandas import DataFrame as DataFrame
    from pandas import concat as concat
    import numpy as np
    import os
    from six.moves import cPickle as pickle

    def reformat(dataset, labels):
        dataset = DataFrame(dataset.reshape((-1, image_size * image_size)).astype(np.float32))
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        columns = list()
        for each_value in dataset.columns.values:
            columns.append(str(each_value))
        dataset.columns = columns
        labels = DataFrame(labels.reshape((-1, 1)).astype(str))
        columns = list()
        for each_value in labels.columns.values:
            columns.append('objective' + str(each_value))
        labels.columns = columns
        # labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
        return dataset, labels

    def convertir_binomial(x):
        if int(x) % 2 == 0:
            return 1
        else:
            return 0

    os.chdir('/Data/Data/datasheets/image-dection/Gdeeplearning-Udacity')
    pickle_file = 'notMNIST.pickle'

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory

    train_dataset = train_dataset[-10001:-1]
    train_labels = train_labels[-10001:-1]

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Validation set', test_dataset.shape, test_labels.shape)

    # In[5]:

    image_size = 28
    num_labels = 10

    pd_train_dataset, pd_train_labels = reformat(train_dataset, train_labels)
    pd_valid_dataset, pd_valid_labels = reformat(valid_dataset, valid_labels)
    pd_test_dataset, pd_test_labels = reformat(valid_dataset, valid_labels)
    pd_train_dataset = concat([pd_train_dataset, pd_train_labels], axis=1)
    pd_valid_dataset = concat([pd_valid_dataset, pd_valid_labels], axis=1)
    pd_test_dataset = concat([pd_test_dataset, pd_test_labels], axis=1)

    print('Training set', pd_train_dataset.shape)
    print('Validation set', pd_valid_dataset.shape)
    print('Test set', pd_test_dataset.shape)


    # Analysis
    controller = Controller()
    if controller.config_checks():
        status, recomendations = controller.exec_analysis(datapath=pd_train_dataset, objective_column='objective0',
                                                          amode=FAST, metric='test_accuracy', deep_impact=3)

        controller.log_model_list(recomendations[0]['model_id'], recomendations, metric='combined', accuracy=True)

        controller.save_models(recomendations, mode=EACH_BEST)
        controller.reconstruct_execution_tree(recomendations, metric='test_accuracy')
        controller.remove_models(recomendations, mode=ALL)

        # Prediction
        print(recomendations[0]['load_path'][0]['value'])
        prediction_frame = controller.exec_prediction(datapath=pd_test_dataset,
                                                      model_file=recomendations[0]['json_path'][0]['value'])
        print(prediction_frame[['objective0', 'predict', 'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9']])

        # Save Pojo
        # controller = Controller()
        result = controller.get_java_model(recomendations[0], 'pojo')
        print(result)

        # Save Mojo
        # controller = Controller()
        result = controller.get_java_model(recomendations[0], 'mojo')
        print(result)

        controller.clean_handlers()
    del controller
