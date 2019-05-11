if __name__ == "__main__":

    from gdayf.handlers.h2ohandler import H2OHandler
    from pandas import DataFrame as DataFrame
    from pandas import concat as concat
    import numpy as np
    import os
    from six.moves import cPickle as pickle
    from time import time

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

    os.chdir('d:/Data/Gdeeplearning-Udacity')
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

    train_dataset = train_dataset[-100001:-1]
    train_labels  = train_labels[-100001:-1]

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Validation set', test_dataset.shape, test_labels.shape)

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


    json_file = open(r'D:\e2its-dayf.svn\gdayf\branches\0.0.3-team03\test\json\algorithm_result(ar)-multinomial.json')
    analysis_list = [(json_file, None)]

    print(analysis_list)

    analysis_models = H2OHandler()
    analysis_results = analysis_models.order_training(analysis_id='PoC-multinmomial_' + str(time()),
                                                      training_frame=concat([pd_train_dataset, pd_valid_dataset], axis=0),
                                                      model=analysis_list)

    for file in os.listdir(r'D:\Data\models\h2o\PoC-multinomial\train\json'):
        analysis_models = H2OHandler()
        json_file = open(r'D:\Data\models\h2o\PoC-multinomial\train\json' + '/' + file)
        analysis_results = analysis_models.predict(pd_test_dataset, json_file)



