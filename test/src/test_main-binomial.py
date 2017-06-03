if __name__ == "__main__":

    from handlers.h2ohandler import H2OHandler
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

    train_dataset = train_dataset[-20001:-1]
    train_labels = train_labels[-20001:-1]

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

    # Binomial_test

    pd_train_dataset['objective0'] = pd_train_dataset['objective0'].apply(convertir_binomial)
    pd_valid_dataset['objective0'] = pd_valid_dataset['objective0'].apply(convertir_binomial)
    pd_test_dataset['objective0'] = pd_test_dataset['objective0'].apply(convertir_binomial)

    json_file = open(r'D:\e2its-dayf.svn\gdayf\branches\0.0.2-jlsanchez\test\json\algorithm_result(ar)-binomial.json')
    analysis_list = [(json_file, None)]

    analysis_models = H2OHandler()
    analysis_results = analysis_models.order_training(analysis_id='PoC_binomial', training_frame=pd_train_dataset,
                                                      valid_frame=pd_train_dataset, analysis_list=analysis_list)
    del analysis_models

    analysis_models = H2OHandler()
    for file in os.listdir(r'D:\Data\models\h2o\PoC-binomial\train\json'):
        json_file = open(r'D:\Data\models\h2o\PoC-binomial\train\json' + '/' + file)
        analysis_results = analysis_models.predict(pd_test_dataset, json_file)
    del analysis_models


