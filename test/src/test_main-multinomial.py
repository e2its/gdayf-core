if __name__ == "__main__":

    from gdayf.handlers.h2ohandler import H2OHandler
    from gdayf.handlers.inputhandler import inputHandlerCSV
    from gdayf.core.adviserastar import AdviserAStar
    from gdayf.common.dfmetada import DFMetada
    from pandas import DataFrame as DataFrame
    from pandas import concat as concat
    import numpy as np
    import os
    from six.moves import cPickle as pickle
    from time import time
    from json import dumps

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

    train_dataset = train_dataset[-1001:-1]
    train_labels  = train_labels[-1001:-1]

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


    analysis_id = 'PoC-multinomial-minst'
    adviser = AdviserAStar(analysis_id=analysis_id, metric='accuracy')
    df = DFMetada().getDataFrameMetadata(pd_train_dataset, 'pandas')
    _, analysis_list = adviser.set_recommendations(dataframe_metadata=df, objective_column='objective0',
                                                             atype=adviser.FAST)


    analysis_models = H2OHandler()
    analysis_results = analysis_models.order_training(analysis_id=adviser.analysis_id,
                                                      training_frame=concat([pd_train_dataset, pd_valid_dataset],
                                                                            axis=0),
                                                      model=analysis_list)

    sorted_list = adviser.priorize_models(analysis_results[0], analysis_results[1])

    for each_model in sorted_list:
        print(dumps(each_model, indent=4))

    '''for file in os.listdir(r'D:\Data\models\h2o\PoC-multinomial-minst\train\json'):
        analysis_models = H2OHandler()
        json_file = open(r'D:\Data\models\h2o\PoC-multinomial-minst\train\json' + '/' + file)
        analysis_results = analysis_models.predict(pd_test_dataset, json_file)'''



