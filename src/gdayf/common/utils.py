## @package gdayf.common.utils
# Define all objects, functions and structs related to common utilities not associated to one concrete object
# and able to be reused on whole context

from hashlib import md5 as md5
from hashlib import sha256 as sha256
from pandas import read_json
from json import dumps
from copy import deepcopy
from numpy.random import rand

dtypes = ['int', 'float', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
ftypes = [ 'float', 'float16', 'float32', 'float64']
itypes = ['int', 'int16', 'int32', 'int64']

## Function oriented to get the hash_key for a file
# @param hash_type in ['MD5', 'SHS256']
# @param filename full path string
# @return Hash_key if success and (1) on error issues
def hash_key(hash_type, filename):
    if hash_type == 'MD5':
        try:
            return md5(open(filename, 'rb').read()).hexdigest()
        except IOError:
            raise
        return 1
    elif hash_type == 'SHA256':
        try:
            return sha256(open(filename, 'rb').read()).hexdigest()
        except IOError:
            raise
        return 1


## Function oriented convert a json dataframe string structure on pandas.dataframe()
# @param json_string containg a DataFrame table
# @param orient split (default) as usual DayF uses
# @return pandas.dataframe
def decode_json_to_dataframe(json_string, orient='split'):
    return read_json(json_string, orient=orient)


## Function oriented compare two dicts based on hash_key(json transformations)
# @param dict1
# @param dict2
# @return True if equals false in other case
def compare_dict(dict1, dict2):
    if dict1 is None or dict2 is None:
        return dict1 is None and dict2 is None
    else:
        return md5(dumps(dict1)) == md5(dumps(dict2))


## Function oriented compare two normalizations_sets based on hash_key(json transformations)
# @param list1 List of Dict
# @param list2 List of Dict
# @return True if equals false in other case
def compare_list_ordered_dict(list1, list2):
    if list1[0] is None or list2[0] is None:
        return list1[0] is None and list2[0] is None
    elif len(list1) != len(list2):
        return False
    else:
        for i in range(0, len(list1)):
            if md5(dumps(list[i])) != md5(dumps(list2[i].encode('utf8')).hexdigest()):
                return False
        return True


## Function to get framework from ar.json model description
def get_model_fw(model):
    return list(model['model_parameters'].keys())[0]


## Function to get normalization_sets structure from ar.json model description
# @param model ArMetadata
# @return ArMetadata deepcopy
def get_model_ns(model):
    return deepcopy(model['normalizations_set'])


## Function to get pandas dataframe split without copy
# @param df Pandas dataframe
# @param train_perc % for train_dataframe
# @return Dict ('trai'n df pointer, 'test' df pointer)
def pandas_split_data(df, train_perc=0.9):
    df['train'] = rand(len(df)) < train_perc
    train = df[df.train == 1].drop('train', axis=1)
    test = df[df.train == 0].drop('train', axis=1)
    return train, test

## Function oriented to extract ArMetadata from json and generate ArMetadata List with all Analysis Data
# @param path path pointed to analysis storage fs
# @return ArMetadata List
def get_arlist_from_path (path):
    pass