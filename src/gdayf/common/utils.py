## @package gdayf.common.utils
# Define all objects, functions and structs related to common utilities not associated to one concrete object
# and able to be reused on whole context

from hashlib import md5 as md5
from hashlib import sha256 as sha256
from pandas import read_json

dtypes = ['int', 'float', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

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

