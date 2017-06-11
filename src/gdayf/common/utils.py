from hashlib import md5 as md5
from hashlib import sha256 as sha256
from pandas import read_json
from pandas import read_json

def hash_key(hash_type, filename):
    if hash_type == 'MD5':
        try:
            return md5(open(filename, 'rb').read()).hexdigest()
        except IOError:
            raise
        return None
    elif hash_type == 'SHA256':
        try:
            return sha256(open(filename, 'rb').read()).hexdigest()
        except IOError:
            raise
        return None


def decode_json_to_dataframe(json_string, orient='split'):
    return read_json (json_string, orient=orient)
