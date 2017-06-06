from json import dump
from os import makedirs as mkdir
from os import path as path
from shutil import copyfile
from hashlib import md5 as md5
from hashlib import sha256 as sha256
from gdayf.common.storagemetadata import StorageMetada


def store_json(storage_json, ar_json):

    assert isinstance(storage_json, StorageMetada)

    for each_storage_type in storage_json:
        if each_storage_type['type'] == 'localfs':
            file = open(each_storage_type['value'], 'w')
            dump(ar_json, file, indent=4)
            file.close()
        elif each_storage_type['type'] == 'hdfs':
            None
        elif each_storage_type['type'] == 'mongoDB':
            None


def hash_key(hash_type, filename):

        if hash_type == 'MD5':
            return md5(open(filename, 'rb').read()).hexdigest()
        elif hash_type == 'SHA256':
            return sha256(open(filename, 'rb').read()).hexdigest()