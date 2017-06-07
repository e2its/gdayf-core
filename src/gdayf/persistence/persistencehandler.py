from json import dump
from os import path as path
from shutil import copyfile
from hashlib import md5 as md5
from hashlib import sha256 as sha256
from gdayf.common.storagemetadata import StorageMetadata

class PersistenceHandler(object):
    def __init__(self):
         None

    @staticmethod
    def replicate_file(type_dest, path_dest, type_source, path_source):

        if type_source == 'localfs':
            if type_dest == 'localfs':
                mkdir(path.dirname(path_dest), 0o0777)
                copyfile(path_source, path_dest)
            elif type_dest == 'hdfs':
                None
            elif type_dest == 'mongoDB':
                None
        elif type_source == 'hdfs':
            if type_dest == 'localfs':
                None
            elif type_dest == 'hdfs':
                None
            elif type_dest == 'mongoDB':
                None
        elif type_source == 'mongoDB':
            if type_dest == 'localfs':
                None
            elif type_dest == 'hdfs':
                None
            elif type_dest == 'mongoDB':
                None

    @staticmethod
    def store_json(storage_json, ar_json):

        assert isinstance(storage_json, StorageMetadata)

        for each_storage_type in storage_json:
            if each_storage_type['type'] == 'localfs':
                file = open(each_storage_type['value'], 'w')
                dump(ar_json, file, indent=4)
                file.close()
            elif each_storage_type['type'] == 'hdfs':
                None
            elif each_storage_type['type'] == 'mongoDB':
                None

    @staticmethod
    def hash_key(hash_type, filename):
            if hash_type == 'MD5':
                try:
                    return md5(open(filename, 'rb').read()).hexdigest()
                except IOError:
                    return None
            elif hash_type == 'SHA256':
                try:
                    return sha256(open(filename, 'rb').read()).hexdigest()
                except IOError:
                    return None