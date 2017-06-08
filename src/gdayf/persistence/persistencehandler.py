from json import dump
from os import path as path
from shutil import copyfile
from gdayf.common.storagemetadata import StorageMetadata
from os import makedirs
from os import path as ospath


class PersistenceHandler(object):
    def __init__(self):
         None

    '''def replicate_file(self, type_dest, path_dest, type_source, path_source):
        if type_source == 'localfs':
            if type_dest == 'localfs':
                self.mkdir(path.dirname(path_dest), 0o0777)
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
    '''
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
    def mkdir(type, path, grants):
        if type == 'localfs':
            try:
                if not ospath.exists(path):
                    makedirs(path, grants)
                return 0
            except IOError:
                return 1
        elif type == 'hdfs':
            return 1
        elif type == 'monoDB':
            return 1