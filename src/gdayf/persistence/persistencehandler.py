from json import dump
from shutil import copyfile
import mmap
from gdayf.common.storagemetadata import StorageMetadata
from os import makedirs
from os import path as ospath
from os.path import dirname
from  gdayf.common.utils import hash_key
from pathlib import Path


class PersistenceHandler(object):
    def __init__(self):
         None

    '''def replicate_bdata(self, filename, storage_ar):
        assert isinstance(storage_ar, StorageMetadata)
        global_op = 0
        try:
            file = open(filename, 'rb')
            mmap_ = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        except IOError:
            return 1

        for each_storage_type in ar:
            if each_storage_type['type'] == 'localfs':
                result, each_storage_type['hash_value'] = self._store_file_to_localfs(storage_json, mmap_)
                global_op += result
            elif each_storage_type['type'] == 'hdfs':
                result, each_storage_type['hash_value'] = self._store_file_to_hdfs(storage_json, mmap_)
                global_op += result
            elif each_storage_type['type'] == 'mongoDB':
                result, each_storage_type['hash_value'] = self._store_file_to_mongoDB(storage_json, mmap_)
                global_op += result

        mmap_.close()
        return global_op'''

    def store_file(self, storage_json, filename):
        '''assert isinstance(storage_json, StorageMetadata)'''
        global_op = 0
        try:
            file = open(filename, 'rb')
            mmap_ = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        except IOError:
            return 1

        for each_storage_type in storage_json:
            if each_storage_type['type'] == 'localfs':
                result, each_storage_type['hash_value'] = self._store_file_to_localfs(each_storage_type, mmap_)
                global_op += result
            elif each_storage_type['type'] == 'hdfs':
                result, each_storage_type['hash_value'] = self._store_file_to_hdfs(each_storage_type, mmap_)
                global_op += result
            elif each_storage_type['type'] == 'mongoDB':
                result, each_storage_type['hash_value'] = self._store_file_to_mongoDB(each_storage_type, mmap_)
                global_op += result

        mmap_.close()
        return global_op

    def _store_file_to_localfs(self, storage_json, mmap_):
        if not ospath.exists(path=storage_json['value']):
            try:
                self._mkdir_localfs(path=dirname(storage_json['value']), grants=0o0777)
                with open(storage_json['value'], 'wb') as wfile:
                    mmap_.seek(0)
                    iterator = 0
                    while iterator < mmap_.size():
                        wfile.write(mmap_.read())
                        wfile.flush()
                        iterator += 1
                    wfile.close()
            except IOError:
                return 1, None

        return 0, hash_key(hash_type=storage_json['hash_type'], filename=storage_json['value'])

    def _store_file_to_hdfs(self, storage_json, mmap_):
        return 1, None

    def _store_file_to_mongoDB(self, storage_json, mmap_):
        return 1, None

    def store_json(self, storage_json, ar_json):
        '''assert isinstance(storage_json, StorageMetadata)'''
        global_op = 0
        for each_storage_type in storage_json:
            if each_storage_type['type'] == 'localfs':
                global_op += self._store_json_to_localfs(each_storage_type, ar_json)
            elif each_storage_type['type'] == 'hdfs':
                global_op += self._store_json_to_hdfs(each_storage_type, ar_json)
            elif each_storage_type['type'] == 'mongoDB':
                global_op += self._store_json_to_mongoDB(each_storage_type, ar_json)
        return global_op


    def _store_json_to_localfs(self, storage_json, ar_json):
        if not ospath.exists(storage_json['value']):
            try:
                self._mkdir_localfs(path=dirname(storage_json['value']), grants=0o0777)
                file = open(storage_json['value'], 'w')
                dump(ar_json, file, indent=4)
                file.close()
                return 0
            except IOError:
                return 1
        return 1

    def _store_json_to_hdfs(self, storage_json, ar_json):
        return 1

    def _store_json_to_mongoDB(self, storage_json, ar_json):
        return 1

    def mkdir(self, type, path, grants):
        if type == 'localfs':
            return self._mkdir_localfs(path, grants)
        elif type == 'localfs':
            return self._mkdir_hdfs(path, grants)
        elif type == 'mongoDB':
            return self._mkdir_mongoDB(path, grants)

    @staticmethod
    def _mkdir_localfs(path, grants):
        try:
            Path(path).mkdir(mode=grants, parents=True, exist_ok=True)
            return 0
        except IOError:
            return 1

    @staticmethod
    def _mkdir_hdfs(path, grants):
        try:
            return 0
        except IOError:
            return 1

    @staticmethod
    def _mkdir_mongoDB(path, grants):
        try:
            return 0
        except IOError:
            return 1