from json import dump, dumps, load, loads
from collections import OrderedDict
import mmap
from os import path as ospath, chmod
from os.path import dirname
from pathlib import Path
from gdayf.common.utils import hash_key
from gdayf.conf.loadconfig import LoadConfig
from gdayf.common.utils import get_model_fw
import gzip
import mimetypes
from hdfs import InsecureClient as Client, HdfsError
from pymongo import MongoClient
from pymongo.errors import *
from copy import deepcopy
import bson
from bson.codec_options import CodecOptions



## @package gdayf.persistence.persistencehandler
# Define all objects, functions and structures related to physically store information on persistence system
#  on an unified way

## Class to manage trasient information between all persistence options and models on an unified way
class PersistenceHandler(object):
    ## Class Constructor
    def __init__(self):
        self._config = LoadConfig().get_config()['storage']

    ## Method used to store a file on one persistence system ['localfs', ' hdfs', ' mongoDB']
    # using mmap structure to manage multi-persistence features
    # @param self object pointer
    # @param storage_json (list of storagemetadata objects or OrderedDict() compatible objects)
    # @param filename file full path string
    # @return global_op state (0 success) (n number of errors)
    def store_file(self, storage_json, filename):
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

    ## Protected method used to store a file on ['localfs']
    # using mmap structure to manage multi-persistence features
    # @param self object pointer
    # @param storage_json (list of storagemetadata objects or OrderedDict() compatible objects)
    # @param mmap mmap structure containing the file to store
    # @returns status (0,1) (hash_key)
    def _store_file_to_hdfs(self, storage_json, mmap_):
        try:
            client = Client(url=storage_json['url'])
        except HdfsError as hexecution_error:
            print(repr(hexecution_error))
            return 1, None
        except IOError as iexecution_error:
            print(repr(iexecution_error))
            return 1, None
        except OSError as oexecution_error:
            print(repr(oexecution_error))
            return 1, None
        try:
            self._mkdir_hdfs(path=dirname(storage_json['value']),
                             grants=int(self._config['grants'], 8),
                             client=client)
            with client.write(storage_json['value'], encoding='utf-8', overwrite=True) as wfile:
                mmap_.seek(0)
                iterator = 0
                while iterator < mmap_.size():
                    wfile.write(mmap_.read())
                    wfile.flush()
                    iterator += 1
                wfile.close()
        except HdfsError as hexecution_error:
            print(repr(hexecution_error))
            return 1, None
        except IOError as iexecution_error:
            print(repr(iexecution_error))
            return 1, None
        except OSError as oexecution_error:
            print(repr(oexecution_error))
            return 1, None
        finally:
            del client

        return 0, hash_key(hash_type=storage_json['hash_type'], filename=storage_json['value'])

    ## Protected method used to store a file on ['hdfs']
    #Not implemented yet !!
    # using mmap structure to manage multi-persistence features
    # @param self object pointer
    # @param storage_json (list of storagemetadata objects or OrderedDict() compatible objects)
    # @param mmap mmap structure containing the file to store
    # @returns status (0,1) (hash_key)
    def _store_file_to_localfs(self, storage_json, mmap_):
        if not ospath.exists(path=storage_json['value']):
            try:
                self._mkdir_localfs(path=dirname(storage_json['value']), grants=int(self._config['grants'], 8))
                with open(storage_json['value'], 'wb') as wfile:
                    mmap_.seek(0)
                    iterator = 0
                    while iterator < mmap_.size():
                        wfile.write(mmap_.read())
                        wfile.flush()
                        iterator += 1
                    wfile.close()
                    chmod(storage_json['value'], int(self._config['grants'], 8))
            except IOError:
                return 1, None

        return 0, hash_key(hash_type=storage_json['hash_type'], filename=storage_json['value'])

    ## Protected method used to store a file on ['mongoDB']
    #Not implemented yet !!
    # using mmap structure to manage multi-persistence features
    # @param self object pointer
    # @param storage_json (list of storagemetadata objects or OrderedDict() compatible objects)
    # @param mmap mmap structure containing the file to store
    # @returns status (0,1) (hash_key)
    def store_file_to_mongoDB(self, storage_json, mmap_):
        # Not implemented not necessary
        return 1, None

    ## Method used to store a json on all persistence system ['localfs', ' hdfs', ' mongoDB']
    # oriented to store full Analysis_results json but useful on whole json
    # @param self object pointer
    # @param storage_json (list of storagemetadata objects or OrderedDict() compatible objects)
    # @param ar_json file ArMetadata Class or OrderedDict() compatible object
    # @return global_op status (0 success) (n number of errors)
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

    ## Protected method used to store a json on ['localfs'] persistence system
    # oriented to store full Analysis_results json but useful on whole json
    # @param self object pointer
    # @param storage_json (list of storagemetadata objects or OrderedDict() compatible objects)
    # @param ar_json file ArMetadata Class or OrderedDict() compatible object
    # @return global_op status (0 success) (1 error)
    def _store_json_to_localfs(self, storage_json, ar_json):
        compress = LoadConfig().get_config()['persistence']['compress_json']
        #if not ospath.exists(storage_json['value']):
        try:
            self._mkdir_localfs(path=dirname(storage_json['value']), grants=int(self._config['grants'], 8))
            if compress:
                file = gzip.GzipFile(storage_json['value'], 'w')
                print(ar_json)
                json_str = dumps(ar_json, indent=4)
                json_bytes = json_str.encode('utf-8')
                file.write(json_bytes)
            else:
                file = open(storage_json['value'], 'w')
                dump(ar_json, file, indent=4)
            file.close()
            chmod(storage_json['value'], int(self._config['grants'], 8))
            return 0
        except IOError as iexecution_error:
            print(repr(iexecution_error))
            return 1


    ## Method used to store a json on ['hdfs'] persistence system
    # oriented to store full Analysis_results json but useful on whole json
    # @param self object pointer
    # @param storage_json (list of storagemetadata objects or OrderedDict() compatible objects)
    # @param ar_json file ArMetadata Class or OrderedDict() compatible object
    # @param client Cliente HDFS
    # @return operation status (0 success) (1 error)
    def _store_json_to_hdfs(self, storage_json, ar_json, client=None):
        remove_client = False
        if client is None:
            client = Client(url=self._config['hdfs']['url'])
            remove_client = True
        compress = LoadConfig().get_config()['persistence']['compress_json']
        #if not ospath.exists(storage_json['value']):
        try:
            self._mkdir_hdfs(path=dirname(storage_json['value']),
                             grants=int(self._config['grants'], 8),
                             client=client)
            if compress:
                json_str = dumps(ar_json, indent=4)
                json_bytes = json_str.encode('utf-8')
                client.write(storage_json['value'],
                             data=gzip.compress(json_bytes),
                             overwrite=True)
            else:
                client.write(storage_json['value'], data=dumps(ar_json, indent=4), encoding='utf-8', overwrite=True)
            return 0
        except HdfsError as hexecution_error:
            print(repr(hexecution_error))
            return 1
        except IOError as iexecution_error:
            print(repr(iexecution_error))
            return 1
        except OSError as oexecution_error:
            print(repr(oexecution_error))
            return 1
        finally:
            if remove_client:
                del client

    ## Protected method used to store a json on ['mongoDB'] persistence system
    # oriented to store full Analysis_results json but useful on whole json
    # @param self object pointer
    # @param storage_json (list of storagemetadata objects or OrderedDict() compatible objects)
    # @param ar_json file ArMetadata Class or OrderedDict() compatible object
    # @param client Cliente MongoClient()
    # @return operation status (0 success) (1 error)
    def _store_json_to_mongoDB(self, storage_json, ar_json, client=None):
        remove_client = False
        if client is None or not isinstance(client(MongoClient)):
            try:
                client = MongoClient(host=self._config['mongoDB']['url'],
                                     port=int(self._config['mongoDB']['port']),
                                     document_class=OrderedDict)
                remove_client = True
            except ConnectionFailure as cexecution_error:
                print(repr(cexecution_error))
                return 1
        try:
            db = client[self._config['mongoDB']['value']]
            collection = db[storage_json['value']]
            model_id = ar_json['model_parameters'][get_model_fw(ar_json)]['parameters']['model_id']['value']
            filter_cond = "model_parameters." + get_model_fw(ar_json) + ".parameters.model_id.value"
            cond =[{filter_cond: model_id}, {"type": ar_json['type']},
                   {"model_id": ar_json['model_id']},  {"timestamp": ar_json['timestamp']}]
            query = {"$and": cond}

            count = collection.find(query).count()
            new_ar_json = deepcopy(ar_json)
            if count == 1:
                collection.delete_one(query)
                collection.insert(new_ar_json, check_keys=False)
                return 0
            elif count == 0:
                collection.insert(new_ar_json, check_keys=False)
                return 0
            else:
                print("Trace: Duplicate Model %s" % model_id)
                return 1
        finally:
            if remove_client:
                client.close()

    ## Method used to recover an experiment as [ar_metadata]
    # Not implemented yet!
    # oriented to store full Analysis_results json but useful on whole json
    # @param self object pointer
    # @param user user_id
    # @param client Cliente MongoClient()
    # @return [ArMetadata]
    def recover_experiment_mongoDB(self, analysis_id, user, client=None):
        execution_list = list()
        remove_client = False
        if client is None or not isinstance(client(MongoClient)):
            try:
                client = MongoClient(host=self._config['mongoDB']['url'],
                                     port=int(self._config['mongoDB']['port']),
                                     document_class=OrderedDict)
                remove_client = True
            except ConnectionFailure as cexecution_error:
                print(repr(cexecution_error))
                return execution_list
        try:
            db = client[self._config['mongoDB']['value']]
            collection = db[user]
            query = {"$and": [{"model_id": analysis_id}, {"type": "train"}]}
            for element in collection.find(query):
                execution_list.append(element)
            for element in execution_list:
                element.pop('_id')
            print(execution_list)
        except PyMongoError as pexecution_error:
            print(repr(pexecution_error))
        finally:
            if remove_client:
                client.close()
            return deepcopy(execution_list)

    ## Method used to check and make directory os similar path structures
    # on all persistence system ['localfs', ' hdfs', ' mongoDB'] over agnostic way
    # @param self object pointer
    # @param type ['localfs', ' hdfs', ' mongoDB']
    # @param path directory or persistence structure to be created
    # @param grants on a 0o#### format (octalpython format)
    # @return operation status (0 success) (1 error)
    def mkdir(self, type, path, grants):
        if type == 'localfs':
            return self._mkdir_localfs(path=path, grants=grants)
        elif type == 'hdfs':
            return self._mkdir_hdfs(path=path, grants=grants)
        elif type == 'mongoDB':
            return self._mkdir_mongoDB(path=path, grants=grants)

    ## Static protected  method used to check and make directory
    # on ['localfs']
    # @param self object pointer
    # @param path directory or persistence structure to be created
    # @param grants on a 0o#### format (octalpython format)
    # @return operation status (0 success) (1 error)
    @staticmethod
    def _mkdir_localfs(path, grants):
        try:
            Path(path).mkdir(mode=grants, parents=True, exist_ok=True)
            return 0
        except IOError:
            return 1

    ## Static protected method used to check and make directory
    # on ['hdfs']
    # Not implemented yet!
    # @param self object pointer
    # @param path directory or persistence structure to be created
    # @param grants on a 0o#### format (octalpython format)
    # @param client Cliente HDFS
    # @return operation status (0 success) (1 error)
    def _mkdir_hdfs(self, path, grants, client=None):
        remove_client = False
        if client is None:
            client = Client(url=self._config['hdfs']['url'])
            remove_client = True
        try:
            if client.status(hdfs_path=path, strict=False) is None:
                client.makedirs(hdfs_path=path, permission=grants)
            return 0
        except HdfsError as hexecution_error:
            print(repr(hexecution_error))
            return 1
        except IOError as iexecution_error:
            print(repr(iexecution_error))
            return 1
        except OSError as oexecution_error:
            print(repr(oexecution_error))
            return 1
        finally:
            if remove_client:
                del client

    ## Static protected method used to check and make directory
    # on ['mongoDB']
    # Not implemented yet!
    # @param self object pointer
    # @param path directory or persistence structure to be created
    # @param grants on a 0o#### format (octalpython format)
    # @return operation status (0 success) (1 error)
    @staticmethod
    def _mkdir_mongoDB(path, grants):
        try:
            return 0
        except IOError:
            return 1

## Function base to get an ArMetadata Structure from file
# @param path FilePath
# @return operation status (0 success /1 error, ArMetadata/None)
def get_ar_from_file(path):
    if ospath.exists(path):
        _, type = mimetypes.guess_type(path)
        if type == 'gzip':
            file = gzip.GzipFile(path, 'r')
            json_bytes = file.read()
            json_str = json_bytes.decode('utf-8')
            ar_metadata = loads(json_str, object_hook=OrderedDict)
        else:
            file = open(path, 'r')
            ar_metadata = load(file, object_hook=OrderedDict)
        file.close()
        return 0, ar_metadata
    else:
        return 1, None
