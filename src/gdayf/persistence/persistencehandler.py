from os import path
from os import makedirs as mkdir
from shutil import copyfile

class PersistenceHandler(object):
    def __init__(self):
         None

    @staticmethod
    def replicate_file(type_dest, path_dest, type_source, path_source):

        if type_source == 'localfs':
            if type_dest == 'localfs':
                if not path.exists(path.dirname(path_dest)):
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
