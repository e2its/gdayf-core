import logging
import logging.config
import os
from json import load


__name__ = 'logHandler'

class LogsHandler (object):

    def __init__(self, log_handler_, module_,  driver, session,
                 filename=r'D:\e2its-dayf.svn\gdayf\branches\0.0.3-team03\src\gdayf\conf\logging.json'):
        self._filename = filename
        self._module = module_
        self._driver = driver
        self._session = session
        self._logger = logging.getLogger(log_handler_)
        if os.path.exists(filename):
            with open(filename, 'rt') as f:
                config = load(f)
            logging.config.dictConfig(config['logging'])
        else:
            logging.basicConfig(level='ERROR')

    def _compose_log_record(self, message):
        record = list()
        record.append('\t')
        record.append(self._driver)
        record.append('\t')
        record.append(self._session)
        record.append('\t')
        record.append('\"')
        record.append(message)
        record.append('\"')
        return ''.join(record)

    def log_info(self, message):
        self._logger.info(self._compose_log_record(message))

    def log_failure(self, message):
        self._logger.fatal(self._compose_log_record(message))

    def log_exec(self, message):
        self._logger.info(self._compose_log_record(message))

    def log_warning(self, message):
        self._logger.warning(self._compose_log_record(message))

    def log_error(self, message):
        self._logger.error(self._compose_log_record(message))

    def log_debug(self, message):
        self._logger.debug(self._compose_log_record(message))



if __name__ == "logHandler":

    logging = LogsHandler('dayf_logger', __name__, 'logging', '234567')
    logging.log_error('Prueba')
    logging.log_info('Prueba')
    logging.log_warning('Prueba')
    logging.log_debug('Prueba')
    logging.log_failure('Prueba')
