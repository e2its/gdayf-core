import logging
import logging.config
import os
from gdayf.conf.loadconfig import LoadConfig
__name__ = 'logs'


class LogsHandler (object):

    def __init__(self, module_name):
        self._conf = LoadConfig().get_config()
        self.logger = logging.getLogger(module_name)
        self.logger.setLevel('DEBUG')
        logging.config.dictConfig(self._conf['logging'])

    def __del__(self):
        logging.shutdown()

    @staticmethod
    def _compose_log_record(trigger, session, message):
        record = list()
        record.append('\t')
        record.append(trigger)
        record.append('\t')
        record.append(session)
        record.append('\t')
        record.append('\"')
        record.append(message)
        record.append('\"')
        return ''.join(record)

    def log_info(self, trigger, session, message):
        self.logger.info(self._compose_log_record(trigger, session, message))

    def log_critical(self, trigger, session, message):
        self.logger.critical(self._compose_log_record(trigger, session, message))

    def log_exec(self, trigger, session, message):
        self.logger.info(self._compose_log_record(trigger, session, message))

    def log_warning(self, trigger, session, message):
        self.logger.warning(self._compose_log_record(trigger, session, message))

    def log_error(self, trigger, session, message):
        self.logger.error(self._compose_log_record(trigger, session, message))

    def log_debug(self, trigger, session, message):
        self.logger.debug(self._compose_log_record(trigger, session, message))
'''
if __name__ == "logs.logHandler":

    logging = LogsHandler()
    logging.log_debug('logging', '234567','Prueba')
    logging.log_info('logging', '234567','Prueba')
    logging.log_exec('logging', '234567','Prueba')
    logging.log_warning('logging', '234567','Prueba')
    logging.log_error('logging', '234567','Prueba')
    logging.log_critical('logging', '234567','Prueba')
'''
