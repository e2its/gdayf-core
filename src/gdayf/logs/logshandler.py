## @package gdayf.logs.logshandler
# Define all objects, functions and structures related to logging event on DayF product logs
#
# Main class LogsHandler

import logging
import logging.config
from gdayf.conf.loadconfig import LoadConfig


## Class oriented to manage all messages and interaction with DayF product logs
class LogsHandler (object):
    ## Constructor
    def __init__(self, module=__name__):
        print(__name__)
        # @var _config
        # protected variable for loading and store DayF whole configuration parameters
        self._conf = LoadConfig().get_config()['logging']
        # @var logger
        # variable for setting log global handlers
        self.logger = logging.getLogger(module)
        self.logger.setLevel('DEBUG')
        logging.config.dictConfig(dict(self._conf))

    ## Static protected method for composing messages
    # @param trigger usually Analysis_id who launch the event log activity
    # @param session system session_id
    # @param message Text to log
    # @return String event log formatted message
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

    ## Method for INFO events
    # @param trigger usually Analysis_id who launch the event log activity
    # @param session system session_id
    # @param message Text to log
    # @return None (event logging)
    def log_info(self, trigger, session, message):
        print(message)
        self.logger.info(self._compose_log_record(trigger, session, message))

    ## Method for CRITICAL events
    # @param trigger usually Analysis_id who launch the event log activity
    # @param session system session_id
    # @param message Text to log
    # @return None (event logging)
    def log_critical(self, trigger, session, message):
        self.logger.critical(self._compose_log_record(trigger, session, message))

    ## Method for EXECUTION (INFO EQUIVALENT) events
    # @param trigger usually Analysis_id who launch the event log activity
    # @param session system session_id
    # @param message Text to log
    # @return None (event logging)
    def log_exec(self, trigger, session, message):
        self.logger.info(self._compose_log_record(trigger, session, message))

    ## Method for WARNING events
    # @param trigger usually Analysis_id who launch the event log activity
    # @param session system session_id
    # @param message Text to log
    # @return None (event logging)
    def log_warning(self, trigger, session, message):
        self.logger.warning(self._compose_log_record(trigger, session, message))

    ## Method for ERROR) events
    # @param trigger usually Analysis_id who launch the event log activity
    # @param session system session_id
    # @param message Text to log
    # @return None (event logging)
    def log_error(self, trigger, session, message):
        self.logger.error(self._compose_log_record(trigger, session, message))

    ## Method for DEBUG events
    # @param trigger usually Analysis_id who launch the event log activity
    # @param session system session_id
    # @param message Text to log
    # @return None (event logging)
    def log_debug(self, trigger, session, message):
        self.logger.debug(self._compose_log_record(trigger, session, message))

if __name__ == "logs.logshandler":
    logging = LogsHandler(__name__)
    logging.log_debug('logging', '234567', 'Prueba')
    logging.log_info('logging', '234567','Prueba')
    logging.log_exec('logging', '234567','Prueba')
    logging.log_warning('logging', '234567','Prueba')
    logging.log_error('logging', '234567','Prueba')
    logging.log_critical('logging', '234567','Prueba')