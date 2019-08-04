## @package gdayf.logs.logshandler
# Define all objects, functions and structures related to logging event on DayF product logs
#
# Main class LogsHandler

'''
 * This file is part of the gDayF AutoML Core Framework project
 * distribution (https://github.com/e2its/gdayf-core).
 * Copyright (c) 2016-2019 Jose Luis Sanchez del Coso <e2its.es@gmail.com>.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 ** Written by Jose L. Sanchez <e2its.es@gmail.com>, 2016-2019
'''

import logging
import logging.config
from gdayf.conf.loadconfig import LoadConfig

## Class oriented to manage all messages and interaction with DayF product logs
# @param e_c context pointer
# @param module __name__
class LogsHandler (object):
    ## Constructor
    def __init__(self, e_c, module=__name__):
        # @var _config
        # protected variable for loading and store DayF whole configuration parameters
        self._ec = e_c
        self._conf = self._ec.config.get_config()['logging']
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
    def _compose_log_record(trigger, session, message, add_message=''):
        record = list()
        record.append('\t')
        record.append(trigger)
        record.append('\t')
        record.append(session)
        record.append('\t')
        record.append(message)
        record.append('\t')
        record.append(str(add_message))
        return ''.join(record)

    ## Method for INFO events
    # @param trigger usually Analysis_id who launch the event log activity
    # @param session system session_id
    # @param message Text to log
    # @param add_message Text for additional message
    # @return None (event logging)
    def log_info(self, trigger, session, message, add_message=''):
        print(self._compose_log_record(trigger, session, message, add_message))
        self.logger.info(self._compose_log_record(trigger, session, message, add_message))

    ## Method for CRITICAL events
    # @param trigger usually Analysis_id who launch the event log activity
    # @param session system session_id
    # @param message Text to log
    # @param add_message Text for additional message
    # @return None (event logging)
    def log_critical(self, trigger, session, message, add_message=''):
        print(self._compose_log_record(trigger, session, message, add_message))
        self.logger.critical(self._compose_log_record(trigger, session, message, add_message))

    ## Method for EXECUTION (INFO EQUIVALENT) events
    # @param trigger usually Analysis_id who launch the event log activity
    # @param session system session_id
    # @param message Text to lognfo
    # @param add_message Text for additional message
    # @return None (event logging)
    def log_exec(self, trigger, session, message, add_message=''):
        self.logger.info(self._compose_log_record(trigger, session, message, add_message))

    ## Method for WARNING events
    # @param trigger usually Analysis_id who launch the event log activity
    # @param session system session_id
    # @param message Text to log
    # @param add_message Text for additional message
    # @return None (event logging)
    def log_warning(self, trigger, session, message, add_message=''):
        self.logger.warning(self._compose_log_record(trigger, session, message, add_message))

    ## Method for ERROR) events
    # @param trigger usually Analysis_id who launch the event log activity
    # @param session system session_id
    # @param message Text to log
    # @param add_message Text for additional message
    # @return None (event logging)
    def log_error(self, trigger, session, message, add_message=''):
        self.logger.error(self._compose_log_record(trigger, session, message))

    ## Method for DEBUG events
    # @param trigger usually Analysis_id who launch the event log activity
    # @param session system session_id
    # @param message Text to log
    # @param add_message Text for additional message
    # @return None (event logging)
    def log_debug(self, trigger, session, message, add_message=''):
        self.logger.debug(self._compose_log_record(trigger, session, message, add_message))
