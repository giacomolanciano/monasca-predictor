"""
    Module containing general utilities
"""

import logging
import os
import sys
import traceback

from monasca_predictor.common.config import PredictorConfig

LOGGING_MAX_BYTES = 5 * 1024 * 1024

log = logging.getLogger(__name__)


def initialize_logging(logger_name):
    """
    Inspired by monasca_agent.common.util.initialize_logging()
    """
    try:
        log_format = (
            "%%(asctime)s | %%(levelname)s | %s | %%(name)s(%%(filename)s:%%(lineno)s) "
            "| %%(message)s" % logger_name
        )
        log_date_format = "%Y-%m-%d %H:%M:%S %Z"
        config = PredictorConfig()
        logging_config = config.get_config(sections="Logging")

        logging.basicConfig(
            format=log_format,
            level=logging_config["log_level"] or logging.INFO,
        )

        # set up file loggers
        log_file = logging_config.get("%s_log_file" % logger_name)
        if log_file is not None and not logging_config["disable_file_logging"]:
            # make sure the log directory is writable
            # NOTE: the entire directory needs to be writable so that rotation works
            if os.access(os.path.dirname(log_file), os.R_OK | os.W_OK):
                if logging_config["enable_logrotate"]:
                    file_handler = logging.handlers.RotatingFileHandler(
                        log_file, maxBytes=LOGGING_MAX_BYTES, backupCount=1
                    )
                else:
                    file_handler = logging.FileHandler(log_file)

                formatter = logging.Formatter(log_format, log_date_format)
                file_handler.setFormatter(formatter)

                root_log = logging.getLogger()
                root_log.addHandler(file_handler)
            else:
                sys.stderr.write("Log file is unwritable: '%s'\n" % log_file)

        # set up syslog
        if logging_config["log_to_syslog"]:
            try:
                syslog_format = (
                    "%s[%%(process)d]: %%(levelname)s (%%(filename)s:%%(lineno)s): "
                    "%%(message)s" % logger_name
                )

                if (
                    logging_config["syslog_host"] is not None
                    and logging_config["syslog_port"] is not None
                ):
                    sys_log_addr = (
                        logging_config["syslog_host"],
                        logging_config["syslog_port"],
                    )
                else:
                    sys_log_addr = "/dev/log"
                    # Special-case macs
                    if sys.platform == "darwin":
                        sys_log_addr = "/var/run/syslog"

                handler = logging.handlers.SysLogHandler(
                    address=sys_log_addr,
                    facility=logging.handlers.SysLogHandler.LOG_DAEMON,
                )
                handler.setFormatter(logging.Formatter(syslog_format, log_date_format))
                root_log = logging.getLogger()
                root_log.addHandler(handler)
            except Exception as err:  # pylint: disable=broad-except
                sys.stderr.write("Error setting up syslog: '%s'\n" % str(err))
                traceback.print_exc()

    except Exception as err:  # pylint: disable=broad-except
        sys.stderr.write("Couldn't initialize logging: %s\n" % str(err))
        traceback.print_exc()

        # if config fails entirely, enable basic stdout logging as a fallback
        logging.basicConfig(
            format=log_format,
            level=logging.INFO,
        )

    # re-get the log after logging is initialized
    global log
    log = logging.getLogger(__name__)
    log.propagate = False