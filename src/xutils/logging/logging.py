import logging
import os
import sys

loggers = {}


def get_logger(file, level=logging.INFO):
    name = str(os.path.basename(file)).rjust(15)
    if name in loggers:
        return loggers[name]

    logger = logging.getLogger(name)
    log_formatter = logging.Formatter(f'%(asctime)s %(levelname)s %(name)s: %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(level)

    loggers[name] = logger

    return logger
