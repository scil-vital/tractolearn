# -*- coding: utf-8 -*-

import logging
import logging.config
import os


def _create_rotating_file_handler_from_filename(
    log_fname, max_bytes, encoding, name, level, formatter
):

    new_hdlr = logging.handlers.RotatingFileHandler(
        log_fname, maxBytes=max_bytes, encoding=encoding
    )
    new_hdlr.name = name
    new_hdlr.level = level
    new_hdlr.formatter = formatter

    return new_hdlr


def set_up(log_fname):

    logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    level = logging.DEBUG
    max_bytes = 10485760  # 10MB
    # backup_count = 20
    encoding = "utf8"
    hdlr_name = "file_handler"

    formatter = logging.Formatter(fmt=logging_format)

    logger = logging.getLogger("root")
    # ToDo
    # This should not be necessary if the level is set to the file handler
    logger.setLevel(level)

    if log_fname is not None:
        log_fname = os.path.abspath(log_fname)
        new_hdlr = _create_rotating_file_handler_from_filename(
            log_fname, max_bytes, encoding, hdlr_name, level, formatter
        )
        logger.addHandler(new_hdlr)
