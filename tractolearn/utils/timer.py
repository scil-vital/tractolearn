# -*- coding: utf-8 -*-
import logging
import time

logger = logging.getLogger("root")


class Timer:
    """Timer class to estimate time in a with statement."""

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info(f"Execution time: {time.time() - self.start}")
