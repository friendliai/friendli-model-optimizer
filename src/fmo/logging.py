# Copyright (c) 2024-present, FriendliAI Inc. All rights reserved.

"""FMO Logger."""

from __future__ import annotations

import logging
import os


class ColorFormatter(logging.Formatter):
    """Customized formatter with ANSI color."""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    default_fmt = "%(asctime)s.%(msecs)05d: %(name)s %(levelname)s: %(message)s"
    default_datefmt = "%Y-%m-%d %H:%M:%S"

    FORMATS = {
        logging.DEBUG: grey + default_fmt + reset,
        logging.INFO: grey + default_fmt + reset,
        logging.WARNING: yellow + default_fmt + reset,
        logging.ERROR: red + default_fmt + reset,
        logging.CRITICAL: bold_red + default_fmt + reset,
    }

    def __init__(self):
        """Initialize CustomFormatter."""
        super().__init__(fmt=self.default_fmt, datefmt=self.default_datefmt)

        # Pre-create Formatter objects for each level to improve efficiency
        self.formatters = {
            level: logging.Formatter(fmt) for level, fmt in self.FORMATS.items()
        }

    def format(self, record):
        """Override format method."""
        formatter = self.formatters.get(record.levelno, self.formatters[logging.INFO])
        return formatter.format(record)


def get_logger(name: str) -> logging.Logger:
    """Get a formatted logger."""
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        for handler in logger.handlers:
            logger.removeHandler(handler)

    handler = logging.StreamHandler()
    formatter = ColorFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("FMO_LOG_LEVEL", "INFO"))

    return logger
