"""Console logging with a consistent format across the pipeline."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FMT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"


def setup_logging(level: int | str = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    for noisy in ("matplotlib", "numexpr", "urllib3", "yfinance", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
