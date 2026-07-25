"""
logger.py
=========

Shared loguru-based logging for QATCH-ML, usable from any module under
``src/`` (systems and utils alike) in place of ad hoc ``logging.getLogger``
calls or the various hand-rolled ``Log.d/i/w/e`` fallback shims scattered
across the codebase for QATCH-app-optional headless operation.

Usage
-----
    from src.utils.logger import logger
    logger.info("message")

For a module-scoped logger carrying a fixed tag (mirrors the ``TAG =
"[Name]"`` convention already used across qmodel_7_onyx modules)::

    from src.utils.logger import get_logger
    log = get_logger("dataset_fetcher")
    log.info("copied {n} files", n=42)

Configuration
-------------
A colorized console sink is installed at import time, at level
``QATCH_LOG_LEVEL`` (environment variable, default ``INFO``). Call
:func:`configure_logging` explicitly — typically once from a CLI's
``main()`` — to change the level and/or add a rotating file sink under
``log_dir``. It is safe to call more than once: existing sinks are removed
first, so re-configuring never duplicates log lines.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[tag]}</cyan> - <level>{message}</level>"
)

_configured = False


def configure_logging(
    level: Optional[str] = None,
    log_dir: Optional[Path] = None,
    log_file: str = "qatch-ml.log",
    rotation: str = "10 MB",
    retention: str = "14 days",
    serialize: bool = False,
) -> None:
    """(Re)configure the shared logger: one colorized console sink, plus an
    optional rotating file sink under ``log_dir``.

    level: log level name; falls back to the QATCH_LOG_LEVEL env var, then "INFO".
    log_dir: if given, also write a rotating file sink here (created if missing).
    log_file: filename within log_dir.
    rotation / retention: passed straight through to loguru's file sink.
    serialize: emit the file sink as JSON lines instead of formatted text.
    """
    global _configured
    logger.remove()
    resolved_level = (level or os.environ.get("QATCH_LOG_LEVEL", "INFO")).upper()
    logger.configure(extra={"tag": "qatch-ml"})
    logger.add(sys.stderr, level=resolved_level, format=_CONSOLE_FORMAT, colorize=True)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / log_file,
            level=resolved_level,
            rotation=rotation,
            retention=retention,
            serialize=serialize,
            encoding="utf-8",
        )
    _configured = True


def get_logger(tag: str):
    """A logger bound with a fixed ``tag`` field, shown in place of the
    default "qatch-ml" tag in the console/file format."""
    if not _configured:
        configure_logging()
    return logger.bind(tag=tag)


configure_logging()

__all__ = ["logger", "configure_logging", "get_logger"]
