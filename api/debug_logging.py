"""Debug logging configuration.

Adds or removes a compound rotating file handler on the root logger.

When enabled, all DEBUG (and above) log records are written to
``data/logs/stash_sense_debug.log``.  The file rotates when it exceeds
5 MB *or* at midnight, keeping at most 10 backup files.

The console/uvicorn stream handlers are explicitly held at WARNING so
that enabling this setting does not spam stdout.
"""

from __future__ import annotations

import datetime
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 10
_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)-40s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Held so we can remove it later
_debug_handler: Optional[logging.Handler] = None


class _CompoundRotatingHandler(RotatingFileHandler):
    """Rotates on size (5 MB) **or** on midnight, whichever comes first."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_date = datetime.date.today()

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        today = datetime.date.today()
        if today != self._last_date:
            self._last_date = today
            return True
        return super().shouldRollover(record)


def configure_debug_logging(enabled: bool, data_dir: Path) -> None:
    """Add or remove the debug file handler on the root logger.

    Safe to call multiple times — removes any previously added handler
    before adding a new one.

    Args:
        enabled: True to activate, False to deactivate.
        data_dir: The sidecar data directory (``DATA_DIR`` env var path).
                  Logs are written to ``data_dir/logs/``.
    """
    global _debug_handler

    root = logging.getLogger()

    # Remove any previously installed debug handler
    if _debug_handler is not None:
        root.removeHandler(_debug_handler)
        try:
            _debug_handler.close()
        except Exception:
            pass
        _debug_handler = None

    if not enabled:
        # Restore root level so non-file handlers stop receiving debug records
        root.setLevel(logging.WARNING)
        logger.warning("Debug logging disabled")
        return

    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "stash_sense_debug.log"

    handler = _CompoundRotatingHandler(
        filename=str(log_path),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
        delay=False,
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    # Clamp all existing console/stream handlers to WARNING so that
    # lowering the root level below WARNING doesn't flood stdout.
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler):
            if h.level == logging.NOTSET or h.level < logging.WARNING:
                h.setLevel(logging.WARNING)

    root.addHandler(handler)
    # Root level must be DEBUG so that DEBUG records are not discarded
    # before reaching the file handler.
    root.setLevel(logging.DEBUG)

    _debug_handler = handler
    logger.warning(
        "Debug logging enabled → %s (max %d MB, %d backups)",
        log_path, _MAX_BYTES // (1024 * 1024), _BACKUP_COUNT,
    )
