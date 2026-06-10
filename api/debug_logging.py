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
import re
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

# Patterns that match name/path fields followed by their numeric ID.
# Each tuple is (compiled_pattern, replacement_template).
# The replacement keeps the ID but drops the human-readable name/path.
_ANON_PATTERNS: list[tuple[re.Pattern, str]] = [
    # 'Scene Title' (id 123) → scene_id=123
    (re.compile(r"'[^']{1,200}'\s*\((?:scene[_ ]?)?id[=:\s]+(\d+)\)", re.IGNORECASE), r"scene_id=\1"),
    # performer 'Name' (id 123) → performer_id=123
    (re.compile(r"performer\s+'[^']{1,200}'\s*(?:\(id[=:\s]+(\d+)\))?", re.IGNORECASE), r"performer_id=\1"),
    # studio 'Name' → studio_id=NNN  (name without id — just redact name)
    (re.compile(r"studio\s+'[^']{1,200}'", re.IGNORECASE), r"studio '<redacted>'"),
    # file paths: /some/path/file.ext → <path redacted>
    (re.compile(r"/[^\s,;\"']{4,200}\.\w{2,5}\b"), r"<file redacted>"),
    # Windows paths
    (re.compile(r"[A-Za-z]:\\[^\s,;\"']{4,200}\.\w{2,5}\b"), r"<file redacted>"),
]


class _AnonymizingFormatter(logging.Formatter):
    """Formatter that strips PII (names, paths) from log messages."""

    def format(self, record: logging.LogRecord) -> str:
        original = record.getMessage()
        cleaned = original
        for pattern, replacement in _ANON_PATTERNS:
            cleaned = pattern.sub(replacement, cleaned)
        if cleaned != original:
            record = logging.makeLogRecord(record.__dict__)
            record.msg = cleaned
            record.args = ()
        return super().format(record)


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


def configure_debug_logging(
    enabled: bool,
    data_dir: Path,
    anonymize: bool = False,
    version: Optional[str] = None,
    plugin_version: Optional[str] = None,
) -> None:
    """Add or remove the debug file handler on the root logger.

    Safe to call multiple times — removes any previously added handler
    before adding a new one.

    Args:
        enabled: True to activate, False to deactivate.
        data_dir: The sidecar data directory (``DATA_DIR`` env var path).
                  Logs are written to ``data_dir/logs/``.
        anonymize: If True, apply PII redaction (names, paths → IDs only).
        version: Optional sidecar version string for the startup log entry.
        plugin_version: Optional plugin version string for the startup log entry.
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
        # Reset the noise-suppression overrides applied below
        logging.getLogger("httpcore").setLevel(logging.NOTSET)
        logging.getLogger("httpx").setLevel(logging.NOTSET)
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
    fmt_cls = _AnonymizingFormatter if anonymize else logging.Formatter
    handler.setFormatter(fmt_cls(_LOG_FORMAT, datefmt=_DATE_FORMAT))

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

    # httpcore emits ~15 DEBUG records per HTTP request (TCP connect, TLS
    # handshake, send/receive headers and body, connection close) that add
    # no diagnostic value beyond httpx's own one-line request/response
    # summary. Silence them; httpx's "HTTP Request: ..." summary stays at
    # INFO so the endpoint URL and outcome are still logged.
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.INFO)

    _debug_handler = handler
    version_parts = []
    if version:
        version_parts.append(f"sidecar v{version}")
    if plugin_version:
        version_parts.append(f"plugin v{plugin_version}")
    version_str = f" ({', '.join(version_parts)})" if version_parts else ""
    logger.warning(
        "Debug logging enabled%s → %s (max %d MB, %d backups, anonymize=%s)",
        version_str, log_path, _MAX_BYTES // (1024 * 1024), _BACKUP_COUNT, anonymize,
    )
