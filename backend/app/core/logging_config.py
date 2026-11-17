"""Structured JSON Logging Configuration for Production Observability."""
import json
import logging
import sys
import time
from typing import Any, Dict


import re

# Regex patterns matching API keys, tokens, and secrets
SENSITIVE_PATTERNS = [
    re.compile(r"gsk_[a-zA-Z0-9]{15,}", re.IGNORECASE),
    re.compile(r"AIza[0-9A-Za-z-_]{20,}", re.IGNORECASE),
    re.compile(r"sk-[a-zA-Z0-9_\-]{15,}", re.IGNORECASE),
    re.compile(r"hf_[a-zA-Z0-9]{15,}", re.IGNORECASE),
    re.compile(r"(Bearer\s+)[a-zA-Z0-9_\-\.]{15,}", re.IGNORECASE),
    re.compile(r"(api[-_]?key\s*[:=]\s*['\"]?)([a-zA-Z0-9_\-]{12,})(['\"]?)", re.IGNORECASE),
]

URI_PASSWORD_PATTERN = re.compile(r"(://[^:\s@]+:)([^@\s]+)(@)", re.IGNORECASE)


def sanitize_sensitive_data(text: str) -> str:
    """Redact sensitive API keys, database credentials, and tokens from strings."""
    if not isinstance(text, str):
        return text
    sanitized = text
    for pattern in SENSITIVE_PATTERNS:
        sanitized = pattern.sub("[REDACTED_API_KEY]", sanitized)
    sanitized = URI_PASSWORD_PATTERN.sub(r"\1[REDACTED_PASSWORD]\3", sanitized)
    return sanitized


class APIKeySanitizingFilter(logging.Filter):
    """Logging filter that scrubs sensitive API keys from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = sanitize_sensitive_data(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: sanitize_sensitive_data(v) if isinstance(v, str) else v for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(sanitize_sensitive_data(v) if isinstance(v, str) else v for v in record.args)
        return True


class JSONLogFormatter(logging.Formatter):
    """Formats log records as parseable, structured single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        raw_msg = record.getMessage()
        sanitized_msg = sanitize_sensitive_data(raw_msg)
        log_payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt) if self.datefmt else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitized_msg,
            "module": record.module,
            "lineno": record.lineno,
        }

        # Include structured metadata if passed via extra
        if hasattr(record, "trace_id"):
            log_payload["trace_id"] = record.trace_id
        if hasattr(record, "error_type"):
            log_payload["error_type"] = record.error_type
        if hasattr(record, "status_code"):
            log_payload["status_code"] = record.status_code
        if hasattr(record, "provider"):
            log_payload["provider"] = record.provider
        if hasattr(record, "latency_ms"):
            log_payload["latency_ms"] = record.latency_ms

        if record.exc_info:
            exc_str = self.formatException(record.exc_info)
            log_payload["exception"] = sanitize_sensitive_data(exc_str)

        return json.dumps(log_payload)


def configure_structured_logging(debug: bool = False) -> None:
    """Apply structured JSON logging to root logger."""
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(APIKeySanitizingFilter())
    handler.setFormatter(JSONLogFormatter())
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
