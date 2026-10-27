"""Process-wide logging configuration.

Every entrypoint (REST, CLI, gateways) calls :func:`configure_logging` once so
diagnostics go through the standard library instead of bare ``print`` calls.
``LOG_LEVEL`` selects the level; the default keeps third-party HTTP logs at
WARNING so a single request does not flood the console.
"""

from __future__ import annotations

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_configured = False


def configure_logging(level: str | int | None = None) -> None:
    """Install a stdout handler on the root logger (idempotent)."""
    global _configured
    if _configured:
        return

    resolved = level if level is not None else os.getenv("LOG_LEVEL", "INFO")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root = logging.getLogger()
    root.setLevel(resolved)
    root.addHandler(handler)
    # Uvicorn installs its own handlers; keep its request log at INFO but
    # silence the very chatty per-request HTTP client logs by default.
    for noisy in ("httpx", "httpcore", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring logging on first use."""
    configure_logging()
    return logging.getLogger(name)
