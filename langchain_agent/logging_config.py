"""
Structured logging configuration using structlog.

Provides:
- configure_logging(): Setup structlog with JSON/console output
- get_logger(name): Get a bound logger for a module
- LogContext: Context manager for adding temporary log context
"""

import logging
import sys
from typing import Any, Dict, Optional
from contextvars import ContextVar

import structlog
from structlog.types import Processor

from config import LOG_LEVEL, LOG_FORMAT, LOG_INCLUDE_TIMESTAMP


# Context variable for request-scoped logging context
_log_context: ContextVar[Dict[str, Any]] = ContextVar("log_context", default={})


def configure_logging() -> None:
    """
    Configure structlog for the application.

    Uses LOG_FORMAT from config to determine output format:
    - "json": JSON output for production (machine-readable)
    - "console": Human-readable console output for development

    Uses LOG_LEVEL from config to set logging verbosity.
    """
    # Shared processors for all output formats
    shared_processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.contextvars.merge_contextvars,
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if LOG_INCLUDE_TIMESTAMP:
        shared_processors.insert(0, structlog.processors.TimeStamper(fmt="iso"))

    # Format-specific renderer
    if LOG_FORMAT == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        # Console format with colors for development
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging to use structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Setup root handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Set levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        BoundLogger instance with structured logging capabilities
    """
    return structlog.get_logger(name)


class LogContext:
    """
    Context manager for adding temporary context to logs.

    Example:
        with LogContext(request_id="abc123", user_id="user1"):
            logger.info("Processing request")
            # Logs will include request_id and user_id
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize log context with key-value pairs.

        Args:
            **kwargs: Context variables to add to logs
        """
        self.context = kwargs
        self.token: Optional[Any] = None

    def __enter__(self) -> "LogContext":
        """Enter the context and bind variables."""
        structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context and unbind variables."""
        structlog.contextvars.unbind_contextvars(*self.context.keys())


def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables to all subsequent logs in this context.

    Args:
        **kwargs: Context variables to bind

    Example:
        bind_context(thread_id="abc123")
        logger.info("Message with thread_id")
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()
