"""Structured logging configuration for recommendation service."""
import logging
import sys
from typing import Any, Dict

import structlog


def setup_structured_logging(
    level: str = "INFO",
    json_logs: bool = True,
    service_name: str = "recommendation-api",
) -> None:
    """
    Configure structured logging with structlog.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Whether to output JSON formatted logs
        service_name: Name of the service for log context
    """
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Add service metadata
    structlog.contextvars.bind_contextvars(
        service=service_name,
        environment="production",
    )

    if json_logs:
        # JSON output for production
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Pretty console output for development
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Optional logger name

    Returns:
        Structured logger
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


# Request ID middleware helper
def add_request_id(request_id: str) -> None:
    """Add request ID to logging context."""
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request_id() -> None:
    """Clear request ID from logging context."""
    structlog.contextvars.unbind_contextvars("request_id")
