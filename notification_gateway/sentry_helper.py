"""Lightweight Sentry helpers for the notification gateway service."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

LOGGER = logging.getLogger(__name__)
_ENABLED = False


def init_sentry(
    *,
    service_name: str = "notification-gateway",
    traces_sample_rate: float = 0.1,
    profiles_sample_rate: float = 0.0,
) -> bool:
    """Initialise Sentry for the notification gateway service."""
    global _ENABLED

    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        LOGGER.warning("SENTRY_DSN not set, skip Sentry initialisation")
        return False

    environment = os.getenv("SENTRY_ENVIRONMENT", "production")
    traces_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", str(traces_sample_rate)))
    profiles_rate = float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", str(profiles_sample_rate)))

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        traces_sample_rate=traces_rate,
        profiles_sample_rate=profiles_rate,
        integrations=[
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
            FlaskIntegration(transaction_style="endpoint"),
        ],
        send_default_pii=True,
    )
    sentry_sdk.set_tag("service", service_name)

    LOGGER.info(
        "Sentry initialised (service=%s, env=%s, traces_rate=%.2f, profiles_rate=%.2f)",
        service_name,
        environment,
        traces_rate,
        profiles_rate,
    )
    _ENABLED = True
    return True


def _apply_context(scope: sentry_sdk.Scope, **context: Any) -> None:
    for key, value in context.items():
        scope.set_extra(key, value)


def capture_message_with_context(
    message: str,
    *,
    level: str = "info",
    **context: Any,
) -> Optional[str]:
    """Capture a message with optional contextual data."""
    if not _ENABLED:
        return None

    with sentry_sdk.push_scope() as scope:
        _apply_context(scope, **context)
        return sentry_sdk.capture_message(message, level=level)


def capture_exception_with_context(
    error: BaseException,
    *,
    level: str = "error",
    fingerprint: Optional[list[str]] = None,
    **context: Any,
) -> Optional[str]:
    """Capture an exception with optional contextual data."""
    if not _ENABLED:
        return None

    with sentry_sdk.push_scope() as scope:
        if fingerprint:
            scope.fingerprint = fingerprint
        _apply_context(scope, **context)
        return sentry_sdk.capture_exception(error)
