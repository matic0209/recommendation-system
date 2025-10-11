"""
Prometheus-friendly metrics endpoint for Airflow health information.

The blueprint exposes `/metrics`, translating the standard Airflow health JSON
into gauge metrics so Prometheus can scrape service liveness without depending
on external plugins.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, Tuple

from flask import Blueprint, Response

from airflow.api_connexion.endpoints import health_endpoint
from airflow.plugins_manager import AirflowPlugin

BLUEPRINT = Blueprint("prometheus_health_metrics", __name__)


def _coerce_status(value: str | None) -> int:
    """Convert textual health status into numeric gauge values."""
    if value == "healthy":
        return 1
    if value == "unhealthy":
        return 0
    return -1


def _iter_heartbeats(health_payload: Dict[str, Dict[str, object]]) -> Iterable[Tuple[str, str]]:
    """Yield `(component, timestamp)` pairs for heartbeat fields present in the payload."""
    heartbeat_keys = {
        "scheduler": "latest_scheduler_heartbeat",
        "triggerer": "latest_triggerer_heartbeat",
        "dag_processor": "latest_dag_processor_heartbeat",
    }
    for component, key in heartbeat_keys.items():
        ts = health_payload.get(component, {}).get(key)
        if isinstance(ts, str):
            yield component, ts


@BLUEPRINT.route("/metrics", methods=["GET"])
def prometheus_metrics() -> Response:
    """Expose gauges derived from the Airflow health endpoint for Prometheus scraping."""
    health = health_endpoint.get_health()  # returns dict

    lines = [
        "# HELP airflow_component_health Airflow component health (-1 unknown, 0 unhealthy, 1 healthy)",
        "# TYPE airflow_component_health gauge",
    ]
    for component, payload in health.items():
        status = _coerce_status(payload.get("status"))
        lines.append(f'airflow_component_health{{component="{component}"}} {status}')

    lines.append("# HELP airflow_component_last_heartbeat Latest heartbeat timestamp as unix epoch seconds")
    lines.append("# TYPE airflow_component_last_heartbeat gauge")
    for component, timestamp in _iter_heartbeats(health):
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            epoch = parsed.timestamp()
            lines.append(f'airflow_component_last_heartbeat{{component="{component}"}} {epoch}')
        except ValueError:
            lines.append(f'airflow_component_last_heartbeat{{component="{component}"}} -1')

    body = "\n".join(lines) + "\n"
    return Response(body, mimetype="text/plain; version=0.0.4; charset=utf-8")


class PrometheusHealthMetricsPlugin(AirflowPlugin):
    """Registers the blueprint with the Airflow webserver."""

    name = "prometheus_health_metrics"
    flask_blueprints = [BLUEPRINT]
