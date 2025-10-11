#!/usr/bin/env python3
"""
Automate end-to-end model release:
- Trigger Airflow DAG run for the offline pipeline.
- Poll until completion/failure.
- Call recommendation API to reload latest model bundle.

Environment variables:
    AIRFLOW_BASE_URL      (e.g. https://airflow.example.com/api/v1)
    AIRFLOW_DAG_ID        (default: recommendation_pipeline)
    AIRFLOW_USERNAME
    AIRFLOW_PASSWORD
    AIRFLOW_TIMEOUT       (seconds, default 3600)
    AIRFLOW_POLL_INTERVAL (seconds, default 30)
    MODEL_RELOAD_URL      (e.g. https://rec-api.example.com/models/reload)
    MODEL_RELOAD_TOKEN    (optional bearer token)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("automate_release")


def _get_env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def trigger_airflow_dag(
    base_url: str,
    dag_id: str,
    auth: HTTPBasicAuth,
    conf: Optional[dict] = None,
) -> str:
    url = f"{base_url}/dags/{dag_id}/dagRuns"
    payload = {"conf": conf or {}, "note": "Automated release pipeline"}
    LOGGER.info("Triggering Airflow DAG %s via %s", dag_id, url)
    response = requests.post(url, auth=auth, json=payload, timeout=30)
    if response.status_code >= 400:
        LOGGER.error("Failed to trigger DAG (%s): %s", response.status_code, response.text)
        response.raise_for_status()
    dag_run_id = response.json().get("dag_run_id")
    if not dag_run_id:
        raise RuntimeError(f"Airflow response missing dag_run_id: {response.text}")
    LOGGER.info("Triggered DAG run %s", dag_run_id)
    return dag_run_id


def poll_airflow_dag(
    base_url: str,
    dag_id: str,
    dag_run_id: str,
    auth: HTTPBasicAuth,
    timeout: int,
    poll_interval: int,
) -> None:
    url = f"{base_url}/dags/{dag_id}/dagRuns/{dag_run_id}"
    deadline = time.time() + timeout
    LOGGER.info("Polling DAG run %s until completion...", dag_run_id)
    while time.time() < deadline:
        response = requests.get(url, auth=auth, timeout=30)
        if response.status_code >= 400:
            LOGGER.error("Failed to poll DAG run (%s): %s", response.status_code, response.text)
            response.raise_for_status()
        state = response.json().get("state", "").lower()
        LOGGER.info("Current DAG state: %s", state)
        if state in {"success"}:
            LOGGER.info("Airflow DAG run succeeded")
            return
        if state in {"failed", "error"}:
            raise RuntimeError(f"Airflow DAG run failed with state={state}")
        time.sleep(poll_interval)
    raise TimeoutError(f"Airflow DAG run {dag_run_id} did not finish within {timeout} seconds")


def trigger_model_reload(url: str, token: Optional[str] = None) -> None:
    LOGGER.info("Calling model reload endpoint: %s", url)
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.post(url, json={"mode": "primary"}, headers=headers, timeout=30)
    if response.status_code >= 400:
        LOGGER.error("Model reload failed (%s): %s", response.status_code, response.text)
        response.raise_for_status()
    LOGGER.info("Model reload triggered successfully")


def main() -> None:
    airflow_base = _get_env("AIRFLOW_BASE_URL", required=True).rstrip("/")
    dag_id = _get_env("AIRFLOW_DAG_ID", "recommendation_pipeline")
    airflow_user = _get_env("AIRFLOW_USERNAME", required=True)
    airflow_pass = _get_env("AIRFLOW_PASSWORD", required=True)
    airflow_timeout = int(_get_env("AIRFLOW_TIMEOUT", "3600"))
    poll_interval = int(_get_env("AIRFLOW_POLL_INTERVAL", "30"))
    reload_url = _get_env("MODEL_RELOAD_URL", required=True)
    reload_token = _get_env("MODEL_RELOAD_TOKEN")

    auth = HTTPBasicAuth(airflow_user, airflow_pass)

    try:
        dag_run_id = trigger_airflow_dag(airflow_base, dag_id, auth)
        poll_airflow_dag(airflow_base, dag_id, dag_run_id, auth, airflow_timeout, poll_interval)
        trigger_model_reload(reload_url, reload_token)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Automation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
