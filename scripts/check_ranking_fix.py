#!/usr/bin/env python3
"""Verify recommend/detail endpoint behaves after ranking fix."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import requests

ERROR_SIGNATURES = [
    "_apply_ranking() missing 1 required keyword-only argument: 'endpoint'",
]


def _call_recommend_detail(
    base_url: str,
    dataset_id: int,
    *,
    limit: int,
    user_id: Optional[int],
    timeout: float,
) -> dict:
    params = {"limit": limit}
    if user_id is not None:
        params["user_id"] = user_id
    url = f"{base_url.rstrip('/')}/recommend/detail/{dataset_id}"
    start = time.perf_counter()
    response = requests.get(url, params=params, timeout=timeout)
    latency = time.perf_counter() - start
    response.raise_for_status()
    payload = response.json()
    return {
        "dataset_id": dataset_id,
        "status": response.status_code,
        "latency_ms": round(latency * 1000, 2),
        "request_id": payload.get("request_id"),
        "variant": payload.get("variant"),
        "recommendation_count": len(payload.get("recommendations", [])),
    }


def _read_new_logs(path: Path, offset: int) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        handle.seek(offset)
        return handle.read()


def _has_error_signature(log_chunk: str) -> bool:
    return any(signature in log_chunk for signature in ERROR_SIGNATURES)


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check recommend/detail endpoint and ensure ranking error no longer appears.",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL of the running recommendation service (e.g. https://api.example.com)",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        action="append",
        dest="dataset_ids",
        required=True,
        help="Dataset ID to request. Repeat flag to test multiple datasets.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Recommendation limit to request (default: 30).",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="Optional user_id to include in requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds for each request (default: 10).",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional application log file to scan for ranking errors after the requests.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    dataset_ids = args.dataset_ids
    log_offset = None
    if args.log_path:
        try:
            log_offset = args.log_path.stat().st_size
        except FileNotFoundError:
            print(f"[WARN] Log file does not exist yet: {args.log_path}", file=sys.stderr)
            log_offset = 0

    results = []
    for dataset_id in dataset_ids:
        try:
            info = _call_recommend_detail(
                args.base_url,
                dataset_id,
                limit=args.limit,
                user_id=args.user_id,
                timeout=args.timeout,
            )
        except requests.RequestException as exc:
            print(f"[FAIL] dataset={dataset_id} request failed: {exc}", file=sys.stderr)
            return 2
        results.append(info)

    for info in results:
        print(
            f"[OK] dataset={info['dataset_id']} status={info['status']} "
            f"latency={info['latency_ms']}ms recommendations={info['recommendation_count']} "
            f"variant={info.get('variant')} request_id={info.get('request_id')}",
        )

    if args.log_path and log_offset is not None:
        try:
            new_logs = _read_new_logs(args.log_path, log_offset)
        except FileNotFoundError as exc:
            print(f"[WARN] Unable to read log file: {exc}", file=sys.stderr)
            return 0
        if _has_error_signature(new_logs):
            print(
                "[FAIL] Ranking error signature detected in logs after requests. "
                "Inspect log output for details.",
                file=sys.stderr,
            )
            return 3
        print("[OK] No ranking error signatures detected in new log entries.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
