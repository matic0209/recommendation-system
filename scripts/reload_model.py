#!/usr/bin/env python3
"""Trigger model reload endpoint for the running FastAPI service."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8000/models/reload"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trigger model reload via HTTP API")
    parser.add_argument("--url", default=DEFAULT_URL, help="Reload endpoint URL (default: %(default)s)")
    parser.add_argument("--mode", default="primary", choices=["primary", "shadow"], help="Reload mode")
    parser.add_argument("--source", help="Optional path to model artifacts (copied when mode=primary)")
    parser.add_argument("--run-id", help="Explicit run_id to record for the bundle")
    parser.add_argument("--rollout", type=float, help="Shadow rollout ratio between 0 and 1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload: dict[str, object] = {"mode": args.mode}
    if args.source:
        payload["source"] = args.source
    if args.run_id:
        payload["run_id"] = args.run_id
    if args.rollout is not None:
        payload["rollout"] = max(0.0, min(1.0, args.rollout))

    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        print(f"Reload failed: {exc.code} {exc.reason}", file=sys.stderr)
        print(exc.read().decode("utf-8"), file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Reload failed: {exc}", file=sys.stderr)
        sys.exit(1)
    else:
        print(body)


if __name__ == "__main__":
    main()
