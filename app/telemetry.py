"""Lightweight telemetry utilities for recording recommendation exposures."""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional

from config.settings import DATA_DIR

LOGGER = logging.getLogger(__name__)
_EXPOSURE_PATH = DATA_DIR / "evaluation" / "exposure_log.jsonl"
_LOCK = threading.Lock()


def record_exposure(
    *,
    request_id: str,
    user_id: Optional[int],
    page_id: Optional[int],
    algorithm_version: Optional[str],
    items: Iterable[Mapping[str, object]],
    context: Optional[Mapping[str, object]] = None,
) -> None:
    """Append a single exposure event to the local JSONL log."""
    payload = {
        "request_id": request_id,
        "user_id": user_id,
        "page_id": page_id,
        "algorithm_version": algorithm_version,
        "items": list(items),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if context:
        payload["context"] = dict(context)

    _EXPOSURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False)
    with _LOCK:
        with _EXPOSURE_PATH.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    LOGGER.debug("Recorded exposure %s", request_id)
