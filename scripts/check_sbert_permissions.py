#!/usr/bin/env python3
"""Validate SBERT model directory permissions inside deployment."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the SBERT model directory is writable.",
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("SBERT_MODEL", "").strip() or "/root/recommendation-system/models/sbert/paraphrase-multilingual-MiniLM-L12-v2",
        help="Path to the SBERT model directory or cache. Defaults to SBERT_MODEL env or common path.",
    )
    parser.add_argument(
        "--parent",
        action="store_true",
        help="If set, check permissions on the parent directory instead of the model directory itself.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _test_write(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".perm_test_", dir=path)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("permission-test")
    os.remove(tmp_path)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    target = Path(args.model_path).expanduser()
    if args.parent:
        target = target.parent
    print(f"[INFO] Checking writability of: {target}")
    try:
        _test_write(target)
    except PermissionError as exc:
        print(f"[FAIL] Permission denied while writing to {target}: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"[FAIL] OS error while testing {target}: {exc}", file=sys.stderr)
        return 3
    print("[OK] Directory is writable. SBERT download should succeed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
