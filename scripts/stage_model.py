from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.model_manager import deploy_from_source, stage_new_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage recommendation model artifacts for hot reload")
    parser.add_argument("source", type=str, help="Directory containing trained model artifacts")
    parser.add_argument("--deploy", action="store_true", help="Copy staged artifacts into active models directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source).resolve()
    if not source_dir.exists():
        raise SystemExit(f"Source directory {source_dir} does not exist")

    staged_dir = stage_new_model(source_dir)
    if args.deploy:
        deploy_from_source(staged_dir)
        LOGGER.info("Deployed staged model from %s", staged_dir)


if __name__ == "__main__":
    main()
