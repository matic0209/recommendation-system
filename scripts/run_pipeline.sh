#!/usr/bin/env bash
# Run the full data pipeline: extract -> build features -> train models.
set -euo pipefail

if [[ "${1:-}" == "--sync-only" ]]; then
  shift || true
  python3 -m pipeline.build_features "$@"
  python3 -m pipeline.data_quality_v2 "$@"
  python3 -m pipeline.train_models "$@"
  python3 -m pipeline.recall_engine_v2 "$@"
  python3 -m pipeline.evaluate "$@"
  exit 0
fi

python3 -m pipeline.extract_load "$@"
python3 -m pipeline.build_features
python3 -m pipeline.data_quality_v2
python3 -m pipeline.train_models
python3 -m pipeline.recall_engine_v2
python3 -m pipeline.evaluate
