#!/usr/bin/env bash
# Run the full data pipeline: extract -> build features -> train models.
set -euo pipefail

python3 -m pipeline.extract_load "$@"
python3 pipeline/build_features.py
python3 pipeline/train_models.py
