#!/usr/bin/env bash
# Run the full data pipeline: extract -> build features -> train models.
set -euo pipefail

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "Pipeline steps that would be executed:"
  echo "  1. python3 -m pipeline.extract_load      # 数据抽取（CDC增量）"
  echo "  2. python3 -m pipeline.build_features    # 特征构建（基础+增强）"
  echo "  3. python3 -m pipeline.data_quality_v2   # 数据质量检查"
  echo "  4. python3 -m pipeline.train_models      # 模型训练（LightGBM）"
  echo "  5. python3 -m pipeline.recall_engine_v2  # 召回索引构建"
  echo "  6. python3 -m pipeline.evaluate          # 离线评估"
  echo ""
  echo "Use '--sync-only' to skip data extraction and only sync features/models"
  exit 0
fi

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
