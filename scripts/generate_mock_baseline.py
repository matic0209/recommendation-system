"""Generate mock Matomo data and exposure log for local evaluation baselines."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

BASE_DIR = Path('data')
MATOMO_DIR = BASE_DIR / 'matomo'
EVAL_DIR = BASE_DIR / 'evaluation'
PROCESSED_DIR = BASE_DIR / 'processed'


def _prepare_dataset_features() -> List[int]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features_path = PROCESSED_DIR / 'dataset_features.parquet'
    if features_path.exists():
        dataset_features = pd.read_parquet(features_path)
    else:
        dataset_features = pd.DataFrame()

    if dataset_features.empty:
        dataset_features = pd.DataFrame(
            [
                {"dataset_id": 101, "dataset_name": "Finance Risk", "description": "finance risk", "tag": "finance;risk", "price": 100},
                {"dataset_id": 102, "dataset_name": "Finance Summary", "description": "finance summary", "tag": "finance", "price": 80},
                {"dataset_id": 103, "dataset_name": "Health Record", "description": "health record", "tag": "health", "price": 55},
            ]
        )
        dataset_features.to_parquet(features_path, index=False)

    ids = dataset_features['dataset_id'].tolist()
    if len(ids) < 3:
        ids.extend([201, 202, 203])
    return ids[:3]


def generate():
    MATOMO_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    dataset_ids = _prepare_dataset_features()

    # log_action
    actions = pd.DataFrame(
        {
            'idaction': range(1, len(dataset_ids) + 1),
            'name': [f'dataDetail/{dataset_id}' for dataset_id in dataset_ids],
            'type': [1] * len(dataset_ids),
        }
    )
    actions.to_parquet(MATOMO_DIR / 'matomo_log_action.parquet', index=False)

    start = datetime(2025, 1, 1, 12, 0, 0)

    # link_visit_action (views/clicks)
    records = []
    for i, dataset_id in enumerate(dataset_ids, start=1):
        for j in range(3):
            records.append(
                {
                    'idlink_va': i * 100 + j,
                    'idsite': 1,
                    'idvisit': i,
                    'idaction_url': i,
                    'idaction_event_action': np.nan,
                    'idaction_event_category': np.nan,
                    'server_time': start + timedelta(minutes=5 * (i + j)),
                }
            )
    pd.DataFrame(records).to_parquet(MATOMO_DIR / 'matomo_log_link_visit_action.parquet', index=False)

    # conversions for first two datasets
    conv_records = []
    for i, dataset_id in enumerate(dataset_ids[:2], start=1):
        conv_records.append(
            {
                'idvisit': i,
                'idsite': 1,
                'idaction_url': i,
                'url': f'https://example.com/dataDetail/{dataset_id}',
                'server_time': start + timedelta(hours=1 + i),
                'revenue': 99.0 + i,
            }
        )
    pd.DataFrame(conv_records).to_parquet(MATOMO_DIR / 'matomo_log_conversion.parquet', index=False)

    # exposure log JSONL with two versions
    exposures = []
    versions = ['baseline_run', 'new_algo_run']
    for idx, version in enumerate(versions):
        for user in range(1, 4):
            rotated = dataset_ids[:: (-1 if idx % 2 else 1)]
            items = []
            for rank, dataset_id in enumerate(rotated[:2], start=1):
                items.append(
                    {
                        'dataset_id': dataset_id,
                        'score': round(1.0 / rank + idx * 0.1, 3),
                        'reason': 'vector+rank' if idx else 'behavior+rank',
                    }
                )
            exposures.append(
                {
                    'request_id': f'{version}-{user}',
                    'user_id': user,
                    'page_id': dataset_ids[user % len(dataset_ids)],
                    'algorithm_version': version,
                    'items': items,
                    'timestamp': (start + timedelta(minutes=idx * 10 + user)).isoformat(),
                }
            )

    with (EVAL_DIR / 'exposure_log.jsonl').open('w', encoding='utf-8') as stream:
        for payload in exposures:
            stream.write(json.dumps(payload, ensure_ascii=False) + '
')

    print('Mock data generated under data/matomo and data/evaluation.')


if __name__ == '__main__':
    generate()
