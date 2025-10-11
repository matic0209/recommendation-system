"""Redis-based feature store for high-performance feature serving."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import redis

from config.settings import DATA_DIR

LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = DATA_DIR / "processed"


class RedisFeatureStore:
    """Redis-based feature store with high-performance access."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 1,  # Use different DB for features
        password: Optional[str] = None,
    ):
        """Initialize Redis feature store."""
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
            )
            self.redis_client.ping()
            LOGGER.info("Connected to Redis feature store: %s:%d/%d", host, port, db)
        except redis.ConnectionError as exc:
            LOGGER.error("Failed to connect to Redis: %s", exc)
            raise

    def sync_user_features(self, user_features: pd.DataFrame) -> int:
        """
        Sync user features to Redis.

        Storage format:
        - Key: user:{user_id}
        - Value: Hash of all user features
        """
        LOGGER.info("Syncing user features to Redis: %d users", len(user_features))

        pipeline = self.redis_client.pipeline()
        synced_count = 0

        for row in user_features.to_dict(orient="records"):
            user_id = row.get("user_id")
            if not user_id:
                continue

            # Convert all values to strings
            feature_dict = {}
            for key, value in row.items():
                if key == "user_id":
                    continue

                # Handle different types
                if pd.isna(value):
                    feature_dict[key] = "0"
                elif isinstance(value, (int, float)):
                    feature_dict[key] = str(value)
                elif isinstance(value, (list, dict)):
                    feature_dict[key] = json.dumps(value)
                else:
                    feature_dict[key] = str(value)

            # Store as hash
            redis_key = f"user:{user_id}"
            pipeline.delete(redis_key)
            pipeline.hset(redis_key, mapping=feature_dict)

            synced_count += 1

            # Execute in batches
            if synced_count % 1000 == 0:
                pipeline.execute()
                pipeline = self.redis_client.pipeline()
                LOGGER.info("Synced %d users...", synced_count)

        # Execute remaining
        if synced_count % 1000 != 0:
            pipeline.execute()

        LOGGER.info("User features synced: %d users", synced_count)
        return synced_count

    def sync_dataset_features(self, dataset_features: pd.DataFrame) -> int:
        """
        Sync dataset features to Redis.

        Storage format:
        - Key: dataset:{dataset_id}
        - Value: Hash of all dataset features
        """
        LOGGER.info("Syncing dataset features to Redis: %d datasets", len(dataset_features))

        pipeline = self.redis_client.pipeline()
        synced_count = 0

        for row in dataset_features.to_dict(orient="records"):
            dataset_id = row.get("dataset_id")
            if not dataset_id:
                continue

            feature_dict = {}
            for key, value in row.items():
                if key == "dataset_id":
                    continue

                if pd.isna(value):
                    feature_dict[key] = "0"
                elif isinstance(value, (int, float)):
                    feature_dict[key] = str(value)
                elif isinstance(value, (list, dict)):
                    feature_dict[key] = json.dumps(value)
                else:
                    feature_dict[key] = str(value)

            redis_key = f"dataset:{dataset_id}"
            pipeline.delete(redis_key)
            pipeline.hset(redis_key, mapping=feature_dict)

            synced_count += 1

            if synced_count % 1000 == 0:
                pipeline.execute()
                pipeline = self.redis_client.pipeline()
                LOGGER.info("Synced %d datasets...", synced_count)

        if synced_count % 1000 != 0:
            pipeline.execute()

        LOGGER.info("Dataset features synced: %d datasets", synced_count)
        return synced_count

    def sync_user_history(self, interactions: pd.DataFrame) -> int:
        """
        Sync user interaction history to Redis.

        Storage format:
        - Key: user_history:{user_id}
        - Value: Sorted Set (score=timestamp, member=dataset_id)
        """
        LOGGER.info("Syncing user history to Redis...")

        pipeline = self.redis_client.pipeline()
        synced_users = 0

        for user_id, group in interactions.groupby("user_id"):
            redis_key = f"user_history:{user_id}"

            # Clear existing
            pipeline.delete(redis_key)

            # Add interactions sorted by timestamp
            for row in group.to_dict(orient="records"):
                dataset_id = row.get("dataset_id")
                timestamp = row.get("last_event_time")

                if timestamp and dataset_id:
                    # Use timestamp as score
                    if isinstance(timestamp, str):
                        score = pd.to_datetime(timestamp).timestamp()
                    else:
                        score = timestamp.timestamp()

                    pipeline.zadd(redis_key, {str(dataset_id): score})

            synced_users += 1

            if synced_users % 1000 == 0:
                pipeline.execute()
                pipeline = self.redis_client.pipeline()
                LOGGER.info("Synced %d users' history...", synced_users)

        if synced_users % 1000 != 0:
            pipeline.execute()

        LOGGER.info("User history synced: %d users", synced_users)
        return synced_users

    def sync_dataset_stats(self, dataset_stats: pd.DataFrame) -> int:
        """
        Sync dataset statistics for quick access.

        Storage format:
        - Key: dataset_stats:{dataset_id}
        - Value: Hash
        """
        LOGGER.info("Syncing dataset stats to Redis...")

        pipeline = self.redis_client.pipeline()
        synced_count = 0

        for row in dataset_stats.to_dict(orient="records"):
            dataset_id = row.get("dataset_id")
            if not dataset_id:
                continue

            redis_key = f"dataset_stats:{dataset_id}"
            stats_dict = {
                "interaction_count": str(row.get("interaction_count", 0)),
                "total_weight": str(row.get("total_weight", 0)),
                "last_event_time": str(row.get("last_event_time", "")),
            }

            pipeline.hset(redis_key, mapping=stats_dict)

            synced_count += 1

            if synced_count % 1000 == 0:
                pipeline.execute()
                pipeline = self.redis_client.pipeline()

        if synced_count % 1000 != 0:
            pipeline.execute()

        LOGGER.info("Dataset stats synced: %d datasets", synced_count)
        return synced_count

    def get_dataset_stats(self, dataset_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Fetch dataset statistics for specified dataset IDs."""
        if not dataset_ids:
            return {}

        pipeline = self.redis_client.pipeline()
        for dataset_id in dataset_ids:
            pipeline.hgetall(f"dataset_stats:{dataset_id}")

        results = pipeline.execute()

        stats: Dict[int, Dict[str, Any]] = {}
        for dataset_id, values in zip(dataset_ids, results):
            if not values:
                continue
            parsed: Dict[str, Any] = {}
            for key, value in values.items():
                if value is None:
                    continue
                try:
                    if key.endswith("_time"):
                        parsed[key] = value
                    elif "." in value:
                        parsed[key] = float(value)
                    else:
                        parsed[key] = int(value)
                except ValueError:
                    parsed[key] = value
            stats[int(dataset_id)] = parsed

        return stats

    def get_user_features(self, user_id: int) -> Dict[str, Any]:
        """Get user features from Redis."""
        redis_key = f"user:{user_id}"
        features = self.redis_client.hgetall(redis_key)

        if not features:
            return {}

        # Convert back to appropriate types
        result = {}
        for key, value in features.items():
            try:
                # Try numeric first
                if "." in value:
                    result[key] = float(value)
                else:
                    result[key] = int(value)
            except ValueError:
                # Keep as string
                result[key] = value

        return result

    def get_dataset_features(self, dataset_id: int) -> Dict[str, Any]:
        """Get dataset features from Redis."""
        redis_key = f"dataset:{dataset_id}"
        features = self.redis_client.hgetall(redis_key)

        if not features:
            return {}

        result = {}
        for key, value in features.items():
            try:
                if "." in value:
                    result[key] = float(value)
                else:
                    result[key] = int(value)
            except ValueError:
                result[key] = value

        return result

    def get_user_history(self, user_id: int, limit: int = 50) -> List[int]:
        """Get user interaction history (most recent first)."""
        redis_key = f"user_history:{user_id}"

        # Get from sorted set (descending by timestamp)
        items = self.redis_client.zrevrange(redis_key, 0, limit - 1)

        return [int(item) for item in items]

    def get_batch_user_features(self, user_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Get features for multiple users in batch."""
        pipeline = self.redis_client.pipeline()

        for user_id in user_ids:
            pipeline.hgetall(f"user:{user_id}")

        results = pipeline.execute()

        batch_features = {}
        for user_id, features in zip(user_ids, results):
            if features:
                result = {}
                for key, value in features.items():
                    try:
                        if "." in value:
                            result[key] = float(value)
                        else:
                            result[key] = int(value)
                    except ValueError:
                        result[key] = value
                batch_features[user_id] = result

        return batch_features

    def get_batch_dataset_features(self, dataset_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Get features for multiple datasets in batch."""
        pipeline = self.redis_client.pipeline()

        for dataset_id in dataset_ids:
            pipeline.hgetall(f"dataset:{dataset_id}")

        results = pipeline.execute()

        batch_features = {}
        for dataset_id, features in zip(dataset_ids, results):
            if features:
                result = {}
                for key, value in features.items():
                    try:
                        if "." in value:
                            result[key] = float(value)
                        else:
                            result[key] = int(value)
                    except ValueError:
                        result[key] = value
                batch_features[dataset_id] = result

        return batch_features

    def sync_all_from_parquet(self) -> Dict[str, int]:
        """Sync all features from parquet files to Redis."""
        LOGGER.info("Syncing all features from parquet to Redis...")

        stats = {}

        # User features
        user_features_path = PROCESSED_DIR / "user_features_v2.parquet"
        if user_features_path.exists():
            user_features = pd.read_parquet(user_features_path)
            stats["user_features"] = self.sync_user_features(user_features)
        else:
            LOGGER.warning("User features not found: %s", user_features_path)

        # Dataset features
        dataset_features_path = PROCESSED_DIR / "dataset_features_v2.parquet"
        if dataset_features_path.exists():
            dataset_features = pd.read_parquet(dataset_features_path)
            stats["dataset_features"] = self.sync_dataset_features(dataset_features)
        else:
            LOGGER.warning("Dataset features not found: %s", dataset_features_path)

        # Interactions
        interactions_path = PROCESSED_DIR / "interactions_v2.parquet"
        if interactions_path.exists():
            interactions = pd.read_parquet(interactions_path)
            stats["user_history"] = self.sync_user_history(interactions)
        else:
            LOGGER.warning("Interactions not found: %s", interactions_path)

        # Dataset stats
        dataset_stats_path = PROCESSED_DIR / "dataset_stats_v2.parquet"
        if dataset_stats_path.exists():
            dataset_stats = pd.read_parquet(dataset_stats_path)
            stats["dataset_stats"] = self.sync_dataset_stats(dataset_stats)
        else:
            LOGGER.warning("Dataset stats not found: %s", dataset_stats_path)

        # Store metadata
        self.redis_client.set(
            "feature_store:last_sync",
            pd.Timestamp.now().isoformat(),
        )
        self.redis_client.set(
            "feature_store:stats",
            json.dumps(stats),
        )

        LOGGER.info("Feature sync complete: %s", stats)

        return stats

    def get_feature_store_info(self) -> Dict[str, Any]:
        """Get feature store metadata."""
        last_sync = self.redis_client.get("feature_store:last_sync")
        stats_json = self.redis_client.get("feature_store:stats")

        info = {
            "last_sync": last_sync,
            "stats": json.loads(stats_json) if stats_json else {},
            "redis_info": {
                "used_memory": self.redis_client.info("memory")["used_memory_human"],
                "total_keys": self.redis_client.dbsize(),
            },
        }

        return info


def main() -> None:
    """Sync features to Redis feature store."""
    import os
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Get Redis config from environment
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_FEATURE_DB", "1"))
    redis_password = os.getenv("REDIS_PASSWORD")

    # Initialize feature store
    feature_store = RedisFeatureStore(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        password=redis_password,
    )

    # Sync all features
    stats = feature_store.sync_all_from_parquet()

    # Print summary
    print("\n" + "=" * 80)
    print("REDIS FEATURE STORE SYNC SUMMARY")
    print("=" * 80)
    for table_name, count in stats.items():
        print(f"{table_name:20s}: {count:,} records")

    # Get feature store info
    info = feature_store.get_feature_store_info()
    print(f"\nLast Sync: {info['last_sync']}")
    print(f"Redis Memory: {info['redis_info']['used_memory']}")
    print(f"Total Keys: {info['redis_info']['total_keys']:,}")
    print("=" * 80 + "\n")

    LOGGER.info("Redis feature store sync completed!")


if __name__ == "__main__":
    main()
