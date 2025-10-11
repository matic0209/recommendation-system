"""Extract raw data from source databases into the local data directory."""
from __future__ import annotations

import argparse
import json
import logging
import time
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import DBAPIError, OperationalError

from config.settings import DATA_DIR, DatabaseConfig, load_database_configs

LOGGER = logging.getLogger(__name__)
DEFAULT_TABLES: Dict[str, tuple[str, ...]] = {
    "business": (
        "user",
        "dataset",
        "task",
        "api_order",
        "dataset_image",
    ),
    "matomo": (
        "matomo_log_visit",
        "matomo_log_link_visit_action",
        "matomo_log_action",
        "matomo_log_conversion",
    ),
}

INCREMENTAL_CANDIDATES: Dict[str, tuple[str, ...]] = {
    "user": ("update_time", "modify_time", "create_time"),
    "dataset": ("update_time", "modify_time", "create_time"),
    "task": ("update_time", "create_time"),
    "api_order": ("update_time", "create_time"),
    "dataset_image": ("update_time", "create_time"),
    "matomo_log_visit": ("visit_last_action_time", "server_time"),
    "matomo_log_link_visit_action": ("server_time",),
    "matomo_log_action": ("server_time",),
    "matomo_log_conversion": ("server_time",),
}

METADATA_DIR = DATA_DIR / "_metadata"
STATE_PATH = METADATA_DIR / "extract_state.json"
EVALUATION_DIR = DATA_DIR / "evaluation"
METRICS_PATH = EVALUATION_DIR / "extract_metrics.json"
PARTITION_PREFIX = "load_time="
CHUNK_SIZE = int(os.getenv("EXTRACT_CHUNK_SIZE", "50000"))
MAX_RETRIES = int(os.getenv("EXTRACT_MAX_RETRIES", "3"))
RETRY_BACKOFF_SECONDS = float(os.getenv("EXTRACT_RETRY_BACKOFF", "3.0"))


@dataclass
class ExtractionResult:
    mode: str
    row_count: int
    partition_path: Optional[str]
    watermark: Optional[str]
    incremental_column: Optional[str]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_state() -> Dict[str, Dict[str, Dict[str, str]]]:
    if not STATE_PATH.exists():
        return {}
    try:
        state = json.loads(STATE_PATH.read_text())
        business_state = state.get("business")
        if isinstance(business_state, dict) and "order_tab" in business_state and "task" not in business_state:
            business_state["task"] = business_state.pop("order_tab")
        return state
    except json.JSONDecodeError:
        LOGGER.warning("Failed to parse state file %s, starting fresh", STATE_PATH)
        return {}


def _save_state(state: Dict[str, Dict[str, Dict[str, str]]]) -> None:
    _ensure_dir(METADATA_DIR)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def _load_metrics() -> list[dict]:
    if not METRICS_PATH.exists():
        return []
    try:
        return json.loads(METRICS_PATH.read_text())
    except json.JSONDecodeError:
        LOGGER.warning("Failed to parse metrics file %s, starting fresh", METRICS_PATH)
        return []


def _save_metrics(metrics: list[dict]) -> None:
    _ensure_dir(EVALUATION_DIR)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


def _candidate_columns(table: str) -> Iterable[str]:
    if table in INCREMENTAL_CANDIDATES:
        yield from INCREMENTAL_CANDIDATES[table]
    # Generic fallbacks
    yield from (
        "update_time",
        "updated_at",
        "modify_time",
        "modified_at",
        "event_time",
        "log_time",
        "server_time",
        "create_time",
        "created_at",
    )


def _select_incremental_column(columns: Iterable[dict], table: str) -> Optional[str]:
    available = {col["name"] for col in columns}
    for candidate in _candidate_columns(table):
        if candidate in available:
            return candidate
    # Fallback to first datetime-like column if available
    for col in columns:
        type_name = str(col.get("type", "")).lower()
        if "date" in type_name or "time" in type_name:
            return col["name"]
    return None


def _format_partition_name() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{PARTITION_PREFIX}{timestamp}.parquet"


def _update_watermark(frame: pd.DataFrame, column: str) -> Optional[str]:
    if frame.empty or column not in frame.columns:
        return None
    value = frame[column].max()
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def _write_parquet_chunk(path: Path, chunk: pd.DataFrame, *, writer_state: dict) -> None:
    """Write dataframe chunk to parquet file, creating writer lazily."""
    if chunk.empty:
        return
    table = pa.Table.from_pandas(chunk, preserve_index=False)
    writer = writer_state.get(path)
    if writer is None:
        writer_state[path] = pq.ParquetWriter(str(path), table.schema)
        writer = writer_state[path]
    writer.write_table(table)


def _append_parquet_chunk(path: Path, chunk: pd.DataFrame, *, schema_cache: dict) -> None:
    """Append chunk to existing parquet file using cached schema."""
    if chunk.empty:
        return
    schema = schema_cache.get(path)
    if schema is None:
        if path.exists():
            schema = pq.read_schema(str(path))
        else:
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            pq.write_table(table, str(path))
            schema_cache[path] = table.schema
            return
        schema_cache[path] = schema
    table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
    pq.write_table(table, str(path), append=True)


def _close_writers(writer_state: dict) -> None:
    for writer in writer_state.values():
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            LOGGER.warning("Failed to close parquet writer", exc_info=True)


def _stream_query(engine, query: str, params: Optional[Dict[str, object]], chunk_size: int) -> Iterable[pd.DataFrame]:
    """Yield query results in chunks with retry."""
    attempt = 0
    while True:
        try:
            with engine.connect().execution_options(stream_results=True) as conn:
                for chunk in pd.read_sql_query(query, con=conn, params=params, chunksize=chunk_size):
                    yield chunk
            break
        except (OperationalError, DBAPIError) as exc:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            sleep_seconds = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            LOGGER.warning(
                "Chunked query failed (attempt=%d/%d): %s. Retrying in %.1fs",
                attempt,
                MAX_RETRIES,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)


def _export_table(
    engine_url: str,
    config: DatabaseConfig,
    table: str,
    output_dir: Path,
    dry_run: bool,
    full_refresh: bool,
    state: Dict[str, Dict[str, Dict[str, str]]],
    source: str,
) -> ExtractionResult:
    source_state = state.setdefault(source, {})
    table_state = source_state.get(table, {})
    start_time = time.monotonic()
    partition_path: Optional[str] = None
    watermark: Optional[str] = None
    mode = "full"
    incremental_column: Optional[str] = None

    output_file = output_dir / f"{table}.parquet"
    table_partition_dir = output_dir / table

    if dry_run:
        LOGGER.info("[dry-run] would export table '%s' from %s", table, config.name)
        return ExtractionResult(mode="dry-run", row_count=0, partition_path=None, watermark=None, incremental_column=None)

    # P0-02 优化: 应用连接池配置
    engine = create_engine(engine_url, **config.get_engine_kwargs())
    try:
        inspector = inspect(engine)
        columns = inspector.get_columns(table)
        incremental_column = _select_incremental_column(columns, table)
        last_watermark = table_state.get("watermark") if table_state else None

        query = f"SELECT * FROM {table}"
        params: Optional[Dict[str, object]] = None
        if not full_refresh and incremental_column:
            if last_watermark:
                mode = "incremental"
                query += f" WHERE {incremental_column} > %(watermark)s ORDER BY {incremental_column}"
                params = {"watermark": last_watermark}
            else:
                mode = "bootstrap"
        else:
            mode = "full"

        LOGGER.info(
            "Exporting table '%s' (mode=%s, incremental_column=%s, watermark=%s)",
            table,
            mode,
            incremental_column,
            last_watermark,
        )

        writer_state: dict[Path, pq.ParquetWriter] = {}
        append_schema_cache: dict[Path, pa.Schema] = {}
        partition_name = _format_partition_name()
        partition_path = table_partition_dir / partition_name
        row_count = 0
        watermark = None

        if mode != "incremental" and output_file.exists():
            output_file.unlink()

        chunk_iter = _stream_query(engine, query, params, CHUNK_SIZE)
        try:
            for chunk in chunk_iter:
                if chunk.empty:
                    continue
                row_count += len(chunk)
                if incremental_column:
                    chunk_watermark = _update_watermark(chunk, incremental_column)
                    if chunk_watermark:
                        watermark = chunk_watermark
                _ensure_dir(output_dir)
                _ensure_dir(table_partition_dir)
                _write_parquet_chunk(partition_path, chunk, writer_state=writer_state)
                if mode == "incremental":
                    _append_parquet_chunk(output_file, chunk, schema_cache=append_schema_cache)
                else:
                    _write_parquet_chunk(output_file, chunk, writer_state=writer_state)
        finally:
            _close_writers(writer_state)

        if row_count == 0:
            LOGGER.info("No new rows for table '%s'", table)
            duration = time.monotonic() - start_time
            if partition_path.exists():
                partition_path.unlink()
            return ExtractionResult(mode=mode, row_count=0, partition_path=None, watermark=last_watermark, incremental_column=incremental_column)

        partition_path_str = str(partition_path)

        if watermark:
            source_state[table] = {
                "watermark": watermark,
                "column": incremental_column,
            }
        elif table_state:
            # Preserve existing watermark if no new rows.
            source_state[table] = table_state

        duration = time.monotonic() - start_time
        LOGGER.info(
            "Saved %s (%d rows, %.2fs)",
            output_file.name,
            row_count,
            duration,
        )
        return ExtractionResult(
            mode=mode,
            row_count=row_count,
            partition_path=partition_path_str,
            watermark=watermark,
            incremental_column=incremental_column,
        )
    finally:
        engine.dispose()


def extract_all(dry_run: bool = True, full_refresh: bool = False) -> None:
    configs = load_database_configs()
    _ensure_dir(DATA_DIR)
    state = _load_state()
    metrics_log = _load_metrics()
    run_started = datetime.now(timezone.utc).isoformat()
    run_metrics = {
        "started_at": run_started,
        "dry_run": dry_run,
        "full_refresh": full_refresh,
        "tables": [],
    }

    for source, tables in DEFAULT_TABLES.items():
        config = configs[source]
        output_dir = DATA_DIR / source
        _ensure_dir(output_dir)
        LOGGER.info("Processing source '%s' (database=%s)", source, config.name)
        engine_url = config.sqlalchemy_url()
        for table in tables:
            result = _export_table(
                engine_url=engine_url,
                config=config,
                table=table,
                output_dir=output_dir,
                dry_run=dry_run,
                full_refresh=full_refresh,
                state=state,
                source=source,
            )
            run_metrics["tables"].append(
                {
                    "source": source,
                    "table": table,
                    "mode": result.mode,
                    "rows": result.row_count,
                    "partition_path": result.partition_path,
                    "incremental_column": result.incremental_column,
                    "watermark": result.watermark,
                }
            )

    run_metrics["finished_at"] = datetime.now(timezone.utc).isoformat()
    metrics_log.append(run_metrics)

    if not dry_run:
        _save_state(state)
        _save_metrics(metrics_log)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Log planned actions without writing outputs.",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        default=False,
        help="Ignore incremental watermarks and perform a full export.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract_all(dry_run=args.dry_run, full_refresh=args.full_refresh)


if __name__ == "__main__":
    main()
