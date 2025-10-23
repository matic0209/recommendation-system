"""Extract raw data from source databases into the local data directory."""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import DBAPIError, OperationalError

from config.settings import (
    DATA_DIR,
    DATA_JSON_DIR,
    DATA_SOURCE,
    SOURCE_DATA_MODES,
    DatabaseConfig,
    load_database_configs,
)

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

JSON_TYPE_CONVERSIONS: Dict[str, Dict[str, str]] = {
    "user": {
        "create_time": "datetime",
        "last_login_time": "datetime",
    },
    "dataset": {
        "price": "float",
        "create_time": "datetime",
        "update_time": "datetime",
    },
    "task": {
        "price": "float",
        "create_time": "datetime",
        "update_time": "datetime",
    },
    "api_order": {
        "price": "float",
        "api_price": "float",
        "create_time": "datetime",
        "update_time": "datetime",
        "pay_time": "datetime",
    },
    "dataset_image": {
        "create_time": "datetime",
        "update_time": "datetime",
    },
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

PRIMARY_KEYS: Dict[str, Dict[str, tuple[str, ...]]] = {
    "business": {
        "user": ("id",),
        "dataset": ("id",),
        "task": ("id",),
        "api_order": ("id",),
        "dataset_image": ("id",),
    },
    "matomo": {
        "matomo_log_visit": ("idvisit",),
        "matomo_log_link_visit_action": ("idlink_va",),
        "matomo_log_action": ("idaction",),
        "matomo_log_conversion": ("idvisit", "idgoal", "buster"),
    },
}


@dataclass
class ExtractionResult:
    mode: str
    row_count: int
    partition_path: Optional[str]
    watermark: Optional[str]
    incremental_column: Optional[str]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_arrow_table(table: pa.Table) -> pa.Table:
    """Promote null-typed columns to string so later chunks with values still match schema."""
    if not any(pa.types.is_null(field.type) for field in table.schema):
        return table

    arrays: list[pa.Array] = []
    fields: list[pa.Field] = []
    for field, column in zip(table.schema, table.itercolumns()):
        if pa.types.is_null(field.type):
            data = column.to_pylist()
            arrays.append(pa.array(data, type=pa.string()))
            fields.append(pa.field(field.name, pa.string()))
        else:
            arrays.append(column)
            fields.append(field)
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def _coerce_frame_to_schema(frame: pd.DataFrame, schema: pa.Schema) -> pd.DataFrame:
    """Coerce DataFrame columns to satisfy an Arrow schema (currently string promotion)."""
    if frame.empty:
        return frame
    result = frame.copy()
    for field in schema:
        name = field.name
        if name not in result.columns:
            continue
        series = result[name]
        if pa.types.is_string(field.type):
            if not (is_object_dtype(series.dtype) or is_string_dtype(series.dtype)):
                result[name] = series.astype("string")
            result[name] = result[name].where(~result[name].isna(), None)
    return result


def _apply_json_type_conversions(table: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply table-specific type conversions for JSON extracted data."""
    if frame.empty:
        return frame
    conversions = JSON_TYPE_CONVERSIONS.get(table)
    if not conversions:
        return frame
    result = frame.copy()
    for column, target_type in conversions.items():
        if column not in result.columns:
            continue
        if target_type == "datetime":
            result[column] = pd.to_datetime(result[column], errors="coerce")
        elif target_type == "float":
            result[column] = pd.to_numeric(result[column], errors="coerce")
        if target_type in {"datetime", "float"}:
            result[column] = result[column].replace([np.inf, -np.inf], pd.NA)
    return result


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

    # Convert to datetime first, handling mixed types
    try:
        time_series = pd.to_datetime(frame[column], errors='coerce')
        value = time_series.max()
        if pd.isna(value):
            return None
        return value.isoformat()
    except Exception:
        # Fallback to string comparison if datetime conversion fails
        value = frame[column].astype(str).max()
        if pd.isna(value) or value == 'nan':
            return None
        return str(value)


def _write_parquet_chunk(path: Path, chunk: pd.DataFrame, *, writer_state: dict) -> None:
    """Write dataframe chunk to parquet file, creating writer lazily."""
    if chunk.empty:
        return
    writer = writer_state.get(path)
    if writer is None:
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        table = _normalize_arrow_table(table)
        writer_state[path] = pq.ParquetWriter(str(path), table.schema)
        writer = writer_state[path]
    else:
        coerced = _coerce_frame_to_schema(chunk, writer.schema)
        table = pa.Table.from_pandas(coerced, schema=writer.schema, preserve_index=False)
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
            table = _normalize_arrow_table(table)
            pq.write_table(table, str(path))
            schema_cache[path] = table.schema
            return
        schema_cache[path] = schema

    coerced = _coerce_frame_to_schema(chunk, schema)
    table = pa.Table.from_pandas(coerced, schema=schema, preserve_index=False)
    table = _normalize_arrow_table(table)
    if path.exists():
        existing = pq.read_table(str(path))
        table = pa.concat_tables([existing, table], promote=True)
    pq.write_table(table, str(path))


def _close_writers(writer_state: dict) -> None:
    for writer in writer_state.values():
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            LOGGER.warning("Failed to close parquet writer", exc_info=True)


def _find_incremental_json_files(json_dir: Path, table: str, last_watermark: Optional[str]) -> list[Path]:
    """Find incremental JSON files newer than the last watermark."""
    import re
    pattern = re.compile(rf"^{re.escape(table)}_(\d{{8}})_(\d{{6}})\.json$")
    files = []

    for file_path in json_dir.glob(f"{table}_*.json"):
        match = pattern.match(file_path.name)
        if match:
            date_str, time_str = match.groups()
            # Parse timestamp from filename: YYYYMMDD_HHMMSS
            timestamp_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
            try:
                file_timestamp = pd.to_datetime(timestamp_str)
                if last_watermark is None or file_timestamp > pd.to_datetime(last_watermark):
                    files.append((file_timestamp, file_path))
            except ValueError:
                LOGGER.warning("Failed to parse timestamp from file: %s", file_path.name)
                continue

    # Sort by timestamp
    files.sort(key=lambda x: x[0])
    return [f[1] for f in files]


def _get_primary_keys(source: str, table: str) -> Optional[tuple[str, ...]]:
    return PRIMARY_KEYS.get(source, {}).get(table)


def _deduplicate_by_keys(frame: pd.DataFrame, primary_keys: Optional[tuple[str, ...]]) -> pd.DataFrame:
    if frame.empty or not primary_keys:
        return frame
    missing = [col for col in primary_keys if col not in frame.columns]
    if missing:
        LOGGER.warning(
            "Primary key columns %s missing in %s table; skipping deduplication for current chunk",
            ", ".join(missing),
            primary_keys,
        )
        return frame
    return frame.drop_duplicates(subset=list(primary_keys), keep="last")


def _merge_incremental_output(output_file: Path, chunk: pd.DataFrame, primary_keys: tuple[str, ...]) -> None:
    if chunk.empty:
        return

    chunk = _deduplicate_by_keys(chunk, primary_keys)
    if output_file.exists():
        try:
            existing = pd.read_parquet(output_file)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to read %s (%s); rewriting from chunk only", output_file, exc)
            existing = pd.DataFrame(columns=chunk.columns)
        if not existing.empty:
            combined = pd.concat([existing, chunk], ignore_index=True, copy=False)
        else:
            combined = chunk
    else:
        combined = chunk

    combined = _deduplicate_by_keys(combined, primary_keys)
    tmp_path = output_file.with_suffix(".tmp.parquet")
    combined.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_file)


def _export_table_from_json(
    table: str,
    json_dir: Path,
    output_dir: Path,
    dry_run: bool,
    full_refresh: bool,
    state: Dict[str, Dict[str, Dict[str, str]]],
    source: str,
) -> ExtractionResult:
    """Export table from JSON files (full or incremental)."""
    source_state = state.setdefault(source, {})
    table_state = source_state.get(table, {})
    start_time = time.monotonic()

    output_file = output_dir / f"{table}.parquet"
    table_partition_dir = output_dir / table

    if dry_run:
        LOGGER.info("[dry-run] would load table '%s' from JSON dir %s", table, json_dir)
        return ExtractionResult(mode="dry-run", row_count=0, partition_path=None, watermark=None, incremental_column=None)

    # Determine mode and files to process
    mode = "full"
    incremental_column = None
    last_watermark = table_state.get("watermark") if table_state else None
    files_to_process = []

    full_file = json_dir / f"{table}.json"

    if full_refresh or not last_watermark:
        # Full refresh: only load the main file
        if full_file.exists():
            files_to_process = [full_file]
            mode = "full" if full_refresh else "bootstrap"
        else:
            LOGGER.warning("Full file not found: %s", full_file)
            return ExtractionResult(mode="error", row_count=0, partition_path=None, watermark=None, incremental_column=None)
    else:
        # Incremental: find files newer than watermark
        incremental_files = _find_incremental_json_files(json_dir, table, last_watermark)
        if incremental_files:
            files_to_process = incremental_files
            mode = "incremental"
        else:
            LOGGER.info("No new incremental files for table '%s'", table)
            return ExtractionResult(mode="incremental", row_count=0, partition_path=None, watermark=last_watermark, incremental_column=None)

    if not files_to_process:
        LOGGER.info("No files to process for table '%s'", table)
        return ExtractionResult(mode=mode, row_count=0, partition_path=None, watermark=last_watermark, incremental_column=None)

    LOGGER.info(
        "Loading table '%s' (mode=%s, files=%d, watermark=%s)",
        table,
        mode,
        len(files_to_process),
        last_watermark,
    )

    # Load and process JSON files
    writer_state: dict[Path, pq.ParquetWriter] = {}
    append_schema_cache: dict[Path, pa.Schema] = {}
    partition_name = _format_partition_name()
    partition_path = table_partition_dir / partition_name
    row_count = 0
    watermark = None

    if mode != "incremental" and output_file.exists():
        output_file.unlink()

    try:
        for json_file in files_to_process:
            LOGGER.info("Processing %s", json_file.name)
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if not data:
                    LOGGER.warning("Empty data in file: %s", json_file.name)
                    continue

                # Convert to DataFrame
                df = pd.DataFrame(data)
                df = _apply_json_type_conversions(table, df)
                primary_keys = _get_primary_keys(source, table)
                df = _deduplicate_by_keys(df, primary_keys)
                if df.empty:
                    continue

                row_count += len(df)

                # Detect incremental column and update watermark
                if incremental_column is None:
                    for candidate in _candidate_columns(table):
                        if candidate in df.columns:
                            incremental_column = candidate
                            break

                if incremental_column and incremental_column in df.columns:
                    chunk_watermark = _update_watermark(df, incremental_column)
                    if chunk_watermark:
                        watermark = chunk_watermark

                # Write to parquet
                _ensure_dir(output_dir)
                _ensure_dir(table_partition_dir)
                _write_parquet_chunk(partition_path, df, writer_state=writer_state)

                if primary_keys:
                    _merge_incremental_output(output_file, df, primary_keys)
                elif mode == "incremental":
                    _append_parquet_chunk(output_file, df, schema_cache=append_schema_cache)
                else:
                    _write_parquet_chunk(output_file, df, writer_state=writer_state)

            except (json.JSONDecodeError, ValueError) as e:
                LOGGER.error("Failed to load JSON file %s: %s", json_file.name, e)
                continue
    finally:
        _close_writers(writer_state)

    if row_count == 0:
        LOGGER.info("No new rows for table '%s'", table)
        if partition_path.exists():
            partition_path.unlink()
        return ExtractionResult(mode=mode, row_count=0, partition_path=None, watermark=last_watermark, incremental_column=incremental_column)

    partition_path_str = str(partition_path)

    # Update state with new watermark
    if watermark:
        source_state[table] = {
            "watermark": watermark,
            "column": incremental_column,
        }
    elif table_state:
        source_state[table] = table_state

    duration = time.monotonic() - start_time
    LOGGER.info(
        "Loaded %s from JSON (%d rows, %.2fs)",
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
        primary_keys = _get_primary_keys(source, table)

        if mode != "incremental" and output_file.exists():
            output_file.unlink()

        chunk_iter = _stream_query(engine, query, params, CHUNK_SIZE)
        try:
            for chunk in chunk_iter:
                if chunk.empty:
                    continue
                chunk = _deduplicate_by_keys(chunk, primary_keys)
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
                if primary_keys:
                    _merge_incremental_output(output_file, chunk, primary_keys)
                elif mode == "incremental":
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
    _ensure_dir(DATA_DIR)
    state = _load_state()
    metrics_log = _load_metrics()
    run_started = datetime.now(timezone.utc).isoformat()
    run_metrics = {
        "started_at": run_started,
        "dry_run": dry_run,
        "full_refresh": full_refresh,
        "data_source": DATA_SOURCE,
        "tables": [],
    }

    LOGGER.info("Default data source mode: %s", DATA_SOURCE)

    configs = None
    for source, tables in DEFAULT_TABLES.items():
        mode = SOURCE_DATA_MODES.get(source, DATA_SOURCE)
        LOGGER.info("Source '%s' configured for mode: %s", source, mode)

        if mode == "json":
            if not DATA_JSON_DIR.exists():
                LOGGER.error("JSON data directory not found: %s", DATA_JSON_DIR)
                raise FileNotFoundError(f"JSON data directory not found: {DATA_JSON_DIR}")
            if source != "business":
                LOGGER.warning("Skipping source '%s' because JSON mode is not supported for it", source)
                continue

            output_dir = DATA_DIR / source
            _ensure_dir(output_dir)
            LOGGER.info("Processing source '%s' from JSON files", source)

            for table in tables:
                result = _export_table_from_json(
                    table=table,
                    json_dir=DATA_JSON_DIR,
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
        elif mode == "database":
            if configs is None:
                configs = load_database_configs()
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
        else:
            LOGGER.warning("Unknown mode '%s' for source '%s'; skipping", mode, source)

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
