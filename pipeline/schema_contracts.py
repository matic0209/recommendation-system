"""Schema contract validation utilities for source datasets."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yaml
from pandas.api import types as pd_types

from config.settings import BASE_DIR, DATA_DIR

LOGGER = logging.getLogger(__name__)
CONTRACT_PATH = BASE_DIR / "config" / "schema_contracts.yaml"


@dataclass
class SchemaViolation:
    column: Optional[str]
    message: str
    severity: str = "warning"
    expected_type: Optional[str] = None
    actual_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "message": self.message,
            "severity": self.severity,
            "expected_type": self.expected_type,
            "actual_type": self.actual_type,
        }


@dataclass
class SchemaValidationResult:
    source: str
    table: str
    path: Path
    passed: bool
    violations: List[SchemaViolation]
    row_count: int = 0
    column_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "table": self.table,
            "path": str(self.path),
            "passed": self.passed,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "violations": [v.to_dict() for v in self.violations],
        }


class SchemaContractValidator:
    """Validate extracted tables against predefined schema contracts."""

    def __init__(self, contracts: Dict[str, Any], data_dir: Path = DATA_DIR):
        self.contracts = contracts
        self.data_dir = data_dir

    @classmethod
    def from_config(cls, path: Path = CONTRACT_PATH) -> Optional["SchemaContractValidator"]:
        if not path.exists():
            LOGGER.warning("Schema contract definition missing at %s", path)
            return None
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as exc:  # noqa: BLE001
            LOGGER.error("Failed to parse schema contract yaml: %s", exc)
            return None
        sources = data.get("sources") or {}
        if not sources:
            LOGGER.warning("No schema contracts defined in %s", path)
            return None
        return cls(contracts=sources)

    def validate_all(self) -> List[SchemaValidationResult]:
        results: List[SchemaValidationResult] = []
        for source, tables in self.contracts.items():
            source_dir = self.data_dir / source
            for table, spec in tables.items():
                results.append(self.validate_table(source, table, spec, source_dir))
        return results

    def validate_table(
        self,
        source: str,
        table: str,
        spec: Dict[str, Any],
        source_dir: Path,
    ) -> SchemaValidationResult:
        table_path = source_dir / f"{table}.parquet"
        violations: List[SchemaViolation] = []

        if not table_path.exists():
            violations.append(
                SchemaViolation(
                    column=None,
                    message="Extracted parquet not found",
                    severity="error",
                )
            )
            return SchemaValidationResult(
                source=source,
                table=table,
                path=table_path,
                passed=False,
                violations=violations,
            )

        try:
            frame = pd.read_parquet(table_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to load table %s.%s: %s", source, table, exc)
            violations.append(
                SchemaViolation(
                    column=None,
                    message=f"Failed to load parquet: {exc}",
                    severity="error",
                )
            )
            return SchemaValidationResult(
                source=source,
                table=table,
                path=table_path,
                passed=False,
                violations=violations,
            )

        expected_columns = (spec.get("required_columns") or {}).items()
        for column_name, column_spec in expected_columns:
            column_type = (column_spec or {}).get("type", "string").lower()
            nullable = bool((column_spec or {}).get("nullable", True))
            if column_name not in frame.columns:
                violations.append(
                    SchemaViolation(
                        column=column_name,
                        message="Missing required column",
                        severity="error",
                        expected_type=column_type,
                    )
                )
                continue

            series = frame[column_name]
            if not self._dtype_matches(series, column_type):
                violations.append(
                    SchemaViolation(
                        column=column_name,
                        message="Column type mismatch",
                        severity="warning",
                        expected_type=column_type,
                        actual_type=str(series.dtype),
                    )
                )

            if not nullable and series.isna().any():
                null_ratio = float(series.isna().mean())
                violations.append(
                    SchemaViolation(
                        column=column_name,
                        message=f"Column contains nulls ({null_ratio:.2%})",
                        severity="error" if null_ratio > 0.01 else "warning",
                        expected_type=column_type,
                        actual_type=str(series.dtype),
                    )
                )

        primary_key = spec.get("primary_key") or []
        if primary_key:
            missing_pk_cols = [col for col in primary_key if col not in frame.columns]
            if missing_pk_cols:
                violations.append(
                    SchemaViolation(
                        column=",".join(missing_pk_cols),
                        message="Primary key columns missing",
                        severity="error",
                    )
                )
            else:
                pk_nulls = frame[primary_key].isna().any(axis=1).sum()
                if pk_nulls:
                    violations.append(
                        SchemaViolation(
                            column=",".join(primary_key),
                            message=f"Primary key contains null rows ({pk_nulls})",
                            severity="error",
                        )
                    )
                duplicate_count = frame.duplicated(subset=primary_key).sum()
                if duplicate_count:
                    violations.append(
                        SchemaViolation(
                            column=",".join(primary_key),
                            message=f"Primary key duplicates detected ({duplicate_count})",
                            severity="error",
                        )
                    )

        passed = len([v for v in violations if v.severity == "error"]) == 0
        return SchemaValidationResult(
            source=source,
            table=table,
            path=table_path,
            passed=passed,
            violations=violations,
            row_count=len(frame.index),
            column_count=len(frame.columns),
        )

    @staticmethod
    def _dtype_matches(series: pd.Series, expected: str) -> bool:
        """Check whether series dtype matches expected logical type."""
        expected = expected.lower()
        if expected == "integer":
            return pd_types.is_integer_dtype(series)
        if expected == "float":
            return pd_types.is_float_dtype(series) or pd_types.is_integer_dtype(series)
        if expected == "numeric":
            return pd_types.is_numeric_dtype(series)
        if expected == "string":
            return pd_types.is_string_dtype(series) or pd_types.is_object_dtype(series)
        if expected == "datetime":
            return pd_types.is_datetime64_any_dtype(series)
        if expected == "boolean":
            return pd_types.is_bool_dtype(series)
        # Default: accept any dtype
        return True


def load_schema_validator() -> Optional[SchemaContractValidator]:
    """Safe helper to build validator."""
    return SchemaContractValidator.from_config()

