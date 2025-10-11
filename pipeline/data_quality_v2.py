"""Enhanced data quality checks with advanced statistics and anomaly detection."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import DATA_DIR
from pipeline.schema_contracts import load_schema_validator

LOGGER = logging.getLogger(__name__)
OUTPUT_DIR = DATA_DIR / "evaluation"
PROCESSED_DIR = DATA_DIR / "processed"


@dataclass
class QualityMetrics:
    """Data quality metrics for a table."""

    table_name: str
    row_count: int
    column_count: int

    # Null values
    null_ratios: Dict[str, float]
    high_null_columns: List[str]

    # Duplicates
    duplicate_count: int
    duplicate_ratio: float

    # Data distribution
    gini_coefficient: Optional[float] = None
    concentration_ratio: Optional[float] = None

    # Anomalies
    anomaly_count: int = 0
    anomaly_details: List[str] = None

    # Coverage
    coverage_metrics: Dict[str, float] = None

    # Quality score (0-100)
    quality_score: float = 0.0
    status: str = "unknown"

    def __post_init__(self):
        if self.anomaly_details is None:
            self.anomaly_details = []
        if self.coverage_metrics is None:
            self.coverage_metrics = {}


class DataQualityChecker:
    """Advanced data quality checker with statistical analysis."""

    def __init__(self):
        """Initialize checker."""
        self.processed_dir = PROCESSED_DIR
        self.output_dir = OUTPUT_DIR
        self.alerts = []
        self.schema_validator = load_schema_validator()
        self.schema_results = []

    def compute_gini_coefficient(self, values: pd.Series) -> float:
        """
        Compute Gini coefficient to measure inequality in distribution.

        Args:
            values: Series of counts or values

        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        if len(values) == 0:
            return 0.0

        sorted_values = np.sort(values.dropna().values)
        n = len(sorted_values)

        if n == 0 or sorted_values.sum() == 0:
            return 0.0

        cumsum = np.cumsum(sorted_values)
        gini = (2 * np.sum((n - np.arange(1, n + 1)) * sorted_values)) / (n * cumsum[-1])

        return float(gini)

    def detect_price_anomalies(
        self,
        prices: pd.Series,
        method: str = "zscore",
        threshold: float = 3.0,
    ) -> Tuple[pd.Series, List[str]]:
        """
        Detect price anomalies using statistical methods.

        Args:
            prices: Series of prices
            method: 'zscore' or 'iqr'
            threshold: Z-score threshold or IQR multiplier

        Returns:
            (anomaly_mask, anomaly_details)
        """
        if len(prices) == 0:
            return pd.Series(dtype=bool), []

        anomalies = pd.Series([False] * len(prices), index=prices.index)
        details = []

        if method == "zscore":
            z_scores = np.abs(stats.zscore(prices.dropna()))
            anomaly_indices = prices.dropna().index[z_scores > threshold]
            anomalies.loc[anomaly_indices] = True

            if len(anomaly_indices) > 0:
                details.append(
                    f"Z-score outliers: {len(anomaly_indices)} prices "
                    f"(>{threshold} std from mean)"
                )

        elif method == "iqr":
            Q1 = prices.quantile(0.25)
            Q3 = prices.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            anomalies = (prices < lower_bound) | (prices > upper_bound)
            anomaly_count = anomalies.sum()

            if anomaly_count > 0:
                details.append(
                    f"IQR outliers: {anomaly_count} prices "
                    f"(outside [{lower_bound:.2f}, {upper_bound:.2f}])"
                )

        return anomalies, details

    def check_time_series_completeness(
        self,
        timestamps: pd.Series,
        expected_frequency: str = "D",
    ) -> Dict[str, Any]:
        """
        Check time series completeness for missing dates.

        Args:
            timestamps: Series of timestamps
            expected_frequency: Expected frequency ('D'=daily, 'H'=hourly)

        Returns:
            Dict with completeness metrics
        """
        if len(timestamps) == 0:
            return {
                "complete": False,
                "missing_periods": 0,
                "completeness_ratio": 0.0,
                "date_range": None,
            }

        timestamps = pd.to_datetime(timestamps, errors="coerce").dropna()

        if len(timestamps) == 0:
            return {
                "complete": False,
                "missing_periods": 0,
                "completeness_ratio": 0.0,
                "date_range": None,
            }

        min_date = timestamps.min()
        max_date = timestamps.max()

        # Create complete date range
        expected_dates = pd.date_range(
            start=min_date,
            end=max_date,
            freq=expected_frequency,
        )

        # Find missing dates
        actual_dates = pd.to_datetime(timestamps.dt.date.unique())
        missing_dates = set(expected_dates.date) - set(actual_dates.date)

        completeness_ratio = 1 - (len(missing_dates) / len(expected_dates))

        return {
            "complete": len(missing_dates) == 0,
            "missing_periods": len(missing_dates),
            "completeness_ratio": float(completeness_ratio),
            "date_range": f"{min_date.date()} to {max_date.date()}",
            "total_expected": len(expected_dates),
            "missing_dates": sorted([str(d) for d in list(missing_dates)[:10]]),
        }

    def check_interactions_quality(self) -> QualityMetrics:
        """Check quality of interactions table."""
        interactions_path = self.processed_dir / "interactions.parquet"

        if not interactions_path.exists():
            LOGGER.warning("Interactions file not found: %s", interactions_path)
            return QualityMetrics(
                table_name="interactions",
                row_count=0,
                column_count=0,
                null_ratios={},
                high_null_columns=[],
                duplicate_count=0,
                duplicate_ratio=0.0,
                status="missing",
            )

        df = pd.read_parquet(interactions_path)
        LOGGER.info("Checking interactions table: %d rows", len(df))

        # Basic metrics
        null_ratios = {col: float(df[col].isnull().mean()) for col in df.columns}
        high_null_columns = [col for col, ratio in null_ratios.items() if ratio > 0.05]

        duplicate_count = df.duplicated(subset=["user_id", "dataset_id"]).sum()
        duplicate_ratio = duplicate_count / len(df) if len(df) > 0 else 0.0

        # Distribution analysis
        user_counts = df["user_id"].value_counts()
        item_counts = df["dataset_id"].value_counts()

        user_gini = self.compute_gini_coefficient(user_counts)
        item_gini = self.compute_gini_coefficient(item_counts)

        # Anomaly detection
        anomalies = []
        anomaly_count = 0

        # Check for negative weights
        if "weight" in df.columns:
            negative_weights = (df["weight"] < 0).sum()
            if negative_weights > 0:
                anomalies.append(f"Negative weights: {negative_weights}")
                anomaly_count += negative_weights

        # Check for future timestamps
        if "last_event_time" in df.columns:
            future_times = (pd.to_datetime(df["last_event_time"]) > datetime.now()).sum()
            if future_times > 0:
                anomalies.append(f"Future timestamps: {future_times}")
                anomaly_count += future_times

        # Check time series completeness
        if "last_event_time" in df.columns:
            time_check = self.check_time_series_completeness(
                pd.to_datetime(df["last_event_time"])
            )
            if time_check["completeness_ratio"] < 0.9:
                anomalies.append(
                    f"Time series incomplete: {time_check['completeness_ratio']:.1%} complete, "
                    f"{time_check['missing_periods']} missing days"
                )

        # Coverage metrics
        total_users = df["user_id"].nunique()
        total_items = df["dataset_id"].nunique()

        coverage_metrics = {
            "unique_users": total_users,
            "unique_items": total_items,
            "user_gini": user_gini,
            "item_gini": item_gini,
            "avg_interactions_per_user": len(df) / total_users if total_users > 0 else 0,
            "avg_interactions_per_item": len(df) / total_items if total_items > 0 else 0,
        }

        # Quality score (0-100)
        quality_score = 100.0

        # Penalize for nulls
        avg_null_ratio = np.mean(list(null_ratios.values()))
        quality_score -= avg_null_ratio * 50

        # Penalize for duplicates
        quality_score -= duplicate_ratio * 30

        # Penalize for anomalies
        anomaly_ratio = anomaly_count / len(df) if len(df) > 0 else 0
        quality_score -= anomaly_ratio * 100

        quality_score = max(0, min(100, quality_score))

        # Status
        if quality_score >= 90:
            status = "excellent"
        elif quality_score >= 75:
            status = "good"
        elif quality_score >= 50:
            status = "fair"
        else:
            status = "poor"

        # Generate alerts
        if user_gini > 0.8:
            self.alerts.append({
                "severity": "warning",
                "table": "interactions",
                "message": f"High user concentration (Gini={user_gini:.2f})",
            })

        if item_gini > 0.8:
            self.alerts.append({
                "severity": "warning",
                "table": "interactions",
                "message": f"High item concentration (Gini={item_gini:.2f})",
            })

        if duplicate_ratio > 0.05:
            self.alerts.append({
                "severity": "error",
                "table": "interactions",
                "message": f"High duplicate ratio: {duplicate_ratio:.1%}",
            })

        return QualityMetrics(
            table_name="interactions",
            row_count=len(df),
            column_count=len(df.columns),
            null_ratios=null_ratios,
            high_null_columns=high_null_columns,
            duplicate_count=int(duplicate_count),
            duplicate_ratio=float(duplicate_ratio),
            gini_coefficient=float((user_gini + item_gini) / 2),
            concentration_ratio=float(user_gini),
            anomaly_count=anomaly_count,
            anomaly_details=anomalies,
            coverage_metrics=coverage_metrics,
            quality_score=float(quality_score),
            status=status,
        )

    def check_dataset_features_quality(self) -> QualityMetrics:
        """Check quality of dataset features."""
        features_path = self.processed_dir / "dataset_features.parquet"

        if not features_path.exists():
            return QualityMetrics(
                table_name="dataset_features",
                row_count=0,
                column_count=0,
                null_ratios={},
                high_null_columns=[],
                duplicate_count=0,
                duplicate_ratio=0.0,
                status="missing",
            )

        df = pd.read_parquet(features_path)
        LOGGER.info("Checking dataset_features table: %d rows", len(df))

        # Basic metrics
        null_ratios = {col: float(df[col].isnull().mean()) for col in df.columns}
        high_null_columns = [col for col, ratio in null_ratios.items() if ratio > 0.1]

        duplicate_count = df.duplicated(subset=["dataset_id"]).sum()
        duplicate_ratio = duplicate_count / len(df) if len(df) > 0 else 0.0

        # Price analysis
        anomalies = []
        anomaly_count = 0

        if "price" in df.columns:
            price_anomalies, price_details = self.detect_price_anomalies(
                df["price"], method="iqr", threshold=3.0
            )
            anomaly_count += price_anomalies.sum()
            anomalies.extend(price_details)

        # Text feature quality
        text_features = ["description", "tag"]
        for col in text_features:
            if col in df.columns:
                empty_ratio = (df[col].fillna("").str.strip() == "").mean()
                if empty_ratio > 0.2:
                    anomalies.append(f"{col} empty ratio: {empty_ratio:.1%}")

        quality_score = 100.0 - (np.mean(list(null_ratios.values())) * 50) - (duplicate_ratio * 30)
        quality_score = max(0, min(100, quality_score))

        status = "excellent" if quality_score >= 90 else "good" if quality_score >= 75 else "fair" if quality_score >= 50 else "poor"

        return QualityMetrics(
            table_name="dataset_features",
            row_count=len(df),
            column_count=len(df.columns),
            null_ratios=null_ratios,
            high_null_columns=high_null_columns,
            duplicate_count=int(duplicate_count),
            duplicate_ratio=float(duplicate_ratio),
            anomaly_count=int(anomaly_count),
            anomaly_details=anomalies,
            quality_score=float(quality_score),
            status=status,
        )

    def check_user_profile_quality(self) -> QualityMetrics:
        """Check quality of user profile."""
        profile_path = self.processed_dir / "user_profile.parquet"

        if not profile_path.exists():
            return QualityMetrics(
                table_name="user_profile",
                row_count=0,
                column_count=0,
                null_ratios={},
                high_null_columns=[],
                duplicate_count=0,
                duplicate_ratio=0.0,
                status="missing",
            )

        df = pd.read_parquet(profile_path)
        LOGGER.info("Checking user_profile table: %d rows", len(df))

        null_ratios = {col: float(df[col].isnull().mean()) for col in df.columns}
        high_null_columns = [col for col, ratio in null_ratios.items() if ratio > 0.2]

        duplicate_count = df.duplicated(subset=["user_id"]).sum()
        duplicate_ratio = duplicate_count / len(df) if len(df) > 0 else 0.0

        quality_score = 100.0 - (np.mean(list(null_ratios.values())) * 50)
        quality_score = max(0, min(100, quality_score))

        status = "excellent" if quality_score >= 90 else "good" if quality_score >= 75 else "fair"

        return QualityMetrics(
            table_name="user_profile",
            row_count=len(df),
            column_count=len(df.columns),
            null_ratios=null_ratios,
            high_null_columns=high_null_columns,
            duplicate_count=int(duplicate_count),
            duplicate_ratio=float(duplicate_ratio),
            quality_score=float(quality_score),
            status=status,
        )

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        LOGGER.info("Running comprehensive data quality checks...")

        schema_contracts = self._run_schema_contract_checks()

        # Check all tables
        interactions_metrics = self.check_interactions_quality()
        dataset_metrics = self.check_dataset_features_quality()
        user_metrics = self.check_user_profile_quality()

        # Overall summary
        all_metrics = [interactions_metrics, dataset_metrics, user_metrics]
        overall_score = np.mean([m.quality_score for m in all_metrics])

        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_quality_score": float(overall_score),
            "overall_status": (
                "excellent" if overall_score >= 90
                else "good" if overall_score >= 75
                else "fair" if overall_score >= 50
                else "poor"
            ),
            "tables": {
                "interactions": asdict(interactions_metrics),
                "dataset_features": asdict(dataset_metrics),
                "user_profile": asdict(user_metrics),
            },
            "schema_contracts": schema_contracts,
            "alerts": self.alerts,
            "recommendations": self._generate_recommendations(all_metrics),
        }

        return report

    def _generate_recommendations(self, metrics: List[QualityMetrics]) -> List[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []

        for metric in metrics:
            if metric.status == "poor":
                recommendations.append(
                    f"CRITICAL: {metric.table_name} has poor quality (score={metric.quality_score:.1f}). "
                    "Review data collection process."
                )

            if metric.duplicate_ratio > 0.05:
                recommendations.append(
                    f"Remove duplicates from {metric.table_name} "
                    f"({metric.duplicate_count} duplicates, {metric.duplicate_ratio:.1%})"
                )

            if metric.high_null_columns:
                recommendations.append(
                    f"Investigate high null ratios in {metric.table_name}: "
                    f"{', '.join(metric.high_null_columns[:3])}"
                )

            if metric.gini_coefficient and metric.gini_coefficient > 0.8:
                recommendations.append(
                    f"High concentration in {metric.table_name} (Gini={metric.gini_coefficient:.2f}). "
                    "Consider cold-start strategies for long-tail items."
                )

        return recommendations

    def _run_schema_contract_checks(self) -> Dict[str, Any]:
        """Validate extracted source tables against schema contracts."""
        if not self.schema_validator:
            LOGGER.info("Schema contract validator not configured; skipping contract checks.")
            return {"enabled": False, "results": [], "summary": {"total": 0, "passed": 0}}

        results = self.schema_validator.validate_all()
        self.schema_results = results
        summary = {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
        }

        for result in results:
            if not result.violations:
                continue
            for violation in result.violations:
                self.alerts.append(
                    {
                        "table": f"{result.source}.{result.table}",
                        "message": violation.message if violation.column is None else f"{violation.column}: {violation.message}",
                        "severity": violation.severity,
                        "expected_type": violation.expected_type,
                        "actual_type": violation.actual_type,
                    }
                )

        return {
            "enabled": True,
            "summary": summary,
            "results": [res.to_dict() for res in results],
        }


def main() -> None:
    """Run data quality checks and generate report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checker = DataQualityChecker()
    report = checker.generate_report()

    # Save JSON report
    json_path = OUTPUT_DIR / "data_quality_report_v2.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    LOGGER.info("Data quality report saved to %s", json_path)

    # Generate HTML report
    html_report = generate_html_report(report)
    html_path = OUTPUT_DIR / "data_quality_report.html"
    html_path.write_text(html_report)
    LOGGER.info("HTML report saved to %s", html_path)

    # Export Prometheus metrics snapshot for AlertManager scraping
    prom_lines = []
    prom_lines.append("# HELP data_quality_score Quality score per table (0-100)\n")
    prom_lines.append("# TYPE data_quality_score gauge\n")
    for table_name, metrics in report["tables"].items():
        prom_lines.append(
            f"data_quality_score{{table=\"{table_name}\"}} {metrics['quality_score']:.3f}\n"
        )

    prom_lines.append("\n# HELP data_quality_anomalies Number of anomalies detected per table\n")
    prom_lines.append("# TYPE data_quality_anomalies gauge\n")
    for table_name, metrics in report["tables"].items():
        prom_lines.append(
            f"data_quality_anomalies{{table=\"{table_name}\"}} {metrics.get('anomaly_count', 0)}\n"
        )

    prom_lines.append("\n# HELP data_quality_alerts Total alerts generated during quality check\n")
    prom_lines.append("# TYPE data_quality_alerts gauge\n")
    alert_counts: Dict[str, int] = {}
    for alert in report.get("alerts", []):
        severity = alert.get("severity", "info")
        alert_counts[severity] = alert_counts.get(severity, 0) + 1
    for severity, count in alert_counts.items():
        prom_lines.append(
            f"data_quality_alerts{{severity=\"{severity}\"}} {count}\n"
        )

    schema_contracts = report.get("schema_contracts", {})
    if schema_contracts.get("enabled"):
        prom_lines.append("\n# HELP data_schema_contract_status Schema contract compliance per table (1=pass,0=fail)\n")
        prom_lines.append("# TYPE data_schema_contract_status gauge\n")
        for result in schema_contracts.get("results", []):
            table_label = f"{result['source']}.{result['table']}"
            status_value = 1 if result.get("passed") else 0
            prom_lines.append(
                f"data_schema_contract_status{{table=\"{table_label}\"}} {status_value}\n"
            )

        prom_lines.append("\n# HELP data_schema_contract_violations Number of schema contract violations per table\n")
        prom_lines.append("# TYPE data_schema_contract_violations gauge\n")
        for result in schema_contracts.get("results", []):
            table_label = f"{result['source']}.{result['table']}"
            violation_count = len(result.get("violations", []))
            prom_lines.append(
                f"data_schema_contract_violations{{table=\"{table_label}\"}} {violation_count}\n"
            )

    prom_path = OUTPUT_DIR / "data_quality_metrics.prom"
    prom_path.write_text("".join(prom_lines))
    LOGGER.info("Prometheus snapshot exported to %s", prom_path)

    # Print summary
    print("\n" + "=" * 80)
    print(f"DATA QUALITY REPORT - {report['timestamp']}")
    print("=" * 80)
    print(f"Overall Quality Score: {report['overall_quality_score']:.1f}/100 ({report['overall_status'].upper()})")
    print("\nTable Scores:")
    for table_name, metrics in report['tables'].items():
        print(f"  - {table_name:20s}: {metrics['quality_score']:5.1f}/100 ({metrics['status']})")

    schema_contracts = report.get("schema_contracts", {})
    if schema_contracts.get("enabled"):
        summary = schema_contracts.get("summary", {})
        total = summary.get("total", 0)
        passed = summary.get("passed", 0)
        failed = summary.get("failed", 0)
        print(f"\nSchema Contracts: {passed}/{total} tables passed ({failed} failed)")
        failing = [res for res in schema_contracts.get("results", []) if not res.get("passed")]
        for res in failing[:5]:
            table_label = f"{res['source']}.{res['table']}"
            print(f"  - {table_label}: {len(res.get('violations', []))} violation(s)")
            for violation in res.get("violations", [])[:2]:
                column = violation.get("column") or "<table>"
                print(f"      [{violation.get('severity','info').upper()}] {column} - {violation.get('message')}")

    if report['alerts']:
        print(f"\nAlerts ({len(report['alerts'])}):")
        for alert in report['alerts'][:5]:
            print(f"  [{alert['severity'].upper()}] {alert['table']}: {alert['message']}")

    if report['recommendations']:
        print(f"\nRecommendations ({len(report['recommendations'])}):")
        for i, rec in enumerate(report['recommendations'][:5], 1):
            print(f"  {i}. {rec}")

    print("=" * 80 + "\n")


def generate_html_report(report: Dict[str, Any]) -> str:
    """Generate HTML visualization of quality report."""
    schema_contracts = report.get("schema_contracts", {})
    schema_section_html = ""
    if schema_contracts.get("enabled"):
        summary = schema_contracts.get("summary", {})
        table_rows = ""
        for result in schema_contracts.get("results", []):
            status_label = "PASS" if result.get("passed") else "FAIL"
            status_class = "good" if result.get("passed") else "poor"
            table_rows += (
                f"<tr>"
                f"<td>{result['source']}.{result['table']}</td>"
                f"<td class='{status_class}'>{status_label}</td>"
                f"<td>{result.get('row_count', 0)}</td>"
                f"<td>{result.get('column_count', 0)}</td>"
                f"<td>{len(result.get('violations', []))}</td>"
                f"</tr>"
            )

        violation_blocks = ""
        for result in schema_contracts.get("results", []):
            if not result.get("violations"):
                continue
            table_label = f"{result['source']}.{result['table']}"
            for violation in result.get("violations", []):
                column = violation.get("column") or "表级"
                severity = violation.get("severity", "warning")
                message = violation.get("message")
                css_class = "error" if severity == "error" else "warning"
                violation_blocks += (
                    f"<div class='alert {css_class}'>"
                    f"<strong>{table_label}</strong> - {column}: {message}"
                    "</div>"
                )

        schema_section_html = f"""
        <h2>Schema Contract Compliance</h2>
        <p>通过 {summary.get('passed', 0)} / {summary.get('total', 0)}（失败 {summary.get('failed', 0)}）</p>
        <table>
            <thead>
                <tr>
                    <th>表</th>
                    <th>状态</th>
                    <th>行数</th>
                    <th>列数</th>
                    <th>违规数量</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        {violation_blocks}
        """

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Quality Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .score {{ font-size: 48px; font-weight: bold; color: #4CAF50; }}
        .score.good {{ color: #8BC34A; }}
        .score.fair {{ color: #FFC107; }}
        .score.poor {{ color: #F44336; }}
        .metric-box {{ display: inline-block; margin: 10px; padding: 15px; background: #f9f9f9; border-radius: 5px; border-left: 4px solid #4CAF50; }}
        .alert {{ padding: 10px; margin: 5px 0; border-radius: 4px; }}
        .alert.warning {{ background: #fff3cd; border-left: 4px solid #FFC107; }}
        .alert.error {{ background: #f8d7da; border-left: 4px solid #F44336; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        td.good {{ color: #4CAF50; font-weight: bold; }}
        td.poor {{ color: #F44336; font-weight: bold; }}
        .recommendation {{ background: #e3f2fd; padding: 10px; margin: 5px 0; border-radius: 4px; border-left: 4px solid #2196F3; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Data Quality Report</h1>
        <p>Generated: {report['timestamp']}</p>

        <h2>Overall Quality</h2>
        <div class="score {report['overall_status']}">
            {report['overall_quality_score']:.1f}/100
        </div>
        <p>Status: <strong>{report['overall_status'].upper()}</strong></p>

        <h2>Table Quality Scores</h2>
        <table>
            <tr>
                <th>Table</th>
                <th>Score</th>
                <th>Status</th>
                <th>Rows</th>
                <th>Duplicates</th>
                <th>Anomalies</th>
            </tr>
    """

    for table_name, metrics in report['tables'].items():
        html += f"""
            <tr>
                <td>{table_name}</td>
                <td>{metrics['quality_score']:.1f}/100</td>
                <td>{metrics['status']}</td>
                <td>{metrics['row_count']:,}</td>
                <td>{metrics['duplicate_count']:,} ({metrics['duplicate_ratio']:.1%})</td>
                <td>{metrics['anomaly_count']}</td>
            </tr>
        """

    html += """
        </table>
    """

    if schema_section_html:
        html += schema_section_html

    # Alerts
    if report['alerts']:
        html += f"""
        <h2>Alerts ({len(report['alerts'])})</h2>
        """
        for alert in report['alerts']:
            html += f"""
        <div class="alert {alert['severity']}">
            <strong>[{alert['severity'].upper()}]</strong> {alert['table']}: {alert['message']}
        </div>
            """

    # Recommendations
    if report['recommendations']:
        html += f"""
        <h2>Recommendations ({len(report['recommendations'])})</h2>
        """
        for i, rec in enumerate(report['recommendations'], 1):
            html += f"""
        <div class="recommendation">
            {i}. {rec}
        </div>
            """

    html += """
    </div>
</body>
</html>
    """

    return html


if __name__ == "__main__":
    main()
