from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd


METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "n_records",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("recovered_modal"))
    parser.add_argument("--tracking-dir", type=Path, default=Path("mlruns"))
    parser.add_argument("--experiment-name", default="ptbxl_ecg_rmt_modal_run")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[name] = value
    return flat


def metric_value(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def log_metrics_from_row(row: pd.Series, prefix: str, columns: list[str]) -> None:
    for column in columns:
        if column not in row:
            continue
        value = metric_value(row[column])
        if value is not None:
            mlflow.log_metric(f"{prefix}_{column}", value)


def log_params(values: dict[str, Any], prefix: str = "") -> None:
    for key, value in flatten(values, prefix).items():
        if value is None:
            continue
        mlflow.log_param(key, str(value))


def log_count_map(values: dict[str, Any], prefix: str) -> None:
    for key, value in values.items():
        if isinstance(value, dict):
            log_count_map(value, f"{prefix}_{key}")
        else:
            logged = metric_value(value)
            if logged is not None:
                mlflow.log_metric(f"{prefix}_{key}", logged)


def sanitize_paths(tracking_dir: Path) -> None:
    for tag in tracking_dir.rglob("tags/mlflow.source.git.commit"):
        tag.unlink()

    resolved_uri = tracking_dir.resolve().as_uri()
    resolved_path = str(tracking_dir.resolve())
    replacements = {
        resolved_uri: "file:./mlruns",
        resolved_path: "./mlruns",
    }
    for path in tracking_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        updated = text
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated)


def export_logs(source_dir: Path, tracking_dir: Path, experiment_name: str, overwrite: bool) -> None:
    metrics_dir = source_dir / "metrics"
    rmt_dir = source_dir / "rmt"
    if overwrite and tracking_dir.exists():
        shutil.rmtree(tracking_dir)
    tracking_dir.mkdir(parents=True, exist_ok=True)

    model_metrics = pd.read_csv(metrics_dir / "model_metrics.csv")
    model_params = load_json(metrics_dir / "model_params.json")
    data_summary = load_json(metrics_dir / "data_summary.json")
    model_rmt = pd.read_csv(metrics_dir / "model_rmt_diagnostics.csv")
    filtering = pd.read_csv(metrics_dir / "best_model_rmt_filtering.csv")
    covariance = pd.read_csv(rmt_dir / "covariance_summary.csv").iloc[0]

    os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "project")
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment(experiment_name)

    for model_name, group in model_metrics.groupby("model", sort=False):
        with mlflow.start_run(run_name=f"model_{model_name}"):
            mlflow.set_tag("run_group", "model_selection")
            mlflow.set_tag("dataset", "PTB-XL 1.0.3")
            mlflow.set_tag("fold_protocol", "official_ptbxl_folds")
            mlflow.log_param("model", model_name)
            params = model_params.get(model_name, {})
            if "best_cv_roc_auc" in params:
                mlflow.log_metric("best_cv_roc_auc", float(params["best_cv_roc_auc"]))
            for key in ["n_trials", "n_complete_trials", "optuna_jobs", "search"]:
                if key in params:
                    mlflow.log_param(key, str(params[key]))
            if isinstance(params.get("best_params"), dict):
                log_params(params["best_params"], "best")
            for _, row in group.iterrows():
                log_metrics_from_row(row, str(row["split"]), METRIC_COLUMNS)
            mlflow.log_artifact(metrics_dir / "model_metrics.csv", artifact_path="tables")
            mlflow.log_artifact(metrics_dir / "model_params.json", artifact_path="tables")

    with mlflow.start_run(run_name="dataset_and_covariance_spectrum"):
        mlflow.set_tag("run_group", "data_rmt_summary")
        mlflow.set_tag("dataset", "PTB-XL 1.0.3")
        for key in [
            "metadata_records",
            "labeled_records",
            "modeling_records",
            "n_numeric_features",
            "n_categorical_features",
            "n_model_features",
        ]:
            if key in data_summary:
                mlflow.log_metric(key, float(data_summary[key]))
        log_count_map(data_summary.get("target_counts", {}), "target")
        log_count_map(data_summary.get("diagnostic_class_counts", {}), "diagnostic_class")
        for column in covariance.index:
            value = metric_value(covariance[column])
            if value is not None:
                mlflow.log_metric(f"covariance_{column}", value)
            else:
                mlflow.log_param(f"covariance_{column}", str(covariance[column]))
        mlflow.log_artifact(metrics_dir / "data_summary.json", artifact_path="tables")
        mlflow.log_artifact(rmt_dir / "covariance_summary.csv", artifact_path="tables")
        mlflow.log_artifact(rmt_dir / "covariance_spectrum.csv", artifact_path="tables")

    with mlflow.start_run(run_name="rmt_subspace_model_diagnostics"):
        mlflow.set_tag("run_group", "rmt_model_diagnostics")
        for _, row in model_rmt.iterrows():
            prefix = f"{row['model']}_{row['split']}"
            for column in [
                "original_roc_auc",
                "signal_subspace_roc_auc",
                "bulk_subspace_roc_auc",
                "signal_score_correlation",
                "bulk_score_correlation",
                "signal_components",
                "bulk_components",
                "signal_variance_fraction",
            ]:
                value = metric_value(row[column])
                if value is not None:
                    mlflow.log_metric(f"{prefix}_{column}", value)
        mlflow.log_artifact(metrics_dir / "model_rmt_diagnostics.csv", artifact_path="tables")

    with mlflow.start_run(run_name="xgboost_rmt_filtering_experiment"):
        mlflow.set_tag("run_group", "best_model_filtering")
        for _, row in filtering.iterrows():
            prefix = f"{row['representation']}_{row['split']}"
            log_metrics_from_row(row, prefix, METRIC_COLUMNS)
        mlflow.log_artifact(metrics_dir / "best_model_rmt_filtering.csv", artifact_path="tables")

    sanitize_paths(tracking_dir)


def main() -> None:
    args = parse_args()
    export_logs(
        source_dir=args.source_dir,
        tracking_dir=args.tracking_dir,
        experiment_name=args.experiment_name,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
