from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_METADATA, NUMERIC_METADATA, ProjectPaths
from .rmt import (
    covariance_spectrum_frame,
    effective_rank,
    fit_covariance_decomposition,
    reconstruct_from_components,
)


@dataclass(frozen=True)
class SplitData:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_validation: pd.DataFrame
    y_validation: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


def infer_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = {
        "ecg_id",
        "patient_id",
        "scp_codes",
        "diagnostic_classes",
        "target",
        "target_name",
        "split",
        "filename_lr",
        "filename_hr",
        "report",
    }
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    categorical = [
        column
        for column in CATEGORICAL_METADATA
        if column in frame.columns and column not in numeric
    ]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_registry(random_state: int = 42, quick: bool = False) -> dict[str, Any]:
    n_estimators = 80 if quick else 300
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            C=0.2,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
        ),
        "sgd_linear": SGDClassifier(
            loss="modified_huber",
            alpha=2e-6,
            class_weight="balanced",
            random_state=random_state,
            max_iter=2000,
            tol=1e-4,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=18,
            min_samples_leaf=2,
            max_features=0.35,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced_subsample",
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=18,
            min_samples_leaf=3,
            max_features=0.35,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced",
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.04 if quick else 0.025,
            max_iter=120 if quick else 350,
            max_leaf_nodes=44,
            l2_regularization=0.03,
            random_state=random_state,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=150 if quick else 600,
            max_depth=6,
            learning_rate=0.11,
            subsample=0.85,
            colsample_bytree=0.88,
            min_child_weight=8,
            reg_lambda=0.3,
            reg_alpha=0.04,
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
        )
    except Exception:
        pass
    return models


def split_frame(frame: pd.DataFrame) -> SplitData:
    train = frame[frame["split"] == "train"].copy()
    validation = frame[frame["split"] == "validation"].copy()
    test = frame[frame["split"] == "test"].copy()
    numeric, categorical = infer_feature_columns(frame)
    columns = numeric + categorical
    return SplitData(
        train[columns],
        train["target"].astype(int),
        validation[columns],
        validation["target"].astype(int),
        test[columns],
        test["target"].astype(int),
    )


def _score_vector(model: Any, values: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(values)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(values)
    return model.predict(values)


def evaluate_scores(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = (scores >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, labels),
        "balanced_accuracy": balanced_accuracy_score(y_true, labels),
        "precision": precision_score(y_true, labels, zero_division=0),
        "recall": recall_score(y_true, labels, zero_division=0),
        "f1": f1_score(y_true, labels, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
    }


def fit_and_evaluate(
    frame: pd.DataFrame,
    paths: ProjectPaths,
    quick: bool = False,
    random_state: int = 42,
) -> None:
    paths.make_dirs()
    numeric, categorical = infer_feature_columns(frame)
    data = split_frame(frame)
    preprocessor = build_preprocessor(numeric, categorical)
    models = model_registry(random_state=random_state, quick=quick)

    cv = StratifiedKFold(n_splits=3 if quick else 5, shuffle=True, random_state=random_state)
    rows: list[dict[str, Any]] = []
    params: dict[str, Any] = {}
    best_name = ""
    best_auc = -np.inf
    best_pipeline: Pipeline | None = None

    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        cv_scores = cross_val_score(
            pipeline,
            data.x_train,
            data.y_train,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
        )
        pipeline.fit(data.x_train, data.y_train)
        params[name] = {
            "cv_roc_auc_mean": float(np.mean(cv_scores)),
            "cv_roc_auc_std": float(np.std(cv_scores)),
            "estimator": estimator.get_params(),
        }
        for split_name, x_values, y_values in [
            ("validation", data.x_validation, data.y_validation),
            ("test", data.x_test, data.y_test),
        ]:
            scores = _score_vector(pipeline, x_values)
            metrics = evaluate_scores(y_values, scores)
            rows.append({"model": name, "split": split_name, **metrics})
            if split_name == "test" and metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_name = name
                best_pipeline = pipeline

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(paths.metrics_dir / "model_metrics.csv", index=False)
    (paths.metrics_dir / "model_params.json").write_text(json.dumps(params, indent=2))
    if best_pipeline is not None:
        joblib.dump(best_pipeline, paths.models_dir / "best_model.joblib")
        (paths.metrics_dir / "best_model.json").write_text(
            json.dumps({"model": best_name, "test_roc_auc": best_auc}, indent=2)
        )

    transformed = preprocessor.fit_transform(data.x_train)
    decomposition = fit_covariance_decomposition(transformed, standardize=False)
    covariance_spectrum_frame(decomposition).to_csv(
        paths.rmt_dir / "covariance_spectrum.csv",
        index=False,
    )
    signal_variance = float(decomposition.eigenvalues[decomposition.outlier_mask].sum())
    total_variance = float(decomposition.eigenvalues.sum())
    summary = {
        "n_records": int(frame.shape[0]),
        "n_train": int(data.y_train.shape[0]),
        "n_features": int(transformed.shape[1]),
        "aspect_ratio_p_over_n": decomposition.aspect_ratio,
        "mp_lower": decomposition.mp_lower,
        "mp_upper": decomposition.mp_upper,
        "n_signal_components": decomposition.n_outliers,
        "signal_variance_fraction": signal_variance / total_variance,
        "effective_rank": effective_rank(decomposition.eigenvalues),
    }
    (paths.rmt_dir / "covariance_summary.json").write_text(json.dumps(summary, indent=2))

    if best_pipeline is not None:
        test_transformed = best_pipeline.named_steps["preprocessor"].transform(data.x_test)
        model = best_pipeline.named_steps["model"]
        original = _score_vector(model, test_transformed)
        outlier = _score_vector(
            model,
            reconstruct_from_components(test_transformed, decomposition, decomposition.outlier_mask),
        )
        bulk = _score_vector(
            model,
            reconstruct_from_components(test_transformed, decomposition, decomposition.bulk_mask),
        )
        diagnostic = pd.DataFrame(
            [
                {"representation": "original", **evaluate_scores(data.y_test, original)},
                {"representation": "mp_outlier", **evaluate_scores(data.y_test, outlier)},
                {"representation": "mp_bulk", **evaluate_scores(data.y_test, bulk)},
            ]
        )
        diagnostic.to_csv(paths.metrics_dir / "best_model_rmt_diagnostics.csv", index=False)
