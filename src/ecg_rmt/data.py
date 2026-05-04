from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import TARGET_ABNORMAL, TARGET_NORMAL


def parse_scp_codes(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(key): float(score) for key, score in value.items()}
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if not isinstance(value, str) or not value.strip():
        return {}
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, dict):
        return {}
    return {str(key): float(score) for key, score in parsed.items()}


def load_ptbxl_metadata(data_root: Path) -> pd.DataFrame:
    path = data_root / "ptbxl_database.csv"
    frame = pd.read_csv(path, index_col="ecg_id")
    frame.index = frame.index.astype(int)
    frame = frame.reset_index()
    frame["scp_codes"] = frame["scp_codes"].map(parse_scp_codes)
    return frame


def load_scp_statements(data_root: Path) -> pd.DataFrame:
    path = data_root / "scp_statements.csv"
    statements = pd.read_csv(path, index_col=0)
    statements.index = statements.index.astype(str)
    return statements


def diagnostic_classes_for_codes(
    scp_codes: dict[str, float],
    statements: pd.DataFrame,
) -> list[str]:
    classes: set[str] = set()
    for code in scp_codes:
        if code not in statements.index:
            continue
        row = statements.loc[code]
        diagnostic = bool(row.get("diagnostic", False))
        superclass = row.get("diagnostic_class")
        if diagnostic and isinstance(superclass, str) and superclass:
            classes.add(superclass)
    return sorted(classes)


def attach_binary_target(
    metadata: pd.DataFrame,
    statements: pd.DataFrame,
) -> pd.DataFrame:
    frame = metadata.copy()
    frame["diagnostic_classes"] = frame["scp_codes"].map(
        lambda codes: diagnostic_classes_for_codes(codes, statements)
    )

    def target_from_classes(classes: list[str]) -> float:
        if not classes:
            return np.nan
        return 0.0 if classes == ["NORM"] else 1.0

    frame["target"] = frame["diagnostic_classes"].map(target_from_classes)
    frame["target_name"] = frame["target"].map(
        {0.0: TARGET_NORMAL, 1.0: TARGET_ABNORMAL}
    )
    return frame


def supervised_records(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.dropna(subset=["target"]).copy()
    result["target"] = result["target"].astype(int)
    return result


def official_split(frame: pd.DataFrame) -> pd.Series:
    folds = frame["strat_fold"].astype(int)
    split = pd.Series("train", index=frame.index, dtype=object)
    split[folds == 9] = "validation"
    split[folds == 10] = "test"
    return split


def resolve_waveform_path(row: pd.Series, sampling_rate: int) -> str:
    if sampling_rate == 500:
        key = "filename_hr"
    else:
        key = "filename_lr"
    return str(row[key])
