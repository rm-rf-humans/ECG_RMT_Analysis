from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from .config import ProjectPaths
from .data import (
    attach_binary_target,
    load_ptbxl_metadata,
    load_scp_statements,
    official_split,
    resolve_waveform_path,
    supervised_records,
)
from .features import extract_ecg_features
from .modeling import fit_and_evaluate
from .plots import save_class_balance, save_covariance_spectrum


def _read_signal(data_root: Path, record_path: str) -> np.ndarray:
    signal, _ = wfdb.rdsamp(str(data_root / record_path))
    return signal


def build_modeling_table(
    data_root: Path,
    sampling_rate: int,
    max_records: int | None = None,
) -> pd.DataFrame:
    metadata = load_ptbxl_metadata(data_root)
    statements = load_scp_statements(data_root)
    frame = supervised_records(attach_binary_target(metadata, statements))
    if max_records is not None:
        frame = frame.head(max_records).copy()

    feature_rows: list[dict[str, float]] = []
    for _, row in frame.iterrows():
        record_path = resolve_waveform_path(row, sampling_rate=sampling_rate)
        try:
            signal = _read_signal(data_root, record_path)
            features = extract_ecg_features(signal, sampling_rate=sampling_rate)
            load_error = 0
        except Exception:
            features = {}
            load_error = 1
        features["ecg_id"] = int(row["ecg_id"])
        features["_load_error"] = load_error
        feature_rows.append(features)

    features = pd.DataFrame(feature_rows)
    merged = frame.merge(features, on="ecg_id", how="inner")
    merged["split"] = official_split(merged)
    return merged


def run_pipeline(
    data_root: Path,
    output_dir: Path,
    sampling_rate: int = 100,
    max_records: int | None = None,
    quick: bool = False,
) -> None:
    paths = ProjectPaths(data_root=data_root, output_dir=output_dir)
    paths.make_dirs()
    table = build_modeling_table(
        data_root=data_root,
        sampling_rate=sampling_rate,
        max_records=max_records,
    )
    processed = output_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    table.to_csv(processed / "modeling_table.csv.gz", index=False)

    summary = {
        "modeling_records": int(table.shape[0]),
        "target_counts": table["target_name"].value_counts().to_dict(),
        "split_counts": table.groupby(["split", "target_name"]).size().unstack(fill_value=0).to_dict(),
    }
    (paths.metrics_dir / "data_summary.json").write_text(json.dumps(summary, indent=2))
    save_class_balance(table, paths.figures_dir / "class_balance.png")
    fit_and_evaluate(table, paths, quick=quick)

    spectrum_path = paths.rmt_dir / "covariance_spectrum.csv"
    if spectrum_path.exists():
        save_covariance_spectrum(pd.read_csv(spectrum_path), paths.figures_dir / "covariance_spectrum.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/ptb-xl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--sampling-rate", type=int, choices=[100, 500], default=100)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(
        data_root=args.data_root,
        output_dir=args.output_dir,
        sampling_rate=args.sampling_rate,
        max_records=args.max_records,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
