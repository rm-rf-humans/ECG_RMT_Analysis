from __future__ import annotations

import argparse
from pathlib import Path

import wfdb


PTBXL_RECORD = "ptb-xl/1.0.3"


def download_ptbxl(data_root: Path, waveforms: str = "none", max_records: int | None = None) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database(
        PTBXL_RECORD,
        dl_dir=str(data_root),
        records=["ptbxl_database.csv", "scp_statements.csv"],
    )
    if waveforms == "none":
        return
    metadata = data_root / "ptbxl_database.csv"
    import pandas as pd

    frame = pd.read_csv(metadata)
    column = "filename_hr" if waveforms == "500" else "filename_lr"
    records = frame[column].astype(str).tolist()
    if max_records is not None:
        records = records[:max_records]
    wfdb.dl_database(PTBXL_RECORD, dl_dir=str(data_root), records=records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/ptb-xl"))
    parser.add_argument("--waveforms", choices=["none", "100", "500"], default="none")
    parser.add_argument("--max-records", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    download_ptbxl(args.data_root, waveforms=args.waveforms, max_records=args.max_records)


if __name__ == "__main__":
    main()
