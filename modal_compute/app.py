from __future__ import annotations

import json
import shutil
from pathlib import Path

import modal


ROOT = Path("/ptbxl_project")
DATA_ROOT = ROOT / "data" / "raw" / "ptb-xl"
OUTPUT_DIR = ROOT / "outputs"
VOLUME = modal.Volume.from_name("ptbxl-rmt-ecg-volume", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1")
    .pip_install_from_requirements("modal_compute/requirements.txt")
    .add_local_dir("src", remote_path="/workspace/src")
    .add_local_dir("scripts", remote_path="/workspace/scripts")
)

app = modal.App("ptbxl-ecg-rmt-analysis", image=image)


def _pythonpath() -> dict[str, str]:
    return {"PYTHONPATH": "/workspace/src"}


@app.function(volumes={str(ROOT): VOLUME}, timeout=60 * 60 * 6)
def download_remote(waveforms: str = "none", max_records: int | None = None) -> dict[str, object]:
    import os
    import subprocess

    command = [
        "python",
        "/workspace/scripts/download_ptbxl.py",
        "--data-root",
        str(DATA_ROOT),
        "--waveforms",
        waveforms,
    ]
    if max_records is not None:
        command += ["--max-records", str(max_records)]
    subprocess.check_call(command, env={**os.environ, **_pythonpath()})
    VOLUME.commit()
    return {"data_root": str(DATA_ROOT), "waveforms": waveforms}


@app.function(volumes={str(ROOT): VOLUME}, timeout=60 * 60 * 12, cpu=16, memory=32768)
def train_remote(
    sampling_rate: int = 100,
    max_records: int | None = None,
    quick: bool = False,
) -> dict[str, object]:
    import os
    import subprocess

    command = [
        "python",
        "/workspace/scripts/run_pipeline.py",
        "--data-root",
        str(DATA_ROOT),
        "--output-dir",
        str(OUTPUT_DIR),
        "--sampling-rate",
        str(sampling_rate),
    ]
    if max_records is not None:
        command += ["--max-records", str(max_records)]
    if quick:
        command += ["--quick"]
    subprocess.check_call(command, env={**os.environ, **_pythonpath()})
    VOLUME.commit()
    metrics = OUTPUT_DIR / "metrics" / "model_metrics.csv"
    return {"outputs": str(OUTPUT_DIR), "metrics_exists": metrics.exists()}


@app.function(volumes={str(ROOT): VOLUME}, timeout=60 * 60 * 12, cpu=16, memory=32768, gpu="T4")
def train_gpu_remote(
    sampling_rate: int = 100,
    max_records: int | None = None,
    quick: bool = False,
) -> dict[str, object]:
    return train_remote.local(
        sampling_rate=sampling_rate,
        max_records=max_records,
        quick=quick,
    )


@app.function(volumes={str(ROOT): VOLUME}, timeout=60 * 60 * 6)
def pack_outputs(zip_name: str = "ptbxl_outputs.zip") -> dict[str, object]:
    import zipfile

    if not zip_name.endswith(".zip"):
        zip_name = f"{zip_name}.zip"
    zip_path = OUTPUT_DIR / zip_name
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in OUTPUT_DIR.rglob("*"):
            if item.is_file() and item != zip_path:
                archive.write(item, item.relative_to(OUTPUT_DIR))
    VOLUME.commit()
    return {"zip_path": str(zip_path), "size_mb": round(zip_path.stat().st_size / 2**20, 2)}


@app.function(volumes={str(ROOT): VOLUME}, timeout=60 * 10)
def status_remote() -> dict[str, object]:
    paths = {
        "data_root": DATA_ROOT.exists(),
        "outputs": OUTPUT_DIR.exists(),
        "metrics": (OUTPUT_DIR / "metrics").exists(),
        "modeling_table": (OUTPUT_DIR / "processed" / "modeling_table.csv.gz").exists(),
    }
    return paths


@app.local_entrypoint()
def download(waveforms: str = "none", max_records: int | None = None) -> None:
    print(json.dumps(download_remote.remote(waveforms=waveforms, max_records=max_records), indent=2))


@app.local_entrypoint()
def train(
    sampling_rate: int = 100,
    max_records: int | None = None,
    quick: bool = False,
    n_trials: int = 8,
) -> None:
    _ = n_trials
    print(
        json.dumps(
            train_remote.remote(
                sampling_rate=sampling_rate,
                max_records=max_records,
                quick=quick,
            ),
            indent=2,
        )
    )


@app.local_entrypoint()
def full_run(
    waveforms: str = "100",
    sampling_rate: int = 100,
    max_records: int | None = None,
    quick: bool = False,
) -> None:
    download_remote.remote(waveforms=waveforms, max_records=max_records)
    print(
        json.dumps(
            train_remote.remote(
                sampling_rate=sampling_rate,
                max_records=max_records,
                quick=quick,
            ),
            indent=2,
        )
    )


@app.local_entrypoint()
def full_run_parallel_gpu(
    waveforms: str = "100",
    sampling_rate: int = 100,
    max_records: int | None = None,
    quick: bool = False,
    shards: int = 8,
    threads_per_shard: int = 8,
    n_jobs: int = 16,
    n_trials: int = 8,
    optuna_jobs: int = 2,
) -> None:
    _ = (shards, threads_per_shard, n_jobs, n_trials, optuna_jobs)
    download_remote.remote(waveforms=waveforms, max_records=max_records)
    print(
        json.dumps(
            train_gpu_remote.remote(
                sampling_rate=sampling_rate,
                max_records=max_records,
                quick=quick,
            ),
            indent=2,
        )
    )


@app.local_entrypoint()
def bundle_outputs(zip_name: str = "ptbxl_outputs.zip") -> None:
    print(json.dumps(pack_outputs.remote(zip_name=zip_name), indent=2))


@app.local_entrypoint()
def status() -> None:
    print(json.dumps(status_remote.remote(), indent=2))
