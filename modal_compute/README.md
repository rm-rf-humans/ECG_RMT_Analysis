# Modal Compute

This directory keeps Modal configuration separate from the local Python environment.

## Setup

```bash
python3 -m venv modal_compute/.venv
source modal_compute/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r modal_compute/requirements.txt
modal setup
```

## Commands

Download metadata and optional waveforms:

```bash
modal run modal_compute/app.py::download --waveforms 100
```

Run training:

```bash
modal run modal_compute/app.py::train --sampling-rate 100 --n-trials 8
```

Download with several containers and then run training:

```bash
modal run modal_compute/app.py::full_run_parallel_gpu \
  --waveforms 100 \
  --sampling-rate 100 \
  --shards 8 \
  --threads-per-shard 8 \
  --n-trials 8
```

Generated outputs are stored in the `ptbxl-rmt-ecg-volume` Modal volume under `/ptbxl_project`.
