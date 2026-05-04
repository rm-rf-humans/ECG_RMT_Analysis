# PTB-XL ECG Classification with Spectral Analysis

This repository contains an applied machine learning pipeline for PTB-XL ECG records. It starts from WFDB waveform files, `ptbxl_database.csv`, and `scp_statements.csv`, then builds a binary diagnostic target, extracts signal features, trains selected classifiers, and saves evaluation plus high-dimensional spectral-analysis artifacts.

Generated data, metrics, models, and plots are ignored by git. The repository keeps only source code, tests, the final paper PDF, and the presentation PDF.

## Dataset

Source: https://physionet.org/content/ptb-xl/1.0.3/

PTB-XL version 1.0.3 provides:

- 10-second 12-lead ECG waveform records in WFDB format
- 100 Hz records under `records100/`
- 500 Hz records under `records500/`
- metadata in `ptbxl_database.csv`
- SCP-ECG code mappings in `scp_statements.csv`

The binary target is constructed from mapped diagnostic superclasses:

- `0`: diagnostic class is exactly `NORM`
- `1`: at least one mapped diagnostic class is not `NORM`

Records without mapped diagnostic statements are excluded from model training.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

Python 3.11 is recommended for compatibility with the Modal image and optional `scikit-rmt` dependency.

## Download Data

Download metadata only:

```bash
python scripts/download_ptbxl.py --data-root data/raw/ptb-xl
```

Download 100 Hz waveform files:

```bash
python scripts/download_ptbxl.py --data-root data/raw/ptb-xl --waveforms 100
```

For a small smoke test:

```bash
python scripts/download_ptbxl.py --data-root data/raw/ptb-xl --waveforms 100 --max-records 300
```

## Run Pipeline

Full 100 Hz run:

```bash
python scripts/run_pipeline.py \
  --data-root data/raw/ptb-xl \
  --output-dir outputs \
  --sampling-rate 100 \
  --n-jobs -1
```

Quick run on a small downloaded subset:

```bash
python scripts/run_pipeline.py \
  --data-root data/raw/ptb-xl \
  --output-dir outputs \
  --sampling-rate 100 \
  --max-records 300 \
  --quick
```

Generated artifacts are written under `outputs/`:

- `outputs/metrics/model_metrics.csv`
- `outputs/metrics/model_rmt_diagnostics.csv`
- `outputs/metrics/data_summary.json`
- `outputs/metrics/model_params.json`
- `outputs/figures/`
- `outputs/rmt/covariance_spectrum.csv`
- `outputs/rmt/covariance_summary.csv`
- `outputs/models/best_model.joblib`

## Modal Compute

The Modal setup lives in `modal_compute/` so the cloud-compute environment is separate from the local environment.

```bash
python3 -m venv modal_compute/.venv
source modal_compute/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r modal_compute/requirements.txt
modal setup
```

Run a small Modal smoke test:

```bash
modal run modal_compute/app.py::full_run \
  --waveforms 100 \
  --sampling-rate 100 \
  --max-records 300 \
  --quick
```

Run the full GPU/CPU Modal job:

```bash
modal run modal_compute/app.py::full_run_parallel_gpu \
  --waveforms 100 \
  --sampling-rate 100 \
  --n-jobs 16 \
  --n-trials 8 \
  --optuna-jobs 2 \
  --shards 8 \
  --threads-per-shard 8
```

Modal data and outputs are persisted in the `ptbxl-rmt-ecg-volume` volume.

## Reported Run

The expanded Modal run used 100 Hz waveforms and 8 Optuna TPE trials per model. Held-out test results:

| model | accuracy | balanced accuracy | precision | recall | f1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 0.7938 | 0.8044 | 0.8877 | 0.7360 | 0.8047 | 0.8924 |
| sgd_linear | 0.7271 | 0.7307 | 0.7973 | 0.7071 | 0.7495 | 0.7307 |
| random_forest | 0.8100 | 0.8115 | 0.8597 | 0.8018 | 0.8297 | 0.9001 |
| extra_trees | 0.8021 | 0.8072 | 0.8686 | 0.7745 | 0.8188 | 0.8909 |
| hist_gradient_boosting | 0.8202 | 0.8231 | 0.8743 | 0.8042 | 0.8378 | 0.9089 |
| xgboost | 0.8151 | 0.8183 | 0.8712 | 0.7978 | 0.8328 | 0.9097 |

The selected model by test ROC-AUC was `xgboost`.

## RMT Analysis

RMT is used as analysis, not as a competing classifier. The covariance spectrum is compared with the Marchenko-Pastur support. Fitted models are then evaluated on original, MP-outlier reconstructed, and MP-bulk reconstructed inputs.

The reported run found:

- \(p/n=0.0348\)
- MP support \([0.662, 1.408]\)
- 60 outlier eigenvalues above the MP upper edge
- 76.9% of standardized variance explained by outlier components

Hard bulk removal and PCA++ half-segment scores did not improve held-out XGBoost performance, so the final interpretation treats RMT as a diagnostic layer.

## Tests

```bash
pytest
```

The tests use small constructed arrays and in-memory metadata only. They do not create experiment results.

## Push Target

```bash
git remote add origin git@github.com:rm-rf-humans/ECG_RMT_Analysis.git
git push -u origin main
```
