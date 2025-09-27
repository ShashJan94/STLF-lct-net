<h1 align="center">STLF‑LCT‑Net — Short‑Term Load Forecasting (LCT‑Net)</h1>
<div align="center">
  <strong>Reproducible STLF experiments and utilities built around a CNN + LCT + attention model</strong><br/>
  <em>Code supporting an academic study on STLF — this repository implements the experiments & reproducible pipeline.</em>
</div>

---

> **NOTE**
> This README is tutorial‑oriented: overview → structure → deep dive → quickstart → CLI → inference & reproducibility → tests → license.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure (Plan‑Oriented Overview)](#project-structure-planoriented-overview)
- [What Is This? (Deep Dive)](#what-is-this-deep-dive)
- [Quickstart Tutorial](#quickstart-tutorial)
  - [Prepare Data](#prepare-data)
  - [Config Example](#config-example)
  - [Run QC](#run-qc)
  - [Preflight Check](#preflight-check)
  - [Train an Experiment](#train-an-experiment)
  - [Inference Example](#inference-example)
- [CLI Reference](#cli-reference)
- [Important Files & Notes](#important-files--notes)
- [Outputs & Artifacts](#outputs--artifacts)
- [Visualization & Evaluation](#visualization--evaluation)
- [Reproducibility Tips](#reproducibility-tips)
- [Results (Sample Run)](#results-sample-run)
- [Tests](#tests)
- [Contributing](#contributing)
- [License (MIT)](#license-mit)
- [Acknowledgements & Citation](#acknowledgements--citation)
- [Contact](#contact)

---

## Introduction

This repository implements a full pipeline for **short‑term load forecasting (STLF)** experiments using a model in the *LCT‑Net* family — a **CNN + Local Context Transformer / attention** architecture (`CNN_LCT_Att`). It bundles utilities for:

- Robust CSV ingestion & **data quality (QC)** checks
- **Weather** merging & time‑zone alignment (Meteostat)
- **Feature enrichment** (cyclic encodings, holiday flags, one‑hots)
- **Rolling** cross‑validation folds & calendar‑aligned splits
- **Windowing** & PyTorch **DataLoaders** (with context)
- Model definition, **training loop** (EMA, early stopping, checkpointing)
- **Evaluation & calibration** (raw / offset / affine)
- **Plotting** helpers for pre / post analysis

Built to support an academic paper; this repo provides the experiment code and a reproducible pipeline.

---

## Project Structure (Plan‑Oriented Overview)

Stage‑by‑stage view of the pipeline and modules.

### Stage 0 — Configuration & Environment
- **Purpose:** centralize paths, defaults, and config parsing.
- **Files:**
  - `src/config/load_config.py` — YAML loader & path expansion (`load_config(path)`)
  - `src/config/paths.py` — constants & directories (`OUTPUTS`, `ARTIFACTS`, `PLOTS`, `CALIB`)
  - `src/config/features.py` — canonical feature list (`FEATS`) and default lookback (`DEFAULT_T`)

### Stage 1 — Data Ingestion & QC
- **Purpose:** read raw CSV robustly, normalize columns, detect gaps/anomalies.
- **Files:**
  - `src/data_io/read_load_csv.py` — enforces `timestamp` & `load`
  - `src/cli/qc_check.py` — CLI to run full QC (optional weather merge)
  - `src/quality/nan_inspector.py` — locate NaN positions & runs
  - `src/quality/qc_report.py` — QC summary stats (ramps, missingness, etc.)
  - `src/quality/audit.py` — anomaly audit helpers

### Stage 2 — Feature Engineering & Weather Enrichment
- **Purpose:** augment load with exogenous features; generate time encodings.
- **Files:**
  - `src/features/enrich_weather.py` — Meteostat merge (tz, lat, lon)
  - `src/config/features.py` — **`FEATS`** ensures consistent feature ordering

### Stage 3 — Splitting & Cross‑Validation
- **Purpose:** realistic temporal splits (rolling‑quarter).
- **Files:**
  - `src/splits/rolling.py` — `FoldSpec`, `rolling_quarter_folds`, `summarize_folds`

### Stage 4 — Windowing & DataLoaders
- **Purpose:** convert series into supervised windows for batch training.
- **Files:**
  - `src/windowing/loaders.py` — `WindowConfig`, `make_loaders_for_fold` → `(train_loader, val_loader, test_loader, invert_y_fn, used_features)`
- **Key params:** `T`, `horizon`, `batch_size`, `scale_load_in_X`, `mask_train`, `use_context_val/test`

### Stage 5 — Model Definition
- **Files:**
  - `src/model/cnn_lct_att.py` — `CNN_LCT_Att` (CNN front‑end + LCT/attention)
- **Notes:** `in_feats = len(used_features)`, `max_len = T`, optional `baseline_lags=(1,24,48,72,...)`

### Stage 6 — Training & Orchestration
- **Files:**
  - `src/training/trainer.py` — `train_model` (EMA, early stopping, ckpt, logging)
  - `src/training/preflight.py` — `run_preflight` sanity checks
  - `src/cli/preflight_ck.py` — CLI entrypoint for end‑to‑end small run

### Stage 7 — Inference & Scaling
- **Files:**
  - `src/inference/scaling.py` — `fit_or_load_minmax(df)`, `prepare_new_data(df, T)`
- **Artifacts:** `outputs/artifacts/scaler_stats.npz` (x_min, x_max, y_min, y_max)

### Stage 8 — Evaluation, Calibration & Visualization
- **Files:**
  - `src/evaluation/metrics.py` — `evaluate_loader`, `metrics_dict`
  - `src/evaluation/calibration.py` — `calibrate_on_val_and_save`
  - `src/viz/pre/*` — EDA plots (weekly mean, heatmaps, ramps)
  - `src/viz/post/*` — diagnostics (learning curves, residuals, overlays)
  - `src/experiments/hub.py` — example glue (model + loaders + training + plots)

### Stage 9 — Tests, Examples & Utilities
- **Files:**
  - `src/tests/test_fold_dummy.py` — windowing + naïve baseline checks
  - `src/experiments/hub.py` — experiment scaffold
  - `configs/` — example YAMLs for runs

**End‑to‑end flow (at a glance):**
1) Load YAML config & raw CSV → **QC** → (optional) weather merge  
2) Create **FEATS** dataframe → build **rolling folds**  
3) Build **WindowConfig** & **DataLoaders**  
4) Instantiate **CNN_LCT_Att** → **train_model** (ckpt/EMA)  
5) **Evaluate**, **calibrate**, **plot** → save **artifacts**

---

## What Is This? (Deep Dive)

**Primary goal:** A fully reproducible STLF research pipeline from QC → features → folds → windows → model → evaluation → inference.

**Design principles**
- Modular (IO, features, quality, splits, model, training, viz)
- Reproducible (checkpoints, scaler stats, metrics, calibration in a structured `outputs/`)
- Experiment‑friendly (config‑driven runs, preflight checks, diagnostics)

**Key concepts**
- **FEATS** (see `src/config/features.py`):  
  `["load","temperature","humidity","hour_sin","hour_cos","day_sin","day_cos","month_sin","month_cos", ...]`  
  > Additional features are used for the five‑year dataset (see training module).
- **DEFAULT_T**: default lookback length (typically `T=72`)
- **Rolling‑quarter folds**: temporal validation splits (`src/splits/rolling.py`)
- **Windowing**: `WindowConfig` + `make_loaders_for_fold` (context‑aware val/test windows)
- **Scaling**: Min‑Max stats saved at `outputs/artifacts/scaler_stats.npz`
- **Model**: `CNN_LCT_Att` — CNN + local‑context attention
- **Training**: EMA, early stopping, Huber loss, gradient clipping; best checkpoint persisted

> [!TIP]
> Weather merging matters: temperature/humidity are strong exogenous predictors.

---

## Quickstart Tutorial

### Clone
```bash
git clone https://github.com/ShashJan94/STLF-lct-net.git
cd STLF-lct-net
```

### Create virtual environment & install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix/Mac: source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt  # if present
# or minimal stack:
# pip install numpy pandas matplotlib pyyaml torch meteostat
```

### Prepare Data
Required columns (after normalization): `timestamp`, `load`  
Optional/desirable: `temperature`, `humidity`  
The reader lowercases headers and normalizes names.

Example:
```csv
timestamp,load,temperature,humidity
2020-01-01 00:00:00,42.5,25.1,80
```

### Config Example
```yaml
data:
  csv_path: "/absolute/path/to/load.csv"
  timezone: "Asia/Dhaka"    # optional; used for weather merge if not skipping
  point_lat: 23.8103
  point_lon: 90.4125
```

### Run QC
```bash
python -m src.cli.qc_check --config configs/my_run.yml --outdir outputs/qc
# If your CSV already contains weather columns:
python -m src.cli.qc_check --config configs/my_run.yml --outdir outputs/qc --skip-weather
```

**QC outputs**
- `outputs/qc/load_nan_positions.csv`
- `outputs/qc/load_nan_runs.csv`
- `outputs/qc/qc_report.json`
- `outputs/qc/audit.json`

### Preflight Check
```bash
python -m src.cli.preflight_ck --csv /absolute/path/to/your_merged.csv --T 72
```
Performs a tiny end‑to‑end check (folds → loaders → model → one‑batch train & forward).

### Train an Experiment
- Build folds with `rolling_quarter_folds(df)`; last months form the test set.
- Create `WindowConfig(T=72, horizon=1, batch_size=64, scale_load_in_X=True)`.
- Use `make_loaders_for_fold(tr_df, va_df, df_test, win, feature_cols, ...)`.

**Instantiate model**
```python
from src.model.cnn_lct_att import CNN_LCT_Att
model = CNN_LCT_Att(in_feats=len(used_feats), max_len=72, baseline_lags=(1,24,48,72))
```

**Train**
```python
from src.training.trainer import train_model

model, ema, history, total_secs = train_model(
    model, train_loader, val_loader,
    invert_y_fn=invert_y,
    max_epochs=100, lr=1e-3, weight_decay=2e-4,
    huber_beta=0.5, ema_decay=0.995,
    patience=6, min_delta=0.25, es_start_epoch=8,
    clip_grad=1.0
)
```

**Artifacts saved**
- `outputs/artifacts/best_cnn_lct.pt`
- `outputs/artifacts/scaler_stats.npz`
- `outputs/metrics.json`
- `outputs/calibration/*`
- `outputs/plots/*`

### Inference Example
```python
from src.inference.scaling import fit_or_load_minmax, prepare_new_data
from src.model.cnn_lct_att import CNN_LCT_Att
import torch

x_min, x_max, y_min, y_max = fit_or_load_minmax(df_train)   # training stats
X_windows = prepare_new_data(df_new, T=72)                   # scaled windows

model = CNN_LCT_Att(in_feats=len(FEATS), max_len=72)
model.load_state_dict(torch.load("outputs/artifacts/best_cnn_lct.pt", map_location="cpu"))
model.eval()
```

---

## CLI Reference
- **Run QC**  
  `python -m src.cli.qc_check --config configs/my_run.yml --outdir outputs/qc [--skip-weather]`
- **Preflight sanity check**  
  `python -m src.cli.preflight_ck --csv /path/to/merged.csv --T 72`
- **Example experiment runner**  
  Edit `src/experiments/hub.py`, then:  
  `python -m src.experiments.hub`

---

## Important Files & Notes
- `src/data_io/read_load_csv.py` — normalizes headers; enforces `timestamp` & `load`
- `src/config/paths.py` — common directories & constants
  - `BEST_CKPT = outputs/artifacts/best_cnn_lct.pt`
  - `SCALE_FILE = outputs/artifacts/scaler_stats.npz`
  - `CALIB_FILE = outputs/calibration/calibration_val.npz`
  - `METRICS_JSON = outputs/metrics.json`
- `src/features/enrich_weather.py` — Meteostat merge (tz, lat, lon)
- `src/windowing/loaders.py` — `WindowConfig`, `make_loaders_for_fold(...)`
- `src/model/cnn_lct_att.py` — `CNN_LCT_Att` model
- `src/training/trainer.py` — training loop, EMA, Huber loss, grad clipping, logging

---

## Outputs & Artifacts
```
outputs/
  artifacts/
    best_cnn_lct.pt
    scaler_stats.npz
  calibration/
    calibration_val.npz
  plots/
    learning_curves.png
    val_scatter.png
    test_scatter.png
    val_residuals.png
    test_residuals.png
    val_overlay.png
  metrics.json
```

---

## Visualization & Evaluation
- **Pre‑EDA** (`src/viz/pre`): weekly mean, ramp histograms, heatmaps, spectral analysis, missingness heatmap
- **Post‑training** (`src/viz/post`): learning curves, pred‑vs‑true scatter, residual histogram, sequence overlays
- **Evaluation helpers**: `src/evaluation/metrics.py`, `src/evaluation/calibration.py`

---

## Reproducibility Tips
- Fix random seeds; record hyperparameters per run.
- Use the saved `scaler_stats.npz` for inference; **do not** refit on inference data.
- Prefer **rolling_quarter_folds** for temporal generalization checks.
- Keep artifacts grouped by run (timestamped subfolders help).
- Capture environment info (Python & library versions).

---

## Results (Sample Run)

**Training excerpt**

```
Start Epoch 
Epoch 001 | train_loss=0.002300 | VAL RAW  RMSE=41.798 MAE=30.902 MSE=1747.109 MBE=12.562 MAPE=0.0316 R2=0.9617 nRMSErng=0.0412 nRMSEmean=0.0422
           VAL +OFF RMSE=39.866 MAE=28.700 MBE=0.000  (offset=+12.562)
           VAL +AFF RMSE=38.893 MAE=27.823 MBE=-0.000  (y≈0.959968*pred+27.611679)
           LR=5.00e-04 | epoch_secs=64.0s
=== Training finished ===
Best RAW Val RMSE: 29.267
Last epoch RAW Val: RMSE=29.154, MAE=19.177, MBE=-0.672, MAPE=0.0196, R2=0.9814
Total training time: 1860.9s
Total training time: 1860.93s; avg/epoch: 74.40s
```

**Final metrics**

| Split | RMSE  | MAE   | R²    | MBE    | MAPE   | nRMSE_range | nRMSE_mean |
|------:|------:|------:|------:|-------:|-------:|------------:|-----------:|
| Train | 31.49 | 20.57 | 0.9820 | +0.021 | 0.0205 | 0.02010     | 0.02959    |
| Val   | 29.05 | 19.03 | 0.9815 | −0.368 | 0.0195 | 0.02863     | 0.02932    |
| Test  | 38.79 | 28.10 | 0.9775 | −2.285 | 0.0200 | 0.02954     | 0.02743    |

> Numbers match the user‑reported run (five‑year dataset; `T=72`, horizon=1). Values are in the original target units.

---

## Tests
Lightweight tests illustrate windowing & naïve baselines.

```bash
pytest -q
```

---

## Contributing
Contributions are welcome. Please:
- Add/modify unit tests in `src/tests/`
- Update README/examples for new CLI flags
- Maintain backward compatibility for saved artifacts when possible

---

## License (MIT)

This project is licensed under the MIT License — full text below.

MIT License — Copyright (c) 2025
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Acknowledgements & Citation

This codebase implements the experiment logic and models used in an associated academic paper.
If you use this repository for research or production, please cite the paper once available.

> Authors. “LCT‑Net for Short‑Term Load Forecasting”. Manuscript in preparation.

---

## Contact

- Repo owner: <https://github.com/ShashJan94>
- Questions/Suggestions: please open a GitHub Issue or a Pull Request.