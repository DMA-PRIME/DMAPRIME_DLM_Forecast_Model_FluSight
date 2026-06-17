# DLM Recursive Forecast Model with EHR Covariates

A **Dynamic Linear Model (DLM)** pipeline for forecasting weekly incident influenza
hospitalizations for the **CDC FluSight** challenge, using electronic health record (EHR)
covariates from two South Carolina health systems (Prisma Health + MUSC). The repository
contains two notebooks: one that **evaluates and selects** the best model configuration, and
one that uses those selected parameters to produce a **real-time weekly submission**.

---

## Model

**Dynamic Linear Model (DLM)** built on statsmodels'
[`UnobservedComponents`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html),
with recursive multi-horizon forecasting.

- **Forecasting strategy:** a single 1-step-ahead model is fit on consecutive (origin → origin+1 week)
  pairs, then applied **recursively** to reach horizons `h = 0, 1, 2, 3` (1–4 weeks ahead). Each
  predicted value is fed back in as the lag input for the next step.
- **Exogenous regressors:** lagged target values, lagged EHR covariates, and holiday flags are
  passed to the model as the `exog` matrix.
- **Covariate handling during forecast:** the target `y` is updated recursively, but EHR
  covariates (positive tests, inpatient) are **held at their last observed value** because future
  EHR values are unknown at forecast time.

---

## Locations

| Code | Name |
|------|------|
| `45` | South Carolina (SC) |
| `US` | United States (national) |

Both locations are modeled. Note that the Prisma/MUSC EHR feeds are **SC-only**. The flag
`APPLY_SC_EHR_TO_US` controls whether the SC EHR series is reused as a covariate for the US model
(see [Important reproducibility notes](#important-reproducibility-notes)).

---

## Data

| Source | Variable | How to obtain |
|--------|----------|---------------|
| **CDC FluSight Hub** | Weekly incident influenza hospitalizations (target) | Pulled live from GitHub: `target-data/target-hospital-admissions.csv` |
| **Prisma Health EHR** | Weekly positive tests, weekly inpatient hospitalizations (SC) | Local CSVs (restricted PHI; not in this repo) |
| **MUSC EHR** | Weekly positive tests, weekly inpatient hospitalizations (SC) | Local CSVs (restricted PHI; not in this repo) |

- The **target** is filtered to `outcome_measure == "wk inc flu hosp"` and the location of interest.
- The **EHR** files are matched by glob on a basename prefix and the **most recent file by
  modification time** is used. Both health systems are summed into combined `pos` and `inp` series.
- The EHR loader keeps rows where `State == "SC"` (when a `State` column exists) and auto-detects the
  week/date column and the positive-test / inpatient columns from a list of candidate names.

> The EHR data contains protected health information and is **not distributed** with this code.
> To reproduce with your own data, place weekly CSVs in the directories configured at the top of
> each notebook (see [Configuration](#configuration)). The public CDC target is fetched automatically.

---

## Features

| Feature type | Description |
|--------------|-------------|
| **Y lags** | Lagged (transformed) hospitalization values. Lag templates range over 1–10 weeks. |
| **EHR positive-test lags** | Lagged combined weekly positive tests (Prisma + MUSC). Templates range over 4–10 weeks. |
| **EHR inpatient lags** | Lagged combined weekly inpatient hospitalizations (Prisma + MUSC). Templates range over 4–10 weeks. |
| **Thanksgiving flag** | Binary; 1 within ±3 days of the 4th Thursday of November. |
| **Christmas flag** | Binary; 1 when December and day ≥ 20. |
| **New Year flag** | Binary; 1 when January and day ≤ 7. |

**Transformation:** `log1p` is applied to the target **and** the EHR covariates for variance
stabilization (`USE_LOG_TRANSFORM = True`); forecasts are mapped back with `expm1` and clipped at 0.
Holiday flags are computed from the **target** date of each forecast.

---

## Repository contents

| File | Purpose |
|------|---------|
| `DMAPRIME_DLM_Model_Evaluation.ipynb` | Hyperparameter search + backtest. Selects the best config per location and writes the optimal-parameters JSON. **Run this first.** |
| `DMAPRIME-DLM_Submission_Weekly.ipynb` | Real-time weekly forecast. Loads the optimal-parameters JSON and writes the FluSight submission CSV (SC + US). |
| `DMAPRIME-DLM.yml` | Conda environment specification. |
| `README.md` | This file. |
| `LICENSE` | License. |

The two notebooks are linked by an artifact: **evaluation produces an `optimal_params_*.json`,
and submission consumes it.** You must run evaluation (or otherwise supply that JSON) before the
submission notebook can run.

---

## Environment setup

Using the provided conda spec:

```bash
conda env create -f DMAPRIME-DLM.yml
conda activate dmaprime-dlm   # use the name defined inside the .yml
```

Or install the core dependencies directly:

```bash
pip install pandas numpy matplotlib statsmodels scipy jupyter
```

| Library | Used for |
|---------|----------|
| `pandas`, `numpy` | Data handling, lag construction |
| `statsmodels` | `UnobservedComponents` DLM |
| `scipy` | `norm` for quantile generation (submission only) |
| `matplotlib` | Diagnostic and forecast plots |

---

## Configuration

Both notebooks define paths at the top that **must be edited** for your machine. They were written
for a Windows + Box environment:

```python
base_dir   = r"...\FluSight_Forecast"   # root for all outputs
PRISMA_DIR = r"...\Prisma Health\...\Latest Weekly Data"
MUSC_DIR   = r"...\MUSC\...\Latest Weekly Data"
```

Key knobs (top of each notebook):

| Setting | Meaning |
|---------|---------|
| `USE_LOG_TRANSFORM` | Apply `log1p`/`expm1` to target + EHR (default `True`). Also sets the `log`/`raw` filename suffix. |
| `APPLY_SC_EHR_TO_US` | Whether to feed the SC EHR series into the US model. **Differs between the two notebooks — see notes below.** |
| `LOCATIONS` | List of `{code, name}` to model (SC `45`, US). |
| `HORIZONS` | `[0, 1, 2, 3]` (1–4 weeks ahead). |
| `MIN_TRAIN_ROWS` | Minimum 1-step training rows required (60). |

---

## Pipeline 1 — Evaluation (`DMAPRIME_DLM_Model_Evaluation.ipynb`)

Selects the best DLM configuration per location by minimizing **validation MAE** of the median
forecast, then saves parameters and diagnostic plots.

**Steps:**

1. **Load target** from the CDC FluSight Hub for each location and apply the transform.
2. **Load EHR** (Prisma + MUSC), keep SC rows, sum into combined `pos` and `inp`, transform.
3. **Build the 1-step dataset:** for each consecutive week pair, assemble the lag features
   (`y_lag*`, `pos_lag*`, `inp_lag*`) and holiday flags; missing early lags fall back to the first
   observed value.
4. **Time-based split** by `target_end_date`:

   | Split | Window |
   |-------|--------|
   | Train | `≤ 2024-06-29` |
   | Validation | `2024-06-29` → `2025-01-31` |
   | Test | `2025-01-31` → `2025-04-26` |

   The model is refit on Train for the validation recursion, and on Train+Validation for the test
   recursion.
5. **Grid search** over every combination of:

   | Grid | Count | Values |
   |------|-------|--------|
   | DLM structures | 4 | level-only / level+trend (each listed twice, `seasonal=None`) |
   | Y-lag templates | 5 | e.g. `[1,2,3,4]` … `[1..10]` |
   | Positive-test lag templates | 5 | e.g. `[4,5]` … `[4..10]` |
   | Inpatient lag templates | 5 | e.g. `[4,5]` … `[4..10]` |

   = **500 configurations per location**. Each is backtested recursively across all horizons; the
   one with the lowest **validation MAE** wins.
6. **Save outputs** (relative to `base_dir`):

   | Output | Path |
   |--------|------|
   | Optimal params (consumed by submission) | `model_eval/optimal_parameters/optimal_params_dlm_recursive_hosp_with_pos_inp_prisma_musc_{log\|raw}.json` |
   | Format-ready evaluation table | `model_eval/eval_files/eval_dlm_recursive_hosp_with_pos_inp_prisma_musc_{log\|raw}_FORMAT.csv` |
   | Per-location diagnostic panels | `model_eval/plots/dlm_recursive_hosp_with_pos_inp_prisma_musc/{loc}_hosp_dlm_recursive_with_pos_inp_panel.png` |

   The params JSON is keyed by `"{loc}_hosp_dlm_recursive_with_pos_inp_prisma_musc"` and stores the
   chosen `y_lags`, `pos_lags`, `inp_lags`, `structure`, and the validation MAE/RMSE.

---

## Pipeline 2 — Weekly Submission (`DMAPRIME-DLM_Submission_Weekly.ipynb`)

Produces the real-time FluSight submission using the parameters selected during evaluation.

**Steps:**

1. **Load optimal params JSON** written by the evaluation notebook (exact path, or newest matching
   file as fallback). A missing key for a location raises an error.
2. **Load target + EHR** as above (latest available data).
3. **Set the forecast origin** = the last observed `target_end_date` in each location's series;
   `reference_date = origin + 7 days`.
4. **Fit on ALL available data** (every 1-step pair), then forecast `h = 0..3` recursively.
5. **Generate 23 quantiles** per horizon: a Normal approximation `μ + z·SE` (using the forecast
   `predicted_mean` and `se_mean`), inverse-transformed (`expm1`), clipped at 0, and **rounded to
   integers**. Quantile levels: `0.01, 0.025, 0.05, 0.10…0.90, 0.95, 0.975, 0.99`.
6. **Save outputs** (relative to `base_dir`, under
   `final_submission/dlm_recursive_hosp_with_pos_inp_prisma_musc_{log\|raw}/realtime_latest_fit_all_data_SC_US/`):

   | Output | File |
   |--------|------|
   | Combined submission (SC + US) | `FluSight_submission_DLM_EHR_SC_US_ref_{reference_date}.csv` |
   | Forecast plot per location | `plots/{loc}_hosp_dlm_realtime_ref_{reference_date}.png` |

   The submission CSV has the FluSight columns `reference_date, target, horizon, target_end_date,
   location, output_type, output_type_id, value` →
   **2 locations × 4 horizons × 23 quantiles = 184 rows.** Saving uses a verified atomic write with
   retries and Desktop/CWD fallbacks (helpful when the target lives in a syncing Box folder).

---

## Reproducing the results

```bash
# 1. Create and activate the environment
conda env create -f DMAPRIME-DLM.yml
conda activate dmaprime-dlm

# 2. Edit base_dir + PRISMA_DIR + MUSC_DIR at the top of BOTH notebooks,
#    and place the weekly EHR CSVs in those directories.

# 3. Run evaluation first (writes optimal_params_*.json)
jupyter nbconvert --to notebook --execute DMAPRIME_DLM_Model_Evaluation.ipynb

# 4. Run the weekly submission (reads that JSON, writes the submission CSV)
jupyter nbconvert --to notebook --execute DMAPRIME-DLM_Submission_Weekly.ipynb
```

The CDC target data updates weekly, so re-running on different dates yields different forecasts;
the model **structure and selected hyperparameters** are what reproduce deterministically from the
saved JSON.

---

