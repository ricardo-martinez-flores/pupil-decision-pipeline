# Statistical analysis of pupil time series (R)

This folder contains the statistical analysis that accompanies the manuscript
*Adapting Pupillometry Preprocessing and Analysis Methods for Older Adults with
MCI: A Decision-by-Decision Approach*.

It reads the output of the Python preprocessing pipeline (`../python_preprocessing`),
or any file with the same structure. It then analyses one within-participant
contrast between two conditions, for example oddball Target vs. Distractor.

## What the analysis does

The analysis asks three questions of the same data. Each question uses the
model suited to it.

| Question | Model | Part |
|---|---|---|
| **How** does the effect unfold over time? | GAMM with by-condition smooths, with and without AR(1) residual correction | 1 |
| **How large** is the effect on average? | Full time-series LMM: every sample in one model, one coefficient | 2 |
| **Which components** of the response differ? | One LMM per trial-level feature, Holm correction | 3 |

Pupil signals are strongly autocorrelated and their residuals are not normal.
For this reason, each model-based p-value or CI is checked with a resampling
method that does not rely on those assumptions:

- **Part 0 – Confound check.** Tests whether data quality differs between
  conditions: per-eye coverage, the coverage ratio and the percentage of
  interpolated samples, used as blink proxies.
- **Part 2b – Block permutation.**
  - Trial labels are shuffled *within participant*.
  - Every trial keeps its own samples, so the autocorrelation is kept intact.
  - Each participant's number of trials per condition is also kept.
- **Part 2c – Calibration.**
  - Small datasets are simulated under a true null: AR(1) noise using the ρ
    estimated in Part 1, and no condition effect.
  - Each dataset is tested with the same permutation scheme.
  - A valid scheme gives about 5% of p < .05.
- **Part 2d – Cluster bootstrap.** Participants are resampled with
  replacement, and a BCa 95% CI is computed with jackknife acceleration.
- **Part 3b – Feature resampling.**
  - Label permutation with max-t family-wise correction (Westfall & Young,
    1993). The same shuffle is applied to all six features, so the correlation
    between features is taken into account.
  - Cluster-bootstrap BCa CIs.
- **Part 4 – Leave-one-subject-out.** Refits the Part 2 model without each
  participant in turn. This checks that no single participant drives the
  estimate.

Permutation p-values are computed as p = (b + 1)/(B + 1) (Phipson & Smyth,
2010), so p is never exactly zero. By default, the analysis uses **5,000
permutations** and **5,000 bootstrap resamples**.

## Installation

The script needs R ≥ 4.1 and these packages:

```r
install.packages(c("yaml", "dplyr", "lme4", "lmerTest", "broom.mixed", "mgcv", "ggplot2"))
```

`parallel` is part of base R.

## Quick start

```bash
# 1. preprocessing (writes ../python_preprocessing/outputs/)
cd ../python_preprocessing
python make_example_data.py
python pupil_pipeline.py --config config.yaml

# 2. analysis
cd ../r_analysis
Rscript pupil_analysis.R config_analysis.yaml
```

**On the article's data.** `config_mci_dataset.yaml` runs the analysis on
`preprocessed_accepted_trials.csv.gz` from the anonymized dataset of the
article (https://doi.org/10.5281/zenodo.23160276). It maps that file's column names
(`stimulus`, `timestamp_ms`) and skips PART 0, because the dataset does not
include the trial-level quality file; running `pupil_pipeline.py` with
`config_mci_dataset.yaml` first produces it.

## Input

**Main input (`input_csv`).** One row per sample of each accepted trial:

| Column (default) | Description |
|---|---|
| `participant` | Participant ID |
| `trial` | Trial ID, unique within participant. Each trial belongs to one condition. |
| `condition` | Condition label |
| `time_ms` | Time from stimulus onset (ms) |
| `pupil_bc` | Baseline-corrected pupil size |

If your column names differ, map them in the `columns:` block of the config.
Use `csv_sep` and `csv_dec` for semicolon or decimal-comma files. Rows from
other conditions are ignored.

**Optional input (`trial_qc_csv`).** A file with one row per trial. It is used
by Part 0 and contains the metrics listed in `qc_metrics`. The
`trial_ledger.csv` written by the Python pipeline already has this format. If
the file has an `accepted` column, only accepted trials are used.

## Configuration

All settings are in `config_analysis.yaml`. The main ones are:

- **`conditions`**: the condition of interest and the reference condition. The
  coefficient that is reported is *of interest − reference*.
- **`features`**: the initial and late windows for the slope features. Fix these
  **a priori**, for example from prior literature. Choosing them after looking at
  the tested contrast makes the test circular.
- **`resampling`**:
  - the number of permutations, bootstrap resamples and calibration datasets
  - the seed
  - the number of cores
  - `timing_test`, which prints the projected run time before each long loop
- **`parts`**: switches each part on or off.
- **`lmm`**: see *Agreement check* below.

## Run time

Each permutation or bootstrap iteration of Parts 2b and 2d refits a mixed model
on every sample. On a large dataset each refit takes tens of seconds. For
example, in the manuscript (about 400,000 samples) one refit took about 40 s
on one core.

The script runs these loops in parallel on one PSOCK cluster, which works on
Windows, macOS and Linux. With `timing_test: true`, it prints the projected
duration before each long loop starts.

For a first test, lower the counts in the config:

```yaml
resampling:
  n_permutations: 50
  n_bootstrap: 50
  n_calibration_datasets: 4
```

Then restore 5,000 for the final run.

## Agreement check for the time-series LMM

The manuscript's model is:

```r
pupil ~ condition + (1 + time_scaled + condition | participant)
```

It has a random slope for time but **no fixed slope for time**. This means the
participants' time slopes are assumed to average zero.

In the manuscript data, this assumption was harmless: the coefficient (0.066 mm)
agreed with the GAMM parametric term (0.065 mm). However, it can bias the
coefficient when the response rises or falls over the window, especially if it
does so differently in the two conditions. In the simulated example data, the
coefficient is 0.030 while the true average difference is 0.093.

For this reason, Part 2 compares the LMM coefficient with two estimates that do
not share this assumption:

- the mean of the participant-level condition differences
- the GAMM parametric term

If the coefficient differs from the participant-level mean by more than
`agreement_tolerance` (20% by default), the script warns you. In that case, set
`lmm: fixed_time_effect: true`. This adds `time_scaled` as a fixed effect and
removes the bias.

## Outputs

```
outputs/
├── analysis_summary.md        # key results of every part
├── config_used.yaml
├── session_info.txt           # R and package versions
├── 00_confound_check/confound_checks.csv
├── 01_gamm/                   # AR1 comparison, smooth and parametric terms,
│                              # difference curve (+ fig_gamm_difference_curve.pdf/.png)
├── 02_time_series_lmm/        # fixed effects, agreement check, block permutation
│                              # (+ null), calibration, bootstrap BCa (+ draws)
├── 03_features/               # features per trial, LMM + Holm,
│                              # robust inference (p_perm, p_maxt, BCa), permutation null
└── 04_loso/                   # per-refit estimates and summary
```

## Adapting the analysis

- **Other features.** Feature extraction is done in the `summarise()` call in
  Part 3. To add a feature:
  1. Add a column to that call.
  2. Add the new feature's name to `FEATURES`.

  The permutation, max-t and bootstrap steps then include it automatically.
- **Covariates.** The models intentionally have no covariates. If your design
  needs them, for example an order effect, add them to the fixed part of the
  formulas:
  - `lmm_formula_signal` (Part 2)
  - the feature formula in Part 3
- **More than two conditions.** The script tests one pairwise contrast. For
  other pairs, run it once per pair by changing `conditions`.

## Citation

If you use this code, please cite the manuscript above (the reference will be
updated upon publication).
