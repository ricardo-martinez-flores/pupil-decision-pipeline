# Pupillometry preprocessing: a decision-by-decision pipeline (Python)

This folder contains the preprocessing code that goes with the manuscript
*Adapting Pupillometry Preprocessing and Analysis Methods for Older Adults with
MCI: A Decision-by-Decision Approach*.

The original dataset is proprietary and is not distributed. This code is a
**generic** version of the workflow. You can run it on any task-evoked
pupillometry dataset that has two conditions (for example, an oddball Target
vs. Distractor).

## What the pipeline does

Most pipelines only *apply* preprocessing parameters. This one also *tests*
them. For every decision it:

1. Starts from the literature default.
2. Evaluates alternatives (literature-anchored and data-driven) on your data.
3. Checks whether the choice changes the effect of interest. This uses a
   participant-level benchmark: the mean of the condition of interest in a
   fixed window, a BCa bootstrap 95% CI and a paired sign-flip permutation test.
4. Writes the evidence to a **decision log**.

The values that are actually applied always come from `config.yaml`. The
pipeline reports what your data suggest. If the two disagree, you can revise
the configuration and re-run, or report why you kept your value.

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.10 or newer.

## Quick start

```bash
python make_example_data.py                # writes data/example_data.csv (simulated data)
python pupil_pipeline.py --config config.yaml
```

With the default 5,000 bootstrap resamples and 5,000 permutations, the example
runs in under a minute. Results go to `outputs/`. Start with
`outputs/decision_report.md`.

## Input format

The input is a single CSV in long format, with **one row per sample**:

| Column (default name) | Required | Description |
|---|---|---|
| `participant` | yes | Participant ID |
| `trial` | yes | Trial ID (unique within participant) |
| `condition` | yes | Condition label (e.g. `Target`, `Distractor`) |
| `time` | yes | Time from stimulus onset. Negative values are the pre-stimulus baseline. |
| `left_pupil`, `right_pupil` | yes | Pupil diameter per eye. Use mm; the range criteria are in mm. |
| `left_valid`, `right_valid` | no | Tracker validity code per eye |
| `gaze_x`, `gaze_y` | no | Gaze position normalized to 0–1 (0 = left/top). Used for the off-screen check and STEP 5. |

**Using your own column names.** You do not need to rename your columns. Map
them in the `columns:` block of `config.yaml`. Set optional columns to `null`
if your file does not have them.

**Validity codes.** `validity.valid_values` lists the codes that mean "valid".
For example, `[0]` for Tobii-style codes, or `[1]` or `[true]` for boolean
flags.

**Timestamps.** If your timestamps are absolute (not relative to onset), first
subtract the stimulus-onset time within each trial. Then set
`recording.time_unit_to_ms` for your units: `0.001` for µs, `1000` for s.

**Separate baseline epoch.** If your baseline is recorded as a separate epoch
(for example, a fixation or mask screen), concatenate it with the stimulus epoch.
The baseline samples must carry negative times.

## Configuration

All parameters live in `config.yaml`. The file is commented. The main blocks
are:

- **`recording`**:
  - sampling rate
  - baseline window (last *N* ms before onset)
  - response window
- **`conditions`**:
  - the condition whose response is the benchmark (`of_interest`)
  - the `reference` condition
- **One block per decision**, each with its applied value and its alternatives:
  - `pupil_range`
  - `coverage`
  - `interocular`
  - `gap`
  - `boundary_edge`
  - `interpolation`
  - `baseline`
  - `smoothing`
- **`analysis`**:
  - `sensitivity_window_ms`: the benchmark window. Fix it **a priori**, from
    prior literature or from a window that does not depend on the tested
    contrast. Choosing it after inspecting the contrast makes the checks
    circular.
  - `n_bootstrap: 5000`
  - `n_permutations: 5000`
  - `seed`
- **`screen`**: screen width and height and the viewing distance, in cm.
  STEP 5 needs these; leave them `null` to skip it.

Parameters that depend on the sampling rate must be rescaled if you record at
a different rate. These are the boundary-edge window (in samples), the
smoothing kernel widths and the elbow step.

## Steps and decisions

| Step | Decision | Default applied here | How it is tested |
|---|---|---|---|
| 1.1 | Minimum pupil diameter | 1.5 mm, tested against 2.0 mm (upper limit 8 mm) | Sample distribution. Interocular agreement in the floor band vs. the band above it. Benchmark with each floor. |
| 1.2 | Minimum per-trial coverage | ≤ 50% missing (better eye) | Literature candidates (20–50% missing) and extended ones (10–80%). Exclusion curve, BCa CI per candidate, pairwise sign-flip tests (Holm-adjusted). |
| 1.3 | Interocular coverage ratio | 0.30 (worse/better eye). Below it, only the better eye is used. | % of trials switched to one eye as a function of the threshold. Stable (slope ≈ 0) region. |
| 1.4 | Maximum gap length | 500 ms | Exclusion curve and marginal-drop elbow |
| 1.5 | Condition balance | – | Exclusion rates by condition. LMMs of per-eye coverage and coverage ratio on condition (a blink proxy). |
| 2 | Interpolation | Linear | Agreement with a local cubic spline at each trial's longest-gap midpoint (Bland–Altman). Ground truth: synthetic gaps with real lengths are injected into near-complete trials. |
| 3 | Baseline | Mean of the last 200 ms; exclude trials with within-participant \|z\| > 2 | Mean/SD vs. median/MAD criterion (MAD × 1.4826 at 2.5; Leys et al., 2013): agreement and κ. Mean vs. median baseline: Bland–Altman and grand averages. |
| 3c | Boundary-edge exclusion | 1 sample | % of trials excluded |
| 4 | Smoothing | Gaussian, σ = 2 samples | Compared with no smoothing, moving average, Savitzky–Golay and a 4 Hz Butterworth filter. Four measures: benchmark, contrast, peak time and residual noise, each with a participant-level BCa CI and sign-flip tests. |
| 5 | Gaze-position confound | Report | Gaze eccentricity (deg) by condition and over time. LMM on condition. |
| 6 | Grand average and export | – | Hierarchical averaging: trials, then participants, then the group. |

**Exclusion order.** Trials are excluded in this order:

1. insufficient coverage
2. missing baseline
3. baseline outlier
4. boundary edge
5. gap longer than the maximum

**Signal pipeline for each accepted trial.** Each accepted trial goes through
these steps in order:

1. Invalid samples are removed.
2. The eyes are combined or selected.
3. Gaps are interpolated on the analysis grid.
4. The baseline is subtracted.
5. The signal is smoothed.

**Permutation p-values.** These are computed as p = (b + 1)/(B + 1) (Phipson &
Smyth, 2010), so p is never exactly zero.

## Outputs

```
outputs/
├── decision_report.md                  # human-readable summary of every decision
├── decision_log.csv                    # same, as a table
├── config_used.json                    # exact configuration of this run
├── trial_ledger.csv                    # one row per trial: quality metrics + exclusion reasons
├── preprocessed_accepted_trials_long.csv   # input for the statistical analysis (R code)
├── 01_validity_screening/              # steps 1.1–1.5 (fig_S1 = extended coverage range)
├── 02_interpolation/
├── 03_baseline/
├── 04_smoothing/
├── 05_gaze_confound/
└── 06_grand_average/
```

`preprocessed_accepted_trials_long.csv` has the columns `participant`, `trial`,
`condition`, `time_ms` and `pupil_bc`. It is the input to the statistical
analysis in the R folder of this repository.

## Adapting the pipeline

- **Other contrasts.** Change `conditions`. Trials from any other conditions are
  kept in the ledger and the export, but the benchmark and contrast use only
  these two.
- **Other candidate values.** Edit the candidate lists: coverage percentages,
  literature floor, smoothing `alternatives`, and so on. They are all tested
  automatically.
- **New decisions.** Each step is a self-contained `stepX_*` function that
  writes to its own folder and calls `log.add(...)`. To add a decision, write a
  new function that follows the same pattern and call it from `main()`.
- **Heuristics.** The data-driven suggestions use explicit, simple rules:
  - Elbow detection: `gap.elbow_*`.
  - Stable interocular region: `interocular.stable_slope_pp_per_0.1`.
  - Pupil floor: a 1.5× increase in the median interocular difference.

  These rules summarize the evidence. They do not replace judgment. Always
  inspect the figures.

## Citation

If you use this code, please cite the manuscript above (the reference will be
updated upon publication).
