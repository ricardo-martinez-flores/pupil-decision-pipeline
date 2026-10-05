# Decision-by-decision pupillometry: preprocessing and analysis

This repository contains the code for the manuscript:

> Martínez-Flores, R., et al. *Adapting Pupillometry Preprocessing and Analysis
> Methods for Older Adults with MCI: A Decision-by-Decision Approach.*
> (Manuscript under review.)

## Why this repository

Most pupillometry guidelines were developed with young, healthy participants.
In populations such as older adults with mild cognitive impairment (MCI), the
data look different:

- more missing data
- longer gaps
- smaller pupils
- more variable data quality between participants

The recommended default parameters may therefore not transfer.

This code does not just *apply* a fixed set of parameters. At every
preprocessing and analysis step, it tests the chosen value against the
literature defaults and against alternatives suggested by the data. It then
checks whether the choice changes the effect of interest, and documents the
decision.

The code is written to be **generic**. It does not contain or require the
original data. You can adapt it to any task-evoked pupillometry dataset with two
conditions by editing a configuration file.

## Structure

```
.
├── python_preprocessing/     # STEP A -- preprocessing (Python)
│   ├── pupil_pipeline.py     #   decision-by-decision preprocessing pipeline
│   ├── config.yaml           #   all parameters and candidate values
│   ├── config_mci_dataset.yaml  # settings for the article's dataset (see below)
│   ├── make_example_data.py  #   generates a simulated example dataset
│   ├── requirements.txt
│   └── README.md
├── r_analysis/               # STEP B -- statistical analysis (R)
│   ├── pupil_analysis.R      #   GAMM, time-series LMM, features, resampling
│   ├── config_analysis.yaml
│   ├── config_mci_dataset.yaml  # settings for the article's dataset
│   └── README.md
├── CITATION.cff
├── LICENSE
└── README.md
```

## Workflow

```
raw samples (CSV) ──► python_preprocessing ──► preprocessed_accepted_trials_long.csv ──► r_analysis
                             │                  trial_ledger.csv                              │
                             ▼                                                                ▼
                  decision_report.md + figures                               analysis_summary.md + tables
```

### Step A: preprocessing (Python)

Each step compares candidate options, and the result goes into a decision log:

| Step | Decision |
|---|---|
| 1.1 | Physiologically plausible pupil range |
| 1.2 | Minimum per-trial data coverage |
| 1.3 | Interocular coverage balance |
| 1.4 | Maximum gap length |
| 1.5 | Balance of data quality across conditions |
| 2 | Interpolation method, including a synthetic ground-truth test |
| 3 | Baseline outliers and baseline summary statistic |
| 4 | Smoothing filter |
| 5 | Gaze-position confound |
| 6 | Grand average and export |

Sensitivity checks use one value per participant. Each comparison reports a
BCa bootstrap 95% CI and a paired sign-flip permutation test.

See [`python_preprocessing/README.md`](python_preprocessing/README.md).

### Step B: statistical analysis (R)

The analysis asks three questions, each with the model suited to it:

1. **How** the effect unfolds over time: a GAMM, with and without an AR(1)
   correction.
2. **How large** the effect is on average: a full time-series LMM.
3. **Which components** of the response differ: trial-level features with Holm
   correction.

The model-based results are checked with methods that do not depend on their
assumptions:

- block permutation
- a calibration of that permutation under a simulated null
- cluster BCa bootstrap
- max-t correction
- leave-one-subject-out refits

See [`r_analysis/README.md`](r_analysis/README.md).

## Quick start

```bash
# Python (>= 3.10)
cd python_preprocessing
pip install -r requirements.txt
python make_example_data.py
python pupil_pipeline.py --config config.yaml

# R (>= 4.1)
cd ../r_analysis
Rscript -e 'install.packages(c("yaml","dplyr","lme4","lmerTest","broom.mixed","mgcv","ggplot2"))'
Rscript pupil_analysis.R config_analysis.yaml
```

The example dataset is **simulated**. It exists only to show the expected input
format and to test the code; its results have no scientific meaning.

## Testing the pipeline on our data

The anonymized data analysed in the article (67 older adults with MCI, visual
oddball task, Tobii 5L at 33 Hz) are openly available at
https://doi.org/10.5281/zenodo.23160276. You can run the pipeline on them to see how each
decision behaves on real data from a hard-to-measure population, or to compare
them with your own data:

```bash
# preprocessing: raw samples, all trials
cd python_preprocessing
# place raw_samples_anonymized.csv.gz in python_preprocessing/data/
python pupil_pipeline.py --config config_mci_dataset.yaml

# statistics: preprocessed accepted trials
cd ../r_analysis
# place preprocessed_accepted_trials.csv.gz in r_analysis/data/
Rscript pupil_analysis.R config_mci_dataset.yaml
```

The two configuration files only map the dataset's column names and recording
settings; the code is the same generic pipeline. The results are therefore close
to, but not necessarily identical with, the values reported in the article.

## Using your own data

1. Export your recording as a long-format CSV with one row per sample. It needs
   these columns:
   - participant
   - trial
   - condition
   - time from stimulus onset (negative = baseline)
   - left and right pupil size
   - optionally, validity codes and gaze position
2. In `python_preprocessing/config.yaml`, map your column names and set:
   - the sampling rate
   - the condition labels
   - the baseline and response windows
3. Run the pipeline and read `outputs/decision_report.md`. If the data suggest a
   different value than the one you applied, revise the configuration and
   re-run, or report why you kept your value.
4. Run the R analysis on the exported file.

## Resampling settings

Both steps use **5,000 bootstrap resamples** and **5,000 permutations** by
default, with a fixed seed (12345). Permutation p-values are computed as
p = (b + 1)/(B + 1) (Phipson & Smyth, 2010).

The time-series refits in R can take several hours on large datasets. Each
script prints a projected run time before every long loop.

## Data availability

The anonymized data analysed in the article are openly available at
https://doi.org/10.5281/zenodo.23160276 (CC BY 4.0). They contain no demographic, clinical,
neuropsychological or biomarker variables and no cohort or site identifiers.
The data are deposited on Zenodo, not in this repository.

## Citation

If you use this code, please cite the manuscript above. The reference will be
updated upon publication. Citation metadata for the software is in
[`CITATION.cff`](CITATION.cff).

## License

This code is released under the MIT License. See [`LICENSE`](LICENSE).

## Contact

Ricardo Martínez-Flores · ORCID [0000-0002-8435-2710](https://orcid.org/0000-0002-8435-2710)
