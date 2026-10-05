#!/usr/bin/env python3
"""
pupil_pipeline.py -- Decision-by-decision pupillometry preprocessing
=====================================================================

Generic, dataset-agnostic implementation of the preprocessing workflow
described in:

    Martinez-Flores, R., et al. Adapting pupillometry preprocessing and
    analysis methods for older adults with mild cognitive impairment:
    A decision-by-decision approach.

The pipeline does not only *apply* a set of preprocessing parameters: at every
step it *tests* the chosen value against literature defaults and data-driven
alternatives, checks whether the choice changes the effect of interest, and
writes the evidence to a decision log. The values actually applied are always
the ones set in the configuration file; the pipeline reports what the data
suggest so that the researcher can revise them and re-run.

Steps (mirroring the manuscript's Section 3):
    STEP 1  Data validity screening
            1.1 Physiologically plausible pupil range
            1.2 Minimum per-trial data coverage
            1.3 Interocular coverage balance
            1.4 Maximum tolerable gap length
            1.5 Balance of exclusions/data quality across conditions
    STEP 2  Interpolation method (linear vs. local cubic spline)
    STEP 3  Baseline reliability (mean/SD vs. median/MAD; mean vs. median)
    STEP 4  Smoothing filter comparison
    STEP 5  Gaze-position confound check (optional)
    STEP 6  Grand-average response and export of the preprocessed data

Every parameter-sensitivity check uses the same participant-level analysis:
one value per participant (mean of the condition of interest in a fixed
window) -> BCa bootstrap 95% CI -> paired sign-flip permutation test.

Usage
-----
    python pupil_pipeline.py --config config.yaml

Author: Ricardo Martinez-Flores
License: see LICENSE in the repository root.
"""

import argparse
import copy
import json
import os
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.stats import bootstrap, wilcoxon

try:
    import yaml
except ImportError:  # YAML is optional; JSON configs also work
    yaml = None

try:
    import statsmodels.formula.api as smf
except ImportError:
    smf = None

warnings.filterwarnings("ignore")

# =============================================================================
# DEFAULT CONFIGURATION
# Every key can be overridden in the YAML/JSON configuration file.
# =============================================================================
DEFAULT_CONFIG = {
    "input_csv": "data/example_data.csv",
    "output_dir": "outputs",

    # Map the column names of YOUR file to the names used internally.
    # Optional columns can be set to null.
    "columns": {
        "participant": "participant",
        "trial": "trial",
        "condition": "condition",
        "time": "time",                    # time relative to stimulus onset (negative = baseline)
        "left_pupil": "left_pupil",        # pupil diameter (mm recommended)
        "right_pupil": "right_pupil",
        "left_valid": "left_valid",        # optional: tracker validity code per eye
        "right_valid": "right_valid",
        "gaze_x": "gaze_x",                # optional: horizontal gaze, normalized 0-1 (0 = left)
        "gaze_y": "gaze_y",                # optional: vertical gaze, normalized 0-1 (0 = top)
        # Alternative to gaze_x/gaze_y: per-eye gaze columns (normalized 0-1),
        # averaged over the eyes with on-screen values.
        "left_gaze_x": None, "left_gaze_y": None, "right_gaze_x": None, "right_gaze_y": None,
        # Optional epoch label column (e.g. "baseline" / "stimulus"); see recording.onset_epoch.
        "epoch": None,
    },

    # Values of the validity columns that mean "valid sample"
    # (e.g. [0] for Tobii-style codes, [1] or [true] for boolean flags).
    "validity": {"valid_values": [0]},

    # The contrast used as benchmark throughout (e.g. oddball Target vs. Distractor).
    "conditions": {"of_interest": "Target", "reference": "Distractor"},

    "recording": {
        "sampling_rate_hz": 60.0,
        "time_unit_to_ms": 1.0,            # multiply the time column by this to get ms
        "baseline_window_ms": 200.0,       # last N ms before stimulus onset
        "response_window_ms": 2000.0,      # analysis window after stimulus onset
        # If the time column is an absolute timestamp, name the epoch label that
        # marks the stimulus: time is then re-zeroed within each trial at the
        # first sample of that epoch (requires columns.epoch).
        "onset_epoch": None,
    },

    # Needed only for the gaze-position check (STEP 5).
    "screen": {"width_cm": None, "height_cm": None, "viewing_distance_cm": None},

    "pupil_range": {
        "min_mm": 1.5,                     # applied floor (tested against the literature floor)
        "max_mm": 8.0,
        "literature_min_mm": 2.0,          # conventional floor, tested as alternative
        "band_width_mm": 0.5,              # width of the bands compared for interocular agreement
    },

    "coverage": {
        "min_valid_pct": 50.0,             # applied: minimum % valid samples (best eye)
        "literature_missing_pct": [20.0, 30.0, 40.0, 50.0],
        "extended_missing_pct": [10.0, 60.0, 70.0, 80.0],
    },

    "interocular": {
        "ratio_threshold": 0.30,           # applied: worse/better eye coverage ratio
        "reference_pct_flagged": 5.0,      # "small percentage" reference for flagged trials
        "stable_slope_pp_per_0.1": 1.0,    # max change in % flagged per 0.1 ratio to count as stable
    },

    "gap": {
        "max_gap_ms": 500.0,               # applied
        "literature_default_ms": 500.0,
        "elbow_step_ms": 10.0,
        "elbow_flat_drop_pp": 0.5,
        "elbow_sustained_steps": 10,
    },

    "boundary_edge": {"enabled": True, "window_samples": 1},

    "interpolation": {
        "method": "linear",                # applied: "linear" or "cubic"
        "cubic_anchor_points": 2,          # per side, local cubic spline
        "longer_gap_ms": 100.0,
        "n_synthetic_gaps": 3000,
        "synthetic_margin_ms": 150.0,
    },

    "baseline": {
        "summary": "mean",                 # applied: "mean" or "median"
        "z_threshold": 2.0,                # mean/SD criterion (Mathot & Vilotijevic, 2023)
        "mad_threshold": 2.5,              # median/MAD criterion (Leys et al., 2013)
    },

    "smoothing": {
        "method": "gaussian",              # applied
        "params": {"sigma": 2},
        "alternatives": [
            {"label": "NS", "method": "none", "params": {}},
            {"label": "MV", "method": "moving_average", "params": {"window": 5}},
            {"label": "SG", "method": "savgol", "params": {"window": 7, "polyorder": 3}},
            {"label": "BL-P", "method": "butterworth", "params": {"cutoff_hz": 4.0, "order": 4}},
        ],
    },

    "analysis": {
        "sensitivity_window_ms": [1000.0, 1500.0],
        "n_bootstrap": 5000,
        "n_permutations": 5000,
        "seed": 12345,
        "min_trials_per_cell": 2,
    },

    "steps": {"gaze_confound": True, "save_figures": True},
}

MAD_SCALE = 1.4826  # Leys et al. (2013): scaled MAD estimates the SD under normality


# =============================================================================
# CONFIGURATION AND LOGGING UTILITIES
# =============================================================================
def deep_update(base, new):
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith((".yml", ".yaml")):
            if yaml is None:
                sys.exit("PyYAML is required for YAML configs (pip install pyyaml), or use a JSON config.")
            user = yaml.safe_load(f)
        else:
            user = json.load(f)
    return deep_update(DEFAULT_CONFIG, user)


class DecisionLog:
    """Collects, for every decision, the literature default, what the data
    suggest, the value applied, and the evidence behind it."""

    def __init__(self):
        self.rows = []

    def add(self, step, decision, literature_default, data_suggestion, applied, evidence, note=""):
        self.rows.append({
            "step": step, "decision": decision,
            "literature_default": literature_default,
            "data_driven_suggestion": data_suggestion,
            "applied_value": applied, "evidence": evidence, "note": note,
        })
        print(f"  -> DECISION [{step}] {decision}: applied={applied} | "
              f"default={literature_default} | data suggest={data_suggestion}")

    def save(self, out_dir):
        df = pd.DataFrame(self.rows)
        df.to_csv(os.path.join(out_dir, "decision_log.csv"), index=False)
        lines = ["# Decision report", "",
                 "Each row documents one preprocessing decision: the literature default, "
                 "what this dataset suggests, the value that was applied (from the configuration), "
                 "and the evidence. If the applied value and the data-driven suggestion disagree, "
                 "revise the configuration and re-run, or report why the applied value was kept.", ""]
        for r in self.rows:
            lines += [f"## {r['step']} -- {r['decision']}", "",
                      f"- **Literature default:** {r['literature_default']}",
                      f"- **Data-driven suggestion:** {r['data_driven_suggestion']}",
                      f"- **Applied value:** {r['applied_value']}",
                      f"- **Evidence:** {r['evidence']}"]
            if r["note"]:
                lines.append(f"- **Note:** {r['note']}")
            lines.append("")
        with open(os.path.join(out_dir, "decision_report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# =============================================================================
# STATISTICS UTILITIES
# =============================================================================
def bca_mean_ci(values, cfg, rng):
    vals = np.asarray([v for v in values if np.isfinite(v)], float)
    if len(vals) < 3 or np.std(vals) == 0:
        m = float(np.mean(vals)) if len(vals) else np.nan
        return m, np.nan, np.nan
    res = bootstrap((vals,), np.mean, n_resamples=cfg["analysis"]["n_bootstrap"],
                    method="BCa", random_state=rng, vectorized=False)
    return float(np.mean(vals)), float(res.confidence_interval.low), float(res.confidence_interval.high)


def paired_signflip_test(dict_a, dict_b, cfg, rng):
    """Paired sign-flip permutation test on participant-level values.
    p = (b + 1) / (B + 1), so it is never exactly zero (Phipson & Smyth, 2010)."""
    common = sorted(set(dict_a) & set(dict_b))
    if len(common) < 3:
        return np.nan, np.nan, len(common)
    diff = np.array([dict_a[p] - dict_b[p] for p in common])
    obs = abs(diff.mean())
    B = cfg["analysis"]["n_permutations"]
    signs = rng.choice([-1, 1], size=(B, len(common)))
    null = np.abs((signs * diff).mean(axis=1))
    p = (np.sum(null >= obs) + 1) / (B + 1)
    return float(obs), float(p), len(common)


def holm(pvals):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p); adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (n - rank) * p[i]))
        adj[i] = running
    return adj


def bland_altman(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return dict(bias=np.nan, sd=np.nan, loa_low=np.nan, loa_high=np.nan, n=len(d))
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    return dict(bias=bias, sd=sd, loa_low=bias - 1.96 * sd, loa_high=bias + 1.96 * sd, n=len(d))


def cohen_kappa(x, y):
    x = np.asarray(x, bool); y = np.asarray(y, bool)
    po = np.mean(x == y)
    px, py = x.mean(), y.mean()
    pe = px * py + (1 - px) * (1 - py)
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def mixedlm_condition(df, y, cfg):
    """Condition effect on a trial-level metric with a by-participant random
    intercept and slope; falls back to random intercept only if needed."""
    if smf is None:
        return {"metric": y, "note": "statsmodels not installed"}
    d = df.dropna(subset=[y]).copy()
    d["cond_code"] = (d["condition"] == cfg["conditions"]["of_interest"]).astype(int)
    if d[y].std() == 0 or d["cond_code"].nunique() < 2:
        return {"metric": y, "note": "constant metric or single condition"}
    for re_formula, label in [("~cond_code", "random intercept + slope"), (None, "random intercept only")]:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                m = smf.mixedlm(f"{y} ~ cond_code", d, groups=d["participant"],
                                re_formula=re_formula).fit(method="lbfgs")
            singular = any("boundary" in str(x.message).lower() or "singular" in str(x.message).lower()
                           for x in w)
            return {"metric": y, "beta": float(m.params["cond_code"]), "se": float(m.bse["cond_code"]),
                    "z": float(m.tvalues["cond_code"]), "p": float(m.pvalues["cond_code"]),
                    "random_effects": label, "singular_or_boundary": singular, "note": ""}
        except Exception as e:  # noqa: BLE001
            last = str(e)
    return {"metric": y, "note": f"model failed: {last}"}


def detect_elbow(values, step, flat_drop_pp, sustained_steps, max_threshold=None):
    """First threshold after which the marginal % of excluded trials stays
    below `flat_drop_pp` for `sustained_steps` consecutive steps."""
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return None, None
    max_threshold = max_threshold or float(np.nanmax(v)) + step
    th = np.arange(0.0, max_threshold + step, step)
    pct = np.array([(v > t).mean() * 100 for t in th])
    drop = -np.diff(pct)
    run = 0
    for i, flat in enumerate(np.abs(drop) < flat_drop_pp):
        run = run + 1 if flat else 0
        if run >= sustained_steps:
            idx = i - sustained_steps + 1
            return float(th[1:][idx]), float(pct[1:][idx])
    return None, None


# =============================================================================
# SIGNAL UTILITIES
# =============================================================================
def interpolate(t, sig, query_t, method="linear", n_anchor=2):
    """Reconstruct `sig` (NaN = missing) at `query_t`.
    linear: straight line between the valid samples bounding each gap.
    cubic : local cubic spline through `n_anchor` valid samples on each side of
            each gap (as in blink-reconstruction algorithms), linear fallback.
    Values outside the first/last valid sample are held constant."""
    t = np.asarray(t, float); sig = np.asarray(sig, float); query_t = np.asarray(query_t, float)
    valid = np.isfinite(sig)
    if valid.sum() < 2:
        return np.full(len(query_t), np.nan)
    out = np.interp(query_t, t[valid], sig[valid])
    if method == "linear":
        return out
    if method != "cubic":
        raise ValueError(f"Unknown interpolation method: {method}")
    order = np.argsort(t); ts, ss, vs = t[order], sig[order], valid[order]
    n, i = len(ts), 0
    while i < n:
        if vs[i]:
            i += 1; continue
        j = i
        while j < n and not vs[j]:
            j += 1
        left = np.where(vs[:i])[0]; right = np.where(vs[j:])[0] + j
        if len(left) >= n_anchor and len(right) >= n_anchor:
            at = np.concatenate([ts[left[-n_anchor:]], ts[right[:n_anchor]]])
            ay = np.concatenate([ss[left[-n_anchor:]], ss[right[:n_anchor]]])
            if len(np.unique(at)) == len(at):
                cs = CubicSpline(at, ay)
                g0, g1 = ts[i - 1] if i > 0 else ts[i], ts[j] if j < n else ts[j - 1]
                seg = (query_t > g0) & (query_t < g1)
                out[seg] = cs(query_t[seg])
        i = j
    return out


def apply_filter(signal, method, params, fs):
    """Low-pass smoothing. Missing samples are temporarily set to 0 and
    restored as NaN after filtering."""
    x = np.asarray(signal, float)
    nm = ~np.isfinite(x)
    if nm.all() or method == "none":
        return x.copy()
    s = x.copy(); s[nm] = 0.0
    if method == "gaussian":
        out = gaussian_filter1d(s, sigma=params.get("sigma", 2), mode="nearest")
    elif method == "moving_average":
        w = int(params.get("window", 5)); k = np.ones(w) / w; p = w // 2
        out = np.convolve(np.pad(s, p, mode="edge"), k, mode="same")[p:p + len(s)]
    elif method == "savgol":
        w = int(params.get("window", 7))
        if w > len(s):
            w = len(s) if len(s) % 2 == 1 else len(s) - 1
        out = savgol_filter(s, window_length=w, polyorder=min(params.get("polyorder", 3), w - 1), mode="nearest")
    elif method == "butterworth":
        b, a = butter(params.get("order", 4), params.get("cutoff_hz", 4.0) / (fs / 2.0), btype="low")
        out = filtfilt(b, a, s, padlen=min(3 * max(len(a), len(b)), len(s) - 1))
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    out[nm] = np.nan
    return out


# =============================================================================
# DATA LOADING
# =============================================================================
def load_raw_trials(cfg):
    """Read the long-format CSV and return one dict of raw arrays per trial."""
    c = cfg["columns"]
    if not os.path.exists(cfg["input_csv"]):
        sys.exit(f"Input file not found: {cfg['input_csv']} (set 'input_csv' in the config; "
                 "run make_example_data.py to create the example dataset).")
    df = pd.read_csv(cfg["input_csv"], low_memory=False)
    rec = cfg["recording"]
    # absolute timestamps + epoch labels -> time relative to stimulus onset
    if rec.get("onset_epoch"):
        ep = c.get("epoch")
        if not ep or ep not in df.columns:
            sys.exit("recording.onset_epoch is set but columns.epoch is missing from the data.")
        df["_time"] = pd.to_numeric(df[c["time"]], errors="coerce")
        is_onset = df[ep].astype(str).str.strip() == str(rec["onset_epoch"])
        onset = df[is_onset].groupby([c["participant"], c["trial"]])["_time"].min().rename("_onset")
        df = df.join(onset, on=[c["participant"], c["trial"]])
        df = df[df["_onset"].notna()].copy()
        df["_time_rel"] = df["_time"] - df["_onset"]
        c = dict(c, time="_time_rel")
    # per-eye gaze -> single gaze position (mean of on-screen eyes; off-screen if none)
    eye_gaze = [c.get(k) for k in ("left_gaze_x", "left_gaze_y", "right_gaze_x", "right_gaze_y")]
    if not (c.get("gaze_x") and c["gaze_x"] in df.columns) and all(k and k in df.columns for k in eye_gaze):
        def on(col):
            v = pd.to_numeric(df[col], errors="coerce")
            return v.where((v >= 0) & (v <= 1))
        for axis, (lc, rc) in (("x", eye_gaze[0::2]), ("y", eye_gaze[1::2])):
            m = pd.concat([on(lc), on(rc)], axis=1).mean(axis=1)
            df[f"_gaze_{axis}"] = m.fillna(-1.0)
        c = dict(c, gaze_x="_gaze_x", gaze_y="_gaze_y")
    required = ["participant", "trial", "condition", "time", "left_pupil", "right_pupil"]
    missing = [k for k in required if not c.get(k) or c[k] not in df.columns]
    if missing:
        sys.exit(f"Missing required columns (check 'columns' in the config): {missing}")

    base_ms, resp_ms = rec["baseline_window_ms"], rec["response_window_ms"]
    valid_values = cfg["validity"]["valid_values"]

    def validity(series):
        if series is None:
            return None
        if valid_values is None:
            return series.fillna(False).astype(bool).to_numpy()
        return series.isin(valid_values).to_numpy()

    has_gaze = bool(c.get("gaze_x")) and c["gaze_x"] in df.columns and bool(c.get("gaze_y")) and c["gaze_y"] in df.columns
    trials = []
    for (pid, tr), g in df.groupby([c["participant"], c["trial"]], sort=True):
        t = pd.to_numeric(g[c["time"]], errors="coerce").to_numpy(float) * rec["time_unit_to_ms"]
        keep = np.isfinite(t) & (t >= -base_ms) & (t <= resp_ms)
        if keep.sum() < 3:
            continue
        g = g.loc[keep]; t = t[keep]
        order = np.argsort(t, kind="stable"); g = g.iloc[order]; t = t[order]
        lv = validity(g[c["left_valid"]]) if c.get("left_valid") and c["left_valid"] in g else None
        rv = validity(g[c["right_valid"]]) if c.get("right_valid") and c["right_valid"] in g else None
        trials.append({
            "participant": str(pid), "trial": tr,
            "condition": str(g[c["condition"]].iloc[0]).strip(),
            "t": t,
            "lp": pd.to_numeric(g[c["left_pupil"]], errors="coerce").to_numpy(float),
            "rp": pd.to_numeric(g[c["right_pupil"]], errors="coerce").to_numpy(float),
            "lv": lv, "rv": rv,
            "gx": pd.to_numeric(g[c["gaze_x"]], errors="coerce").to_numpy(float) if has_gaze else None,
            "gy": pd.to_numeric(g[c["gaze_y"]], errors="coerce").to_numpy(float) if has_gaze else None,
        })
    conds = {tr["condition"] for tr in trials}
    for k in ("of_interest", "reference"):
        if cfg["conditions"][k] not in conds:
            sys.exit(f"Condition '{cfg['conditions'][k]}' not found in data. Found: {sorted(conds)}")
    print(f"Loaded {len(trials)} trials from {len({t['participant'] for t in trials})} participants.")
    return trials, has_gaze


# =============================================================================
# CORE TRIAL PROCESSING
# =============================================================================
def eye_valid_masks(raw, pupil_min, pupil_max):
    """Per-eye sample validity: tracker flag, finite, within the plausible
    pupil range and (if gaze is available) gaze within the screen."""
    ok_l = np.isfinite(raw["lp"]) & (raw["lp"] >= pupil_min) & (raw["lp"] <= pupil_max)
    ok_r = np.isfinite(raw["rp"]) & (raw["rp"] >= pupil_min) & (raw["rp"] <= pupil_max)
    if raw["lv"] is not None: ok_l &= raw["lv"]
    if raw["rv"] is not None: ok_r &= raw["rv"]
    if raw["gx"] is not None:
        on = (raw["gx"] >= 0) & (raw["gx"] <= 1) & (raw["gy"] >= 0) & (raw["gy"] <= 1)
        ok_l &= on; ok_r &= on
    return ok_l, ok_r


def process_trial(raw, p, grid, fs):
    """Apply one set of parameters `p` to one trial. Returns a dict with all
    quantities needed for exclusion decisions and analysis."""
    t = raw["t"]
    ok_l, ok_r = eye_valid_masks(raw, p["pupil_min"], p["pupil_max"])
    lp = np.where(ok_l, raw["lp"], np.nan); rp = np.where(ok_r, raw["rp"], np.nan)
    resp = t >= 0; base = t < 0

    # coverage of each eye in the response window
    n_resp = int(resp.sum())
    pct_l = 100.0 * ok_l[resp].sum() / n_resp if n_resp else 0.0
    pct_r = 100.0 * ok_r[resp].sum() / n_resp if n_resp else 0.0
    best, worst = max(pct_l, pct_r), min(pct_l, pct_r)
    ratio = worst / best if best > 0 else np.nan

    # eye selection: single better eye below the ratio threshold, otherwise
    # average of both eyes (samples valid in only one eye taken from that eye)
    if best > 0 and ratio < p["ratio_threshold"]:
        comb = lp.copy() if pct_l >= pct_r else rp.copy()
        eye_sel = "left_only" if pct_l >= pct_r else "right_only"
    else:
        comb = np.nanmean(np.vstack([lp, rp]), axis=0)
        eye_sel = "binocular_average"

    # gaps in the response window (between valid samples)
    tr_, cr_ = t[resp], comb[resp]
    tv = tr_[np.isfinite(cr_)]
    if len(tv) >= 2:
        max_gap = float(np.diff(tv).max()); lead = float(tv[0]); trail = float(p["resp_ms"] - tv[-1])
    else:
        max_gap = lead = trail = np.nan

    # baseline: per-sample binocular value within the baseline window
    b_both = np.nanmean(np.vstack([lp[base], rp[base]]), axis=0) if base.any() else np.array([])
    b_both = b_both[np.isfinite(b_both)]
    b_mean = float(np.mean(b_both)) if len(b_both) else np.nan
    b_median = float(np.median(b_both)) if len(b_both) else np.nan

    # boundary edge: samples adjacent to stimulus onset invalid in both eyes
    w = p["edge_window"]
    both_bad = ~(ok_l | ok_r)
    edge = bool(both_bad[base][-w:].any() or both_bad[resp][:w].any()) if (base.any() and resp.any()) else True

    # reconstruct on the analysis grid, baseline-correct, smooth
    recon = interpolate(tr_, cr_, grid, p["interp_method"], p["n_anchor"])
    b_val = b_mean if p["baseline_summary"] == "mean" else b_median
    bc = recon - b_val
    smoothed = apply_filter(bc, p["smooth_method"], p["smooth_params"], fs)

    return {
        "participant": raw["participant"], "trial": raw["trial"], "condition": raw["condition"],
        "pct_left": pct_l, "pct_right": pct_r, "max_coverage": best, "coverage_ratio": ratio,
        "eye_selection": eye_sel,
        "pct_interpolated": 100.0 * (1 - np.isfinite(cr_).mean()) if len(cr_) else np.nan,
        "max_gap_ms": max_gap, "leading_edge_ms": lead,
        "trailing_edge_ms": trail, "baseline_mean": b_mean, "baseline_median": b_median,
        "boundary_edge": edge, "t_resp": tr_, "raw_resp": cr_,
        "bc_unsmoothed": bc, "pupil": smoothed,
    }


def params_from_cfg(cfg, **overrides):
    p = {
        "pupil_min": cfg["pupil_range"]["min_mm"], "pupil_max": cfg["pupil_range"]["max_mm"],
        "ratio_threshold": cfg["interocular"]["ratio_threshold"],
        "min_valid_pct": cfg["coverage"]["min_valid_pct"],
        "z_threshold": cfg["baseline"]["z_threshold"],
        "mad_threshold": cfg["baseline"]["mad_threshold"],
        "max_gap_ms": cfg["gap"]["max_gap_ms"],
        "edge_enabled": cfg["boundary_edge"]["enabled"],
        "edge_window": int(cfg["boundary_edge"]["window_samples"]),
        "interp_method": cfg["interpolation"]["method"],
        "n_anchor": int(cfg["interpolation"]["cubic_anchor_points"]),
        "baseline_summary": cfg["baseline"]["summary"],
        "smooth_method": cfg["smoothing"]["method"], "smooth_params": cfg["smoothing"]["params"],
        "resp_ms": cfg["recording"]["response_window_ms"],
    }
    p.update(overrides)
    return p


def run_cascade(raw_trials, p, grid, fs):
    """Process all trials with parameters `p` and apply the exclusion cascade
    in order: coverage -> baseline z-score -> boundary edge -> max gap."""
    trials = [process_trial(r, p, grid, fs) for r in raw_trials]
    for t in trials:
        t.update(excl_coverage=t["max_coverage"] < p["min_valid_pct"], excl_no_baseline=False,
                 excl_baseline_z=False, excl_boundary_edge=False, excl_max_gap=False,
                 baseline_z=np.nan, baseline_robust_z=np.nan, flag_mad=False)
    surv = [t for t in trials if not t["excl_coverage"]]
    for t in surv:
        if not np.isfinite(t["baseline_mean"]):
            t["excl_no_baseline"] = True
    surv = [t for t in surv if not t["excl_no_baseline"]]

    # baseline z-scores within participant (mean/SD) and robust (median/scaled MAD)
    by_p = {}
    for t in surv:
        by_p.setdefault(t["participant"], []).append(t)
    for ts in by_p.values():
        b = np.array([t["baseline_mean"] for t in ts])
        sd = b.std(ddof=1) if len(b) > 1 else 0.0
        med = np.median(b); mad = MAD_SCALE * np.median(np.abs(b - med))
        for t, bi in zip(ts, b):
            t["baseline_z"] = (bi - b.mean()) / sd if sd > 0 else 0.0
            t["baseline_robust_z"] = (bi - med) / mad if mad > 0 else 0.0
            t["excl_baseline_z"] = abs(t["baseline_z"]) > p["z_threshold"]
            t["flag_mad"] = abs(t["baseline_robust_z"]) > p["mad_threshold"]
    surv = [t for t in surv if not t["excl_baseline_z"]]
    if p["edge_enabled"]:
        for t in surv:
            t["excl_boundary_edge"] = t["boundary_edge"]
        surv = [t for t in surv if not t["excl_boundary_edge"]]
    for t in surv:
        t["excl_max_gap"] = bool(np.isfinite(t["max_gap_ms"]) and t["max_gap_ms"] > p["max_gap_ms"])
    for t in trials:
        t["accepted"] = not any(t[k] for k in ("excl_coverage", "excl_no_baseline", "excl_baseline_z",
                                               "excl_boundary_edge", "excl_max_gap"))
    return trials


def participant_curves(trials, cond, field, min_trials):
    out = {}
    for pid in sorted({t["participant"] for t in trials}):
        pt = [t[field] for t in trials if t["participant"] == pid and t["condition"] == cond]
        if len(pt) >= min_trials:
            out[pid] = np.nanmean(np.vstack(pt), axis=0)
    return out


def window_means(curves, grid, window):
    m = (grid >= window[0]) & (grid <= window[1])
    return {pid: float(np.nanmean(c[m])) for pid, c in curves.items()}


def benchmark_values(trials, cfg, grid, field="pupil"):
    """Participant-level benchmark: mean of the condition of interest in the
    sensitivity window, using accepted trials only."""
    acc = [t for t in trials if t["accepted"]]
    curves = participant_curves(acc, cfg["conditions"]["of_interest"], field, cfg["analysis"]["min_trials_per_cell"])
    return window_means(curves, grid, cfg["analysis"]["sensitivity_window_ms"])


# =============================================================================
# FIGURE HELPERS
# =============================================================================
def savefig(fig, path, cfg):
    if cfg["steps"]["save_figures"]:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def bar_ci(ax, labels, means, lows, highs, colors=None):
    x = np.arange(len(labels))
    ax.bar(x, means, color=colors or "steelblue", alpha=0.85)
    yerr = [np.array(means) - np.array(lows), np.array(highs) - np.array(means)]
    ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.spines[["top", "right"]].set_visible(False)


# =============================================================================
# STEP 1 -- DATA VALIDITY SCREENING
# =============================================================================
def step1_pupil_range(raw_trials, cfg, grid, fs, log, out, rng):
    print("\n[1.1] Physiologically plausible pupil range")
    pr = cfg["pupil_range"]
    allp = np.concatenate([np.concatenate([r["lp"], r["rp"]]) for r in raw_trials])
    allp = allp[np.isfinite(allp) & (allp > 0)]
    lo_app, lo_lit, hi = pr["min_mm"], pr["literature_min_mm"], pr["max_mm"]
    lo_band = min(lo_app, lo_lit)
    pct_below = float((allp < lo_band).mean() * 100)
    pct_between = float(((allp >= lo_band) & (allp < max(lo_app, lo_lit))).mean() * 100)
    pct_above = float((allp > hi).mean() * 100)
    print(f"  samples < {lo_band} mm: {pct_below:.1f}% | between {lo_band}-{max(lo_app, lo_lit)} mm: "
          f"{pct_between:.1f}% | > {hi} mm: {pct_above:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, ttl in [(axes[0], allp, "A. Full range"), (axes[1], allp[allp <= 4], "B. Lower tail (<= 4 mm)")]:
        ax.hist(data, bins=100, color="steelblue", alpha=0.8)
        for v, col in [(lo_app, "orange"), (lo_lit, "red"), (hi, "red")]:
            if v <= data.max() + 0.1:
                ax.axvline(v, color=col, ls="--")
        ax.set_xlabel("Pupil diameter (mm)"); ax.set_ylabel("Samples"); ax.set_title(ttl)
        ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, os.path.join(out, "fig_1_1_pupil_distribution.png"), cfg)

    # interocular agreement in the floor band vs. the adjacent band above it
    bw = pr["band_width_mm"]
    bands = [(lo_band, lo_band + bw), (lo_band + bw, lo_band + 2 * bw)]
    L = np.concatenate([r["lp"] for r in raw_trials]); R = np.concatenate([r["rp"] for r in raw_trials])
    ok = np.isfinite(L) & np.isfinite(R) & (L > 0) & (R > 0)
    mean_lr = (L[ok] + R[ok]) / 2; diff_lr = L[ok] - R[ok]
    ba_rows = []
    for lo, hi_b in bands:
        m = (mean_lr >= lo) & (mean_lr < hi_b)
        ba = bland_altman(L[ok][m], R[ok][m])
        mad_lr = float(np.median(np.abs(diff_lr[m]))) if m.any() else np.nan
        ba_rows.append({"band_mm": f"{lo:.2f}-{hi_b:.2f}", **ba, "median_abs_LR_diff": mad_lr})
        print(f"  interocular agreement {lo:.2f}-{hi_b:.2f} mm: bias={ba['bias']:+.3f}, "
              f"LOA=[{ba['loa_low']:.3f}, {ba['loa_high']:.3f}] (n={ba['n']})")
    pd.DataFrame(ba_rows).to_csv(os.path.join(out, "pupil_range_interocular_agreement.csv"), index=False)

    # does the floor change the benchmark?
    cand = sorted({lo_app, lo_lit})
    vals, rows = {}, []
    for floor in cand:
        tr = run_cascade(raw_trials, params_from_cfg(cfg, pupil_min=floor), grid, fs)
        vals[floor] = benchmark_values(tr, cfg, grid)
        m, l, h = bca_mean_ci(vals[floor].values(), cfg, rng)
        rows.append({"pupil_min_mm": floor, "n_participants": len(vals[floor]), "mean_mm": m, "ci_low": l, "ci_high": h})
    p_perm = np.nan
    if len(cand) == 2:
        _, p_perm, _ = paired_signflip_test(vals[cand[0]], vals[cand[1]], cfg, rng)
    df = pd.DataFrame(rows); df["paired_permutation_p"] = p_perm
    df.to_csv(os.path.join(out, "pupil_range_benchmark_sensitivity.csv"), index=False)
    print(df.to_string(index=False))

    # excess disagreement: median |left - right| in the floor band > 1.5 x the band above it
    wider = bool(ba_rows[0]["median_abs_LR_diff"] > 1.5 * ba_rows[1]["median_abs_LR_diff"]) \
        if ba_rows[0]["n"] >= 3 and ba_rows[1]["n"] >= 3 else False
    suggestion = (f"keep {lo_lit} mm (lower band shows excess interocular disagreement)" if wider
                  else f"lower floor ({lo_band} mm) is defensible: no excess interocular disagreement")
    log.add("1.1", "Minimum pupil diameter", f"{lo_lit} mm", suggestion, f"{lo_app} mm",
            f"{pct_between:.1f}% of samples between {lo_band} and {max(lo_app, lo_lit)} mm; median |L-R| "
            f"{ba_rows[0]['median_abs_LR_diff']:.3f} mm ({ba_rows[0]['band_mm']}) vs "
            f"{ba_rows[1]['median_abs_LR_diff']:.3f} mm ({ba_rows[1]['band_mm']}); benchmark "
            f"{', '.join(f'{r.pupil_min_mm} mm: {r.mean_mm:.4f} (n={r.n_participants})' for r in df.itertuples())}; "
            f"paired permutation p={p_perm:.3f}")


def step1_min_coverage(raw_trials, cfg, grid, fs, log, out, rng):
    print("\n[1.2] Minimum per-trial data coverage")
    cov = cfg["coverage"]
    lit, ext = list(cov["literature_missing_pct"]), list(cov["extended_missing_pct"])
    cands = sorted(set(lit + ext))
    base = run_cascade(raw_trials, params_from_cfg(cfg), grid, fs)
    mc = np.array([t["max_coverage"] for t in base])

    vals, rows = {}, []
    for m in cands:
        tr = run_cascade(raw_trials, params_from_cfg(cfg, min_valid_pct=100.0 - m), grid, fs)
        vals[m] = benchmark_values(tr, cfg, grid)
        mean, lo, hi = bca_mean_ci(vals[m].values(), cfg, rng)
        rows.append({"missing_pct_tolerated": m, "literature_anchored": m in lit,
                     "pct_trials_excluded_by_coverage": float((mc < 100 - m).mean() * 100),
                     "pct_trials_accepted": float(np.mean([t["accepted"] for t in tr]) * 100),
                     "n_participants": len(vals[m]), "mean_mm": mean, "ci_low": lo, "ci_high": hi})
    df = pd.DataFrame(rows)
    perm = []
    for a, b in combinations(cands, 2):
        obs, p, n = paired_signflip_test(vals[a], vals[b], cfg, rng)
        perm.append({"a": a, "b": b, "abs_mean_diff_mm": obs, "p": p, "n": n,
                     "both_literature": a in lit and b in lit})
    dp = pd.DataFrame(perm)
    # Holm correction within the literature-anchored family and across all pairs
    dp["p_holm_all"] = holm(dp["p"].to_numpy())
    dp["p_holm_literature"] = np.nan
    lm = dp["both_literature"].to_numpy()
    if lm.any():
        dp.loc[lm, "p_holm_literature"] = holm(dp.loc[lm, "p"].to_numpy())
    df.to_csv(os.path.join(out, "coverage_candidates_benchmark.csv"), index=False)
    dp.to_csv(os.path.join(out, "coverage_candidates_pairwise_permutation.csv"), index=False)
    print(df[["missing_pct_tolerated", "pct_trials_accepted", "mean_mm", "ci_low", "ci_high"]].to_string(index=False))

    # main figure: literature-anchored candidates; supplementary: full range
    xs = np.arange(0, 91)
    for subset, fname, ttl in [(lit, "fig_1_2_min_coverage.png", "literature range"),
                               (cands, "fig_S1_min_coverage_extended.png", "extended range")]:
        sub = df[df["missing_pct_tolerated"].isin(subset)]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(xs, [(mc < 100 - x).mean() * 100 for x in xs], color="black")
        axes[0].axvspan(min(lit), max(lit), color="gray", alpha=0.15, label="literature range")
        for m in subset:
            axes[0].scatter([m], [(mc < 100 - m).mean() * 100], color="#D62828" if m in lit else "#4361EE", zorder=5)
        axes[0].set_xlabel("Missing data tolerated (%)"); axes[0].set_ylabel("Trials excluded (%)")
        axes[0].set_title(f"A. Exclusion curve ({ttl})"); axes[0].legend(frameon=False)
        axes[0].spines[["top", "right"]].set_visible(False)
        bar_ci(axes[1], [f"{int(m)}%" for m in sub["missing_pct_tolerated"]], sub["mean_mm"], sub["ci_low"],
               sub["ci_high"], ["#D62828" if l else "#4361EE" for l in sub["literature_anchored"]])
        axes[1].set_ylabel("Benchmark mean (mm)"); axes[1].set_title("B. Benchmark by candidate")
        savefig(fig, os.path.join(out, fname), cfg)

    lit_pairs = dp[dp["both_literature"]]
    any_sig = bool((lit_pairs["p_holm_literature"] < 0.05).any()) if len(lit_pairs) else False
    sug = ("literature range differs -> choose with care (see pairwise tests)" if any_sig else
           f"no detectable bias within {min(lit)}-{max(lit)}% -> most permissive ({max(lit)}%) retains most data")
    log.add("1.2", "Minimum per-trial coverage (max % missing)",
            f"{min(lit)}-{max(lit)}% (no consensus)", sug, f"{100 - cov['min_valid_pct']:.0f}% missing tolerated",
            (f"largest benchmark difference within literature range = {lit_pairs['abs_mean_diff_mm'].max():.4f} mm; "
             f"min Holm-adjusted p = {lit_pairs['p_holm_literature'].min():.3f}") if len(lit_pairs) else "n/a",
            "Extended candidates are reported to show whether the result holds outside the literature range.")
    return base


def step1_interocular(base, cfg, log, out):
    print("\n[1.3] Interocular coverage balance")
    io = cfg["interocular"]
    d = [t for t in base if not t["excl_coverage"] and np.isfinite(t["coverage_ratio"])]
    r = np.array([t["coverage_ratio"] for t in d])
    th = np.round(np.arange(0, 1.0001, 0.01), 2)
    pct = np.array([(r < x).mean() * 100 for x in th])
    slope = np.abs(np.gradient(pct, th)) * 0.1  # percentage points per 0.1 ratio
    stable = slope < io["stable_slope_pp_per_0.1"]
    # longest contiguous stable region
    best, cur, start = (None, None), 0, None
    for i, s in enumerate(stable):
        if s:
            start = i if cur == 0 else start; cur += 1
            if best[0] is None or cur > best[1] - best[0] + 1:
                best = (start, i)
        else:
            cur = 0
    applied = io["ratio_threshold"]
    flagged = float((r < applied).mean() * 100)
    region = f"{th[best[0]]:.2f}-{th[best[1]]:.2f}" if best[0] is not None else "none found"
    print(f"  applied ratio {applied}: {flagged:.1f}% of trials switched to single eye; stable region: {region}")
    pd.DataFrame({"ratio_threshold": th, "pct_flagged": pct, "stable": stable}).to_csv(
        os.path.join(out, "interocular_ratio_curve.csv"), index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(th, pct, color="black"); ax.axvline(applied, color="red", ls="--")
    if best[0] is not None:
        ax.axvspan(th[best[0]], th[best[1]], color="green", alpha=0.12, label="stable region")
        ax.legend(frameon=False)
    ax.set_xlabel("Coverage ratio threshold (worse / better eye)"); ax.set_ylabel("Trials switched to one eye (%)")
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, os.path.join(out, "fig_1_3_interocular_ratio.png"), cfg)
    in_region = best[0] is not None and th[best[0]] <= applied <= th[best[1]]
    log.add("1.3", "Interocular coverage ratio threshold", "no generally adopted criterion",
            f"any value in stable region {region}", applied,
            f"{flagged:.1f}% of trials switched to single-eye data (reference ~{io['reference_pct_flagged']}%); "
            f"applied value {'inside' if in_region else 'OUTSIDE'} the stable region")


def step1_gap(base, cfg, log, out):
    print("\n[1.4] Maximum tolerable gap length")
    gp = cfg["gap"]
    d = [t for t in base if not t["excl_coverage"] and not t["excl_no_baseline"]
         and not t["excl_baseline_z"] and not t["excl_boundary_edge"] and np.isfinite(t["max_gap_ms"])]
    g = np.array([t["max_gap_ms"] for t in d])
    elbow, pct_elbow = detect_elbow(g, gp["elbow_step_ms"], gp["elbow_flat_drop_pp"], gp["elbow_sustained_steps"])
    pct_default = float((g > gp["literature_default_ms"]).mean() * 100)
    pct_applied = float((g > gp["max_gap_ms"]).mean() * 100)
    print(f"  default {gp['literature_default_ms']} ms -> {pct_default:.2f}% excluded; "
          f"elbow {elbow} ms -> {pct_elbow if pct_elbow is None else round(pct_elbow, 2)}%")
    th = np.arange(0, g.max() + gp["elbow_step_ms"], gp["elbow_step_ms"]) if len(g) else np.array([0])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(th, [(g > x).mean() * 100 for x in th], color="black")
    ax.axvline(gp["literature_default_ms"], color="firebrick", ls="--", label="literature default")
    if elbow is not None:
        ax.axvline(elbow, color="seagreen", ls=":", label="elbow (slope ~ 0)")
    ax.set_xlabel("Candidate maximum gap (ms)"); ax.set_ylabel("Trials excluded (%)")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, os.path.join(out, "fig_1_4_gap_length.png"), cfg)
    pd.DataFrame([{"reference": "literature_default", "ms": gp["literature_default_ms"], "pct_excluded": pct_default},
                  {"reference": "elbow", "ms": elbow, "pct_excluded": pct_elbow},
                  {"reference": "applied", "ms": gp["max_gap_ms"], "pct_excluded": pct_applied}]).to_csv(
        os.path.join(out, "gap_length_diagnostic.csv"), index=False)
    log.add("1.4", "Maximum gap length", f"{gp['literature_default_ms']} ms",
            f"elbow at {elbow} ms" if elbow is not None else "no elbow detected",
            f"{gp['max_gap_ms']} ms",
            f"excluded: default {pct_default:.2f}%, elbow {pct_elbow if pct_elbow is None else round(pct_elbow, 2)}%, "
            f"applied {pct_applied:.2f}%",
            "Check STEP 2: the interpolation method should remain accurate up to the applied maximum gap.")


def step1_balance(base, cfg, log, out):
    print("\n[1.5] Balance of exclusions and data quality across conditions")
    df = pd.DataFrame([{k: t[k] for k in ("participant", "condition", "pct_left", "pct_right", "coverage_ratio",
                                          "pct_interpolated",
                                          "excl_coverage", "excl_no_baseline", "excl_baseline_z",
                                          "excl_boundary_edge", "excl_max_gap", "accepted")} for t in base])
    rates = df.groupby("condition")[["excl_coverage", "excl_baseline_z", "excl_boundary_edge",
                                     "excl_max_gap", "accepted"]].mean() * 100
    rates.to_csv(os.path.join(out, "exclusion_rates_by_condition.csv"))
    print(rates.round(2).to_string())
    models = [mixedlm_condition(df, y, cfg) for y in ("pct_left", "pct_right", "coverage_ratio", "pct_interpolated")]
    pd.DataFrame(models).to_csv(os.path.join(out, "data_quality_by_condition_lmm.csv"), index=False)
    for m in models:
        if "beta" in m:
            print(f"  {m['metric']}: beta={m['beta']:.4f}, SE={m['se']:.4f}, p={m['p']:.4f} "
                  f"({m['random_effects']}{', singular/boundary' if m['singular_or_boundary'] else ''})")
    sig = [m["metric"] for m in models if "p" in m and m["p"] < 0.05]
    log.add("1.5", "Condition balance of data quality", "should not differ; report if it does",
            "differs for: " + ", ".join(sig) if sig else "no difference detected", "reported (no correction)",
            "; ".join(f"{m['metric']} beta={m.get('beta', np.nan):.3f} p={m.get('p', np.nan):.3f}" for m in models),
            "A difference is not necessarily a problem (e.g. fewer blinks on salient trials), but must be reported.")


# =============================================================================
# STEP 2 -- INTERPOLATION
# =============================================================================
def step2_interpolation(accepted, cfg, fs, log, out, rng):
    ip = cfg["interpolation"]; n_anchor = ip["cubic_anchor_points"]
    print("\n[2.1] Linear vs. cubic: value at the midpoint of each trial's longest gap")
    rows = []
    for t in accepted:
        tt, s = t["t_resp"], t["raw_resp"]; v = np.isfinite(s)
        if v.sum() < 2 * n_anchor + 1 or not np.isfinite(t["max_gap_ms"]):
            continue
        tv = tt[v]; i = int(np.argmax(np.diff(tv))); mid = (tv[i] + tv[i + 1]) / 2
        lin = interpolate(tt, s, [mid], "linear")[0]; cub = interpolate(tt, s, [mid], "cubic", n_anchor)[0]
        rows.append({"participant": t["participant"], "max_gap_ms": t["max_gap_ms"], "linear": lin, "cubic": cub})
    d = pd.DataFrame(rows)
    summ = []
    for label, sub in [("all", d), (f"gap > {ip['longer_gap_ms']} ms", d[d["max_gap_ms"] > ip["longer_gap_ms"]])]:
        if len(sub) < 3:
            continue
        ba = bland_altman(sub["linear"], sub["cubic"])
        summ.append({"subset": label, "n": len(sub), "r": float(np.corrcoef(sub["linear"], sub["cubic"])[0, 1]),
                     **ba, "pct_abs_diff_gt_0.05mm": float(((sub["linear"] - sub["cubic"]).abs() > 0.05).mean() * 100)})
    pd.DataFrame(summ).to_csv(os.path.join(out, "interpolation_midpoint_agreement.csv"), index=False)
    for s in summ:
        print(f"  {s['subset']}: r={s['r']:.3f}, bias={s['bias']:.4f} mm, LOA=[{s['loa_low']:.3f}, {s['loa_high']:.3f}]")

    print("\n[2.2] Ground truth: synthetic gaps injected into near-complete trials")
    step_ms = 1000.0 / fs
    donors = [t for t in accepted if np.isfinite(t["raw_resp"]).mean() >= 0.98
              and np.isfinite(t["max_gap_ms"]) and t["max_gap_ms"] <= 1.5 * step_ms]
    pool = np.array([t["max_gap_ms"] for t in accepted if np.isfinite(t["max_gap_ms"]) and t["max_gap_ms"] > 1.5 * step_ms])
    res = []
    if len(donors) >= 5 and len(pool) >= 10:
        attempts = 0
        while len(res) < ip["n_synthetic_gaps"] and attempts < 10 * ip["n_synthetic_gaps"]:
            attempts += 1
            t = donors[rng.integers(len(donors))]; tt, s = t["t_resp"], t["raw_resp"]
            v = np.isfinite(s); tv = tt[v]
            gl = pool[rng.integers(len(pool))]
            lo, hi = tv.min() + ip["synthetic_margin_ms"] + gl / 2, tv.max() - ip["synthetic_margin_ms"] - gl / 2
            if hi <= lo:
                continue
            c = rng.uniform(lo, hi)
            # evaluate at the real sample nearest to the gap centre (true, unresampled value)
            k = int(np.argmin(np.where(v, np.abs(tt - c), np.inf)))
            blank = s.copy(); blank[(tt >= c - gl / 2) & (tt <= c + gl / 2)] = np.nan
            if np.isfinite(blank[k]):
                continue
            lin = interpolate(tt, blank, [tt[k]], "linear")[0]; cub = interpolate(tt, blank, [tt[k]], "cubic", n_anchor)[0]
            if np.isfinite(lin) and np.isfinite(cub):
                res.append({"gap_ms": gl, "err_linear": lin - s[k], "err_cubic": cub - s[k]})
    g = pd.DataFrame(res)
    if len(g):
        g.to_csv(os.path.join(out, "synthetic_ground_truth_draws.csv"), index=False)
        edges = np.unique(np.concatenate([[0], np.quantile(g["gap_ms"], [0.25, 0.5, 0.75]), [g["gap_ms"].max()]]))
        g["gap_bin"] = pd.cut(g["gap_ms"], edges, include_lowest=True)
        agg = g.groupby("gap_bin", observed=True).agg(
            n=("gap_ms", "size"), center_ms=("gap_ms", "mean"),
            rmse_linear=("err_linear", lambda x: np.sqrt(np.mean(x ** 2))),
            rmse_cubic=("err_cubic", lambda x: np.sqrt(np.mean(x ** 2))),
            pct_linear_closer=("err_linear", lambda x: np.nan)).reset_index()
        for i, b in enumerate(agg["gap_bin"]):
            sub = g[g["gap_bin"] == b]
            agg.loc[i, "pct_linear_closer"] = (sub["err_linear"].abs() < sub["err_cubic"].abs()).mean() * 100
        rmse_l = float(np.sqrt(np.mean(g["err_linear"] ** 2))); rmse_c = float(np.sqrt(np.mean(g["err_cubic"] ** 2)))
        closer = float((g["err_linear"].abs() < g["err_cubic"].abs()).mean() * 100)
        try:
            p_w = float(wilcoxon(g["err_linear"].abs(), g["err_cubic"].abs()).pvalue)
        except ValueError:
            p_w = np.nan
        agg.astype({"gap_bin": str}).to_csv(os.path.join(out, "synthetic_ground_truth_by_gap.csv"), index=False)
        print(f"  {len(g)} synthetic gaps: RMSE linear={rmse_l:.4f}, cubic={rmse_c:.4f}; linear closer in {closer:.1f}%")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        if len(d):
            axes[0].scatter((d["linear"] + d["cubic"]) / 2, d["linear"] - d["cubic"], s=5, alpha=0.4)
            if summ:
                for yv, ls in [(summ[0]["bias"], "-"), (summ[0]["loa_low"], ":"), (summ[0]["loa_high"], ":")]:
                    axes[0].axhline(yv, color="black", ls=ls)
        axes[0].set_xlabel("Mean of methods"); axes[0].set_ylabel("Linear - cubic")
        axes[0].set_title("A. Agreement at longest-gap midpoint"); axes[0].spines[["top", "right"]].set_visible(False)
        axes[1].plot(agg["center_ms"], agg["rmse_linear"], "o-", label="Linear")
        axes[1].plot(agg["center_ms"], agg["rmse_cubic"], "s-", label="Cubic spline")
        axes[1].set_xlabel("Synthetic gap length (ms)"); axes[1].set_ylabel("RMSE vs. ground truth")
        axes[1].set_title("B. Accuracy by gap length"); axes[1].legend(frameon=False)
        axes[1].spines[["top", "right"]].set_visible(False)
        savefig(fig, os.path.join(out, "fig_2_interpolation.png"), cfg)
        suggestion = "linear" if rmse_l <= rmse_c else "cubic"
        log.add("2", "Interpolation method", "cubic spline (e.g. blink reconstruction) or linear",
                f"{suggestion} (lower RMSE against ground truth)", ip["method"],
                f"RMSE linear={rmse_l:.4f}, cubic={rmse_c:.4f}; linear closer in {closer:.1f}% of "
                f"{len(g)} gaps (Wilcoxon p={p_w:.3g}); midpoint agreement r={summ[0]['r'] if summ else np.nan:.3f}",
                "Ground truth is the real sample nearest to the centre of each injected gap.")
    else:
        print("  Not enough near-complete donor trials or real gaps for the ground-truth check.")
        log.add("2", "Interpolation method", "cubic spline or linear", "not testable (insufficient donors)",
                ip["method"], "synthetic ground-truth check skipped")


# =============================================================================
# STEP 3 -- BASELINE
# =============================================================================
def step3_baseline(trials, cfg, grid, log, out):
    print("\n[3] Baseline reliability")
    bl = cfg["baseline"]
    pool = [t for t in trials if not t["excl_coverage"] and not t["excl_no_baseline"]]
    z = np.array([t["baseline_z"] for t in pool]); rz = np.array([t["baseline_robust_z"] for t in pool])
    ex_z = np.abs(z) > bl["z_threshold"]; ex_m = np.abs(rz) > bl["mad_threshold"]
    ba = bland_altman(z, rz); kappa = cohen_kappa(ex_z, ex_m)
    conc = {"n_trials": len(pool), "r": float(np.corrcoef(z, rz)[0, 1]), **{f"ba_{k}": v for k, v in ba.items()},
            "agreement_pct": float(np.mean(ex_z == ex_m) * 100), "kappa": kappa,
            "pct_excluded_both": float(np.mean(ex_z & ex_m) * 100),
            "pct_excluded_meanSD_only": float(np.mean(ex_z & ~ex_m) * 100),
            "pct_excluded_MAD_only": float(np.mean(~ex_z & ex_m) * 100)}
    pd.DataFrame([conc]).to_csv(os.path.join(out, "baseline_meanSD_vs_MAD.csv"), index=False)
    print(f"  mean/SD vs median/MAD: agreement={conc['agreement_pct']:.1f}%, kappa={kappa:.2f}, "
          f"mean/SD-only={conc['pct_excluded_meanSD_only']:.2f}%, MAD-only={conc['pct_excluded_MAD_only']:.2f}%")

    bm = np.array([t["baseline_mean"] for t in pool]); bmed = np.array([t["baseline_median"] for t in pool])
    ba2 = bland_altman(bm, bmed)
    mm = {"n_trials": len(pool), "r": float(np.corrcoef(bm, bmed)[0, 1]), **ba2,
          "pct_abs_diff_gt_0.05mm": float((np.abs(bm - bmed) > 0.05).mean() * 100)}
    pd.DataFrame([mm]).to_csv(os.path.join(out, "baseline_mean_vs_median.csv"), index=False)

    acc = [t for t in trials if t["accepted"]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    axes[0].hist(z[~ex_z], bins=60, color="steelblue", label="retained")
    axes[0].hist(z[ex_z], bins=60, color="firebrick", label="excluded")
    axes[0].set_xlabel("Baseline z-score (within participant)"); axes[0].legend(frameon=False)
    axes[0].set_title("A. Baseline exclusions")
    axes[1].scatter(bm, bmed, s=4, alpha=0.3); axes[1].set_xlabel("Baseline mean"); axes[1].set_ylabel("Baseline median")
    axes[1].set_title(f"B. Mean vs. median (r = {mm['r']:.3f})")
    peaks = []
    for cond, col in [(cfg["conditions"]["of_interest"], "#D62828"), (cfg["conditions"]["reference"], "#4361EE")]:
        for summary, ls in [("mean", "-"), ("median", "--")]:
            shift = 0.0
            field = []
            for t in acc:
                if t["condition"] != cond:
                    continue
                s = t["pupil"] if summary == cfg["baseline"]["summary"] else \
                    t["pupil"] + (t["baseline_mean"] - t["baseline_median"]) * (1 if summary == "median" else -1)
                field.append((t["participant"], s))
            pc = {}
            for pid, s in field:
                pc.setdefault(pid, []).append(s)
            curves = [np.nanmean(np.vstack(v), axis=0) for v in pc.values() if len(v) >= cfg["analysis"]["min_trials_per_cell"]]
            if curves:
                gm = np.nanmean(np.vstack(curves), axis=0)
                axes[2].plot(grid, gm, color=col, ls=ls, label=f"{cond} ({summary})")
                k = int(np.nanargmax(gm)); peaks.append({"condition": cond, "baseline": summary,
                                                         "peak_mm": float(gm[k]), "peak_ms": float(grid[k])})
    axes[2].set_xlabel("Time from onset (ms)"); axes[2].legend(frameon=False, fontsize=8)
    axes[2].set_title("C. Grand averages by baseline summary")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    savefig(fig, os.path.join(out, "fig_3_baseline.png"), cfg)
    pd.DataFrame(peaks).to_csv(os.path.join(out, "baseline_summary_curve_peaks.csv"), index=False)

    n_edge = np.mean([t["excl_boundary_edge"] for t in trials if not t["excl_coverage"]
                      and not t["excl_no_baseline"] and not t["excl_baseline_z"]]) * 100
    log.add("3a", "Baseline exclusion criterion", f"|z| > {bl['z_threshold']} (mean/SD)",
            "mean/SD adequate" if conc["pct_excluded_MAD_only"] < 1.0 else "robust MAD criterion flags additional trials",
            f"mean/SD, |z| > {bl['z_threshold']}",
            f"agreement {conc['agreement_pct']:.1f}% (kappa={kappa:.2f}); MAD-only exclusions {conc['pct_excluded_MAD_only']:.2f}%")
    log.add("3b", "Baseline summary statistic", "mean",
            "mean adequate" if abs(mm["bias"]) < 0.01 else "mean and median differ systematically",
            cfg["baseline"]["summary"], f"r={mm['r']:.4f}, bias={mm['bias']:.4f} mm, LOA=[{mm['loa_low']:.3f}, {mm['loa_high']:.3f}]")
    log.add("3c", "Boundary edge exclusion", "reject samples next to gaps (window scaled to sampling rate)",
            "-", f"{'on' if cfg['boundary_edge']['enabled'] else 'off'}, {cfg['boundary_edge']['window_samples']} sample(s)",
            f"{n_edge:.2f}% of remaining trials excluded")


# =============================================================================
# STEP 4 -- SMOOTHING
# =============================================================================
def step4_smoothing(accepted, cfg, grid, fs, log, out, rng):
    print("\n[4] Smoothing filter comparison")
    sm = cfg["smoothing"]; an = cfg["analysis"]
    methods = [{"label": "Applied", "method": sm["method"], "params": sm["params"]}] + [
        a for a in sm["alternatives"] if not (a["method"] == sm["method"] and a["params"] == sm["params"])]
    ci, co = cfg["conditions"]["of_interest"], cfg["conditions"]["reference"]
    win = (grid >= an["sensitivity_window_ms"][0]) & (grid <= an["sensitivity_window_ms"][1])
    metrics, rows = {}, []
    for m in methods:
        for t in accepted:
            t["_f"] = apply_filter(t["bc_unsmoothed"], m["method"], m["params"], fs)
        pt = participant_curves(accepted, ci, "_f", an["min_trials_per_cell"])
        pr = participant_curves(accepted, co, "_f", an["min_trials_per_cell"])
        common = sorted(set(pt) & set(pr))
        d = {"benchmark_mean": {p: float(np.nanmean(pt[p][win])) for p in pt},
             "contrast_mean": {p: float(np.nanmean(pt[p][win] - pr[p][win])) for p in common},
             "peak_time_ms": {p: float(grid[int(np.nanargmax(pt[p]))]) for p in pt},
             "noise_sd": {}}
        for pid in {t["participant"] for t in accepted}:
            sig = [t["_f"] for t in accepted if t["participant"] == pid]
            if len(sig) >= 3:
                d["noise_sd"][pid] = float(np.nanmean(np.nanstd(np.vstack(sig), axis=0, ddof=1)))
        metrics[m["label"]] = d
        row = {"filter": m["label"], "method": m["method"], "params": json.dumps(m["params"])}
        for k, v in d.items():
            mean, lo, hi = bca_mean_ci(v.values(), cfg, rng)
            row.update({f"{k}": mean, f"{k}_ci_low": lo, f"{k}_ci_high": hi})
        rows.append(row)
    for t in accepted:
        t.pop("_f", None)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "smoothing_comparison.csv"), index=False)
    perm = []
    for m in methods[1:]:
        for k in ("benchmark_mean", "contrast_mean", "peak_time_ms", "noise_sd"):
            obs, p, n = paired_signflip_test(metrics["Applied"][k], metrics[m["label"]][k], cfg, rng)
            perm.append({"applied_vs": m["label"], "metric": k, "abs_mean_diff": obs, "p": p, "n": n})
    dp = pd.DataFrame(perm); dp.to_csv(os.path.join(out, "smoothing_pairwise_permutation.csv"), index=False)
    print(df[["filter", "benchmark_mean", "contrast_mean", "peak_time_ms", "noise_sd"]].round(4).to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (k, ttl) in zip(axes.flat, [("benchmark_mean", "A. Benchmark mean (mm)"), ("contrast_mean", "B. Contrast (mm)"),
                                         ("peak_time_ms", "C. Peak time (ms)"), ("noise_sd", "D. Residual noise (SD, mm)")]):
        bar_ci(ax, df["filter"], df[k], df[f"{k}_ci_low"], df[f"{k}_ci_high"],
               ["firebrick" if f == "Applied" else "steelblue" for f in df["filter"]])
        ax.set_title(ttl)
    savefig(fig, os.path.join(out, "fig_4_smoothing.png"), cfg)

    lowest = df.loc[df["noise_sd"].idxmin(), "filter"]
    rel = (df["benchmark_mean"] - df.loc[df["filter"] == "Applied", "benchmark_mean"].iloc[0]).abs().max()
    base_val = abs(df.loc[df["filter"] == "Applied", "benchmark_mean"].iloc[0])
    log.add("4", "Smoothing filter", "none, or low-pass (e.g. 4 Hz Butterworth)",
            f"lowest residual noise: {lowest}", f"{sm['method']} {sm['params']}",
            f"max change in benchmark across filters = {rel:.4f} mm "
            f"({(rel / base_val * 100 if base_val else np.nan):.1f}% of the applied value); "
            f"timing differences p>=.05 for {int((dp[dp.metric == 'peak_time_ms']['p'] >= .05).sum())}/"
            f"{int((dp.metric == 'peak_time_ms').sum())} comparisons",
            "Paired tests can detect tiny but systematic shifts; judge practical size, not only p.")


# =============================================================================
# STEP 5 -- GAZE-POSITION CONFOUND
# =============================================================================
def step5_gaze(raw_trials, accepted, cfg, log, out):
    sc = cfg["screen"]
    if not all(sc.get(k) for k in ("width_cm", "height_cm", "viewing_distance_cm")):
        print("\n[5] Gaze check skipped: set screen width/height and viewing distance in the config.")
        return
    print("\n[5] Gaze position by condition")
    keep = {(t["participant"], t["trial"]) for t in accepted}
    rows, tc = [], []
    for r in raw_trials:
        if (r["participant"], r["trial"]) not in keep:
            continue
        resp = r["t"] >= 0
        x = (r["gx"][resp] - 0.5) * sc["width_cm"]; y = (0.5 - r["gy"][resp]) * sc["height_cm"]
        on = (r["gx"][resp] >= 0) & (r["gx"][resp] <= 1) & (r["gy"][resp] >= 0) & (r["gy"][resp] <= 1)
        D = sc["viewing_distance_cm"]
        h = np.degrees(np.arctan(np.abs(x) / D)); v = np.degrees(np.arctan(np.abs(y) / D))
        tot = np.degrees(np.arctan(np.sqrt(x ** 2 + y ** 2) / D))
        h, v, tot = [np.where(on, a, np.nan) for a in (h, v, tot)]
        rows.append({"participant": r["participant"], "trial": r["trial"], "condition": r["condition"],
                     "horizontal_deg": np.nanmean(h), "vertical_deg": np.nanmean(v), "total_deg": np.nanmean(tot)})
        tc.append(pd.DataFrame({"condition": r["condition"], "t": r["t"][resp], "total_deg": tot}))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(out, "gaze_trial_level.csv"), index=False)
    summary = df.groupby("condition")[["horizontal_deg", "vertical_deg", "total_deg"]].mean()
    summary.to_csv(os.path.join(out, "gaze_summary_by_condition.csv")); print(summary.round(3).to_string())
    models = [mixedlm_condition(df, y, cfg) for y in ("horizontal_deg", "vertical_deg", "total_deg")]
    pd.DataFrame(models).to_csv(os.path.join(out, "gaze_by_condition_lmm.csv"), index=False)
    tcd = pd.concat(tc); tcd["bin"] = (tcd["t"] // 100) * 100 + 50
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond, col in [(cfg["conditions"]["of_interest"], "#D62828"), (cfg["conditions"]["reference"], "#4361EE")]:
        s = tcd[tcd["condition"] == cond].groupby("bin")["total_deg"].mean()
        ax.plot(s.index, s.values, color=col, label=cond)
    ax.set_xlabel("Time from onset (ms)"); ax.set_ylabel("Gaze eccentricity (deg)")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, os.path.join(out, "fig_5_gaze.png"), cfg)
    diff = float(summary["total_deg"].max() - summary["total_deg"].min())
    log.add("5", "Gaze-position confound", "check covariates by condition (report; correct only if needed)",
            "negligible" if summary["total_deg"].max() < 8 and diff < 1 else "inspect: large eccentricity or difference",
            "reported (no correction)",
            f"mean eccentricity {summary['total_deg'].min():.2f}-{summary['total_deg'].max():.2f} deg; "
            f"difference {diff:.2f} deg; " + "; ".join(f"{m['metric']} p={m.get('p', np.nan):.3f}" for m in models),
            "Pupil foreshortening error is small within about +/-8 deg horizontally and +/-6 deg vertically (Hayes & Petrov, 2016).")


# =============================================================================
# STEP 6 -- GRAND AVERAGE AND EXPORT
# =============================================================================
def step6_grand_average(trials, cfg, grid, out, out_root):
    print("\n[6] Grand average and export")
    acc = [t for t in trials if t["accepted"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond, col in [(cfg["conditions"]["of_interest"], "#D62828"), (cfg["conditions"]["reference"], "#4361EE")]:
        pc = participant_curves(acc, cond, "pupil", cfg["analysis"]["min_trials_per_cell"])
        if len(pc) < 2:
            continue
        mat = np.vstack(list(pc.values())); m = np.nanmean(mat, 0); se = np.nanstd(mat, 0, ddof=1) / np.sqrt(len(mat))
        ax.plot(grid, m, color=col, lw=2, label=f"{cond} (n={len(pc)})"); ax.fill_between(grid, m - se, m + se, color=col, alpha=0.2)
        pd.DataFrame({"time_ms": grid, "mean": m, "sem": se}).to_csv(os.path.join(out, f"grand_average_{cond}.csv"), index=False)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Time from stimulus onset (ms)"); ax.set_ylabel("Baseline-corrected pupil size")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, os.path.join(out, "fig_6_grand_average.png"), cfg)

    # long-format preprocessed data (input for the statistical analysis in R)
    long = [pd.DataFrame({"participant": t["participant"], "trial": t["trial"], "condition": t["condition"],
                          "time_ms": grid, "pupil_bc": t["pupil"]}) for t in acc]
    pd.concat(long).to_csv(os.path.join(out_root, "preprocessed_accepted_trials_long.csv"), index=False)
    ledger_keys = ["participant", "trial", "condition", "pct_left", "pct_right", "max_coverage", "coverage_ratio",
                   "eye_selection", "pct_interpolated", "max_gap_ms", "leading_edge_ms", "trailing_edge_ms", "baseline_mean",
                   "baseline_median", "baseline_z", "baseline_robust_z", "boundary_edge", "excl_coverage",
                   "excl_no_baseline", "excl_baseline_z", "excl_boundary_edge", "excl_max_gap", "accepted"]
    pd.DataFrame([{k: t[k] for k in ledger_keys} for t in trials]).to_csv(
        os.path.join(out_root, "trial_ledger.csv"), index=False)
    print(f"  exported {len(acc)} accepted trials -> preprocessed_accepted_trials_long.csv")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Decision-by-decision pupillometry preprocessing.")
    ap.add_argument("--config", default=None, help="YAML or JSON configuration file")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg["analysis"]["seed"])
    fs = float(cfg["recording"]["sampling_rate_hz"])
    grid = np.arange(0.0, cfg["recording"]["response_window_ms"] + 1e-9, 1000.0 / fs)

    root = cfg["output_dir"]
    dirs = {k: os.path.join(root, v) for k, v in {
        1: "01_validity_screening", 2: "02_interpolation", 3: "03_baseline",
        4: "04_smoothing", 5: "05_gaze_confound", 6: "06_grand_average"}.items()}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(root, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    log = DecisionLog()
    raw, has_gaze = load_raw_trials(cfg)

    section("STEP 1 -- DATA VALIDITY SCREENING")
    step1_pupil_range(raw, cfg, grid, fs, log, dirs[1], rng)
    base = step1_min_coverage(raw, cfg, grid, fs, log, dirs[1], rng)
    step1_interocular(base, cfg, log, dirs[1])
    step1_gap(base, cfg, log, dirs[1])
    step1_balance(base, cfg, log, dirs[1])
    n = len(base)
    summary = {k: float(np.mean([t[k] for t in base]) * 100) for k in
               ("excl_coverage", "excl_no_baseline", "excl_baseline_z", "excl_boundary_edge", "excl_max_gap", "accepted")}
    pd.DataFrame([summary]).to_csv(os.path.join(dirs[1], "exclusion_cascade_summary.csv"), index=False)
    print("\n  Exclusion cascade (% of all trials): " + ", ".join(f"{k}={v:.2f}" for k, v in summary.items()))
    accepted = [t for t in base if t["accepted"]]
    if len(accepted) < 10:
        sys.exit("Too few accepted trials to continue; check the configuration.")

    section("STEP 2 -- INTERPOLATION")
    step2_interpolation(accepted, cfg, fs, log, dirs[2], rng)
    section("STEP 3 -- BASELINE")
    step3_baseline(base, cfg, grid, log, dirs[3])
    section("STEP 4 -- SMOOTHING")
    step4_smoothing(accepted, cfg, grid, fs, log, dirs[4], rng)
    section("STEP 5 -- GAZE-POSITION CONFOUND")
    if cfg["steps"]["gaze_confound"] and has_gaze:
        step5_gaze(raw, accepted, cfg, log, dirs[5])
    else:
        print("  skipped (no gaze columns or disabled in config)")
    section("STEP 6 -- GRAND AVERAGE AND EXPORT")
    step6_grand_average(base, cfg, grid, dirs[6], root)

    log.save(root)
    print(f"\nDone. {len(accepted)}/{n} trials accepted. Decision report: {os.path.join(root, 'decision_report.md')}")


if __name__ == "__main__":
    main()
