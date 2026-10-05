#!/usr/bin/env python3
"""Generate a small synthetic dataset in the input format expected by
pupil_pipeline.py. The data are simulated (no real participants) and exist
only to test the pipeline and to show the expected file layout."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
FS, BASE_MS, RESP_MS = 60.0, 200.0, 2000.0
t = np.arange(-BASE_MS, RESP_MS + 1e-9, 1000.0 / FS)
rows = []
for p in range(1, 21):
    pid = f"P{p:02d}"
    size = rng.uniform(2.6, 4.5)                 # participant-specific pupil size (mm)
    amp = rng.uniform(0.05, 0.25)                # participant-specific Target response
    for tr in range(1, 61):
        cond = "Target" if rng.random() < 0.3 else "Distractor"
        b = size + rng.normal(0, 0.12)
        resp = np.where(t > 0, (amp if cond == "Target" else 0.2 * amp) * (t / 1100) * np.exp(1 - t / 1100), 0)
        drift = np.cumsum(rng.normal(0, 0.004, len(t)))
        pupil = b + resp + drift
        left = pupil + rng.normal(0, 0.02, len(t)); right = pupil + 0.05 + rng.normal(0, 0.02, len(t))
        lv = np.zeros(len(t), int); rv = np.zeros(len(t), int)
        for _ in range(rng.poisson(1.2)):        # blinks: both eyes invalid
            s = rng.integers(0, len(t)); L = rng.integers(3, 25)
            lv[s:s + L] = 4; rv[s:s + L] = 4
            left[s:s + L] = -1; right[s:s + L] = -1
        if rng.random() < 0.08:                  # tracking loss in one eye
            s = rng.integers(0, len(t) // 2); lv[s:] = 4; left[s:] = -1
        if rng.random() < 0.05:                  # implausibly small samples
            s = rng.integers(0, len(t) - 10); left[s:s + 10] = rng.uniform(1.0, 1.8, 10)
        gx = 0.5 + rng.normal(0, 0.03, len(t)); gy = 0.5 + rng.normal(0, 0.03, len(t))
        rows.append(pd.DataFrame({"participant": pid, "trial": tr, "condition": cond, "time": t,
                                  "left_pupil": left, "right_pupil": right,
                                  "left_valid": lv, "right_valid": rv, "gaze_x": gx, "gaze_y": gy}))
pd.concat(rows).round(4).to_csv("data/example_data.csv", index=False)
print("wrote data/example_data.csv")
