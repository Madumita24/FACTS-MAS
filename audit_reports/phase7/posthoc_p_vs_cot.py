"""
POST HOC (not pre-registered): FACTS-MAS (P) and FA-ETS-d against the NEXUS-prompt CoT forecast on the holdout,
smoothed cut. Requested by Umang on 2026-10-22, after the holdout MAPEs were seen; descriptive only.
Uses the runner's own statistics (stats_lib.pair_stats; block length as in run_holdout.holdout_stats).
Lag 2 uses only weeks where all 15 MSAs have a CoT forecast (FINAL_CONFIG 11.5).
Writes results_tables/phase7_posthoc_p_vs_cot.csv.

Usage: python audit_reports/phase7/posthoc_p_vs_cot.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "phase4_scripts"))
import stats_lib as sl  # noqa: E402

rows = []
for lag in (0, 2):
    D = pd.read_csv(os.path.join(HERE, f"results_20261022_smoothed_lag{lag}", "cell_losses.csv"))
    bad = D[D.cot.isna()][["h", "origin"]].drop_duplicates().assign(_x=1)
    D = D.merge(bad, on=["h", "origin"], how="left")
    D = D[D._x.isna()].drop(columns="_x")
    for h in (4, 8, 13):
        sub = D[D.h == h]
        W = sl.weekly(sub, ["P", "fa_ets_d", "cot"])
        T = len(W)
        Lb = h if h <= T / 2 else int(np.floor(T / 2))
        idx = sl.block_indices(T, sl.SEED + h, block_len=Lb)
        for a, b in (("P", "cot"), ("fa_ets_d", "cot")):
            r = sl.pair_stats(sub, W, idx, a, b, h)
            r.update(lag=lag, block_len=Lb, ci_flag_unreliable=h > T / 2 or h == 13,
                     mape_A=float(sub[a].mean()), mape_B=float(sub[b].mean()))
            rows.append(r)
out = pd.DataFrame(rows)
p = os.path.join(HERE, "..", "results_tables", "phase7_posthoc_p_vs_cot.csv")
with open(p, "w", encoding="utf-8", newline="") as f:
    f.write("# CONFIG: POST HOC, not pre-registered (requested after holdout MAPEs were seen) | holdout smoothed cut | "
            "CoT gpt-5.6-luna NEXUS prompt | stats_lib.pair_stats, block L=h (13wk floor(T/2)), B=5000 | lag 2: complete "
            "weeks only | 13wk descriptive\n")
    out.to_csv(f, index=False)
print(out[["lag", "h", "A", "B", "mape_A", "mape_B", "mean_diff", "ci95_lo", "ci95_hi", "panel_dm_p",
           "msas_A_lower", "BH_A_better", "BH_B_better", "T_weeks"]].round(4).to_string(index=False))
