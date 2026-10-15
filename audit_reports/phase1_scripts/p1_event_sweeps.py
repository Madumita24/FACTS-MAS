"""
Phase 1 (H1.2, H1.3): re-run the two event calibration sweeps that fed the
paper, unbounded (as originally run) and with targets bounded to train_end.

VALIDATION windows only. No test window is read. Uses the capped panel.

Mirrors:
  impact scale  -> calibrate_event_impact_scale.py score()   (paper Table 10)
  window        -> calibrate_event_window.py via calibration.calibrate_target
                   with the July event agent (magnitude x 0.10)  (calibration.md Study 1)
The originals cannot be imported: they read the uncapped CSV at import time.

Run from repo root:  python audit_reports/phase1_scripts/p1_event_sweeps.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "tests_audit"))
import audit_core as ac  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from facts_mas.calibration import _validation_window  # noqa: E402
from facts_mas.run_backtest import FOLDS, HORIZONS, compute_mape  # noqa: E402

W = ac.load_panel()
MSAS = sorted(W["msa"].unique())
ev = pd.read_csv("event_agent_weekly.csv", parse_dates=["week"])
IDX = {}
for (msa, week), g in ev.groupby(["msa", "week"], sort=False):
    r = g.iloc[0]
    b = r["incident_begin_date"]
    IDX[(msa, week.date())] = (float(r["impact_magnitude"]),
                               (week - pd.Timestamp(b)).days / 7.0 if pd.notna(b) else None)
BY_MSA = {m: g.sort_values("date").reset_index(drop=True) for m, g in W.groupby("msa")}


def iter_cells(start, end, bound):
    for msa in MSAS:
        md = BY_MSA[msa]
        for i in np.where((md["date"] >= pd.Timestamp(start)) & (md["date"] <= pd.Timestamp(end)))[0]:
            o = md["date"].iloc[i]
            last = float(md["inventory_count"].iloc[i])
            for h in HORIZONS:
                fut = md.iloc[i + 1:i + 1 + h]
                if len(fut) < h:
                    continue
                if bound and fut["date"].max() > pd.Timestamp(end):
                    continue
                yield msa, o.date(), h, last, fut["inventory_count"].values.astype(float)


def impact_scale_sweep(bound):
    scales = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
    table = {}
    for f in FOLDS:
        s, e = _validation_window(f)
        res = {x: [] for x in scales}
        for msa, o, h, last, act in iter_cells(s, e, bound):
            hit = IDX.get((msa, o))
            if hit is None:
                continue
            imp, age = hit
            if imp == 0.0 or (age is not None and age > 8):
                continue
            for x in scales:
                res[x].append(compute_mape(act, np.full(h, max(0.0, last * (1 + imp * x)))))
        table[f.fold] = {x: (float(np.mean(v)) if v else float("nan"), len(v)) for x, v in res.items()}
    return scales, table


def window_sweep(cands, bound):
    lo, hi = min(cands), max(cands)
    table = {}
    for f in FOLDS:
        s, e = _validation_window(f)
        res = {w: [] for w in cands}
        for msa, o, h, last, act in iter_cells(s, e, bound):
            hit = IDX.get((msa, o))
            if hit is None or hit[1] is None or not (lo < hit[1] <= hi):
                continue
            imp, age = hit
            for w in cands:
                fac = 1.0 + imp * 0.10 if age <= w else 1.0
                fc = np.full(h, max(0.0, round(last * fac, 2)))
                res[w].append(compute_mape(act, fc))
        table[f.fold] = {w: (float(np.mean(v)) if v else float("inf"), len(v)) for w, v in res.items()}
    return cands, table


def report(name, cands, table, baseline, min_n=0):
    means = {c: float(np.nanmean([table[f][c][0] for f in table if np.isfinite(table[f][c][0])])) for c in cands}
    best = min(means, key=means.get)
    dirs = []
    for f in sorted(table):
        b, a, n = table[f][baseline][0], table[f][best][0], table[f][best][1]
        if not (np.isfinite(a) and np.isfinite(b)) or n < min_n:
            dirs.append("neutral")
        else:
            dirs.append("decrease" if a < b - 0.01 else "increase" if a > b + 0.01 else "neutral")
    print(f"\n{name}")
    print("  mean MAPE: " + "  ".join(f"{c:g}:{means[c]:.3f}" for c in cands))
    print("  n per fold (best cand): " + str({f: table[f][best][1] for f in sorted(table)}))
    print(f"  best={best:g}  baseline={baseline:g}  folds improving: {dirs.count('decrease')}/6  dirs={dirs}")
    return means


if __name__ == "__main__":
    print("VALIDATION ONLY. Panel capped at", ac.DATA_CAP.date())
    for bound in (False, True):
        tag = "BOUNDED targets<=train_end" if bound else "UNBOUNDED (as originally run)"
        c, t = impact_scale_sweep(bound)
        report(f"IMPACT SCALE [{tag}]  paper Table 10: 5.813 5.982 6.227 6.529 7.229 8.242 9.357", c, t, 0.10)
        c, t = window_sweep((2, 4, 6, 8, 12, 16, 26), bound)
        report(f"WINDOW grid A [{tag}]  calibration.md: 5.442 6.067 6.530 6.671 6.797 7.943 8.905", c, t, 8, min_n=30)
        c, t = window_sweep((0, 1, 2, 3, 4, 6, 8), bound)
        report(f"WINDOW grid B [{tag}]  calibration.md: 5.321 5.844 6.094 6.318 6.532 6.855 6.974", c, t, 8, min_n=30)
