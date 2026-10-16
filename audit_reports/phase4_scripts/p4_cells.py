"""
Phase 4 step 1: per-cell test losses (MAPE over steps 1..H) for every compared method.

P (production) = Phase 3 rule outcome with event and intrinsic retired:
  4wk  R1c protocol, channels {Holt, seasonal, macro}
  8/13 C2  (temporal = FA-ETS-d, damped chosen per fold on validation), channels {FA-ETS-d, seasonal, macro}
Usage (repo root): python audit_reports/phase4_scripts/p4_cells.py <phase3 baselines.pkl>
Writes audit_reports/results_tables/phase4_cell_losses.csv (bleed rows flagged; DISCLOSURE rows included).
"""
import os
import pickle
import subprocess
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "tests_audit"))
import audit_core as ac  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from facts_mas.agent_cache import AgentCache  # noqa: E402
from facts_mas.run_backtest import FOLDS, HORIZONS  # noqa: E402

COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
R1C = dict(min_support_weeks=13, bound_targets=True)
KEEP = ["ar", "seasonality", "macro"]

cells = ac.cells_from_cache(AgentCache.load("agent_output_cache.pkl"), ac.load_panel())
B = pickle.load(open(sys.argv[1], "rb"))
R1 = ac.r1_values_fn(ac.load_r1_seasonal())
key = lambda c: (c.msa, c.origin, c.h)
FOLD = {f.fold: f for f in FOLDS}
STAND = ["naive", "snaive", "naive2", "holt", "holt_d", "fa_ets", "fa_ets_d", "theta", "macro", "intrinsic"]


def stand(c, n):
    if n == "naive":
        return c.naive
    if n == "holt":
        return np.maximum(c.values["ar"], 0)
    if n in ("macro", "intrinsic"):
        return np.maximum(c.values[n], 0)
    return B[key(c)][n]


val = {f.fold: [c for c in cells if ac._in_val(c, f, True)] for f in FOLDS}
# damped choice for C2 (identical rule to Phase 3)
ch = {}
for f in FOLDS:
    mu = np.mean([ac.mape_pct(c.actuals, stand(c, "fa_ets")) for c in val[f.fold]])
    md = np.mean([ac.mape_pct(c.actuals, stand(c, "fa_ets_d")) for c in val[f.fold]])
    ch[f.fold] = "fa_ets_d" if md < mu else "fa_ets"
print("C2 damped choice:", ch)


def c2_fn(c, fold):
    v = dict(R1(c, fold))
    v["ar"] = stand(c, ch[fold])
    return v


# SelBest-model (same definition as Phase 3) and parity with the Phase 3 table
sel = {}
for f in FOLDS:
    for h in HORIZONS:
        vm = {n: np.mean([ac.mape_pct(c.actuals, stand(c, n)) for c in val[f.fold] if c.h == h]) for n in STAND}
        sel[(f.fold, h)] = min(vm, key=vm.get)
p3 = pd.read_csv(os.path.join("audit_reports", "results_tables", "phase3_validation_selection.csv"), comment="#")
assert all(sel[(r.fold, r.h)] == r.selbest_model for r in p3.itertuples()), "SelBest differs from Phase 3"
print("SelBest-model matches Phase 3 selections")

M = {
    "R1c": ac.build_models(cells, R1, **R1C),
    "C2": ac.build_models(cells, c2_fn, **R1C),
    "P4": ac.build_models(cells, R1, agents=KEEP, **R1C),
    "P813": ac.build_models(cells, c2_fn, agents=KEEP, **R1C),
}
EWTS = {"ar": 0.5, "seasonality": 0.5}
rows = []
tc = ac.test_cells(cells)
ac.assert_scoring_allowed([d for c in tc for d in c.target_dates], disclosure=True)  # bleed rows kept, flagged
for c in tc:
    f = c.test_fold
    v1, v2 = R1(c, f), c2_fn(c, f)
    fc = {
        "P": ac.fuse(v1, ac.weights_for(c, M["P4"]), c.h) if c.h == 4 else ac.fuse(v2, ac.weights_for(c, M["P813"]), c.h),
        "P_5ch": ac.fuse(v1, ac.weights_for(c, M["R1c"]), c.h) if c.h == 4 else ac.fuse(v2, ac.weights_for(c, M["C2"]), c.h),
        "naive": c.naive, "snaive": stand(c, "snaive"), "naive2": stand(c, "naive2"), "fa_ets_d": stand(c, "fa_ets_d"),
        "selbest": stand(c, sel[(f, c.h)]),
        "C2": ac.fuse(v2, ac.weights_for(c, M["C2"]), c.h),
        "R1c": ac.fuse(v1, ac.weights_for(c, M["R1c"]), c.h),
        "ew_ts": ac.fuse(v1, EWTS, c.h),
        "hz": ac.fuse(v1, M["R1c"][f].horizon_weights[c.h], c.h),
    }
    r = dict(h=c.h, fold=f, msa=c.msa, origin=c.origin, bleed=bool(c.last_target >= ac.RESERVED_START))
    r.update({k: ac.mape_pct(c.actuals, x) for k, x in fc.items()})
    rows.append(r)
D = pd.DataFrame(rows)
print(D[~D.bleed].groupby("h")[[k for k in D.columns if k not in ("h", "fold", "msa", "origin", "bleed")]].mean().round(4).to_string())
out = os.path.join("audit_reports", "results_tables", "phase4_cell_losses.csv")
with open(out, "w", newline="") as fh:
    fh.write(f"# CONFIG: commit={COMMIT} | TEST cells, per-cell MAPE over steps 1..H | bleed=True rows are the 165 "
             f"fold-6 reserved-window cells (DISCLOSURE only) | P = R1c(4wk, Holt+seasonal+macro) / C2(8,13wk, "
             f"FA-ETS-d+seasonal+macro); P_5ch = same with event+intrinsic | R1c protocol, T=0.05 PROVISIONAL\n")
    D.to_csv(fh, index=False)
print("written", out, len(D))
