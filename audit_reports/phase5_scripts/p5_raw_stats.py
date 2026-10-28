"""
Phase 5 step 0 (reporting only, no selection): CIs and DM tests on the RAW Zillow cut.

Raw cells as in p3_raw.py (all channels recomputed on raw; targets and naive from raw). Methods:
naive, Holt, seasonal (Naive2), R1c (5 ch), FA-ETS-d, C2 (5 ch), P (production definition:
R1c channels Holt+seasonal+macro at 4wk, C2 channels FA-ETS-d+seasonal+macro at 8/13wk).
Stats exactly as Phase 4 (stats_lib.py). CLEANBLEED.

Usage (repo root): python audit_reports/phase5_scripts/p5_raw_stats.py <raw_agents.pkl> <raw_baselines.pkl>
"""
import os
import pickle
import subprocess
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "tests_audit"))
sys.path.insert(0, os.path.join(os.getcwd(), "audit_reports", "phase4_scripts"))
import audit_core as ac  # noqa: E402
import stats_lib as sl  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from facts_mas.agent_cache import AgentCache  # noqa: E402
from facts_mas.run_backtest import FOLDS  # noqa: E402

COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
R1C = dict(min_support_weeks=13, bound_targets=True)
KEEP = ["ar", "seasonality", "macro"]
RA = pickle.load(open(sys.argv[1], "rb"))
RB = pickle.load(open(sys.argv[2], "rb"))
smooth = ac.cells_from_cache(AgentCache.load("agent_output_cache.pkl"), ac.load_panel())
cells = [ac.Cell(c.msa, c.origin, c.h, c.regime, RA[(c.msa, c.origin, c.h)]["actuals"], RA[(c.msa, c.origin, c.h)]["naive"],
                 c.last_target, c.target_dates, {a: RA[(c.msa, c.origin, c.h)][a] for a in ac.AGENTS}, c.test_fold)
         for c in smooth]
key = lambda c: (c.msa, c.origin, c.h)
ch = {}
for f in FOLDS:
    val = [c for c in cells if ac._in_val(c, f, True)]
    mu = np.mean([ac.mape_pct(c.actuals, RB[key(c)]["fa_ets"]) for c in val])
    md = np.mean([ac.mape_pct(c.actuals, RB[key(c)]["fa_ets_d"]) for c in val])
    ch[f.fold] = "fa_ets_d" if md < mu else "fa_ets"
base = lambda c, f: c.values


def c2fn(c, f):
    v = dict(c.values)
    v["ar"] = RB[key(c)][ch[f]]
    return v


M = {"R1c": ac.build_models(cells, base, **R1C), "C2": ac.build_models(cells, c2fn, **R1C),
     "P4": ac.build_models(cells, base, agents=KEEP, **R1C), "P813": ac.build_models(cells, c2fn, agents=KEEP, **R1C)}
rows = []
for c in ac.test_cells(cells, exclude_bleed=True):
    f = c.test_fold
    fc = {"naive": c.naive, "holt": np.maximum(c.values["ar"], 0), "naive2": c.values["seasonality"],
          "fa_ets_d": RB[key(c)]["fa_ets_d"],
          "R1c": ac.fuse(c.values, ac.weights_for(c, M["R1c"]), c.h),
          "C2": ac.fuse(c2fn(c, f), ac.weights_for(c, M["C2"]), c.h),
          "P": ac.fuse(c.values, ac.weights_for(c, M["P4"]), c.h) if c.h == 4 else ac.fuse(c2fn(c, f), ac.weights_for(c, M["P813"]), c.h)}
    rows.append(dict(h=c.h, fold=f, msa=c.msa, origin=c.origin, **{k: ac.mape_pct(c.actuals, v) for k, v in fc.items()}))
D = pd.DataFrame(rows)
METH = ["P", "naive", "holt", "naive2", "fa_ets_d", "C2", "R1c"]
lv, pr, mc = [], [], []
for h in (4, 8, 13):
    sub = D[D.h == h]
    W = sl.weekly(sub, METH)
    idx = sl.block_indices(len(W), sl.SEED + h)
    for m in METH:
        est, lo, hi = sl.level_ci(W, idx, m)
        lv.append(dict(h=h, method=m, mape=est, ci95_lo=lo, ci95_hi=hi))
    for a, b in (("P", "naive"), ("P", "fa_ets_d"), ("P", "naive2"), ("P", "R1c"), ("C2", "fa_ets_d"),
                 ("R1c", "naive2"), ("fa_ets_d", "naive"), ("holt", "naive")):
        r = sl.pair_stats(sub, W, idx, a, b, h)
        r["equivalent_2pct"] = (a == "C2" or (a == "P" and b == "fa_ets_d")) and b == "fa_ets_d" and \
            bool(r["ci90_lo"] > -0.02 * W["fa_ets_d"].mean() and r["ci90_hi"] < 0.02 * W["fa_ets_d"].mean())
        pr.append(r)
    p, dup = sl.mcs(W, idx, ["P", "naive", "holt", "naive2", "fa_ets_d", "C2", "R1c"])
    mc += [dict(h=h, method=m, mcs_p=v, in_MCS90=v >= 0.10, identical_to=dup.get(m, "")) for m, v in p.items()]
pd.set_option("display.width", 250)
LV, PR, MC = pd.DataFrame(lv), pd.DataFrame(pr), pd.DataFrame(mc)
print("C2 damped choice (raw):", ch)
print(LV.pivot(index="method", columns="h", values="mape").round(3).to_string())
print(PR.round(4).to_string(index=False))
print(MC.pivot(index="method", columns="h", values="mcs_p").round(3).to_string())
hdr = (f"# CONFIG: commit={COMMIT} | RAW Zillow cut (sha256 18dd108c..., vintage 2026-10-15, capped 2026-02-21), all channels "
       f"recomputed on raw | TEST CLEANBLEED | Phase 4 stats (NW lag h-1, block L=13, B=5000) | REPORTING ONLY, no selection\n")
for name, df in (("levels", LV), ("pairs", PR), ("mcs", MC)):
    with open(os.path.join("audit_reports", "results_tables", f"phase5_raw_stats_{name}.csv"), "w", newline="") as fh:
        fh.write(hdr)
        df.to_csv(fh, index=False)
