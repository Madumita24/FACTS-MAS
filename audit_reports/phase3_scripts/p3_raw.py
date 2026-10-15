"""
Phase 3 (A2): raw vs smoothed Zillow cut. Same cells, folds, origins, protocol (R1c).

Raw: all five channels recomputed on the raw series (p3_build_raw_agents.py),
targets and naive from the raw series. Smoothed: the standard cells.
Pre-registered methods: naive, temporal (Holt), seasonal (w7), FACTS-MAS R1c.
Addition (logged before running, robustness only): FA-ETS-d and C2.

Usage (repo root): python audit_reports/phase3_scripts/p3_raw.py <raw_agents.pkl> <raw_baselines.pkl> <smoothed_baselines.pkl>
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
RAW_PANEL = os.path.join("audit_reports", "artifacts", "zillow_raw_vintage20261015_capped20260221.csv")
R1C = dict(min_support_weeks=13, bound_targets=True)

cache = AgentCache.load("agent_output_cache.pkl")
RA = pickle.load(open(sys.argv[1], "rb"))
RB = pickle.load(open(sys.argv[2], "rb"))
SB = pickle.load(open(sys.argv[3], "rb"))
raw_panel = pd.read_csv(RAW_PANEL, parse_dates=["date"])
assert raw_panel["date"].max() <= ac.DATA_CAP
smooth_cells = ac.cells_from_cache(cache, ac.load_panel())
W7 = ac.load_r1_seasonal()

# raw Cell objects: same keys / regimes / target dates; raw actuals, naive and channel values
raw_cells = []
for c in smooth_cells:
    r = RA[(c.msa, c.origin, c.h)]
    raw_cells.append(ac.Cell(c.msa, c.origin, c.h, c.regime, r["actuals"], r["naive"], c.last_target, c.target_dates,
                             {a: r[a] for a in ac.AGENTS}, c.test_fold))


def damped_choice(cells, B, fold):
    val = [c for c in cells if ac._in_val(c, fold, True)]
    mu = np.mean([ac.mape_pct(c.actuals, B[(c.msa, c.origin, c.h)]["fa_ets"]) for c in val])
    md = np.mean([ac.mape_pct(c.actuals, B[(c.msa, c.origin, c.h)]["fa_ets_d"]) for c in val])
    return "fa_ets_d" if md < mu else "fa_ets"


def run(cells, B, seasonal_override=None):
    base_fn = (lambda c, f: c.values) if seasonal_override is None else seasonal_override
    ch = {f.fold: damped_choice(cells, B, f) for f in FOLDS}

    def c2_fn(c, f):
        v = dict(base_fn(c, f))
        v["ar"] = B[(c.msa, c.origin, c.h)][ch[f]]
        return v
    M = ac.build_models(cells, base_fn, **R1C)
    M2 = ac.build_models(cells, c2_fn, **R1C)
    rows = []
    for c in ac.test_cells(cells, exclude_bleed=True):
        v = base_fn(c, c.test_fold)
        rows.append(dict(h=c.h, fold=c.test_fold, msa=c.msa,
                         naive=ac.mape_pct(c.actuals, c.naive),
                         temporal=ac.mape_pct(c.actuals, np.maximum(v["ar"], 0)),
                         seasonal=ac.mape_pct(c.actuals, v["seasonality"]),
                         R1c=ac.mape_pct(c.actuals, ac.fuse(v, ac.weights_for(c, M), c.h)),
                         fa_ets_d=ac.mape_pct(c.actuals, B[(c.msa, c.origin, c.h)]["fa_ets_d"]),
                         C2=ac.mape_pct(c.actuals, ac.fuse(c2_fn(c, c.test_fold), ac.weights_for(c, M2), c.h))))
    return pd.DataFrame(rows), ch


S, chs = run(smooth_cells, SB, ac.r1_values_fn(W7))
R, chr_ = run(raw_cells, RB)
print("damped choice smoothed:", chs, "| raw:", chr_)
cols = ["naive", "temporal", "seasonal", "R1c", "fa_ets_d", "C2"]
out = []
for lab, d in (("smoothed", S), ("raw", R)):
    g = d.groupby("h")[cols].mean()
    for h in HORIZONS:
        rec = dict(cut=lab, h=h, **g.loc[h].to_dict())
        dd = d[d.h == h]
        pm = dd.groupby("msa")[cols].mean()
        rec.update({f"msa_wins_{m}_vs_naive": int((pm[m] < pm["naive"]).sum()) for m in cols[1:]})
        rec["msa_wins_fa_ets_d_vs_R1c"] = int((pm["fa_ets_d"] < pm["R1c"]).sum())
        pf = dd.groupby("fold")[cols].mean()
        rec["folds_fa_ets_d_better_than_R1c"] = int((pf["fa_ets_d"] < pf["R1c"]).sum())
        out.append(rec)
T = pd.DataFrame(out)
pd.set_option("display.width", 250)
print(T.round(3).to_string(index=False))
pf = pd.concat([S.assign(cut="smoothed"), R.assign(cut="raw")]).groupby(["cut", "h", "fold"])[cols].mean().reset_index()
print(pf.round(3).to_string(index=False))
p = os.path.join("audit_reports", "results_tables", "phase3_raw_vs_smoothed.csv")
with open(p, "w", newline="") as fh:
    fh.write(f"# CONFIG: commit={COMMIT} | TEST CLEANBLEED | R1c protocol | raw = Zillow raw weekly (sha256 18dd108c...), "
             f"vintage 2026-10-15, capped 2026-02-21, all channels recomputed on raw; smoothed vintage identical to July | "
             f"seasonal window 7 not re-selected on raw | fa_ets_d and C2 = logged addition, robustness only\n")
    T.to_csv(fh, index=False)
with open(p.replace(".csv", "_per_fold.csv"), "w", newline="") as fh:
    fh.write("# CONFIG: as phase3_raw_vs_smoothed.csv, per fold\n")
    pf.to_csv(fh, index=False)
