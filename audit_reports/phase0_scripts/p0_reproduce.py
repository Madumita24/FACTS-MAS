"""Phase 0: reproduce headline numbers from agent_output_cache.pkl. Read-only.

Run from the FACTS-MAS repo root. Writes nothing into the repo.
"""
import sys, os, json, pickle
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd

import run_phase34  # top-level import is side-effect free apart from stdout reconfig
from facts_mas.agent_cache import AgentCache, make_cached_runners
from facts_mas.ablation import fuse_entry
from facts_mas.run_backtest import FOLDS, HORIZONS, compute_mape, compute_rmse

df = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
msas = sorted(df["msa"].unique().tolist())
cache = AgentCache.load("agent_output_cache.pkl")
print("CACHE:", cache.summary())
runners = make_cached_runners(cache)

adaptive_w, fold_w_fn, fold_w, mode_sel = run_phase34.build_weight_providers(df, runners, msas, FOLDS)

def fold_of(o):
    for f in FOLDS:
        if f.test_start <= o <= f.test_end:
            return f.fold

out = {"mode_selection": {f"{k[0]}_{k[1]}": v for k, v in mode_sel.items()},
       "fold_weights": {str(k): v for k, v in fold_w.items()}}
rows = []
for h in HORIZONS:
    cells = cache.cells(h, test_only=True)
    for e in cells:
        fa = fuse_entry(e, adaptive_w(e))
        ff = fuse_entry(e, fold_w_fn(e))
        rows.append(dict(h=h, fold=fold_of(e.origin), msa=e.msa, origin=e.origin, regime=e.regime,
                         naive=compute_mape(e.actuals, e.naive),
                         adaptive=compute_mape(e.actuals, fa),
                         foldw=compute_mape(e.actuals, ff),
                         **{f"solo_{a}": compute_mape(e.actuals, np.maximum(e.agent_values[a], 0)) for a in cache.agent_names}))
R = pd.DataFrame(rows)
R.to_csv(sys.argv[1], index=False)  # required arg: write OUTSIDE the repo

print("\n(A) cell-level mean over test cells (ablation.py method)")
print(R.groupby("h")[["naive", "adaptive", "foldw"] + [c for c in R if c.startswith("solo_")]].mean().round(3).to_string())
print("\ncells per horizon:", R.groupby("h").size().to_dict())

# (B) run_backtest-style: BacktestResult row = mean over origins within (fold, msa, h, regime);
#     headline = unweighted mean over rows
B = R.groupby(["h", "fold", "msa", "regime"])[["naive", "adaptive"]].mean().reset_index()
print("\n(B) unweighted mean of (fold,msa,h,regime) rows")
print(B.groupby("h")[["naive", "adaptive"]].mean().round(3).to_string())
# (C) n-weighted mean of rows == (A)
print("\nmode selection:", out["mode_selection"])
for k, v in fold_w.items():
    print("fold", k, {a: round(w, 4) for a, w in v.items()})
