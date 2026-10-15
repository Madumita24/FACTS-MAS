"""
Run the PRODUCTION weighting code once (unchanged, from facts_mas/) on the
committed cache and save its exact outputs as a fixture. test_regression_pin.py
compares the fast audit mirror against this file.

Uses the capped panel (load_panel, <= 2026-02-21). Production reads actuals
for validation origins only from the panel, and those all end before
2025-05-03, so capping cannot change anything; the fixture proves it.

Run from anywhere:  python tests_audit/make_production_fixture.py   (~10 min)
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit_core as ac  # noqa: E402  (puts the repo on sys.path)

os.chdir(ac.REPO)
import numpy as np  # noqa: E402

import run_phase34  # noqa: E402
from facts_mas.agent_cache import AgentCache, make_cached_runners  # noqa: E402
from facts_mas.factor_attribution import compute_all_regime_weights, evaluate_validation_mape  # noqa: E402
from facts_mas.fusion_baseline import compute_fold_weights  # noqa: E402
from facts_mas.ablation import fuse_entry  # noqa: E402
from facts_mas.run_backtest import FOLDS, HORIZONS, compute_mape  # noqa: E402

panel = ac.load_panel()
cache = AgentCache.load("agent_output_cache.pkl")
runners = make_cached_runners(cache)
msas = sorted(panel["msa"].unique().tolist())

fx = {"commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip(),
      "cache_built_at": cache.built_at, "folds": {}}
for f in FOLDS:
    fw = compute_fold_weights(panel, runners, msas, train_end=f.train_end)
    grid = compute_all_regime_weights(panel, runners, msas, f)["grid"]
    rec = {"fold_weights": {k: float(v) for k, v in fw.items()},
           "grid": {r: {h: {"weights": {a: float(w) for a, w in grid[r][h]["weights"].items()},
                            "n": int(grid[r][h]["n"])} for h in grid[r]} for r in grid},
           "val_mape": {}, "mode": {}}
    for h in HORIZONS:
        fm, rm = evaluate_validation_mape(panel, runners, msas, f, h, fw, grid)
        rec["val_mape"][str(h)] = [fm, rm]
        rec["mode"][str(h)] = "fold" if fm <= rm else "regime"
    fx["folds"][str(f.fold)] = rec
    print("fold", f.fold, rec["mode"], flush=True)

# Production adaptive provider -> test-cell MAPE and a fixed slice of fused values.
adaptive_w, _, _, _ = run_phase34.build_weight_providers(panel, runners, msas, FOLDS)
fx["test_mape"], fx["slice"] = {}, {}
for h in HORIZONS:
    cells = cache.cells(h, test_only=True)
    fx["test_mape"][str(h)] = {
        "fused": float(np.mean([compute_mape(e.actuals, fuse_entry(e, adaptive_w(e))) for e in cells])),
        "naive": float(np.mean([compute_mape(e.actuals, e.naive) for e in cells])),
        "n": len(cells)}
    for e in cells:
        if e.msa in ("Phoenix", "Miami") and e.origin.isoformat() in ("2021-06-05", "2023-01-07", "2025-06-07"):
            fx["slice"][f"{e.msa}|{e.origin}|{h}"] = [float(x) for x in fuse_entry(e, adaptive_w(e))]

out = os.path.join(HERE, "fixtures", "production_pin.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    json.dump(fx, fh, indent=1)
print("written", out, fx["test_mape"])
