"""
Phase 5: apply the promotion rule to B5 vs current production P at 4 weeks (P = R1c protocol, channels
Holt + seasonal + macro, naive skill reference). Rule: validation win (mean over folds of each fold's own
validation fused MAPE) AND test better in >= 5 of 6 folds (CLEANBLEED).
Usage (repo root): python audit_reports/phase5_scripts/p5_b5_vs_p_4wk.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "tests_audit"))
import audit_core as ac  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from facts_mas.agent_cache import AgentCache  # noqa: E402
from facts_mas.run_backtest import FOLDS  # noqa: E402

cells = ac.cells_from_cache(AgentCache.load("agent_output_cache.pkl"), ac.load_panel())
R1 = ac.r1_values_fn(ac.load_r1_seasonal())
M = ac.build_models(cells, R1, agents=["ar", "seasonality", "macro"], min_support_weeks=13, bound_targets=True)
pv = {f.fold: ac.val_fused_mape(cells, f, M[f.fold], R1, 4) for f in FOLDS}
v5 = pd.read_csv("audit_reports/results_tables/phase5_validation.csv", comment="#")
pf = pd.read_csv("audit_reports/results_tables/phase5_per_fold.csv", comment="#")
b5v = v5[v5.h == 4].set_index("fold").B5
t = pf[pf.h == 4].set_index("fold")
print("validation 4wk per fold  P:", {k: round(v, 4) for k, v in pv.items()})
print("validation 4wk per fold B5:", b5v.round(4).to_dict())
print(f"mean validation: P {np.mean(list(pv.values())):.4f}  B5 {b5v.mean():.4f}  -> B5 val win: {b5v.mean() < np.mean(list(pv.values()))}")
fb = int((t.B5 < t.P).sum())
print(f"test folds where B5 < P: {fb}/6 ; pooled test B5 {t.B5.mean():.4f} vs P {t.P.mean():.4f} (fold means)")
print("RULE: B5 replaces P at 4wk =", bool(b5v.mean() < np.mean(list(pv.values())) and fb >= 5))

# Same rule at 8 and 13 weeks: B5 (FA-ETS-d skill reference) vs P (C2 channels, naive skill reference).
import pickle  # noqa: E402
B = pickle.load(open(sys.argv[1], "rb")) if len(sys.argv) > 1 else None
if B is not None:
    key = lambda c: (c.msa, c.origin, c.h)

    def c2fn(c, f):
        v = dict(R1(c, f))
        v["ar"] = B[key(c)]["fa_ets_d"]   # damped chosen in every fold (Phase 3 and 5)
        return v
    M2 = ac.build_models(cells, c2fn, agents=["ar", "seasonality", "macro"], min_support_weeks=13, bound_targets=True)
    for h in (8, 13):
        pv = {f.fold: ac.val_fused_mape(cells, f, M2[f.fold], c2fn, h) for f in FOLDS}
        b5v = v5[v5.h == h].set_index("fold").B5
        t = pf[pf.h == h].set_index("fold")
        win = b5v.mean() < np.mean(list(pv.values()))
        fb = int((t.B5 < t.P).sum())
        print(f"h={h}: mean val P {np.mean(list(pv.values())):.4f} B5 {b5v.mean():.4f} (B5 win {win}); "
              f"test folds B5 < P {fb}/6; RULE B5 replaces P = {bool(win and fb >= 5)}")
