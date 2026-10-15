"""
Phase 2: fused MAPE by regime (R1 and R1c, Router on/off) and the Router
effect by fold. Regenerates phase2_regime_cells.csv and phase2_router_by_fold.csv.

Run from repo root:  python audit_reports/phase2_scripts/p2_regime_router.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "tests_audit"))
import audit_core as ac  # noqa: E402

import pandas as pd  # noqa: E402

from facts_mas.agent_cache import AgentCache  # noqa: E402

cells = ac.cells_from_cache(AgentCache.load("agent_output_cache.pkl"), ac.load_panel())
fn = ac.r1_values_fn(ac.load_r1_seasonal())
CONFIGS = {
    "PROVISIONAL": (ac.build_models(cells), ac.production_values),
    "R1": (ac.build_models(cells, fn), fn),
    "R1 combined": (ac.build_models(cells, fn, min_support_weeks=13, bound_targets=True), fn),
}
OUT = os.path.join("audit_reports", "results_tables")

reg, rout = [], []
for lab, (M, f) in CONFIGS.items():
    a = ac.score(cells, M, f, "adaptive", exclude_bleed=True)
    b = ac.score(cells, M, f, "fold", exclude_bleed=True)
    if lab != "PROVISIONAL":
        for sch, d in (("adaptive", a), ("fold", b)):
            for (h, r), g in d.groupby(["h", "regime"]):
                reg.append(dict(config=lab, scheme=sch, h=h, regime=r, n=len(g), fused=g.fused.mean(), naive=g.naive.mean()))
    e = (b.groupby(["h", "fold"]).fused.sum() - a.groupby(["h", "fold"]).fused.sum()) / a.groupby("h").size()
    for (h, fo), v in e.items():
        rout.append(dict(config=lab, h=h, fold=fo, router_contrib=v))

R = pd.DataFrame(reg)
print(R.pivot_table(index=["h", "regime", "n", "naive"], columns=["config", "scheme"], values="fused").round(3).to_string())
with open(os.path.join(OUT, "phase2_regime_cells.csv"), "w", newline="") as fh:
    fh.write("# CONFIG: R1 and R1+fix2+F0.10 | TEST CLEANBLEED | fused MAPE by regime of origin | "
             "scheme adaptive=production Router, fold=Router removed\n")
    R.to_csv(fh, index=False)
D = pd.DataFrame(rout)
print(D.pivot_table(index=["h", "fold"], columns="config", values="router_contrib").round(3).to_string())
print(D.groupby(["config", "h"]).router_contrib.sum().round(3).unstack())
with open(os.path.join(OUT, "phase2_router_by_fold.csv"), "w", newline="") as fh:
    fh.write("# CONFIG: TEST CLEANBLEED | router_contrib = (fold-weights MAPE - adaptive MAPE) summed over the "
             "fold's cells / all cells at h; positive = Router helps\n")
    D.to_csv(fh, index=False)
