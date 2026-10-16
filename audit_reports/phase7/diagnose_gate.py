"""
Diagnose the 4-week gate gap: compare the runner's Holt refits with the cached AR values on every cell,
and the resulting fold weights / modes. Capped panel only. Usage (repo root): python audit_reports/phase7/diagnose_gate.py
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.getcwd(), "tests_audit"))
import audit_core as ac  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def refit(args):
    msa, origins = args
    from ar_agent import fit_and_forecast
    panel = ac.load_panel()
    md = panel[panel.msa == msa].sort_values("date").set_index("date")["inventory_count"]
    out = {}
    for o in origins:
        hist = md[md.index <= pd.Timestamp(o)]
        out[o] = (np.array(fit_and_forecast(hist, 13)["point_forecast"]), np.array(fit_and_forecast(hist, 4)["point_forecast"]))
    return msa, out


def main():
    from facts_mas.agent_cache import AgentCache
    from facts_mas.run_backtest import FOLDS
    cache = AgentCache.load("agent_output_cache.pkl")
    by = {}
    for (m, o, h) in cache.entries:
        if h == 4:
            by.setdefault(m, []).append(o)
    res = {}
    with ProcessPoolExecutor(max_workers=14) as ex:
        for m, out in ex.map(refit, list(by.items())):
            res[m] = out
    rows = []
    for (m, o, h), e in cache.entries.items():
        if h != 4:
            continue
        r13, r4 = res[m][o]
        c = e.agent_values["ar"]
        rows.append(dict(msa=m, origin=o, rel_13slice_vs_cache=float(np.max(np.abs(r13[:4] - c) / np.maximum(c, 1))),
                         rel_4_vs_cache=float(np.max(np.abs(r4 - c) / np.maximum(c, 1))),
                         rel_13slice_vs_4=float(np.max(np.abs(r13[:4] - r4) / np.maximum(r4, 1)))))
    D = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for col in ("rel_13slice_vs_cache", "rel_4_vs_cache", "rel_13slice_vs_4"):
        q = D[col].quantile([0.5, 0.9, 0.99, 1.0]).to_dict()
        print(f"{col}: median {q[0.5]:.2e}  p90 {q[0.9]:.2e}  p99 {q[0.99]:.2e}  max {q[1.0]:.2e}  n>1e-4: {(D[col] > 1e-4).sum()}")
    print(D.sort_values("rel_13slice_vs_cache", ascending=False).head(8).to_string(index=False))
    D.to_csv(os.path.join("audit_reports", "phase7", "diagnose_gate_ar.csv"), index=False)


if __name__ == "__main__":
    main()
