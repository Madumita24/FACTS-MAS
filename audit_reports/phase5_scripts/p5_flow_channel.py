"""
Phase 5, P5a: listing-flow channel (pre-registered in phase5_preregistration.md, Addendum A).

At origin o (all 15 MSAs at once):
  * flows usable at any date t are the values dated t - L (L = 2 weeks; timing not confirmable, Addendum A item 2)
  * x_t = log(NL_{t-L} / NP_{t-L}),  p_t = price-cut share_{t-L}
  * window-7 seasonal factors f computed at o from inventory on or before o
  * training pairs (t, t+k) with t + k <= o (expanding window, refit at every origin; Addendum A item 7):
      z_{t,k} = log(I_{t+k}/I_t) - log(f(w_{t+k}) / f(w_t))
      pooled OLS across MSAs, one regression per step k = 1..13:  z = a_k + b_k x_t + c_k p_t
  * forecast: I_o * f(w_{o+k}) / f(w_o) * exp(a_k + b_k x_o + c_k p_o)
    (future week index follows the seasonal agent: ((w_o - 1 + k) mod 53) + 1)

Usage (repo root): python audit_reports/phase5_scripts/p5_flow_channel.py <out.pkl>
"""
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.getcwd(), "tests_audit"))
import audit_core as ac  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LAG_WEEKS = 2
H_MAX = 13
FLOWS = os.path.join("audit_reports", "artifacts", "zillow_flows_sm_vintage20261016_capped20260221.csv")


def load_flows(path=FLOWS):
    f = pd.read_csv(path, parse_dates=["date"])
    assert f["date"].max() <= ac.DATA_CAP
    return f


def _factors(panel, msa, o):
    from facts_mas.agents import seasonality_agent as sa
    from facts_mas.calibration import override_constant
    with override_constant(sa, "SEASONAL_SMOOTHING_WINDOW", 7):
        return sa.compute_seasonal_factors(panel, msa, o)


def flow_forecasts(panel, flows, o, lag=LAG_WEEKS):
    """{msa: 13-step forecast} at origin o using inventory <= o and flows dated <= o - lag."""
    ots = pd.Timestamp(o)
    cutoff_flow = ots - pd.Timedelta(weeks=lag)
    fl = flows[flows["date"] <= cutoff_flow].copy()
    fl["date"] = fl["date"] + pd.Timedelta(weeks=lag)          # value dated d becomes usable at d + lag
    fl["x"] = np.log(fl["new_listings"] / fl["new_pending"])
    fl = fl[["date", "msa", "x", "price_cut_share"]]
    inv = panel[panel["date"] <= ots][["date", "msa", "inventory_count"]]
    rows_x, rows_z = {k: [] for k in range(1, H_MAX + 1)}, {k: [] for k in range(1, H_MAX + 1)}
    state = {}
    for msa, g in inv.groupby("msa"):
        g = g.sort_values("date").merge(fl[fl.msa == msa].drop(columns="msa"), on="date", how="left").reset_index(drop=True)
        f = _factors(panel, msa, o)
        wk = g["date"].apply(lambda d: d.isocalendar()[1]).to_numpy()
        li = np.log(g["inventory_count"].to_numpy(float)) - np.log(f[wk - 1])   # deseasonalized log inventory
        X = g[["x", "price_cut_share"]].to_numpy(float)
        n = len(g)
        for k in range(1, H_MAX + 1):
            if n <= k:
                continue
            z = li[k:] - li[:-k]
            xx = X[:-k]
            ok = np.isfinite(z) & np.isfinite(xx).all(1)
            rows_x[k].append(xx[ok])
            rows_z[k].append(z[ok])
        last_wk = int(wk[-1])
        fut = np.array([((last_wk - 1 + k) % 53) + 1 for k in range(1, H_MAX + 1)])
        state[msa] = (float(g["inventory_count"].iloc[-1]), f[fut - 1] / f[last_wk - 1], X[-1])
    coef = {}
    for k in range(1, H_MAX + 1):
        Xk = np.vstack(rows_x[k])
        A = np.column_stack([np.ones(len(Xk)), Xk])
        coef[k] = np.linalg.lstsq(A, np.concatenate(rows_z[k]), rcond=None)[0]
    out = {}
    for msa, (last, ratio, x0) in state.items():
        if not np.isfinite(x0).all():
            raise ValueError(f"no lagged flow value for {msa} at {o}")
        step = np.array([coef[k] @ np.r_[1.0, x0] for k in range(1, H_MAX + 1)])
        out[msa] = np.maximum(last * ratio * np.exp(step), 0.0)
    return out, coef


def _job(origins):
    panel, flows = ac.load_panel(), load_flows()
    return {o: flow_forecasts(panel, flows, o)[0] for o in origins}


def main():
    from facts_mas.agent_cache import AgentCache
    cache = AgentCache.load("agent_output_cache.pkl")
    origins = sorted({e.origin for e in cache.entries.values()})
    chunks = [origins[i::14] for i in range(14)]
    res = {}
    with ProcessPoolExecutor(max_workers=14) as ex:
        for part in ex.map(_job, chunks):
            res.update(part)
    cells = {(e.msa, e.origin, e.horizon): res[e.origin][e.msa][:e.horizon].copy() for e in cache.entries.values()}
    with open(sys.argv[1], "wb") as fh:
        pickle.dump(cells, fh)
    print("written", sys.argv[1], len(cells), "cells from", len(origins), "origins")


if __name__ == "__main__":
    main()
