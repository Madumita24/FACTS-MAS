"""
Phase 3: standalone baseline forecasts for every cached cell (validation + test).

  snaive, naive2 (window-7 factors), holt_d (damped Holt on levels),
  fa_ets / fa_ets_d (factor-adjusted ETS, undamped / damped), theta (factor-adjusted)

Every model sees only data on or before the origin (panel capped at 2026-02-21
and then truncated at the origin). ETS / Theta are fitted once per
(MSA, origin) and sliced to each horizon (fits are horizon-independent).

Self-checks: naive2 == cached R1 seasonal (window 7); a refit of undamped Holt
on a sample == cached AR.

Usage (repo root):  python audit_reports/phase3_scripts/p3_build_baselines.py <out.pkl> [--panel <capped long csv>]
--panel lets the raw-Zillow robustness run reuse this builder.
"""
import argparse
import os
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.getcwd(), "tests_audit"))
import audit_core as ac  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

H_MAX = 13


def factors_w7(panel, msa, origin):
    from facts_mas.agents import seasonality_agent as sa
    from facts_mas.calibration import override_constant
    with override_constant(sa, "SEASONAL_SMOOTHING_WINDOW", 7):
        return sa.compute_seasonal_factors(panel, msa, origin)


def ets(y, damped):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(y, trend="add", damped_trend=damped, seasonal=None,
                                   initialization_method="estimated").fit()
        return np.asarray(fit.forecast(H_MAX), dtype=float)


def theta(y):
    from statsmodels.tsa.forecasting.theta import ThetaModel
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(ThetaModel(y, period=1, deseasonalize=False).fit().forecast(H_MAX), dtype=float)


def forecasts_at(panel, msa, o):
    """All new baselines at one (msa, origin), from data on or before the origin only."""
    md = panel[panel["msa"] == msa].sort_values("date").reset_index(drop=True)
    hist = md[md["date"] <= pd.Timestamp(o)]
    y = hist["inventory_count"].to_numpy(dtype=float)
    dates = hist["date"]
    n = len(y)
    f = factors_w7(panel, msa, o)
    wk = dates.apply(lambda d: d.isocalendar()[1]).to_numpy()
    last_wk = int(wk[-1])
    fut_wk = np.array([((last_wk - 1 + k) % 53) + 1 for k in range(1, H_MAX + 1)])
    adj = y / f[wk - 1]
    sn = np.array([y[n - 1 + k - 52] for k in range(1, H_MAX + 1)])
    return {
        "snaive": sn,
        "naive2": np.maximum(y[-1] * f[fut_wk - 1] / f[last_wk - 1], 0.0),
        "holt_d": np.maximum(ets(y, True), 0.0),
        "fa_ets": np.maximum(ets(adj, False) * f[fut_wk - 1], 0.0),
        "fa_ets_d": np.maximum(ets(adj, True) * f[fut_wk - 1], 0.0),
        "theta": np.maximum(theta(adj) * f[fut_wk - 1], 0.0),
    }


def one_msa(args):
    msa, origins, panel_path = args
    panel = pd.read_csv(panel_path, parse_dates=["date"]) if panel_path else ac.load_panel()
    panel = panel[panel["date"] <= ac.DATA_CAP]
    out = {}
    for o in origins:
        out[(msa, o)] = forecasts_at(panel, msa, o)
    return msa, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--panel", default=None)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    from facts_mas.agent_cache import AgentCache
    cache = AgentCache.load("agent_output_cache.pkl")
    pairs = sorted({(e.msa, e.origin) for e in cache.entries.values()})
    by_msa = {}
    for m, o in pairs:
        by_msa.setdefault(m, []).append(o)
    print(f"{len(pairs)} (msa, origin) pairs over {len(by_msa)} MSAs", flush=True)
    res = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for msa, out in ex.map(one_msa, [(m, os_, args.panel) for m, os_ in by_msa.items()]):
            res.update(out)
            print(f"  {msa} done ({len(out)} origins)", flush=True)
    # slice to cells
    cells = {}
    for e in cache.entries.values():
        r = res[(e.msa, e.origin)]
        cells[(e.msa, e.origin, e.horizon)] = {k: v[:e.horizon].copy() for k, v in r.items()}
    if args.panel is None:
        w7 = ac.load_r1_seasonal()
        worst = max(float(np.max(np.abs(cells[k]["naive2"] - w7[k]))) for k in cells)
        print(f"self-check naive2 vs R1 seasonal (window 7): max abs diff {worst:.3g}")
        panel = ac.load_panel()
        import random
        random.seed(3)
        sample = random.sample(sorted(cache.entries), 40)
        from ar_agent import fit_and_forecast
        wd = 0.0
        for k in sample:
            e = cache.entries[k]
            s = panel[(panel.msa == e.msa) & (panel.date <= pd.Timestamp(e.origin))].sort_values("date").set_index("date")["inventory_count"]
            wd = max(wd, float(np.max(np.abs(np.array(fit_and_forecast(s, e.horizon)["point_forecast"]) - e.agent_values["ar"])
                                     / np.maximum(e.agent_values["ar"], 1))))
        print(f"self-check Holt refit vs cached AR (40 cells): max rel diff {wd:.3g}")
    with open(args.out, "wb") as fh:
        pickle.dump(cells, fh)
    print("written", args.out, len(cells), "cells")


if __name__ == "__main__":
    main()
