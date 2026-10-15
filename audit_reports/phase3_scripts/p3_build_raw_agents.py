"""
Phase 3 (A2): the five R1 channels recomputed on another Zillow cut (the raw
series) for every cached (msa, origin, horizon), plus actuals and naive from
that cut.

  ar          production ETS core (ar_agent.fit_and_forecast), fitted once per
              (msa, origin) at h=13 and sliced (horizon-consistent, Phase 2)
  seasonality window-7 factors on the raw cut (window not re-selected; pre-registered)
  macro       production macro adapter on the raw panel
  event       last raw value (the production event channel is naive)
  intrinsic   production intrinsic agent on the raw panel (per horizon; it is
              horizon-inconsistent, Phase 2)

Usage (repo root): python audit_reports/phase3_scripts/p3_build_raw_agents.py <capped_long.csv> <out.pkl>
"""
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.getcwd(), "tests_audit"))
import audit_core as ac  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def one_msa(args):
    msa, keys, panel_path = args
    from ar_agent import fit_and_forecast
    from facts_mas.agents import seasonality_agent as sa
    from facts_mas.agents.intrinsic_agent import run_intrinsic_agent, train_intrinsic_model
    from facts_mas.agents.macro_agent import run_macro_agent
    from facts_mas.calibration import override_constant
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    assert panel["date"].max() <= ac.DATA_CAP
    static = pd.read_csv("intrinsic_static.csv")
    md = panel[panel.msa == msa].sort_values("date").reset_index(drop=True)
    s_all = md.set_index("date")["inventory_count"]
    out = {}
    origins = sorted({o for o, _ in keys})
    for o in origins:
        ots = pd.Timestamp(o)
        hist = s_all[s_all.index <= ots]
        fut = s_all[s_all.index > ots]
        ar13 = np.array(fit_and_forecast(hist, 13)["point_forecast"])
        with override_constant(sa, "SEASONAL_SMOOTHING_WINDOW", 7):
            f = sa.compute_seasonal_factors(panel, msa, o)
        model = train_intrinsic_model(panel, static, o)
        for h in [h for oo, h in keys if oo == o]:
            last = float(hist.iloc[-1])
            out[(msa, o, h)] = {
                "actuals": fut.iloc[:h].to_numpy(dtype=float),
                "naive": np.full(h, last),
                "ar": ar13[:h].copy(),
                "seasonality": np.array(sa.run_seasonality_agent(panel, msa, o, h, seasonal_factors=f).values),
                "macro": np.array(run_macro_agent(panel, msa, o, h).values),
                "event": np.full(h, last),
                "intrinsic": np.array(run_intrinsic_agent(panel, static, msa, o, h, model=model).values),
            }
    return msa, out


def main():
    panel_path, out_path = sys.argv[1], sys.argv[2]
    from facts_mas.agent_cache import AgentCache
    cache = AgentCache.load("agent_output_cache.pkl")
    by_msa = {}
    for (m, o, h) in cache.entries:
        by_msa.setdefault(m, []).append((o, h))
    res = {}
    with ProcessPoolExecutor(max_workers=14) as ex:
        for msa, out in ex.map(one_msa, [(m, k, panel_path) for m, k in by_msa.items()]):
            res.update(out)
            print(f"  {msa}: {len(out)} cells", flush=True)
    bad = [k for k, v in res.items() if len(v["actuals"]) < k[2]]
    assert not bad, f"{len(bad)} cells lack actuals"
    with open(out_path, "wb") as fh:
        pickle.dump(res, fh)
    print("written", out_path, len(res))


if __name__ == "__main__":
    main()
