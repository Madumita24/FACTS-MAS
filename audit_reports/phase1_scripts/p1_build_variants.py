"""
Phase 1: alternative agent outputs for every cached cell (validation + test).

  seasonal_w{1,3,5,7}  seasonality agent with SEASONAL_SMOOTHING_WINDOW = w
                       (w=1 means no smoothing, the July behaviour)
  event_july           event agent exactly as at commit da77e8e (magnitude x 0.10,
                       8-week window), source loaded from git, not re-typed

Self-checks: seasonal_w5 must equal the cached seasonality values; event_july
must equal naive wherever no declaration is active.

Output (pickle, written OUTSIDE the repo):  python ... <out.pkl>
"""
import os
import pickle
import subprocess
import sys
import types

sys.path.insert(0, os.path.join(os.getcwd(), "tests_audit"))
import audit_core as ac  # noqa: E402

import numpy as np  # noqa: E402

from facts_mas.agent_cache import AgentCache  # noqa: E402
from facts_mas.agents import seasonality_agent as sa  # noqa: E402
from facts_mas.calibration import override_constant  # noqa: E402

out_path = sys.argv[1]
panel = ac.load_panel()
cache = AgentCache.load("agent_output_cache.pkl")

# July event agent, from git
src = subprocess.check_output(["git", "show", "da77e8e:facts_mas/agents/event_agent.py"], text=True)
july_event = types.ModuleType("event_agent_july_da77e8e")
exec(compile(src, "event_agent@da77e8e", "exec"), july_event.__dict__)

pairs = sorted({(e.msa, e.origin) for e in cache.entries.values()})
print(f"{len(pairs)} (msa, origin) pairs, {len(cache.entries)} cells")
var = {}
for i, (msa, o) in enumerate(pairs):
    hs = [h for h in ac.HORIZONS if cache.has(msa, o, h)]
    for w in (1, 3, 5, 7):
        with override_constant(sa, "SEASONAL_SMOOTHING_WINDOW", w):
            f = sa.compute_seasonal_factors(panel, msa, o)
        for h in hs:
            var.setdefault((msa, o, h), {})[f"seasonal_w{w}"] = np.array(
                sa.run_seasonality_agent(panel, msa, o, h, seasonal_factors=f).values)
    for h in hs:
        var[(msa, o, h)]["event_july"] = np.array(july_event.run_event_agent(panel, msa, o, h).values)
    if i % 500 == 0:
        print(f"  {i}/{len(pairs)}", flush=True)

# self-checks
worst = max(float(np.max(np.abs(var[k]["seasonal_w5"] - e.agent_values["seasonality"])))
            for k, e in cache.entries.items())
n_ev_diff = sum(not np.allclose(var[k]["event_july"], e.naive) for k, e in cache.entries.items())
print(f"self-check: max |seasonal_w5 - cached seasonality| = {worst:.3g}")
print(f"event_july differs from naive in {n_ev_diff} of {len(cache.entries)} cells")
with open(out_path, "wb") as fh:
    pickle.dump(var, fh)
print("written", out_path)
