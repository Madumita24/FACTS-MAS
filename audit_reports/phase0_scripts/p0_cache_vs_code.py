"""Phase 0: does agent_output_cache.pkl match what the CURRENT agent code produces?
Random sample of test-window cells, all five agents, live recompute. Read-only."""
import sys, os, random
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from facts_mas.agent_cache import AgentCache
from facts_mas.run_backtest import build_agent_runners

df = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
cache = AgentCache.load("agent_output_cache.pkl")
runners = build_agent_runners()
random.seed(20261014)
cells = []
for h in (4, 8, 13):
    cs = cache.cells(h, test_only=True)
    cells += random.sample(cs, 12)
worst = {a: 0.0 for a in runners}
for e in cells:
    for a, r in runners.items():
        live = np.array(r(df, e.msa, e.origin, e.horizon).values)
        d = float(np.max(np.abs(live - e.agent_values[a]) / np.maximum(np.abs(e.agent_values[a]), 1)))
        worst[a] = max(worst[a], d)
    # cached actuals/naive vs data
    md = df[df.msa == e.msa].sort_values("date")
    fut = md[md.date > pd.Timestamp(e.origin)].head(e.horizon)["inventory_count"].values
    assert np.allclose(fut, e.actuals), (e.msa, e.origin)
print(f"checked {len(cells)} cells x {len(runners)} agents")
print("max relative |live - cached| per agent:", {a: f"{v:.2e}" for a, v in worst.items()})
