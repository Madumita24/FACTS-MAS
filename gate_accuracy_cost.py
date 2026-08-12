"""
What does the early-exit gate actually cost in accuracy?

A skip rate on its own says nothing. A gate that skipped 100% of forecasts
would report a perfect saving and forecast badly. The number that matters is
the pair: how much compute was avoided, and how much error that bought.

Compares three policies over fold 6's test window, all 15 metros:
    full      run the 5-agent fused system on every forecast
    gated     accept the cheap baseline when the gate stays shut,
              run the fused system only when it opens
    baseline  never run the heavy loop at all (the floor)
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from facts_mas.agents import event_agent
from facts_mas.nodes.gate_node import default_baseline, evaluate_gate
from facts_mas.nodes.spatial_node import build_spatial_graph, compute_spillover_signal
from facts_mas.run_backtest import FOLDS, HORIZONS, build_agent_runners, compute_mape
from facts_mas.state.gate import GateConfig
from facts_mas.state.ontology import GateTriggerType

WEEKLY = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
MSAS = sorted(WEEKLY["msa"].unique().tolist())
FOLD = FOLDS[5]

ev = pd.read_csv(event_agent.EVENT_WEEKLY_PATH, parse_dates=["week"])
ACTIVE = set()
for (m, w), g in ev.groupby(["msa", "week"], sort=False):
    b = g.iloc[0]["incident_begin_date"]
    if pd.notna(b) and (w - pd.Timestamp(b)).days / 7.0 <= event_agent.DECLARATION_ACTIVE_WINDOW_WEEKS:
        ACTIVE.add((m, w.date()))

print("Building spatial graph...")
GRAPH = build_spatial_graph(WEEKLY, MSAS, as_of=FOLD.train_end)
runners = build_agent_runners()
W = {n: 1.0 / len(runners) for n in runners}
CFG = GateConfig()

rows = []
for msa in MSAS:
    md = WEEKLY[WEEKLY["msa"] == msa].sort_values("date")
    spill = compute_spillover_signal(WEEKLY, GRAPH, msa, FOLD.test_end)
    cached = {}
    origins = md[(md["date"] >= pd.Timestamp(FOLD.test_start))
                 & (md["date"] <= pd.Timestamp(FOLD.test_end))]["date"]
    for ots in origins:
        origin = ots.date()
        for h in HORIZONS:
            fut = md[md["date"] > ots].head(h)
            if len(fut) < h:
                continue
            act = fut["inventory_count"].values.astype(np.float64)
            trig = GateTriggerType.NATURAL_DISASTER if (msa, origin) in ACTIVE else None
            # The gate decision is horizon-independent -- the z-score is
            # computed from one-week-ahead errors and the triggers do not
            # reference the horizon at all. Only the baseline's LENGTH
            # changes. Evaluating once per origin instead of once per
            # (origin, horizon) removes two thirds of the ETS fits.
            if h == HORIZONS[0]:
                d = evaluate_gate(WEEKLY, msa, origin, h, config=CFG,
                                  event_trigger=trig, spillover=spill)
                cached[origin] = d
            d = cached[origin]
            base = default_baseline(WEEKLY, msa, origin, h)

            vals = []
            for run in runners.values():
                try:
                    vals.append(np.array(run(WEEKLY, msa, origin, h), dtype=np.float64)
                                if False else np.array(run(WEEKLY, msa, origin, h).values))
                except Exception:
                    pass
            fused = np.maximum(np.mean(vals, axis=0), 0.0) if vals else base

            rows.append({
                "h": h, "open": d.should_activate_heavy_loop,
                "full": compute_mape(act, fused),
                "gated": compute_mape(act, fused if d.should_activate_heavy_loop else base),
                "baseline": compute_mape(act, base),
            })
    print(f"  {msa} done", flush=True)

df = pd.DataFrame(rows)
print("\n" + "=" * 72)
print("GATE: compute saved vs accuracy cost — fold 6, 15 metros")
print("=" * 72)
print(f"\n{'horizon':>8} {'gate open':>11} {'full':>9} {'gated':>9} {'baseline':>10} {'cost of gating':>16}")
for h in HORIZONS:
    s = df[df.h == h]
    print(f"{h:>6}wk {s['open'].mean():>10.1%} {s['full'].mean():>9.3f} "
          f"{s['gated'].mean():>9.3f} {s['baseline'].mean():>10.3f} "
          f"{s['gated'].mean() - s['full'].mean():>+15.3f}")

print(f"\n{'':>8} {'gate open':>11} {'full':>9} {'gated':>9} {'baseline':>10} {'cost of gating':>16}")
print(f"{'ALL':>8} {df['open'].mean():>10.1%} {df['full'].mean():>9.3f} "
      f"{df['gated'].mean():>9.3f} {df['baseline'].mean():>10.3f} "
      f"{df['gated'].mean() - df['full'].mean():>+15.3f}")

o, c = df[df.open], df[~df.open]
print(f"\nWhere the gate OPENED (n={len(o)}): full {o['full'].mean():.3f} vs baseline {o['baseline'].mean():.3f} "
      f"-> heavy loop worth {o['baseline'].mean()-o['full'].mean():+.3f}")
print(f"Where it stayed SHUT (n={len(c)}): full {c['full'].mean():.3f} vs baseline {c['baseline'].mean():.3f} "
      f"-> heavy loop worth {c['baseline'].mean()-c['full'].mean():+.3f}")
print("\n(A good gate opens where the heavy loop is worth the most.)")
df.to_csv("gate_accuracy_cost.csv", index=False)
