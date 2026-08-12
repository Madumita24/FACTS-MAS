"""
Measure NEXUS Modifications A/B/C/D on real data.

Each modification exists to produce a specific number, and this script
produces it. Passing tests say the code is correct; these numbers say
whether the modification is worth having.

    Mod A  what fraction of forecasts skip the heavy loop
    Mod B  how connected the 15 metros actually are
    Mod C  how often the fused forecast breaks a physical bound
    Mod D  how many LLM calls batching removes

Run:  python evaluate_nexus_mods.py
"""
import sys
import json

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from facts_mas.nodes.batched_reasoning import batching_savings
from facts_mas.nodes.boundary_node import boundary_report, enforce_boundaries
from facts_mas.nodes.gate_node import evaluate_gate, gate_savings_report
from facts_mas.nodes.spatial_node import build_spatial_graph, compute_spillover_signal
from facts_mas.run_backtest import FOLDS, HORIZONS, build_agent_runners
from facts_mas.state.boundary import BoundaryConfig
from facts_mas.state.gate import GateConfig
from facts_mas.state.ontology import GateTriggerType

WEEKLY = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
MSAS = sorted(WEEKLY["msa"].unique().tolist())
FOLD = FOLDS[5]
OUT = {}


def hr(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


# ═══════════════════════════════════════════════════════════════════════════
# Mod B — spatial spillover graph
# ═══════════════════════════════════════════════════════════════════════════

hr("MOD B — Cross-Regional Spatial Spillover")
print("Building the Granger graph over all 15 metros...")
GRAPH = build_spatial_graph(WEEKLY, MSAS, as_of=FOLD.train_end)

sig = [e for e in GRAPH.edges if e.is_significant]
print(f"  ordered pairs tested : {len(GRAPH.edges)}")
print(f"  significant edges    : {len(sig)}  ({len(sig)/len(GRAPH.edges):.1%} density)")
print(f"  mean optimal lag     : {np.mean([e.optimal_lag_weeks for e in sig]):.1f} weeks"
      if sig else "  no significant edges")

print("\n  strongest directed links:")
for e in sorted(sig, key=lambda x: x.p_value)[:8]:
    print(f"    {e.source_msa_id:>14} -> {e.target_msa_id:<14} "
          f"lag {e.optimal_lag_weeks:>2}wk   p={e.p_value:.2e}")

indeg = {m: len(GRAPH.neighbors_of(m)) for m in MSAS}
print("\n  most-influenced metros (in-degree):")
for m, d in sorted(indeg.items(), key=lambda kv: -kv[1])[:5]:
    print(f"    {m:<16} {d} significant inbound neighbours")

isolated = [m for m, d in indeg.items() if d == 0]
print(f"\n  metros with no significant neighbours: {isolated or 'none'}")

OUT["mod_b"] = {
    "pairs_tested": len(GRAPH.edges),
    "significant_edges": len(sig),
    "density": len(sig) / len(GRAPH.edges),
    "mean_lag_weeks": float(np.mean([e.optimal_lag_weeks for e in sig])) if sig else None,
    "isolated_msas": isolated,
    "top_edges": [{"source": e.source_msa_id, "target": e.target_msa_id,
                   "lag": e.optimal_lag_weeks, "p": e.p_value}
                  for e in sorted(sig, key=lambda x: x.p_value)[:8]],
}


# ═══════════════════════════════════════════════════════════════════════════
# Mod A — early-exit gate
# ═══════════════════════════════════════════════════════════════════════════

hr("MOD A — Smart Early-Exit Gate")
print(f"Evaluating over fold {FOLD.fold}'s test window "
      f"({FOLD.test_start} .. {FOLD.test_end}), all 15 metros.")
print("Baseline: ETS (TimesFM substitute). Event triggers from FEMA declarations.\n")

# FEMA declaration lookup -> natural_disaster trigger
from facts_mas.agents import event_agent
ev = pd.read_csv(event_agent.EVENT_WEEKLY_PATH, parse_dates=["week"])
ACTIVE = set()
for (msa, week), grp in ev.groupby(["msa", "week"], sort=False):
    b = grp.iloc[0]["incident_begin_date"]
    if pd.notna(b) and (week - pd.Timestamp(b)).days / 7.0 <= event_agent.DECLARATION_ACTIVE_WINDOW_WEEKS:
        ACTIVE.add((msa, week.date()))

CFG = GateConfig()
decisions = []
for msa in MSAS:
    md = WEEKLY[WEEKLY["msa"] == msa].sort_values("date")
    origins = md[(md["date"] >= pd.Timestamp(FOLD.test_start))
                 & (md["date"] <= pd.Timestamp(FOLD.test_end))]["date"]
    spill = compute_spillover_signal(WEEKLY, GRAPH, msa, FOLD.test_end)
    for ots in origins:
        origin = ots.date()
        trig = GateTriggerType.NATURAL_DISASTER if (msa, origin) in ACTIVE else None
        try:
            decisions.append(evaluate_gate(WEEKLY, msa, origin, 13, config=CFG,
                                           event_trigger=trig, spillover=spill))
        except Exception as exc:
            print(f"    gate failed at {msa} {origin}: {exc}")
    print(f"  {msa:<16} done ({len(decisions)} decisions)", flush=True)

rep = gate_savings_report(decisions)
print(f"\n  forecasts evaluated  : {rep['n_forecasts']}")
print(f"  heavy loop activated : {rep['n_activated']}  ({rep['activation_rate']:.1%})")
print(f"  skipped (cheap path) : {rep['heavy_loop_calls_saved']}  ({rep['skip_rate']:.1%})")
print("\n  trigger breakdown:")
for k, v in sorted(rep["by_trigger"].items(), key=lambda kv: -kv[1]):
    print(f"    {k:<22} {v:>4}  ({v/rep['n_forecasts']:.1%})")
OUT["mod_a"] = rep


# ═══════════════════════════════════════════════════════════════════════════
# Mod C — boundary constraints on real fused forecasts
# ═══════════════════════════════════════════════════════════════════════════

hr("MOD C — Statistical Boundary Constraints")
print("Checking real 5-agent fused forecasts against per-metro historical bounds.\n")

runners = build_agent_runners()
outcomes, raw_agent_outcomes = [], []
EQUAL_W = {n: 1.0 / len(runners) for n in runners}

for msa in MSAS:
    md = WEEKLY[WEEKLY["msa"] == msa].sort_values("date")
    hist = md[md["date"] <= pd.Timestamp(FOLD.train_end)]["inventory_count"].tolist()
    cfg = BoundaryConfig.from_historical_series(msa, hist)

    origins = md[(md["date"] >= pd.Timestamp(FOLD.test_start))
                 & (md["date"] <= pd.Timestamp(FOLD.test_end))]["date"].tolist()[::4]

    for ots in origins:
        origin = ots.date()
        past = md[md["date"] <= ots]
        if past.empty:
            continue
        last = float(past["inventory_count"].iloc[-1])
        baseline = [last] * 13

        vals = {}
        for name, run in runners.items():
            try:
                vals[name] = np.array(run(WEEKLY, msa, origin, 13).values, dtype=np.float64)
            except Exception:
                pass
        if not vals:
            continue

        fused = np.mean([vals[n] for n in vals], axis=0)
        outcomes.append(enforce_boundaries([float(v) for v in fused], last, cfg, baseline))

        # Also check each agent's RAW output. The fused number is an average
        # of five, so extremes cancel; the individual agents are where a
        # boundary violation would actually originate.
        for name, arr in vals.items():
            raw_agent_outcomes.append(
                (name, enforce_boundaries([float(v) for v in arr], last, cfg, baseline))
            )
    print(f"  {msa:<16} checked", flush=True)

frep = boundary_report(outcomes)
print(f"\n  FUSED forecasts checked : {frep['n_forecasts']}")
print(f"    clean                 : {frep['clean']}")
print(f"    corrected             : {frep['corrected']}")
print(f"    fell back to baseline : {frep['fell_back_to_baseline']}")
print(f"    violation rate        : {frep['violation_rate']:.1%}")
print(f"    violations by type    : {frep['violations_by_type'] or 'none'}")

print("\n  per-agent RAW output (before fusion averages extremes away):")
for name in sorted(runners):
    sub = [o for n, o in raw_agent_outcomes if n == name]
    if sub:
        r = boundary_report(sub)
        print(f"    {name:<14} violation rate {r['violation_rate']:>6.1%}  "
              f"types {r['violations_by_type'] or '{}'}")

OUT["mod_c"] = {
    "fused": frep,
    "per_agent": {n: boundary_report([o for m, o in raw_agent_outcomes if m == n])
                  for n in sorted(runners)},
}


# ═══════════════════════════════════════════════════════════════════════════
# Mod D — batching savings
# ═══════════════════════════════════════════════════════════════════════════

hr("MOD D — Batched Micro-Reasoning")
n_forecasts = len(MSAS) * len(FOLDS) * 29   # every origin, all folds
print(f"Full backtest scale: {len(MSAS)} metros x {len(FOLDS)} folds x 29 origins "
      f"= {n_forecasts} forecasts per horizon\n")
print(f"  {'horizon':>8} {'sequential':>12} {'batched':>10} {'saved':>10} {'reduction':>11}")
mod_d = {}
for h in HORIZONS:
    s = batching_savings(h, n_forecasts)
    mod_d[h] = s
    print(f"  {h:>6}wk {s['sequential_calls']:>12,} {s['batched_calls']:>10,} "
          f"{s['calls_saved']:>10,} {s['reduction']:>10.1%}")

s26 = batching_savings(26, n_forecasts)
print(f"\n  At the paper's own 26-week horizon: {s26['sequential_calls']:,} calls "
      f"-> {s26['batched_calls']:,}  ({s26['reduction']:.1%} reduction)")
lossy = batching_savings(13, n_forecasts, batch_failure_rate=0.10)
print(f"  With a 10% batch failure rate at 13wk: {lossy['batched_calls']:,} calls "
      f"({lossy['reduction']:.1%} reduction, failures charged honestly)")
OUT["mod_d"] = {"by_horizon": mod_d, "h26": s26, "h13_10pct_failure": lossy}


# ═══════════════════════════════════════════════════════════════════════════

with open("nexus_mods_results.json", "w", encoding="utf8") as fh:
    json.dump(OUT, fh, indent=2, default=str)
hr("done — written to nexus_mods_results.json")
