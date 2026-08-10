"""
Which gating design does the data actually support?

Madumita's proposal: stop letting the Event Agent modify the forecast
numerically, and use event presence as a gate/confidence signal instead.
Three candidate mechanisms were floated:
    (a) binary weight bump toward some agent when an event is active
    (b) confidence-based weighting driven by the LLM's own confidence
    (c) downweight the other agents while an event is active

All three assume the same underlying premise: that SOME agent's reliability
changes when a disaster declaration is active. If no agent's skill shifts on
event weeks, none of the three can help and Event should simply be dropped.

This script tests that premise directly. For each agent it measures skill
vs. naive (skill = 1 - agent_mape/naive_mape, positive = beats naive) on
declaration-active origins vs. quiet origins, per horizon.

Scored on fold 6's 104-week TRAINING validation window, so this is a design
input and not a test-set result.
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from facts_mas.agents import event_agent
from facts_mas.calibration import _validation_window
from facts_mas.run_backtest import FOLDS, HORIZONS, build_agent_runners, compute_mape

WEEKLY = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
MSAS = sorted(WEEKLY["msa"].unique().tolist())
WINDOW = _validation_window(FOLDS[5])
ACTIVE_WEEKS = event_agent.DECLARATION_ACTIVE_WINDOW_WEEKS

ev = pd.read_csv(event_agent.EVENT_WEEKLY_PATH, parse_dates=["week"])
EV = {}
for (msa, week), grp in ev.groupby(["msa", "week"], sort=False):
    r = grp.iloc[0]
    b = r["incident_begin_date"]
    age = (week - pd.Timestamp(b)).days / 7.0 if pd.notna(b) else None
    EV[(msa, week.date())] = (age, float(r["confidence"]), float(r["impact_magnitude"]))


def event_state(msa, origin):
    """(is_active, confidence) using the production 8-week window."""
    hit = EV.get((msa, origin))
    if hit is None:
        return False, 0.0
    age, conf, _ = hit
    if age is None or age > ACTIVE_WEEKS:
        return False, 0.0
    return True, conf


runners = build_agent_runners()
rows = []
start, end = WINDOW
print(f"scoring fold 6 validation window {start} .. {end}, {len(MSAS)} MSAs")

for msa in MSAS:
    md = WEEKLY[WEEKLY["msa"] == msa].sort_values("date")
    origins = md[(md["date"] >= pd.Timestamp(start)) & (md["date"] <= pd.Timestamp(end))]["date"]
    for ots in origins:
        origin = ots.date()
        active, conf = event_state(msa, origin)
        past = md[md["date"] <= ots]
        if past.empty:
            continue
        last = float(past["inventory_count"].iloc[-1])
        for h in HORIZONS:
            fut = md[md["date"] > ots].head(h)
            if len(fut) < h:
                continue
            act = fut["inventory_count"].values.astype(np.float64)
            nm = compute_mape(act, np.full(h, last))
            if not np.isfinite(nm) or nm <= 0:
                continue
            rec = {"msa": msa, "origin": origin, "h": h, "active": active,
                   "conf": conf, "naive": nm}
            for name, run in runners.items():
                try:
                    out = run(WEEKLY, msa, origin, h)
                    rec[name] = 1.0 - (compute_mape(act, np.array(out.values)) / nm)
                except Exception:
                    rec[name] = np.nan
            rows.append(rec)
    print(f"  {msa} done ({len(rows)} rows)", flush=True)

df = pd.DataFrame(rows)
df.to_csv("event_gate_evidence.csv", index=False)
AG = list(runners.keys())

print("\n" + "=" * 78)
print("PREMISE TEST: does any agent's skill change when a declaration is active?")
print("=" * 78)
print("skill = 1 - agent_mape/naive_mape.  positive = beats naive.\n")

for h in HORIZONS:
    s = df[df.h == h]
    a, q = s[s.active], s[~s.active]
    print(f"h={h:>2}wk   (active n={len(a)}, quiet n={len(q)})")
    print(f"  {'agent':>12} {'quiet':>9} {'active':>9} {'shift':>9}")
    for ag in AG:
        if a[ag].notna().sum() == 0:
            continue
        qs, as_ = q[ag].mean(), a[ag].mean()
        print(f"  {ag:>12} {qs:>9.3f} {as_:>9.3f} {as_-qs:>+9.3f}")
    print()

print("=" * 78)
print("Does the LLM's confidence predict anything? (active origins only)")
act = df[df.active]
if len(act) > 10:
    for ag in AG:
        v = act[[ag, "conf"]].dropna()
        if len(v) > 10 and v["conf"].std() > 0:
            print(f"  corr(confidence, {ag} skill) = {v[ag].corr(v['conf']):+.3f}   n={len(v)}")
print("=" * 78)
