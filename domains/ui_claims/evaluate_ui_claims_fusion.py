"""
UI claims: skill-weighted adaptive fusion, not an equal-weight blend.

WHY THIS EXISTS
The pilot (evaluate_ui_claims_pilot.py) reported a naive (AR + Macro) / 2
blend and found it strictly worse than AR alone on 15 of 15 states at every
horizon. That result is correct, but it does not test the FACTS-MAS fusion
mechanism -- it tests the thing that mechanism was built to avoid. Averaging
a useful forecaster with a null one hurts, which is exactly the failure mode
that motivated replacing inverse-MAPE weighting with skill-weighted softmax
in the housing domain.

The paper stakes a transfer hypothesis on this ("H3: after reconfiguration,
adaptive fusion should remain competitive with the new domain's simple
reference and strong standalone models"). Until skill-weighted fusion is
actually run on a second domain, H3 is untested.

WHAT THIS DOES
Uses the real housing fusion primitives -- skill_score_median and
compute_weights_softmax from facts_mas.fusion_baseline, imported not
reimplemented -- on the UI claims agents. Weights are estimated on a
validation window ending at train_end, strictly before the embargo, then
held fixed across the test window.

The UI agents return plain dicts rather than AgentOutput, so thin adapters
wrap them into the shared contract. That wrapping is itself part of the
transfer claim: if a new domain's agents can be made to satisfy the same
interface, the downstream machinery needs no changes.

Run:  python domains/ui_claims/evaluate_ui_claims_fusion.py
"""
import datetime
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from domains.ui_claims.agents.ar_agent_ui import run_ar_agent_ui
from domains.ui_claims.agents.macro_agent_ui import run_macro_agent_ui
from facts_mas.fusion_baseline import compute_weights_softmax, skill_score_median
from facts_mas.run_backtest import compute_mape, compute_rmse
from facts_mas.schema import AgentOutput

DATA = "domains/ui_claims/aligned_weekly_ui_claims.csv"
TRAIN_END = datetime.date(2023, 6, 24)
TEST_START = datetime.date(2023, 9, 23)
TEST_END = datetime.date(2026, 8, 22)
VAL_WEEKS = 104          # same lookback the housing fusion uses
HORIZONS = [4, 8, 13]
TEMPERATURE = 0.05       # same as housing

DF = pd.read_csv(DATA, parse_dates=["date"])
STATES = sorted(DF["state"].unique().tolist())


# ── Adapters: dict -> AgentOutput ─────────────────────────────────────────

def _adapt(runner, name):
    """Wrap a UI agent so it satisfies the shared AgentOutput contract.

    Non-negativity is enforced here rather than trusted: claims counts have
    the same physical floor inventory does, and the schema rejects negatives,
    so an ETS extrapolation that dips below zero would raise instead of
    silently producing an invalid forecast.
    """
    def _runner(df, unit, origin, horizon):
        out = runner(df, unit, origin, horizon)
        vals = [max(0.0, float(v)) for v in out["values"]]
        return AgentOutput(agent_name=name, msa=unit, forecast_origin=origin,
                           horizon_weeks=horizon, values=vals)
    return _runner


RUNNERS = {"ar": _adapt(run_ar_agent_ui, "ar"),
           "macro": _adapt(run_macro_agent_ui, "macro")}


# ── Weight estimation on the validation window only ───────────────────────

def estimate_weights(horizon):
    """
    Skill-weighted weights from the pre-test validation window.

    Identical procedure to the housing domain: per-origin skill against the
    persistence reference, median-aggregated, softmax-converted. The median
    matters for the same reason it did there -- when the reference error is
    near zero at an origin, the skill ratio explodes and a mean is dominated
    by a handful of points.
    """
    val_start = TRAIN_END - datetime.timedelta(weeks=VAL_WEEKS)
    points = {k: [] for k in RUNNERS}

    for state in STATES:
        sd = DF[DF["state"] == state].sort_values("date")
        origins = sd[(sd["date"] >= pd.Timestamp(val_start))
                     & (sd["date"] <= pd.Timestamp(TRAIN_END))]["date"]
        for ots in origins:
            fut = sd[sd["date"] > ots].head(horizon)
            if len(fut) < horizon:
                continue
            act = fut["claims_count"].values.astype(np.float64)
            past = sd[sd["date"] <= ots]
            if past.empty:
                continue
            last = float(past["claims_count"].iloc[-1])
            naive_e = compute_mape(act, np.full(horizon, last))
            if not np.isfinite(naive_e) or naive_e <= 0:
                continue
            for name, run in RUNNERS.items():
                try:
                    e = compute_mape(act, np.array(run(DF, state, ots.date(), horizon).values))
                except Exception:
                    continue
                if np.isfinite(e):
                    points[name].append(1.0 - e / naive_e)

    skills = {k: skill_score_median(v) if v else -1.0 for k, v in points.items()}
    return compute_weights_softmax(skills, temperature=TEMPERATURE), skills


# ── Test-window evaluation ────────────────────────────────────────────────

print("=" * 76)
print("UI CLAIMS: skill-weighted adaptive fusion vs the pilot's equal-weight blend")
print("=" * 76)
print(f"{len(STATES)} states | validation {TRAIN_END - datetime.timedelta(weeks=VAL_WEEKS)}"
      f" .. {TRAIN_END} | test {TEST_START} .. {TEST_END}\n")

rows = []
for horizon in HORIZONS:
    weights, skills = estimate_weights(horizon)
    print(f"h={horizon:>2}wk  validation skill: "
          + "  ".join(f"{k}={v:+.3f}" for k, v in skills.items())
          + "   ->  weights: " + "  ".join(f"{k}={v:.3f}" for k, v in weights.items()),
          flush=True)

    for state in STATES:
        sd = DF[DF["state"] == state].sort_values("date")
        origins = sd[(sd["date"] >= pd.Timestamp(TEST_START))
                     & (sd["date"] <= pd.Timestamp(TEST_END))]["date"]
        for ots in origins:
            fut = sd[sd["date"] > ots].head(horizon)
            if len(fut) < horizon:
                continue
            act = fut["claims_count"].values.astype(np.float64)
            past = sd[sd["date"] <= ots]
            if past.empty:
                continue
            last = float(past["claims_count"].iloc[-1])
            naive_v = np.full(horizon, last)

            vals = {}
            for name, run in RUNNERS.items():
                try:
                    vals[name] = np.array(run(DF, state, ots.date(), horizon).values)
                except Exception:
                    pass
            if len(vals) < len(RUNNERS):
                continue

            blend = (vals["ar"] + vals["macro"]) / 2.0            # pilot's rule
            tw = sum(weights[k] for k in vals)
            fused = np.maximum(sum(weights[k] * vals[k] for k in vals) / tw, 0.0)

            rows.append({
                "state": state, "horizon": horizon,
                "naive": compute_mape(act, naive_v),
                "ar": compute_mape(act, vals["ar"]),
                "macro": compute_mape(act, vals["macro"]),
                "blend": compute_mape(act, blend),
                "fused": compute_mape(act, fused),
                "naive_rmse": compute_rmse(act, naive_v),
                "fused_rmse": compute_rmse(act, fused),
                "ar_rmse": compute_rmse(act, vals["ar"]),
            })

df = pd.DataFrame(rows)
df.to_csv("domains/ui_claims/ui_claims_fusion_per_origin.csv", index=False)

per_state = df.groupby(["horizon", "state"]).mean(numeric_only=True).reset_index()
per_state.to_csv("domains/ui_claims/ui_claims_fusion_per_state.csv", index=False)

print("\n" + "=" * 76)
print("MEAN MAPE ACROSS 15 STATES (mean of per-state means)")
print("=" * 76)
print(f"{'h':>4} {'naive':>8} {'AR':>8} {'Macro':>8} {'blend':>8} {'FUSED':>9}")
for h in HORIZONS:
    s = per_state[per_state.horizon == h]
    print(f"{h:>4} {s.naive.mean():>8.3f} {s.ar.mean():>8.3f} {s.macro.mean():>8.3f} "
          f"{s.blend.mean():>8.3f} {s.fused.mean():>9.3f}")

print(f"\n{'h':>4} {'naive':>8} {'AR':>8} {'blend':>8} {'FUSED':>9}   (medians)")
for h in HORIZONS:
    s = per_state[per_state.horizon == h]
    print(f"{h:>4} {s.naive.median():>8.3f} {s.ar.median():>8.3f} "
          f"{s.blend.median():>8.3f} {s.fused.median():>9.3f}")

print("\n" + "=" * 76)
print("WIN COUNTS (of 15 states)")
print("=" * 76)
print(f"{'h':>4} {'AR>naive':>10} {'blend>naive':>13} {'FUSED>naive':>13} "
      f"{'blend>AR':>10} {'FUSED>=AR':>11}")
for h in HORIZONS:
    s = per_state[per_state.horizon == h]
    print(f"{h:>4} {int((s.ar < s.naive).sum()):>9}/15 {int((s.blend < s.naive).sum()):>12}/15 "
          f"{int((s.fused < s.naive).sum()):>12}/15 {int((s.blend < s.ar).sum()):>9}/15 "
          f"{int((s.fused <= s.ar + 1e-9).sum()):>10}/15")

print("\n" + "=" * 76)
print("VERDICT ON H3")
print("=" * 76)
for h in HORIZONS:
    s = per_state[per_state.horizon == h]
    f_vs_b = s.fused.mean() - s.blend.mean()
    f_vs_a = s.fused.mean() - s.ar.mean()
    f_vs_n = s.fused.mean() - s.naive.mean()
    print(f"  h={h:>2}: fused vs blend {f_vs_b:+.3f} | vs AR alone {f_vs_a:+.3f} "
          f"| vs naive {f_vs_n:+.3f}")
print("\n  (negative = fused is better. The blend column is the pilot's rule;")
print("   the point is whether skill weighting avoids the damage equal")
print("   weighting caused, not whether it beats a strong standalone agent.)")
print("\nwritten to ui_claims_fusion_per_state.csv and _per_origin.csv")
