"""
Calibration study: the Intrinsic Agent's mean-reversion cap (rho_max).

The paper states the gap plainly: "The Intrinsic Agent's mean-reversion cap
rho_max = 0.30 is still a modeling choice and should eventually pass through
the same calibration protocol used for event scale."

This is that pass. Same protocol as the Event Agent's IMPACT_SCALE sweep:
score each candidate on every fold's 104-week TRAINING window, never the
test window, and require the direction of improvement to hold in at least
5 of 6 folds individually before promoting a value.

rho_max is a function-local constant (TOTAL_REVERSION) inside
run_intrinsic_agent, so override_constant() cannot reach it. Rather than
edit the agent, this reproduces the two lines that consume it --

    reversion_frac = TOTAL_REVERSION * (step / horizon_weeks) ** 2
    projected      = last + reversion_frac * (structural - last)

-- against the same trained model the agent uses. A parity check against
the real agent at the production value runs first, so the reproduction
cannot silently drift.

Run:  python calibrate_rho_max.py
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from facts_mas.agents.intrinsic_agent import (
    INTRINSIC_FEATURES, run_intrinsic_agent, train_intrinsic_model,
)
from facts_mas.calibration import (
    MIN_CONSISTENT_FOLDS, NEUTRAL_BAND_MAPE, _validation_window,
)
from facts_mas.run_backtest import FOLDS, HORIZONS, compute_mape
from facts_mas.state.micro_reasoning import CalibrationGuideline, FoldResult

PRODUCTION = 0.30
CANDIDATES = (0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80)

WEEKLY = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
STATIC = pd.read_csv("intrinsic_static.csv")
MSAS = sorted(WEEKLY["msa"].unique().tolist())


def forecast(last, structural, horizon, rho):
    steps = np.arange(1, horizon + 1, dtype=np.float64)
    frac = rho * (steps / horizon) ** 2
    return np.maximum(last + frac * (structural - last), 0.0)


def _parity_check(model, n=25):
    """Assert the reproduction matches run_intrinsic_agent at rho=0.30."""
    checked = 0
    for msa in MSAS:
        md = WEEKLY[WEEKLY["msa"] == msa].sort_values("date")
        structural = float(model.predict(
            STATIC[STATIC["msa"] == msa][INTRINSIC_FEATURES].values)[0])
        for ots in md["date"].iloc[-40::7]:
            past = md[md["date"] <= ots]
            if past.empty:
                continue
            last = float(past["inventory_count"].iloc[-1])
            mine = forecast(last, structural, 4, PRODUCTION)
            theirs = np.array(run_intrinsic_agent(
                WEEKLY, STATIC, msa, ots.date(), 4, model=model).values)
            if np.abs(mine - theirs).max() > 0.02:
                raise AssertionError(
                    f"reproduction diverges at {msa} {ots.date()}: "
                    f"{mine[:2]} vs {theirs[:2]}")
            checked += 1
            if checked >= n:
                print(f"parity check: {checked} origins match run_intrinsic_agent")
                return
    print(f"parity check: {checked} origins match run_intrinsic_agent")


print("=" * 72)
print("CALIBRATING: Intrinsic Agent rho_max (mean-reversion cap)")
print("=" * 72)
print(f"candidates: {CANDIDATES}   production value: {PRODUCTION}")
print("scored on each fold's 104-week TRAINING window; test windows unread.\n")

table = {}
for fold in FOLDS:
    model = train_intrinsic_model(WEEKLY, STATIC, fold.train_end)
    if fold.fold == 1:
        _parity_check(model)
        print()
    start, end = _validation_window(fold)
    row = {r: [] for r in CANDIDATES}
    for msa in MSAS:
        md = WEEKLY[WEEKLY["msa"] == msa].sort_values("date")
        structural = float(model.predict(
            STATIC[STATIC["msa"] == msa][INTRINSIC_FEATURES].values)[0])
        origins = md[(md["date"] >= pd.Timestamp(start))
                     & (md["date"] <= pd.Timestamp(end))]["date"]
        for ots in origins:
            past = md[md["date"] <= ots]
            if past.empty:
                continue
            last = float(past["inventory_count"].iloc[-1])
            for h in HORIZONS:
                fut = md[md["date"] > ots].head(h)
                if len(fut) < h:
                    continue
                act = fut["inventory_count"].values.astype(np.float64)
                for r in CANDIDATES:
                    row[r].append(compute_mape(act, forecast(last, structural, h, r)))
    table[fold.fold] = {r: float(np.mean(row[r])) for r in CANDIDATES}
    print(f"  fold {fold.fold}: " + "  ".join(
        f"{r:g}:{table[fold.fold][r]:7.3f}" for r in CANDIDATES), flush=True)

means = {r: float(np.mean([table[f][r] for f in table])) for r in CANDIDATES}
best = min(means, key=means.get)

print("\nmean validation MAPE by rho_max:")
for r in CANDIDATES:
    tag = "  <- production" if r == PRODUCTION else ""
    star = " *best*" if r == best else ""
    print(f"  rho_max={r:<5g} MAPE={means[r]:8.3f}{star}{tag}")

fold_results = []
for fold in FOLDS:
    before, after = table[fold.fold][PRODUCTION], table[fold.fold][best]
    d = ("decrease" if after < before - NEUTRAL_BAND_MAPE
         else "increase" if after > before + NEUTRAL_BAND_MAPE else "neutral")
    fold_results.append(FoldResult(fold_index=fold.fold, correction_direction=d,
                                   mape_before=before, mape_after=after))

g = CalibrationGuideline(
    guideline_text=(f"Set rho_max = {best:g} (currently {PRODUCTION:g}). "
                    f"Mean validation MAPE {means[best]:.3f} vs {means[PRODUCTION]:.3f}."),
    fold_results=fold_results, min_consistent_folds=MIN_CONSISTENT_FOLDS)

approved = g.passes_consistency_check and g.dominant_direction == "decrease"
print(f"\nfold-consistency gate (>= {MIN_CONSISTENT_FOLDS}/6 must improve):")
print(f"  {g.guideline_text}")
print(f"  '{g.dominant_direction}' in {g.fold_consistency_count}/6 folds -> "
      f"{'APPROVED' if approved else 'REJECTED'}")

print("\nINTERPRETATION")
if best == 0.0:
    print("  Fitted cap is zero: structural mean reversion is unsupported. The")
    print("  Intrinsic agent would be better off holding the last value than")
    print("  pulling toward its regression-implied level at all.")
elif abs(best - PRODUCTION) < 1e-9:
    print("  The hand-picked 0.30 survives the sweep. It is now a validated")
    print("  value rather than an untested assumption, which is what the paper")
    print("  asked for.")
else:
    print(f"  Fitted cap is {best:g}, not the hand-picked {PRODUCTION:g}.")
    print(f"  Interior optimum, so reversion helps but the production value was")
    print(f"  {'too aggressive' if best < PRODUCTION else 'too timid'}.")

pd.DataFrame(table).T.to_csv("rho_max_calibration.csv")
print("\nwritten to rho_max_calibration.csv")
