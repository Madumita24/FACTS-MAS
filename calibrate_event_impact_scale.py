"""
Calibration study: the Event Agent's IMPACT_SCALE.

Follow-up to calibrate_event_window.py, which found that shortening
DECLARATION_ACTIVE_WINDOW_WEEKS improves MAPE monotonically all the way down
to zero -- i.e. the best decay window is "never apply an event impact at
all". That is not a result about the window. It says the magnitude being
applied is wrong, and the window sweep was only reducing how often the wrong
magnitude got used.

The magnitude is IMPACT_SCALE, which event_agent.py flags itself:

    "ARBITRARY, UNVALIDATED CHOICE -- there is no ground-truth data in this
     project connecting qualitative severity scores to actual pct inventory
     change, so 0.10 is a documented guess, not a fitted value."

and, in a TODO immediately below it, predicts its own failure mode:

    "a generic 'WILDFIRES' declaration scored strongly NEGATIVE ... actual
     decline was only ~-1.5% while the -7.5% adjustment overshot it roughly
     5x."

A 5x overshoot at scale 0.10 implies a fitted scale near 0.02. This script
tests that prediction directly, and asks the sharper question the window
sweep could not: is there ANY scale at which the event signal helps, or does
its useful content stop at direction?

NOTE ON METHOD -- why this study does not call run_event_agent().
IMPACT_SCALE is a function-local variable inside run_event_agent(), not a
module attribute, so calibration.override_constant() cannot reach it. Rather
than edit event_agent.py (owned by Madumita), this script reproduces the one
line that consumes it --

    factor = 1.0 + impact_magnitude * IMPACT_SCALE

-- against the same cached event_agent_weekly.csv the agent itself reads,
with the same lookup, gating and age rules. Verified identical to the real
agent at the production scale in _verify_matches_agent() below, which is run
on every invocation so this parity claim cannot quietly rot.

Recommendation that follows from this: promote IMPACT_SCALE to a module
constant so future calibration can sweep it without a reimplementation.

Run:  python calibrate_event_impact_scale.py
"""
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from facts_mas.agents import event_agent
from facts_mas.agents.event_agent import run_event_agent
from facts_mas.calibration import (
    MIN_CONSISTENT_FOLDS,
    NEUTRAL_BAND_MAPE,
    _validation_window,
)
from facts_mas.run_backtest import FOLDS, HORIZONS, compute_mape
from facts_mas.state.micro_reasoning import CalibrationGuideline, FoldResult

PRODUCTION_SCALE = 0.10
PRODUCTION_WINDOW = 8
# Spans "signal is worthless" (0.0) through the production guess (0.10).
# 0.02 is included because event_agent.py's own overshoot note predicts it.
SCALES = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)

WEEKLY = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
MSAS = sorted(WEEKLY["msa"].unique().tolist())


# ═══════════════════════════════════════════════════════════════════════════
# Event lookup -- mirrors run_event_agent's own rules exactly
# ═══════════════════════════════════════════════════════════════════════════

def build_event_index() -> dict:
    """(msa, week) -> (impact_magnitude, age_weeks), first row on a tie."""
    ev = pd.read_csv(event_agent.EVENT_WEEKLY_PATH, parse_dates=["week"])
    idx = {}
    for (msa, week), grp in ev.groupby(["msa", "week"], sort=False):
        row = grp.iloc[0]
        begin = row["incident_begin_date"]
        age = (week - pd.Timestamp(begin)).days / 7.0 if pd.notna(begin) else None
        idx[(msa, week.date())] = (float(row["impact_magnitude"]), age)
    return idx


EVENT_INDEX = build_event_index()


def event_factor(msa: str, snapped_week, scale: float, window: int) -> float:
    """The multiplicative factor run_event_agent would apply, at `scale`."""
    hit = EVENT_INDEX.get((msa, snapped_week))
    if hit is None:
        return 1.0
    impact, age = hit
    if age is not None and age > window:
        return 1.0
    return 1.0 + impact * scale


def _verify_matches_agent(n_checks: int = 40) -> None:
    """
    Assert this script's reimplementation equals run_event_agent at the
    production scale. Runs every invocation: a parity claim that is not
    re-checked is a parity claim that eventually becomes false.
    """
    checked = 0
    for msa in MSAS:
        md = WEEKLY[WEEKLY["msa"] == msa].sort_values("date")
        # Prefer origins with an active declaration -- those are the only
        # ones where the two implementations could possibly diverge.
        cands = [d for d in md["date"] if (msa, d.date()) in EVENT_INDEX]
        for origin_ts in cands[:4]:
            origin = origin_ts.date()
            past = md[md["date"] <= origin_ts]
            if past.empty:
                continue
            last_inv = float(past["inventory_count"].iloc[-1])
            snapped = past["date"].iloc[-1].date()
            mine = max(0.0, round(last_inv * event_factor(
                msa, snapped, PRODUCTION_SCALE, PRODUCTION_WINDOW), 2))
            theirs = run_event_agent(WEEKLY, msa, origin, 4).values[0]
            if abs(mine - theirs) > 0.011:
                raise AssertionError(
                    f"reimplementation diverges from run_event_agent at "
                    f"{msa} {origin}: {mine} vs {theirs}"
                )
            checked += 1
            if checked >= n_checks:
                print(f"parity check: {checked} origins match run_event_agent exactly")
                return
    print(f"parity check: {checked} origins match run_event_agent exactly")


# ═══════════════════════════════════════════════════════════════════════════
# Sweep
# ═══════════════════════════════════════════════════════════════════════════

def score(scale: float, window: int, start, end) -> tuple[float, int]:
    apes, n = [], 0
    for msa in MSAS:
        md = WEEKLY[WEEKLY["msa"] == msa].sort_values("date")
        origins = md[(md["date"] >= pd.Timestamp(start))
                     & (md["date"] <= pd.Timestamp(end))]["date"]
        for origin_ts in origins:
            key = (msa, origin_ts.date())
            if key not in EVENT_INDEX:
                continue  # scale is inert with no active declaration
            impact, age = EVENT_INDEX[key]
            if impact == 0.0 or (age is not None and age > window):
                continue  # inert regardless of scale
            past = md[md["date"] <= origin_ts]
            if past.empty:
                continue
            last_inv = float(past["inventory_count"].iloc[-1])
            factor = event_factor(msa, origin_ts.date(), scale, window)
            for h in HORIZONS:
                fut = md[md["date"] > origin_ts].head(h)
                if len(fut) < h:
                    continue
                actuals = fut["inventory_count"].values.astype(np.float64)
                fc = np.full(h, max(0.0, last_inv * factor))
                apes.append(compute_mape(actuals, fc))
                n += 1
    finite = [a for a in apes if np.isfinite(a)]
    return (float(np.mean(finite)) if finite else float("inf")), n


def main():
    print("=" * 72)
    print("CALIBRATING: event_agent IMPACT_SCALE  (window held at production 8)")
    print("=" * 72)
    _verify_matches_agent()
    print(f"candidates: {SCALES}   current production value: {PRODUCTION_SCALE}")
    print("scored on each fold's 104-week TRAINING window; test windows unread.")
    print()

    table = {}
    for fold in FOLDS:
        start, end = _validation_window(fold)
        row = {}
        for s in SCALES:
            m, n = score(s, PRODUCTION_WINDOW, start, end)
            row[s] = m
        table[fold.fold] = row
        cells = "  ".join(f"{s:g}:{row[s]:6.3f}" for s in SCALES)
        print(f"  fold {fold.fold}: {cells}")

    df = pd.DataFrame(table).T
    means = df.mean()
    print()
    print("mean validation MAPE by scale:")
    best = means.idxmin()
    for s in SCALES:
        tag = "  <- current" if s == PRODUCTION_SCALE else ""
        star = " *best*" if s == best else ""
        print(f"  IMPACT_SCALE={s:<6g}  MAPE={means[s]:7.3f}{star}{tag}")
    print()

    # ── Fold-consistency gate on the best non-production scale ───────────
    fold_results = []
    for fold in FOLDS:
        before = table[fold.fold][PRODUCTION_SCALE]
        after = table[fold.fold][best]
        if after < before - NEUTRAL_BAND_MAPE:
            d = "decrease"
        elif after > before + NEUTRAL_BAND_MAPE:
            d = "increase"
        else:
            d = "neutral"
        fold_results.append(FoldResult(
            fold_index=fold.fold, correction_direction=d,
            mape_before=float(before), mape_after=float(after),
        ))

    g = CalibrationGuideline(
        guideline_text=(
            f"Set IMPACT_SCALE = {best:g} (currently {PRODUCTION_SCALE:g}). "
            f"Mean validation MAPE {means[best]:.3f} vs {means[PRODUCTION_SCALE]:.3f}."
        ),
        fold_results=fold_results,
        min_consistent_folds=MIN_CONSISTENT_FOLDS,
    )
    print(f"fold-consistency gate (>= {MIN_CONSISTENT_FOLDS}/6 folds must improve):")
    print(f"  {g.guideline_text}")
    print(f"  dominant direction '{g.dominant_direction}' in "
          f"{g.fold_consistency_count}/6 folds -> "
          f"{'APPROVED' if g.passes_consistency_check and g.dominant_direction == 'decrease' else 'REJECTED'}")
    print()

    interior = best not in (0.0,)
    print("INTERPRETATION")
    if not interior:
        print("  The fitted scale is 0.0: no positive magnitude improves on applying")
        print("  no event adjustment at all. The Event Agent's severity score carries")
        print("  no usable MAGNITUDE information against weekly inventory -- consistent")
        print("  with event_agent.md's own finding that it resolves to a 4-tier")
        print("  classifier rather than a continuous scale.")
        print("  This does NOT say events are irrelevant. It says this agent should")
        print("  not be emitting a point forecast built from that score.")
    else:
        print(f"  Interior optimum at {best:g}, roughly {PRODUCTION_SCALE/best:.0f}x smaller than")
        print(f"  the production 0.10 -- matching the ~5x overshoot event_agent.py's")
        print("  own TODO predicted from the WILDFIRES case. The direction of the")
        print("  event signal is right; only its magnitude was miscalibrated.")


if __name__ == "__main__":
    main()
