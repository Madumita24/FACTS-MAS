"""
Calibration study: the Event Agent's DECLARATION_ACTIVE_WINDOW_WEEKS.

facts_mas/agents/event_agent.py sets this constant to 8 and says so plainly:

    "8 weeks is chosen to roughly match the duration of short-lived disasters
     that scored WELL in that same diagnostic (e.g. Tropical Storm Imelda,
     6 weeks) -- a reasonable-but-arbitrary starting assumption pending real
     calibration data."

This script produces that calibration data. It sweeps candidate window
lengths through facts_mas/calibration.py, which scores each one on every
fold's 104-week TRAINING validation window (never the test window) and
gates the winner on the NEXUS Mod-D fold-consistency requirement.

Run:  python calibrate_event_window.py
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
    CalibrationTarget,
    calibrate_target,
    evaluate_on_test_window,
    format_calibration_report,
)
from facts_mas.run_backtest import FOLDS

# Candidate grid. Spans "very short-lived shock" (2 weeks) through "a full
# quarter of disruption" (26), bracketing the current 8 on both sides so the
# sweep can move the constant either direction rather than only confirm it.
CANDIDATES = (2, 4, 6, 8, 12, 16, 26)
BASELINE = 8

WEEKLY = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
MSAS = sorted(WEEKLY["msa"].unique().tolist())


# ═══════════════════════════════════════════════════════════════════════════
# Restrict scoring to origins the constant can actually change
# ═══════════════════════════════════════════════════════════════════════════

def build_declaration_age_index() -> dict:
    """
    (msa, week) -> age of the active declaration in weeks.

    Mirrors run_event_agent's own lookup exactly (first row on a tie, age
    measured from incident_begin_date) so the filter and the agent agree on
    which origins have an active declaration and how old it is.
    """
    events = pd.read_csv(event_agent.EVENT_WEEKLY_PATH, parse_dates=["week"])
    index = {}
    for (msa, week), grp in events.groupby(["msa", "week"], sort=False):
        row = grp.iloc[0]
        begin = row["incident_begin_date"]
        if pd.isna(begin):
            continue
        age = (week - pd.Timestamp(begin)).days / 7.0
        index[(msa, week.date())] = age
    return index


AGE_INDEX = build_declaration_age_index()
LO, HI = min(CANDIDATES), max(CANDIDATES)


def discriminating_origin(msa: str, origin) -> bool:
    """
    True only where the candidate windows disagree.

    run_event_agent applies an impact when age <= window and suppresses it
    when age > window. So all candidates behave identically unless
    LO < age <= HI -- outside that band every candidate produces a
    byte-identical forecast and the origin carries zero information about
    which window is better. Scoring those origins anyway would add the same
    constant to every candidate's MAPE, shrinking the visible gap between
    candidates without changing their order: it makes a real difference look
    small rather than making a small difference look real, but it is still
    noise, so it is excluded.
    """
    age = AGE_INDEX.get((msa, origin))
    return age is not None and LO < age <= HI


TARGET = CalibrationTarget(
    name="Event Agent declaration decay window",
    module=event_agent,
    attribute="DECLARATION_ACTIVE_WINDOW_WEEKS",
    candidates=CANDIDATES,
    baseline=BASELINE,
    description=(
        "How many weeks after incident_begin_date a FEMA declaration keeps "
        "moving the forecast. FEMA's own 'open' status tracks funding "
        "availability, not market disruption (COVID stayed open 172 weeks), "
        "so the agent needs its own expiry."
    ),
)


def main():
    n_disc = sum(1 for _ in AGE_INDEX.values())
    in_band = sum(1 for a in AGE_INDEX.values() if LO < a <= HI)
    print("=" * 72)
    print("CALIBRATING: event_agent.DECLARATION_ACTIVE_WINDOW_WEEKS")
    print("=" * 72)
    print(f"declaration-active (msa, week) cells: {n_disc}")
    print(f"  of those, in the {LO}<age<={HI} band where candidates disagree: {in_band}")
    print(f"candidates: {CANDIDATES}   current production value: {BASELINE}")
    print()
    print("Scoring each candidate on every fold's 104-week TRAINING window")
    print("(ends at train_end -- test windows are never read here).")
    print()

    scores, result = calibrate_target(
        TARGET, WEEKLY, run_event_agent, MSAS,
        origin_filter=discriminating_origin,
        verbose=True,
    )

    print()
    print("=" * 72)
    print(format_calibration_report(TARGET, scores, result))
    print("=" * 72)

    # ── Out-of-sample check, AFTER the choice is made ────────────────────
    ranked = sorted(scores.items(), key=lambda kv: kv[1].mean_mape())
    best_value = ranked[0][0]

    print()
    print("Held-out check (reporting only -- played no part in the choice above):")
    print(f"  test-window MAPE per fold, current value {BASELINE} vs best-scoring {best_value}")

    base_test = evaluate_on_test_window(
        TARGET, BASELINE, WEEKLY, run_event_agent, MSAS,
        origin_filter=discriminating_origin,
    )
    best_test = evaluate_on_test_window(
        TARGET, best_value, WEEKLY, run_event_agent, MSAS,
        origin_filter=discriminating_origin,
    )

    print(f"    {'fold':>6} {'window=' + str(BASELINE):>14} {'window=' + str(best_value):>14} {'delta':>9}")
    deltas = []
    for fold in FOLDS:
        b = base_test.get(fold.fold, float("nan"))
        c = best_test.get(fold.fold, float("nan"))
        d = c - b
        if np.isfinite(d):
            deltas.append(d)
        fmt = lambda x: f"{x:14.3f}" if np.isfinite(x) else f"{'n/a':>14}"
        dfmt = f"{d:+9.3f}" if np.isfinite(d) else f"{'n/a':>9}"
        print(f"    {fold.fold:>6} {fmt(b)} {fmt(c)} {dfmt}")

    if deltas:
        improved = sum(1 for d in deltas if d < 0)
        print(f"    mean test-window delta: {np.mean(deltas):+.3f} MAPE "
              f"({improved}/{len(deltas)} folds improved)")

    print()
    if result.approved_guidelines:
        print("RECOMMENDATION")
        for g in result.approved_guidelines:
            print(f"  - {g.guideline_text}")
            print(f"    consistent in {g.fold_consistency_count}/{len(FOLDS)} folds")
        print()
        print("  Apply by editing DECLARATION_ACTIVE_WINDOW_WEEKS in")
        print("  facts_mas/agents/event_agent.py -- that file is Madumita's under")
        print("  the one-file-one-owner rule, so this is a recommendation to her,")
        print("  not a change made here.")
    else:
        print("RECOMMENDATION: keep the current value.")
        print("  No candidate improved MAPE consistently enough across folds to")
        print("  clear the Mod-D gate. The 8-week window stands -- now as a value")
        print("  that survived a sweep, not merely an untested assumption.")


if __name__ == "__main__":
    main()
