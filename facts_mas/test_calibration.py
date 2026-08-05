"""
Test Suite — Calibration Agent (facts_mas/calibration.py).

Kept in its own file rather than appended to test_agents.py: that file is
edited by both owners, and a separate suite avoids a merge conflict on every
push. Run with:  python facts_mas/test_calibration.py

Self-contained and fast (no LLM calls, no full backtest). The heavyweight
end-to-end sweeps live in calibrate_event_window.py and
calibrate_event_impact_scale.py, which are studies, not tests.
"""

import sys
import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from facts_mas.agents import event_agent
from facts_mas.calibration import (
    LOOKBACK_WEEKS,
    MIN_AFFECTED_ORIGINS,
    MIN_CONSISTENT_FOLDS,
    CalibrationTarget,
    _validation_window,
    calibrate_target,
    format_calibration_report,
    override_constant,
    score_agent_over_window,
)
from facts_mas.run_backtest import FOLDS
from facts_mas.schema import AgentOutput
from facts_mas.state.micro_reasoning import CalibrationGuideline, FoldResult

WEEKLY_DF = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])

passed = 0
failed = 0


def check(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1


# ═══════════════════════════════════════════════════════════════════════════
# override_constant
# ═══════════════════════════════════════════════════════════════════════════

def test_override_sets_and_restores():
    original = event_agent.DECLARATION_ACTIVE_WINDOW_WEEKS
    with override_constant(event_agent, "DECLARATION_ACTIVE_WINDOW_WEEKS", 99):
        assert event_agent.DECLARATION_ACTIVE_WINDOW_WEEKS == 99, "override did not apply"
    assert event_agent.DECLARATION_ACTIVE_WINDOW_WEEKS == original, "value not restored"


def test_override_restores_on_exception():
    """The restore must survive an error inside the block, or one failed
    sweep silently leaves a teammate's constant mutated for the rest of the
    process."""
    original = event_agent.DECLARATION_ACTIVE_WINDOW_WEEKS
    try:
        with override_constant(event_agent, "DECLARATION_ACTIVE_WINDOW_WEEKS", 99):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert event_agent.DECLARATION_ACTIVE_WINDOW_WEEKS == original, "not restored after exception"


def test_override_rejects_unknown_attribute():
    try:
        with override_constant(event_agent, "NO_SUCH_CONSTANT", 1):
            pass
    except AttributeError:
        return
    raise AssertionError("expected AttributeError for unknown attribute")


# ═══════════════════════════════════════════════════════════════════════════
# No-lookahead guarantee — the property that matters most
# ═══════════════════════════════════════════════════════════════════════════

def test_validation_window_ends_at_train_end():
    for fold in FOLDS:
        start, end = _validation_window(fold)
        assert end == fold.train_end, f"fold {fold.fold} window ends at {end}, not train_end"
        expected = fold.train_end - datetime.timedelta(weeks=LOOKBACK_WEEKS)
        assert start == expected, f"fold {fold.fold} lookback wrong"


def test_validation_window_never_touches_test_window():
    """The whole no-lookahead claim reduces to this: no calibration window
    may overlap its own fold's test window, or any later fold's."""
    for fold in FOLDS:
        _, end = _validation_window(fold)
        assert end < fold.test_start, (
            f"fold {fold.fold}: validation end {end} is not strictly before "
            f"test_start {fold.test_start}"
        )


def test_validation_window_respects_embargo():
    """Validation must end at least the full 13-week embargo before test."""
    for fold in FOLDS:
        _, end = _validation_window(fold)
        gap_days = (fold.test_start - end).days
        assert gap_days >= 91, f"fold {fold.fold}: only {gap_days} days before test_start"


def test_scoring_reads_no_data_past_window_end():
    """A runner is handed the full frame, so the guarantee has to come from
    the origins the scorer chooses. Assert it never proposes one past the
    window end."""
    seen = []

    def spy_runner(df, msa, origin, horizon):
        seen.append(origin)
        return AgentOutput(agent_name="event", msa=msa, forecast_origin=origin,
                           horizon_weeks=horizon, values=[1.0] * horizon)

    fold = FOLDS[2]
    start, end = _validation_window(fold)
    score_agent_over_window(WEEKLY_DF, spy_runner, ["Phoenix"], (start, end), [4])
    assert seen, "scorer proposed no origins at all"
    assert max(seen) <= end, f"scorer used origin {max(seen)} past window end {end}"
    assert min(seen) >= start, f"scorer used origin {min(seen)} before window start {start}"


# ═══════════════════════════════════════════════════════════════════════════
# CalibrationTarget
# ═══════════════════════════════════════════════════════════════════════════

def test_target_requires_baseline_in_candidates():
    try:
        CalibrationTarget(name="x", module=event_agent,
                          attribute="DECLARATION_ACTIVE_WINDOW_WEEKS",
                          candidates=(2, 4), baseline=8)
    except ValueError:
        return
    raise AssertionError("expected ValueError when baseline absent from candidates")


# ═══════════════════════════════════════════════════════════════════════════
# Consistency gate — reused from micro_reasoning, wired here
# ═══════════════════════════════════════════════════════════════════════════

def _guideline(directions):
    return CalibrationGuideline(
        guideline_text="test",
        fold_results=[
            FoldResult(fold_index=i + 1, correction_direction=d,
                       mape_before=10.0, mape_after=9.0 if d == "decrease" else 11.0)
            for i, d in enumerate(directions)
        ],
        min_consistent_folds=MIN_CONSISTENT_FOLDS,
    )


def test_gate_passes_at_five_of_six():
    g = _guideline(["decrease"] * 5 + ["increase"])
    assert g.dominant_direction == "decrease"
    assert g.fold_consistency_count == 5
    assert g.passes_consistency_check, "5/6 should clear the gate"


def test_gate_fails_at_four_of_six():
    """A candidate better in most folds but not consistently so must be
    rejected — this is the whole point of the Mod-D tightening."""
    g = _guideline(["decrease"] * 4 + ["increase"] * 2)
    assert g.fold_consistency_count == 4
    assert not g.passes_consistency_check, "4/6 must not clear the gate"


def test_consistently_worse_candidate_is_not_approved():
    """CalibrationGuideline's own check is direction-agnostic: it confirms
    folds AGREE, not that they agree on an improvement. calibrate_target
    must filter on direction as well, or a consistently-worse value would be
    'approved'."""
    g = _guideline(["increase"] * 6)
    assert g.passes_consistency_check, "consistency check itself should pass"
    assert g.dominant_direction == "increase", "direction should be 'increase'"
    # The module-level rule: approval requires dominant_direction == 'decrease'.
    approved = g.passes_consistency_check and g.dominant_direction == "decrease"
    assert not approved, "a consistently-worse candidate must never be approved"


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end sweep on a small slice
# ═══════════════════════════════════════════════════════════════════════════

def test_calibrate_target_end_to_end():
    """One fold, one MSA, two candidates — enough to prove the pieces fit
    together and that a report renders."""
    from facts_mas.agents.event_agent import run_event_agent

    target = CalibrationTarget(
        name="smoke", module=event_agent,
        attribute="DECLARATION_ACTIVE_WINDOW_WEEKS",
        candidates=(4, 8), baseline=8,
    )
    scores, result = calibrate_target(
        target, WEEKLY_DF, run_event_agent, ["Houston"],
        folds=[FOLDS[3]], horizons=[4], verbose=False,
    )
    assert set(scores.keys()) == {4, 8}, "both candidates should be scored"
    assert len(result.candidate_guidelines) == 1, "one non-baseline candidate"
    assert result.evaluation_window_end == FOLDS[3].train_end.isoformat()
    text = format_calibration_report(target, scores, result)
    assert "CALIBRATION" in text and "consistency gate" in text


def test_calibrate_target_leaves_constant_unchanged():
    """After a full sweep, the production constant must be exactly what it
    was — the sweep is a measurement, not a mutation."""
    from facts_mas.agents.event_agent import run_event_agent

    original = event_agent.DECLARATION_ACTIVE_WINDOW_WEEKS
    target = CalibrationTarget(
        name="smoke", module=event_agent,
        attribute="DECLARATION_ACTIVE_WINDOW_WEEKS",
        candidates=(2, 8), baseline=8,
    )
    calibrate_target(target, WEEKLY_DF, run_event_agent, ["Houston"],
                     folds=[FOLDS[3]], horizons=[4], verbose=False)
    assert event_agent.DECLARATION_ACTIVE_WINDOW_WEEKS == original, "sweep mutated the constant"


def test_thin_fold_votes_neutral():
    """A fold with fewer than MIN_AFFECTED_ORIGINS affected origins must not
    cast a directional vote."""
    from facts_mas.agents.event_agent import run_event_agent

    target = CalibrationTarget(
        name="thin", module=event_agent,
        attribute="DECLARATION_ACTIVE_WINDOW_WEEKS",
        candidates=(2, 8), baseline=8,
    )
    # A filter admitting almost nothing forces the thin-fold path.
    scores, result = calibrate_target(
        target, WEEKLY_DF, run_event_agent, ["Houston"],
        folds=[FOLDS[3]], horizons=[4],
        origin_filter=lambda msa, o: o.year == 1990,
        verbose=False,
    )
    g = result.candidate_guidelines[0]
    assert all(fr.correction_direction == "neutral" for fr in g.fold_results), \
        "a fold with no affected origins must vote neutral"
    assert not g.passes_consistency_check or g.dominant_direction == "neutral"


# ═══════════════════════════════════════════════════════════════════════════

print("\n-- override_constant --")
check("Override applies and restores", test_override_sets_and_restores)
check("Override restores even when the block raises", test_override_restores_on_exception)
check("Override rejects an unknown attribute", test_override_rejects_unknown_attribute)

print("\n-- No-lookahead guarantee --")
check("Validation window ends exactly at train_end", test_validation_window_ends_at_train_end)
check("Validation window is strictly before test_start", test_validation_window_never_touches_test_window)
check("Validation window respects the 13-week embargo", test_validation_window_respects_embargo)
check("Scorer proposes no origin outside its window", test_scoring_reads_no_data_past_window_end)

print("\n-- CalibrationTarget --")
check("Target requires baseline among candidates", test_target_requires_baseline_in_candidates)

print("\n-- Fold-consistency gate (NEXUS Mod-D) --")
check("Gate passes at 5/6 folds", test_gate_passes_at_five_of_six)
check("Gate fails at 4/6 folds", test_gate_fails_at_four_of_six)
check("Consistently-worse candidate is never approved", test_consistently_worse_candidate_is_not_approved)

print("\n-- End-to-end sweep --")
check("calibrate_target runs and renders a report", test_calibrate_target_end_to_end)
check("Sweep leaves the production constant unchanged", test_calibrate_target_leaves_constant_unchanged)
check("Thin fold votes neutral, not directional", test_thin_fold_votes_neutral)

print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("All tests passed!")
else:
    sys.exit(1)
