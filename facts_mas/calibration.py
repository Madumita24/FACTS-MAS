"""
Calibration Agent -- FACTS-MAS Phase 2 extension.

Learns constants from backtest error instead of hand-setting them.

Several forecasting constants in this project are, by their own authors'
admission, reasoned guesses rather than fitted values:

    event_agent.DECLARATION_ACTIVE_WINDOW_WEEKS = 8
        "a reasonable-but-arbitrary starting assumption pending real
         calibration data"
    event_agent.IMPACT_SCALE = 0.10
        "ARBITRARY, UNVALIDATED CHOICE -- 0.10 is a documented guess,
         not a fitted value"

This module supplies the "real calibration data" those comments are waiting
for. It sweeps candidate values for such a constant, measures the resulting
change in MAPE fold by fold, and only promotes a new value if the direction
of improvement is *consistent across folds* -- not merely better when all
folds are pooled together.

Reuses the NEXUS Mod-D calibration models already built and tested in
facts_mas/state/micro_reasoning.py (FoldResult, CalibrationGuideline,
TightenedCalibrationResult) rather than reimplementing them. Those models
already encode the fold-consistency gate this module needs; until now they
had no producer wired to real backtest output. This is that producer.

Two properties this module treats as non-negotiable:

1. NO LOOKAHEAD. Candidates are scored only on a validation window that
   ends at fold.train_end, using the same 104-week lookback the fusion
   layer uses (fusion_baseline.compute_fold_weights). A fold's test window
   is never read during calibration. `evaluate_on_test_window()` exists to
   report what the chosen value would have done out-of-sample, and is
   deliberately a separate call that plays no part in choosing anything.

2. NO EDITS TO AGENTS OWNED BY OTHERS. The constants being calibrated live
   in modules owned by a teammate. Sweeping them requires varying them, so
   `override_constant()` sets the module attribute inside a context manager
   and restores it on exit, leaving the source file untouched. Calibration
   reports a recommendation; a human applies it.
"""
from __future__ import annotations

import contextlib
import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from facts_mas.run_backtest import FOLDS, HORIZONS, FoldSpec, compute_mape
from facts_mas.state.micro_reasoning import (
    CalibrationGuideline,
    FoldResult,
    TightenedCalibrationResult,
)

# Same 104-week validation lookback the fusion layer uses
# (fusion_baseline.compute_fold_weights' default), so calibration and
# weighting are calibrated on the same slice of history rather than two
# silently different ones.
LOOKBACK_WEEKS = 104

# NEXUS Proposal Mod-D specifies "4 of the 5 training folds". This backtest
# has 6 folds, so the equivalent proportion is 5 of 6 (83%, vs. the
# proposal's 80%) -- the nearest integer that is at least as strict as the
# proposal. Stated explicitly because it is a translation of the spec to a
# different fold count, not a threshold invented here.
MIN_CONSISTENT_FOLDS = 5

# A fold whose validation window contains fewer affected origins than this
# cannot cast a meaningful vote on direction, so it is recorded as
# "neutral" rather than allowed to swing the consistency count on noise.
# Matches factor_attribution.MIN_RELIABLE_N, deliberately: the two modules
# are making the same kind of small-sample judgment and should not disagree
# on where "too small to trust" begins.
MIN_AFFECTED_ORIGINS = 30

# Below this absolute MAPE change (percentage points), a fold is recorded as
# "neutral" rather than "increase"/"decrease". Without a dead band, folds
# where the constant barely matters still cast a full-strength directional
# vote based on floating-point noise, which inflates apparent consistency.
NEUTRAL_BAND_MAPE = 0.01


# ═══════════════════════════════════════════════════════════════════════════
# Non-invasive constant override
# ═══════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def override_constant(module: Any, name: str, value: Any):
    """
    Temporarily set `module.name = value`, restoring the original on exit.

    Used instead of editing the agent modules directly: the Event Agent is
    owned by a teammate under this project's "one file, one owner" rule, and
    a calibration sweep is not a reason to rewrite someone else's source. The
    override is scoped to the sweep; the file on disk never changes.
    """
    if not hasattr(module, name):
        raise AttributeError(f"{module.__name__} has no attribute '{name}'")
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


# ═══════════════════════════════════════════════════════════════════════════
# Candidate specification
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CalibrationTarget:
    """
    One constant to calibrate, and everything needed to sweep it.

    module/attribute identify the constant. `candidates` is the value grid.
    `baseline` is the value currently in production -- every candidate is
    scored as a delta against it, which is what makes a FoldResult's
    mape_before/mape_after pair meaningful.
    """

    name: str
    module: Any
    attribute: str
    candidates: tuple
    baseline: Any
    description: str = ""

    def __post_init__(self):
        if self.baseline not in self.candidates:
            raise ValueError(
                f"{self.name}: baseline {self.baseline!r} must appear in "
                f"candidates {self.candidates!r} so it can be scored on equal "
                f"footing with the alternatives."
            )


@dataclass
class CandidateScore:
    """Per-fold MAPE for one candidate value, plus how many origins it moved."""

    value: Any
    fold_mapes: dict[int, float] = field(default_factory=dict)
    fold_affected: dict[int, int] = field(default_factory=dict)

    def mean_mape(self) -> float:
        vals = [m for m in self.fold_mapes.values() if np.isfinite(m)]
        return float(np.mean(vals)) if vals else float("inf")


# ═══════════════════════════════════════════════════════════════════════════
# Scoring: run an agent over a validation window under one constant value
# ═══════════════════════════════════════════════════════════════════════════

def _validation_window(fold: FoldSpec, lookback_weeks: int = LOOKBACK_WEEKS):
    """
    The slice of TRAINING history a fold is allowed to calibrate on.

    Ends at train_end -- strictly before the embargo, and therefore strictly
    before the test window. This is the single most important line in the
    module: everything downstream inherits its no-lookahead guarantee from
    the fact that no calibration ever sees a date after train_end.
    """
    return (fold.train_end - datetime.timedelta(weeks=lookback_weeks), fold.train_end)


def score_agent_over_window(
    weekly_df: pd.DataFrame,
    runner: Callable,
    msas: list[str],
    window: tuple[datetime.date, datetime.date],
    horizons: list[int],
    origin_filter: Optional[Callable[[str, datetime.date], bool]] = None,
) -> tuple[float, int, dict[tuple, float]]:
    """
    Mean MAPE of `runner` over every weekly origin in `window`.

    `origin_filter` restricts scoring to origins the constant actually
    affects. This matters more than it looks: FEMA declarations are active
    in a small minority of (msa, week) cells, so scoring the Event Agent's
    decay window over ALL origins averages the signal into nothing -- the
    same MAPE appears for every candidate because most origins are identical
    under all of them. Filtering to affected origins is what makes the sweep
    able to see anything at all.

    Returns (mean_mape, n_origins, per_origin_mape) -- the per-origin dict is
    keyed by (msa, origin, horizon) so callers can diff two candidates origin
    by origin rather than only in aggregate.
    """
    start, end = window
    per_origin: dict[tuple, float] = {}

    for msa in msas:
        msa_data = weekly_df[weekly_df["msa"] == msa].sort_values("date")
        origins = msa_data[
            (msa_data["date"] >= pd.Timestamp(start))
            & (msa_data["date"] <= pd.Timestamp(end))
        ]["date"].values

        for origin_ts in origins:
            origin = pd.Timestamp(origin_ts).date()

            if origin_filter is not None and not origin_filter(msa, origin):
                continue

            for horizon in horizons:
                future = msa_data[msa_data["date"] > pd.Timestamp(origin)].head(horizon)
                if len(future) < horizon:
                    continue
                actuals = future["inventory_count"].values.astype(np.float64)

                try:
                    output = runner(weekly_df, msa, origin, horizon)
                except Exception:
                    # An agent that fails on an origin contributes no
                    # evidence about the constant. Skipping is correct here,
                    # unlike in fusion weighting where a failure is penalised
                    # -- we are grading the constant, not the agent.
                    continue

                per_origin[(msa, origin, horizon)] = compute_mape(
                    actuals, np.array(output.values, dtype=np.float64)
                )

    if not per_origin:
        return float("inf"), 0, per_origin

    finite = [m for m in per_origin.values() if np.isfinite(m)]
    mean = float(np.mean(finite)) if finite else float("inf")
    return mean, len(per_origin), per_origin


# ═══════════════════════════════════════════════════════════════════════════
# The calibration sweep
# ═══════════════════════════════════════════════════════════════════════════

def calibrate_target(
    target: CalibrationTarget,
    weekly_df: pd.DataFrame,
    runner: Callable,
    msas: list[str],
    folds: Optional[list[FoldSpec]] = None,
    horizons: Optional[list[int]] = None,
    origin_filter: Optional[Callable[[str, datetime.date], bool]] = None,
    verbose: bool = True,
) -> tuple[dict, TightenedCalibrationResult]:
    """
    Sweep every candidate value for `target` across every fold's validation
    window, then gate the winner on fold-direction consistency.

    Returns (scores_by_value, TightenedCalibrationResult).

    The returned TightenedCalibrationResult carries one CalibrationGuideline
    per non-baseline candidate. A candidate is APPROVED only if it improves
    MAPE in at least MIN_CONSISTENT_FOLDS of the folds individually -- the
    NEXUS Mod-D requirement. A candidate that wins on pooled mean MAPE but
    only because one fold improved enormously while the others got slightly
    worse is exactly what that gate exists to reject.
    """
    folds = folds or FOLDS
    horizons = horizons or HORIZONS

    scores: dict[Any, CandidateScore] = {v: CandidateScore(value=v) for v in target.candidates}

    for fold in folds:
        window = _validation_window(fold)
        if verbose:
            print(f"  fold {fold.fold}: validation window "
                  f"{window[0]} .. {window[1]}")

        for value in target.candidates:
            with override_constant(target.module, target.attribute, value):
                mape, n, _ = score_agent_over_window(
                    weekly_df, runner, msas, window, horizons, origin_filter
                )
            scores[value].fold_mapes[fold.fold] = mape
            scores[value].fold_affected[fold.fold] = n
            if verbose:
                shown = "inf" if not np.isfinite(mape) else f"{mape:7.3f}"
                print(f"      {target.attribute}={value!r:>6}  "
                      f"MAPE={shown}  n={n}")

    # ── Build a fold-consistency guideline per candidate ──────────────────
    baseline_score = scores[target.baseline]
    guidelines = []

    for value in target.candidates:
        if value == target.baseline:
            continue

        fold_results = []
        for fold in folds:
            before = baseline_score.fold_mapes.get(fold.fold, float("inf"))
            after = scores[value].fold_mapes.get(fold.fold, float("inf"))
            n_affected = scores[value].fold_affected.get(fold.fold, 0)

            if not (np.isfinite(before) and np.isfinite(after)):
                direction = "neutral"
            elif n_affected < MIN_AFFECTED_ORIGINS:
                # Too few affected origins in this fold to vote (see
                # MIN_AFFECTED_ORIGINS). Recorded, not silently dropped.
                direction = "neutral"
            elif after < before - NEUTRAL_BAND_MAPE:
                direction = "decrease"   # MAPE decreased == improvement
            elif after > before + NEUTRAL_BAND_MAPE:
                direction = "increase"   # MAPE increased == regression
            else:
                direction = "neutral"

            fold_results.append(FoldResult(
                fold_index=fold.fold,
                correction_direction=direction,
                # FoldResult requires finite non-negative MAPEs; an infinite
                # fold is represented by its neutral direction above, with
                # 0.0 standing in for "no measurement" rather than a real
                # score of zero.
                mape_before=float(before) if np.isfinite(before) else 0.0,
                mape_after=float(after) if np.isfinite(after) else 0.0,
            ))

        guideline = CalibrationGuideline(
            guideline_text=(
                f"Set {target.module.__name__}.{target.attribute} = {value!r} "
                f"(currently {target.baseline!r}). "
                f"Mean validation MAPE {scores[value].mean_mape():.3f} vs "
                f"baseline {baseline_score.mean_mape():.3f}."
            ),
            fold_results=fold_results,
            min_consistent_folds=MIN_CONSISTENT_FOLDS,
        )
        guidelines.append(guideline)

    # A guideline "passes" only when its dominant direction is an actual
    # improvement. CalibrationGuideline's own check is direction-agnostic --
    # it confirms the folds AGREE, not that they agree on something good --
    # so consistently-worse candidates are filtered out here.
    approved = [
        g for g in guidelines
        if g.passes_consistency_check and g.dominant_direction == "decrease"
    ]

    windows = [_validation_window(f) for f in folds]

    result = TightenedCalibrationResult(
        # Aggregate MAPE reported here is the BASELINE's, i.e. what the
        # constant currently in production scores -- the number any approved
        # guideline is proposing to improve on.
        mape=float(baseline_score.mean_mape()) if np.isfinite(baseline_score.mean_mape()) else 0.0,
        # RMSE is deliberately not used as a calibration criterion. Inventory
        # levels differ by more than an order of magnitude across MSAs (New
        # York ~8M housing units vs Seattle ~1.7M), so an RMSE-driven sweep
        # would be decided almost entirely by the largest metros. MAPE is
        # scale-free and gives every MSA equal say. Reported as 0.0 to mean
        # "not scored on", not "zero error".
        rmse=0.0,
        evaluation_window_start=min(w[0] for w in windows).isoformat(),
        evaluation_window_end=max(w[1] for w in windows).isoformat(),
        candidate_guidelines=guidelines,
        approved_guidelines=approved,
    )
    return scores, result


def evaluate_on_test_window(
    target: CalibrationTarget,
    value: Any,
    weekly_df: pd.DataFrame,
    runner: Callable,
    msas: list[str],
    folds: Optional[list[FoldSpec]] = None,
    horizons: Optional[list[int]] = None,
    origin_filter: Optional[Callable[[str, datetime.date], bool]] = None,
) -> dict[int, float]:
    """
    What `value` would have scored on each fold's TEST window.

    Reporting only. Kept as a separate function, never called by
    calibrate_target(), so that no execution path can let a test-window
    number influence which value gets chosen. Use it to state out-of-sample
    performance honestly after the choice is already made.
    """
    folds = folds or FOLDS
    horizons = horizons or HORIZONS
    out = {}
    for fold in folds:
        with override_constant(target.module, target.attribute, value):
            mape, _, _ = score_agent_over_window(
                weekly_df, runner, msas,
                (fold.test_start, fold.test_end), horizons, origin_filter,
            )
        out[fold.fold] = mape
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def format_calibration_report(
    target: CalibrationTarget,
    scores: dict,
    result: TightenedCalibrationResult,
) -> str:
    """Plain-language summary, in the style of factor_attribution's report."""
    lines = []
    lines.append(f"CALIBRATION: {target.name}")
    lines.append(f"  constant : {target.module.__name__}.{target.attribute}")
    lines.append(f"  current  : {target.baseline!r}")
    if target.description:
        lines.append(f"  rationale: {target.description}")
    lines.append("")

    lines.append("  Mean validation MAPE by candidate value (lower is better):")
    ordered = sorted(scores.items(), key=lambda kv: kv[1].mean_mape())
    for value, score in ordered:
        tag = "  <- current" if value == target.baseline else ""
        best = " *best*" if value == ordered[0][0] else ""
        lines.append(f"    {target.attribute}={value!r:>6}  "
                     f"MAPE={score.mean_mape():7.3f}{best}{tag}")
    lines.append("")

    lines.append(f"  Fold-consistency gate: a candidate must improve MAPE in "
                 f">= {MIN_CONSISTENT_FOLDS} of {len(FOLDS)} folds individually.")
    for g in result.candidate_guidelines:
        verdict = ("APPROVED" if g in result.approved_guidelines
                   else f"rejected ({g.dominant_direction} in "
                        f"{g.fold_consistency_count} folds)")
        lines.append(f"    {g.guideline_text}")
        lines.append(f"      -> {verdict}")
    lines.append("")

    if result.approved_guidelines:
        lines.append(f"  {len(result.approved_guidelines)} candidate(s) passed "
                     f"the consistency gate.")
    else:
        lines.append("  No candidate passed the consistency gate. The current "
                     "value stands -- not because it is optimal, but because "
                     "nothing beat it consistently enough to justify a change.")
    return "\n".join(lines)
