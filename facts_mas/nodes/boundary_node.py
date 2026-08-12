"""
NEXUS Modification C — Statistical Boundary Constraints.

Solves Limitation 3: nothing in the published architecture would catch a
forecast that violates physical reality -- a negative inventory count, or a
weekly swing far larger than anything ever observed for that city.

`state/boundary.py` already detects violations and handles the final
fallback. What was missing, and what this module adds, is the loop between
them: intercept the bad forecast, describe exactly how far out of range it
fell, ask for a revision, and re-check. That correction cycle is the actual
modification; detection alone just reports the problem.

Two design points from the proposal worth restating, because both are easy
to get backwards:

    The bound is on week-over-week PERCENTAGE CHANGE, not on the value.
    Housing inventory trends over multiple years, so a bound derived from
    the historical average level would flag a perfectly legitimate
    continuation of that trend as an error. A bound on rate of change adapts
    to whatever trend is already present. A separate absolute floor at zero
    handles the physical impossibility.

    Correction retries are capped at two, then the system falls back to the
    cheap baseline for the offending timesteps. Without a cap, an unbounded
    retry loop becomes its own source of unreliability -- the guarantee that
    matters is that this always terminates with a usable, bounded forecast.

No LLM is required. The default corrector is arithmetic: clamp the offending
step to the edge of its allowed range. A `corrector` callable is accepted
for the case where the revision should go back to a language model, which is
how the proposal describes it, but the deterministic path is the default
because a clamp cannot hallucinate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from facts_mas.state.boundary import (
    BoundaryCheckResult,
    BoundaryConfig,
    BoundaryViolation,
)


@dataclass
class BoundaryOutcome:
    """What the correction loop did, and why."""

    final_forecast: list[float]
    result: BoundaryCheckResult
    attempts: int
    corrected: bool
    fell_back: bool
    violation_log: list[str]


# ═══════════════════════════════════════════════════════════════════════════
# Deterministic corrector
# ═══════════════════════════════════════════════════════════════════════════

def clamp_corrector(
    forecast: list[float],
    violations: list[BoundaryViolation],
    last_value: float,
    config: BoundaryConfig,
) -> list[float]:
    """
    Pull each offending step back to the nearest edge of its allowed range.

    Walks forward rather than correcting each violation in isolation,
    because the bound is defined relative to the *previous* value. Fixing
    step 3 changes what step 4 is allowed to be, so a pass that used the
    original neighbours would leave a forecast that still violates.

    Clamping to the boundary rather than to something safely inside it is
    deliberate: the model's direction is preserved, only its magnitude is
    capped. Pulling further would substitute our judgement for the model's
    on a point where the model is not actually wrong, only extreme.
    """
    out = list(forecast)
    bad = {v.timestep_index for v in violations}
    prev = last_value

    for i, val in enumerate(out):
        if i in bad and prev != 0:
            hi = prev * (1.0 + config.max_historical_pct_change)
            lo = prev * (1.0 - config.max_historical_pct_change)
            val = float(min(max(val, lo), hi))
        out[i] = max(float(config.physical_floor), float(val))
        prev = out[i]

    return out


def _sweep_to_bounds(
    forecast: list[float],
    last_value: float,
    config: BoundaryConfig,
) -> list[float]:
    """
    Force every step inside the allowed band, walking forward.

    Unconditional, unlike clamp_corrector which only touches steps that were
    flagged. Used as the final guarantee after the retry budget is spent.
    """
    out, prev = [], last_value
    for val in forecast:
        v = float(val)
        if prev != 0:
            hi = prev * (1.0 + config.max_historical_pct_change)
            lo = prev * (1.0 - config.max_historical_pct_change)
            v = min(max(v, lo), hi)
        v = max(float(config.physical_floor), v)
        out.append(v)
        prev = v
    return out


# ═══════════════════════════════════════════════════════════════════════════
# The correction loop
# ═══════════════════════════════════════════════════════════════════════════

def enforce_boundaries(
    forecast: list[float],
    last_historical_value: float,
    config: BoundaryConfig,
    baseline_forecast: Optional[list[float]] = None,
    corrector: Callable = clamp_corrector,
) -> BoundaryOutcome:
    """
    Check, correct, re-check, up to the configured retry limit.

    Terminates in one of three states, and the caller can tell which:
        clean       passed on the first check, nothing changed
        corrected   passed after one or two revisions
        fell_back   still out of bounds after the limit, so the offending
                    timesteps were replaced with the cheap baseline

    Every violation is logged with the range it broke, so a downstream
    report can say what was wrong rather than only that something was.
    """
    log: list[str] = []
    current = list(forecast)
    attempts = 0

    while True:
        result = BoundaryCheckResult.check_forecast(
            forecast_values=current,
            last_historical_value=last_historical_value,
            config=config,
            retry_count=attempts,
            baseline_forecast=baseline_forecast,
        )

        if result.is_within_bounds:
            return BoundaryOutcome(
                final_forecast=current, result=result, attempts=attempts,
                corrected=attempts > 0, fell_back=False, violation_log=log,
            )

        for v in result.violations:
            log.append(
                f"attempt {attempts} · step {v.timestep_index} · "
                f"{v.violation_type} · {v.allowed_range_description}"
            )

        # Retry budget exhausted: take whatever fallback check_forecast
        # produced, or clamp as a last resort if no baseline was supplied.
        if attempts >= config.max_correction_retries:
            final = result.corrected_forecast
            if final is None:
                final = corrector(current, result.violations,
                                  last_historical_value, config)
                fell_back = False
            else:
                fell_back = result.fell_back_to_baseline

            # Final sweep, and it is load-bearing rather than belt-and-braces.
            # check_forecast measures each step against the step BEFORE it in
            # the same forecast, so a series that jumps once and then holds an
            # absurd level records exactly one violation. Its fallback then
            # substitutes the baseline at that one index and leaves the rest,
            # which makes the *following* step violate against its new
            # predecessor. Patching a subset of a chained series cannot be
            # correct on its own. Sweeping forward re-bases every step on the
            # value actually preceding it, so what we return provably
            # satisfies the bound -- which is the guarantee this module is
            # here to make.
            final = _sweep_to_bounds([float(v) for v in final],
                                     last_historical_value, config)
            return BoundaryOutcome(
                final_forecast=final, result=result,
                attempts=attempts, corrected=True, fell_back=fell_back,
                violation_log=log,
            )

        current = corrector(current, result.violations,
                            last_historical_value, config)
        attempts += 1


# ═══════════════════════════════════════════════════════════════════════════
# Measuring what the constraint catches
# ═══════════════════════════════════════════════════════════════════════════

def boundary_report(outcomes: list[BoundaryOutcome]) -> dict:
    """
    How often the constraint fired, and what it caught.

    The honest reading of a low number here is that the fusion layer was
    already well-behaved, which is itself worth reporting: a safety net that
    never catches anything is evidence about the system, not a wasted
    component. It becomes load-bearing the moment an LLM is allowed to
    produce the final number, which is the configuration the proposal is
    actually guarding against.
    """
    n = len(outcomes)
    if n == 0:
        return {}
    clean = sum(1 for o in outcomes if not o.corrected)
    corrected = sum(1 for o in outcomes if o.corrected and not o.fell_back)
    fell_back = sum(1 for o in outcomes if o.fell_back)

    kinds: dict[str, int] = {}
    for o in outcomes:
        for v in o.result.violations:
            kinds[v.violation_type] = kinds.get(v.violation_type, 0) + 1

    return {
        "n_forecasts": n,
        "clean": clean,
        "corrected": corrected,
        "fell_back_to_baseline": fell_back,
        "violation_rate": 1.0 - clean / n,
        "violations_by_type": kinds,
        "mean_attempts": float(np.mean([o.attempts for o in outcomes])),
    }
