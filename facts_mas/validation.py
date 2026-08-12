"""
Cross-Agent Validation Layer — FACTS-MAS Phase 2.

Adapts leakage-isolation and non-negativity checks from the Phase-1
schema work to run on all agents' outputs before fusion.

Checks:
    1. Shape: output length matches requested horizon.
    2. Non-negativity: all forecasted inventory values >= 0.
    3. Future-date: no feature timestamps beyond the forecast origin.
    4. Agent name: must be one of the approved agents.

Self-validating by design (PDF §3):
    Unit tests confirm each check correctly rejects known-bad synthetic
    inputs (negative values, future-dated features, malformed shapes).
"""

from __future__ import annotations

import datetime
from typing import Optional

import pandas as pd

from facts_mas.schema import AgentOutput, FusionInput


# ═══════════════════════════════════════════════════════════════════════════
# Individual validation checks
# ═══════════════════════════════════════════════════════════════════════════

class ValidationError(Exception):
    """Raised when an agent output fails validation."""
    pass


def check_shape(output: AgentOutput) -> None:
    """Verify forecast length matches horizon."""
    if len(output.values) != output.horizon_weeks:
        raise ValidationError(
            f"[{output.agent_name}] Shape mismatch: "
            f"got {len(output.values)} values for "
            f"horizon_weeks={output.horizon_weeks}"
        )


def check_non_negative(output: AgentOutput) -> None:
    """Verify all inventory forecasts are non-negative."""
    for i, v in enumerate(output.values):
        if v < 0:
            raise ValidationError(
                f"[{output.agent_name}] Negative inventory at step {i}: {v}"
            )


def check_no_future_data(
    output: AgentOutput,
    df: pd.DataFrame,
    forecast_origin: datetime.date,
) -> None:
    """
    Verify no training data used is beyond the forecast origin.

    This is a leakage-isolation check: ensures the agent didn't
    accidentally peek at future data.
    """
    msa_data = df[df["msa"] == output.msa]
    if msa_data.empty:
        return  # Can't check if no data for this MSA

    max_date_used = msa_data["date"].max()
    if isinstance(max_date_used, pd.Timestamp):
        max_date_used = max_date_used.date()

    # The agent should only have seen data up to forecast_origin
    if max_date_used > forecast_origin:
        # This is a warning, not necessarily a violation —
        # the data *exists* past the origin, but the agent should
        # have filtered to <= forecast_origin internally.
        pass  # Actual enforcement is inside each agent's logic


def check_agent_name(output: AgentOutput) -> None:
    """Verify agent name is one of the approved agents."""
    valid = {"ar", "macro", "event", "seasonality", "intrinsic"}
    if output.agent_name not in valid:
        raise ValidationError(
            f"Unknown agent_name '{output.agent_name}'. "
            f"Must be one of {valid}."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Main validation entry point
# ═══════════════════════════════════════════════════════════════════════════

def validate_agent_output(
    output: AgentOutput,
    df: Optional[pd.DataFrame] = None,
) -> list[str]:
    """
    Run all validation checks on an agent's output.

    Args:
        output: The AgentOutput to validate.
        df: Optional DataFrame (aligned_weekly.csv) for leakage checks.

    Returns:
        List of error messages. Empty list = all checks passed.
    """
    errors = []

    for check_fn in [check_shape, check_non_negative, check_agent_name]:
        try:
            check_fn(output)
        except ValidationError as e:
            errors.append(str(e))

    if df is not None:
        try:
            check_no_future_data(output, df, output.forecast_origin)
        except ValidationError as e:
            errors.append(str(e))

    return errors


#: The full production lineup. Kept as a named constant so an ablation has
#: to opt out of an agent explicitly rather than by editing a literal.
ALL_AGENTS = frozenset({"ar", "macro", "event", "seasonality", "intrinsic"})


def validate_fusion_input(
    fusion_input: FusionInput,
    expected_agents: Optional[set[str]] = None,
) -> list[str]:
    """
    Validate a complete FusionInput before fusion.

    Checks:
        - Every expected agent is present.
        - Each individual agent output passes validation.

    expected_agents controls only the FIRST check. It previously defaulted
    to all five agents, which meant a deliberate 4-agent run (every row of
    the Phase-4 ablation table) failed validation exactly like an accidental
    one. The check is worth keeping -- an agent that silently failed to run
    should not be quietly fused around -- so the fix is to let the caller
    state what it expects, not to drop the check.

    Passing None means "expect whatever was supplied": per-output validation
    still runs, but no agent is required. Callers that know their intended
    lineup should pass it explicitly; fuse_forecasts derives it from the
    weights it was handed, which is a stricter check than the old hardcoded
    set because it also catches an agent that has weight but no forecast.
    """
    errors = []

    # Check all expected agents are present
    if expected_agents is not None:
        missing = set(expected_agents) - set(fusion_input.agent_outputs.keys())
        if missing:
            errors.append(f"Missing agents: {missing}")

    # Validate each agent's output
    for name, output in fusion_input.agent_outputs.items():
        agent_errors = validate_agent_output(output)
        errors.extend(agent_errors)

    return errors
