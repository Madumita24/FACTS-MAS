"""
Rolling-Weight Fusion Baseline — FACTS-MAS Phase 2.

Combines all agents via weights recomputed per backtest fold as new weekly
data rolls in, not one static weight set for the whole test period
(per Prof. Pan's note on adjustable weighting).

Method:
    Inverse-MAPE weighting: each agent's weight is proportional to
    1 / (its MAPE on the most recent training fold).
    Agents with lower error get higher weight.
    Weights are normalized to sum to 1.0.
"""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
import pandas as pd

from facts_mas.schema import AgentOutput, FusionInput
from facts_mas.validation import validate_fusion_input


def compute_agent_weights(
    agent_errors: dict[str, float],
) -> dict[str, float]:
    """
    Compute fusion weights from per-agent MAPE scores.

    Method: inverse-MAPE weighting.
        weight_i = (1 / mape_i) / sum(1 / mape_j for all j)

    If an agent has MAPE = 0 (perfect), it gets a large finite weight.
    If all agents have the same MAPE, weights are uniform.

    Args:
        agent_errors: {agent_name: mape_on_validation_data}

    Returns:
        {agent_name: weight}, normalized to sum to 1.0.
    """
    if not agent_errors:
        return {}

    # Clamp MAPEs to a small floor to avoid division by zero
    eps = 1e-6
    inverse_mapes = {
        name: 1.0 / max(mape, eps) for name, mape in agent_errors.items()
    }
    total = sum(inverse_mapes.values())

    return {name: inv / total for name, inv in inverse_mapes.items()}


def fuse_forecasts(
    fusion_input: FusionInput,
    weights: dict[str, float],
) -> list[float]:
    """
    Combine agent forecasts using the given weights.

    forecast[t] = sum(weight_i * agent_i.values[t]) for each timestep t.

    Args:
        fusion_input: Validated FusionInput with all agents' outputs.
        weights: {agent_name: weight}, should sum to ~1.0.

    Returns:
        Fused forecast values (one per horizon week).
    """
    # Validate before fusing
    errors = validate_fusion_input(fusion_input)
    if errors:
        raise ValueError(f"Fusion input validation failed: {errors}")

    horizon = fusion_input.horizon_weeks
    fused = np.zeros(horizon, dtype=np.float64)

    for agent_name, output in fusion_input.agent_outputs.items():
        w = weights.get(agent_name, 0.0)
        fused += w * np.array(output.values)

    # Non-negativity floor
    fused = np.maximum(fused, 0.0)

    return [round(float(v), 2) for v in fused]


def compute_fold_weights(
    weekly_df: pd.DataFrame,
    agent_runners: dict[str, callable],
    msas: list[str],
    val_start: datetime.date,
    val_end: datetime.date,
    horizon_weeks: int,
) -> dict[str, float]:
    """
    Compute per-fold agent weights by evaluating each agent on a
    validation window and measuring MAPE.

    This is the "rolling" part: weights are recomputed each fold
    using the most recent realized data.

    Args:
        weekly_df: aligned_weekly.csv.
        agent_runners: {agent_name: callable(df, msa, origin, horizon) -> AgentOutput}
        msas: List of MSA names to evaluate on.
        val_start: Start of the validation window.
        val_end: End of the validation window.
        horizon_weeks: Horizon to evaluate at.

    Returns:
        {agent_name: weight}
    """
    agent_mapes: dict[str, list[float]] = {name: [] for name in agent_runners}

    for msa in msas:
        msa_data = weekly_df[weekly_df["msa"] == msa].sort_values("date")
        msa_dates = msa_data["date"].values

        # Find forecast origins within the validation window
        origins = msa_data[
            (msa_data["date"] >= pd.Timestamp(val_start))
            & (msa_data["date"] <= pd.Timestamp(val_end))
        ]["date"].values

        for origin_ts in origins:
            origin = pd.Timestamp(origin_ts).date()

            # Get actuals for this origin + horizon
            future_mask = msa_data["date"] > pd.Timestamp(origin)
            future_data = msa_data[future_mask].head(horizon_weeks)

            if len(future_data) < horizon_weeks:
                continue  # Not enough future data for evaluation

            actuals = future_data["inventory_count"].values

            for agent_name, runner in agent_runners.items():
                try:
                    output = runner(weekly_df, msa, origin, horizon_weeks)
                    forecast = np.array(output.values)

                    # MAPE for this origin
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ape = np.abs((actuals - forecast) / actuals)
                    ape = ape[np.isfinite(ape)]
                    if len(ape) > 0:
                        agent_mapes[agent_name].append(float(np.mean(ape)))
                except Exception:
                    # If an agent fails, give it a high error penalty
                    agent_mapes[agent_name].append(1.0)

    # Average MAPE per agent across all origins and MSAs
    avg_mapes = {}
    for name, mapes in agent_mapes.items():
        avg_mapes[name] = float(np.mean(mapes)) if mapes else 1.0

    return compute_agent_weights(avg_mapes)
