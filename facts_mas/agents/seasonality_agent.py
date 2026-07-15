"""
Seasonality Agent — FACTS-MAS Phase 2.

Calendar/seasonality adjustment using week-of-year, month, quarter,
and spring-selling-season flags from the aligned weekly panel.

Design:
    - 100% deterministic: same calendar inputs → same output, no randomness.
    - Learns seasonal factors from historical inventory data per MSA.
    - Outputs a seasonally-adjusted forecast as inventory counts.

Validation (from PDF §3):
    - Unit test: output is deterministic given the same calendar inputs.
    - Sanity check: adjustment direction matches known housing seasonality
      (higher inventory in spring/summer).
"""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
import pandas as pd

from facts_mas.schema import AgentOutput


# ═══════════════════════════════════════════════════════════════════════════
# Seasonal factor computation (pure NumPy — deterministic)
# ═══════════════════════════════════════════════════════════════════════════

def compute_seasonal_factors(
    df: pd.DataFrame,
    msa: str,
    train_end: datetime.date,
) -> np.ndarray:
    """
    Compute week-of-year seasonal factors for a given MSA from historical data.

    Method: ratio-to-moving-average.
        1. For each week, compute inventory / 52-week centered moving average.
        2. Group by week-of-year (1–53), take the median ratio.
        3. Normalize so factors average to 1.0.

    This captures the known housing seasonality:
        - Spring/summer (weeks ~12–30): higher inventory → factors > 1.0
        - Winter (weeks ~45–8): lower inventory → factors < 1.0

    Args:
        df: aligned_weekly.csv loaded as a DataFrame.
        msa: MSA name to filter.
        train_end: Last allowable training date (no lookahead).

    Returns:
        Array of 53 seasonal factors indexed by ISO week number (1-53).
    """
    msa_df = df[(df["msa"] == msa) & (df["date"] <= pd.Timestamp(train_end))].copy()
    msa_df = msa_df.sort_values("date").reset_index(drop=True)

    inv = msa_df["inventory_count"].values.astype(np.float64)

    # 52-week centered moving average (26 weeks each side)
    if len(inv) < 53:
        # Not enough data — return flat factors (no seasonal adjustment)
        return np.ones(53, dtype=np.float64)

    cma = pd.Series(inv).rolling(window=52, center=True, min_periods=26).mean().values

    # Ratio to moving average
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = inv / cma
    ratios[~np.isfinite(ratios)] = np.nan

    # Extract ISO week numbers
    weeks = msa_df["date"].apply(lambda d: d.isocalendar()[1]).values

    # Median ratio per week-of-year
    factors = np.ones(53, dtype=np.float64)
    for w in range(1, 54):
        mask = weeks == w
        week_ratios = ratios[mask]
        week_ratios = week_ratios[np.isfinite(week_ratios)]
        if len(week_ratios) >= 2:
            factors[w - 1] = np.median(week_ratios)

    # Normalize so mean factor = 1.0
    mean_factor = np.mean(factors)
    if mean_factor > 0:
        factors = factors / mean_factor

    return factors


# ═══════════════════════════════════════════════════════════════════════════
# Seasonality Agent — main entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_seasonality_agent(
    df: pd.DataFrame,
    msa: str,
    forecast_origin: datetime.date,
    horizon_weeks: int,
    *,
    seasonal_factors: Optional[np.ndarray] = None,
) -> AgentOutput:
    """
    Generate a seasonally-adjusted inventory forecast.

    Method:
        1. Start from the last observed inventory value.
        2. Compute the expected seasonal trajectory by applying week-of-year
           factors to the recent level.
        3. Return the seasonal forecast as absolute inventory counts.

    This is fully deterministic — no randomness, no LLM, no stochastic model.

    Args:
        df: aligned_weekly.csv as DataFrame with 'date' already parsed.
        msa: MSA name.
        forecast_origin: Last date of data the agent can see.
        horizon_weeks: Number of weeks to forecast (4, 8, or 13).
        seasonal_factors: Pre-computed factors (optional, computed if None).

    Returns:
        AgentOutput conforming to the shared schema.
    """
    if seasonal_factors is None:
        seasonal_factors = compute_seasonal_factors(df, msa, forecast_origin)

    # Get the most recent inventory value as of forecast_origin
    msa_df = df[
        (df["msa"] == msa) & (df["date"] <= pd.Timestamp(forecast_origin))
    ].sort_values("date")

    if msa_df.empty:
        raise ValueError(f"No data for MSA '{msa}' before {forecast_origin}")

    last_inventory = float(msa_df["inventory_count"].iloc[-1])
    last_date = msa_df["date"].iloc[-1]
    last_week = last_date.isocalendar()[1]

    # Get the factor for the "current" week (baseline reference)
    current_factor = seasonal_factors[last_week - 1]

    # Project forward: scale inventory by the ratio of future-week factor
    # to current-week factor
    forecast_values = []
    for step in range(1, horizon_weeks + 1):
        future_week = ((last_week - 1 + step) % 53) + 1
        future_factor = seasonal_factors[future_week - 1]

        if current_factor > 0:
            adjusted = last_inventory * (future_factor / current_factor)
        else:
            adjusted = last_inventory

        # Enforce non-negativity (physical floor)
        forecast_values.append(max(0.0, round(adjusted, 2)))

    return AgentOutput(
        agent_name="seasonality",
        msa=msa,
        forecast_origin=forecast_origin,
        horizon_weeks=horizon_weeks,
        values=forecast_values,
    )
