"""
Intrinsic Agent — FACTS-MAS Phase 2.

Regression/GBM model using Census/NOAA static features (population, income,
housing units, density, hazard_index) from intrinsic_static.csv.

Design:
    - Learns how static MSA characteristics relate to inventory levels.
    - Uses a GradientBoostingRegressor (scikit-learn) trained on historical
      inventory averages per MSA.
    - Validation uses held-out MSAs (leave-one-out) to check the model isn't
      just memorizing city identity.

Validation (from PDF §3):
    - Unit test: output shape and range checks.
    - Backtest: MAPE/RMSE contribution isolated using held-out MSAs.
"""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from facts_mas.schema import AgentOutput


# ═══════════════════════════════════════════════════════════════════════════
# Feature columns from intrinsic_static.csv
# ═══════════════════════════════════════════════════════════════════════════

INTRINSIC_FEATURES = [
    "population",
    "median_income",
    "housing_units",
    "density",
    "hazard_index",
]


# ═══════════════════════════════════════════════════════════════════════════
# Model training — deterministic (fixed random_state)
# ═══════════════════════════════════════════════════════════════════════════

def train_intrinsic_model(
    weekly_df: pd.DataFrame,
    static_df: pd.DataFrame,
    train_end: datetime.date,
    *,
    exclude_msa: Optional[str] = None,
) -> GradientBoostingRegressor:
    """
    Train a GBM that predicts average inventory level from static MSA features.

    The target variable is the average weekly inventory_count per MSA
    over the training period. This gives the model a structural "expected
    level" for each city based on its demographics and hazard exposure.

    Args:
        weekly_df: aligned_weekly.csv with 'date' column as datetime.
        static_df: intrinsic_static.csv.
        train_end: Last allowable training date.
        exclude_msa: If set, hold out this MSA for validation (leave-one-out).

    Returns:
        Trained GradientBoostingRegressor.
    """
    # Compute average inventory per MSA over training period
    train_data = weekly_df[weekly_df["date"] <= pd.Timestamp(train_end)]
    avg_inventory = (
        train_data.groupby("msa")["inventory_count"]
        .mean()
        .reset_index()
        .rename(columns={"inventory_count": "avg_inventory"})
    )

    # Merge with static features
    merged = avg_inventory.merge(static_df, on="msa", how="inner")

    if exclude_msa:
        merged = merged[merged["msa"] != exclude_msa]

    if len(merged) < 3:
        raise ValueError(
            f"Not enough MSAs for training (got {len(merged)} after exclusion)."
        )

    X = merged[INTRINSIC_FEATURES].values
    y = merged["avg_inventory"].values

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,  # deterministic
    )
    model.fit(X, y)

    return model


# ═══════════════════════════════════════════════════════════════════════════
# Intrinsic Agent — main entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_intrinsic_agent(
    weekly_df: pd.DataFrame,
    static_df: pd.DataFrame,
    msa: str,
    forecast_origin: datetime.date,
    horizon_weeks: int,
    *,
    model: Optional[GradientBoostingRegressor] = None,
) -> AgentOutput:
    """
    Generate an intrinsic-level inventory forecast.

    Method:
        1. Use the GBM to predict the structural average inventory for this MSA.
        2. Compute the ratio of recent actual inventory to the structural level.
        3. Project forward, gradually reverting toward the structural mean
           over the forecast horizon (mean-reversion assumption).

    This captures the idea that cities will tend to revert toward their
    structurally-implied inventory level over time.

    Args:
        weekly_df: aligned_weekly.csv with 'date' as datetime.
        static_df: intrinsic_static.csv.
        msa: MSA name.
        forecast_origin: Last date of data the agent can see.
        horizon_weeks: Number of weeks to forecast (4, 8, or 13).
        model: Pre-trained model (trained if None).

    Returns:
        AgentOutput conforming to the shared schema.
    """
    if model is None:
        model = train_intrinsic_model(weekly_df, static_df, forecast_origin)

    # Get static features for this MSA
    msa_static = static_df[static_df["msa"] == msa]
    if msa_static.empty:
        raise ValueError(f"MSA '{msa}' not found in intrinsic_static.csv")

    X_msa = msa_static[INTRINSIC_FEATURES].values
    structural_level = float(model.predict(X_msa)[0])

    # Get the most recent actual inventory
    msa_recent = weekly_df[
        (weekly_df["msa"] == msa)
        & (weekly_df["date"] <= pd.Timestamp(forecast_origin))
    ].sort_values("date")

    if msa_recent.empty:
        raise ValueError(f"No data for MSA '{msa}' before {forecast_origin}")

    last_inventory = float(msa_recent["inventory_count"].iloc[-1])

    # Mean-reversion forecast:
    # At step t, forecast = last_inventory + reversion_rate * t * (structural - last)
    # reversion_rate chosen so that at horizon_weeks, we've reverted ~30%
    reversion_rate = 0.30 / max(horizon_weeks, 1)

    forecast_values = []
    for step in range(1, horizon_weeks + 1):
        reversion_frac = min(reversion_rate * step, 1.0)
        projected = last_inventory + reversion_frac * (
            structural_level - last_inventory
        )
        forecast_values.append(max(0.0, round(projected, 2)))

    return AgentOutput(
        agent_name="intrinsic",
        msa=msa,
        forecast_origin=forecast_origin,
        horizon_weeks=horizon_weeks,
        values=forecast_values,
    )
