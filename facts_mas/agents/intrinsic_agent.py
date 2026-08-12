"""
Intrinsic Agent — FACTS-MAS Phase 2.

Regularised linear model using Census/NOAA static features (population, income,
housing units, density, hazard_index) from intrinsic_static.csv.

Design:
    - Learns how static MSA characteristics relate to inventory levels.
    - Uses StandardScaler + Ridge regression (scikit-learn) to prevent
      overfitting on the small 15-MSA dataset.
    - Validation uses held-out MSAs (leave-one-out) to check the model isn't
      just memorizing city identity.

History:
    - v1 used GradientBoostingRegressor — overfitted badly (0.1% train,
      48-52% leave-one-out error) because 100-tree GBM memorises 15 samples.
    - v2 (current) uses Ridge with L2 regularisation + feature scaling,
      forcing the model to learn general demographic relationships.

Validation (from PDF §3):
    - Unit test: output shape and range checks.
    - Backtest: MAPE/RMSE contribution isolated using held-out MSAs.
"""

from __future__ import annotations

import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
# Model training — deterministic, regularised
# ═══════════════════════════════════════════════════════════════════════════

def train_intrinsic_model(
    weekly_df: pd.DataFrame,
    static_df: pd.DataFrame,
    train_end: datetime.date,
    *,
    exclude_msa: Optional[str] = None,
) -> Pipeline:
    """
    Train a Ridge regression pipeline that predicts average inventory level
    from static MSA features.

    The target variable is the average weekly inventory_count per MSA
    over the training period. This gives the model a structural "expected
    level" for each city based on its demographics and hazard exposure.

    StandardScaler is essential because population (~millions) and
    hazard_index (~0-10) live on vastly different scales; without it
    Ridge penalises all coefficients equally by magnitude, not importance.

    Args:
        weekly_df: aligned_weekly.csv with 'date' column as datetime.
        static_df: intrinsic_static.csv.
        train_end: Last allowable training date.
        exclude_msa: If set, hold out this MSA for validation (leave-one-out).

    Returns:
        Trained sklearn Pipeline (StandardScaler → Ridge).
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

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),  # L2 regularisation, deterministic
    ])
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
    model: Optional[Pipeline] = None,
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

    # Mean-reversion forecast, quadratic curve (changed from linear):
    # reversion_frac(step) = TOTAL_REVERSION * (step / horizon_weeks)^2
    #
    # Original design reverted LINEARLY (reversion_frac = 0.30/horizon_weeks
    # * step), meaning the fraction reverted per step was constant regardless
    # of horizon length -- e.g. at h=4, fully 25% of the step's progress
    # toward horizon_weeks happens in week 1 alone. Diagnosed via a live
    # sanity check: Intrinsic's skill vs. naive was worst at h=4 (mean
    # -1.98) and improved monotonically toward h=13 (mean -0.28) -- the
    # signature of reverting too far too soon, before the actual series has
    # had time to move that much.
    # The quadratic curve keeps the SAME total ~30% reversion by the end of
    # a 13-week horizon (reversion_frac(horizon_weeks) = TOTAL_REVERSION,
    # unchanged from the original design intent), but redistributes WHEN
    # that reversion happens: at h=4 step=1, reversion_frac is now
    # 0.30*(1/4)^2 = 0.019 (vs. the old linear 0.075) -- a much gentler
    # start, with most of the pull arriving later in the window.
    TOTAL_REVERSION = 0.30

    forecast_values = []
    for step in range(1, horizon_weeks + 1):
        reversion_frac = TOTAL_REVERSION * (step / horizon_weeks) ** 2
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


# ═══════════════════════════════════════════════════════════════════════════
# Historical-mean baseline -- a third structural-target variant, evaluated
# alongside GBM (v1) and Ridge (v2), not replacing either here.
#
# No model training at all: the "structural level" is just each MSA's own
# historical mean inventory over the training window. This has no features
# to generalize from, so MSA-level leave-one-out (as run for GBM/Ridge)
# doesn't apply in any meaningful sense here -- there's no cross-MSA
# relationship being tested, since nothing is being predicted FROM other
# MSAs' data. MSA-LEVEL LEAVE-ONE-OUT DOESN'T APPLY TO A NO-FEATURE MODEL;
# FOLD-LEVEL BACKTEST PERFORMANCE IS THE CORRECT COMPARISON INSTEAD -- see
# scratch_intrinsic_variants_comparison.py for that evaluation (not wired
# into build_agent_runners() here; this is a comparison candidate).
# ═══════════════════════════════════════════════════════════════════════════

class HistoricalMeanModel:
    """Maps msa -> its historical mean inventory over some training window.
    Not a scikit-learn model -- no fitting, no features, just a lookup."""

    def __init__(self, means: pd.Series):
        self._means = means

    def predict_for_msa(self, msa: str) -> float:
        if msa not in self._means.index:
            raise ValueError(f"No historical mean available for msa='{msa}'")
        return float(self._means[msa])


def train_historical_mean_model(
    weekly_df: pd.DataFrame,
    train_end: datetime.date,
    *,
    exclude_msa: Optional[str] = None,
) -> HistoricalMeanModel:
    """No training in the model-fitting sense -- just a groupby mean over
    the training window. exclude_msa is accepted for interface parity with
    train_intrinsic_model, but excluding an MSA from a per-MSA mean lookup
    has no effect on any OTHER msa's mean (unlike GBM/Ridge, where excluding
    an MSA changes every other MSA's fitted coefficients) -- kept only so
    callers written against the GBM/Ridge interface don't need a special
    case for this variant.
    """
    train_data = weekly_df[weekly_df["date"] <= pd.Timestamp(train_end)]
    means = train_data.groupby("msa")["inventory_count"].mean()
    if exclude_msa is not None and exclude_msa in means.index:
        means = means.drop(exclude_msa)
    return HistoricalMeanModel(means)


def run_intrinsic_agent_historical_mean(
    weekly_df: pd.DataFrame,
    static_df: pd.DataFrame,
    msa: str,
    forecast_origin: datetime.date,
    horizon_weeks: int,
    *,
    model: Optional[HistoricalMeanModel] = None,
) -> AgentOutput:
    """Same (weekly_df, static_df, msa, forecast_origin, horizon_weeks) ->
    AgentOutput contract and the same quadratic reversion mechanics as
    run_intrinsic_agent -- only the structural-level source differs
    (historical mean, not a fitted regression). static_df is accepted for
    interface parity with run_intrinsic_agent (so this drops into
    build_agent_runners()-style wiring identically) but isn't actually
    used, since this variant has no features to read from it.
    """
    if model is None:
        model = train_historical_mean_model(weekly_df, forecast_origin)

    structural_level = model.predict_for_msa(msa)

    msa_recent = weekly_df[
        (weekly_df["msa"] == msa) & (weekly_df["date"] <= pd.Timestamp(forecast_origin))
    ].sort_values("date")
    if msa_recent.empty:
        raise ValueError(f"No data for MSA '{msa}' before {forecast_origin}")
    last_inventory = float(msa_recent["inventory_count"].iloc[-1])

    TOTAL_REVERSION = 0.30
    forecast_values = []
    for step in range(1, horizon_weeks + 1):
        reversion_frac = TOTAL_REVERSION * (step / horizon_weeks) ** 2
        projected = last_inventory + reversion_frac * (structural_level - last_inventory)
        forecast_values.append(max(0.0, round(projected, 2)))

    return AgentOutput(
        agent_name="intrinsic",
        msa=msa,
        forecast_origin=forecast_origin,
        horizon_weeks=horizon_weeks,
        values=forecast_values,
    )
