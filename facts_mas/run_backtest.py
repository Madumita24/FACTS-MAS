"""
Backtest Harness — FACTS-MAS Phase 2.

6-fold expanding-window backtest per evaluation_protocol.md.

Fold table (from evaluation_protocol.md):
    | Fold | Train start | Train end  | Test start | Test end   |
    |------|-------------|------------|------------|------------|
    | 1    | 2019-02-02  | 2021-01-23 | 2021-05-01 | 2021-09-11 |
    | 2    | 2019-02-02  | 2021-09-11 | 2021-12-18 | 2022-04-30 |
    | 3    | 2019-02-02  | 2022-04-30 | 2022-08-06 | 2022-12-17 |
    | 4    | 2019-02-02  | 2022-12-17 | 2023-03-25 | 2023-08-05 |
    | 5    | 2019-02-02  | 2023-08-05 | 2023-11-11 | 2024-03-23 |
    | 6    | 2019-02-02  | 2024-03-23 | 2024-06-29 | 2024-11-09 |

Key constraints:
    - 13-week embargo between train_end and test_start per fold.
    - Training data bounded to pre-Jan-2026 (Prof. Pan Round-2).
    - Horizons: 4, 8, 13 weeks — reported separately, never averaged.
    - Regime-stratified: hiking/cutting/stable from Fed funds 13-week change.
    - Fusion weights logged per fold to confirm they change (not frozen).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from facts_mas.schema import AgentOutput, BacktestResult


# ═══════════════════════════════════════════════════════════════════════════
# Fold definitions (from evaluation_protocol.md)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FoldSpec:
    fold: int
    train_start: datetime.date
    train_end: datetime.date
    test_start: datetime.date
    test_end: datetime.date


FOLDS = [
    FoldSpec(1, datetime.date(2019, 2, 2), datetime.date(2021, 1, 23),
             datetime.date(2021, 5, 1), datetime.date(2021, 9, 11)),
    FoldSpec(2, datetime.date(2019, 2, 2), datetime.date(2021, 9, 11),
             datetime.date(2021, 12, 18), datetime.date(2022, 4, 30)),
    FoldSpec(3, datetime.date(2019, 2, 2), datetime.date(2022, 4, 30),
             datetime.date(2022, 8, 6), datetime.date(2022, 12, 17)),
    FoldSpec(4, datetime.date(2019, 2, 2), datetime.date(2022, 12, 17),
             datetime.date(2023, 3, 25), datetime.date(2023, 8, 5)),
    FoldSpec(5, datetime.date(2019, 2, 2), datetime.date(2023, 8, 5),
             datetime.date(2023, 11, 11), datetime.date(2024, 3, 23)),
    FoldSpec(6, datetime.date(2019, 2, 2), datetime.date(2024, 3, 23),
             datetime.date(2024, 6, 29), datetime.date(2024, 11, 9)),
]

HORIZONS = [4, 8, 13]


# ═══════════════════════════════════════════════════════════════════════════
# Regime classification (from evaluation_protocol.md)
# ═══════════════════════════════════════════════════════════════════════════

def classify_regime(
    df: pd.DataFrame, msa: str, origin_date: datetime.date
) -> str:
    """
    Classify the Fed funds regime at a forecast origin.

    From evaluation_protocol.md:
        - hiking:  fed_funds[t] - fed_funds[t-13wk] > +0.25
        - cutting: fed_funds[t] - fed_funds[t-13wk] < -0.25
        - stable:  everything in between

    Uses the regime at the time the forecast was made, not the target date.
    """
    msa_data = df[df["msa"] == msa].sort_values("date")

    current = msa_data[msa_data["date"] <= pd.Timestamp(origin_date)]
    if len(current) < 14:
        return "stable"  # Not enough history

    current_ff = float(current["fed_funds"].iloc[-1])
    past_ff = float(current["fed_funds"].iloc[-14])  # ~13 weeks back
    delta = current_ff - past_ff

    if delta > 0.25:
        return "hiking"
    elif delta < -0.25:
        return "cutting"
    else:
        return "stable"


# ═══════════════════════════════════════════════════════════════════════════
# Embargo validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_embargo(fold: FoldSpec) -> None:
    """
    Assert the 13-week (91-day) embargo between train_end and test_start.
    """
    gap = (fold.test_start - fold.train_end).days
    if gap < 91:
        raise ValueError(
            f"Fold {fold.fold}: embargo violation. "
            f"Gap between train_end ({fold.train_end}) and "
            f"test_start ({fold.test_start}) is {gap} days, "
            f"need >= 91 (13 weeks)."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Metric computation (pure NumPy)
# ═══════════════════════════════════════════════════════════════════════════

def compute_mape(actuals: np.ndarray, forecast: np.ndarray) -> float:
    """MAPE = (100/n) * sum(|y - yhat| / |y|)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs((actuals - forecast) / actuals)
    ape = ape[np.isfinite(ape)]
    return float(100.0 * np.mean(ape)) if len(ape) > 0 else float("inf")


def compute_rmse(actuals: np.ndarray, forecast: np.ndarray) -> float:
    """RMSE = sqrt(mean((y - yhat)^2))"""
    return float(np.sqrt(np.mean((actuals - forecast) ** 2)))


# ═══════════════════════════════════════════════════════════════════════════
# Naive last-value baseline (acceptance bar comparison)
# ═══════════════════════════════════════════════════════════════════════════

def naive_forecast(last_value: float, horizon: int) -> list[float]:
    """
    Naive last-value baseline: repeat the last observed inventory.

    The fused 4-agent forecast must beat this on at least 2 of 3 horizons.
    """
    return [last_value] * horizon


# ═══════════════════════════════════════════════════════════════════════════
# Main backtest loop
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest(
    df: pd.DataFrame,
    agent_runners: dict[str, callable],
    fusion_fn: callable,
    msas: Optional[list[str]] = None,
    folds: Optional[list[FoldSpec]] = None,
    horizons: Optional[list[int]] = None,
) -> list[BacktestResult]:
    """
    Execute the 6-fold expanding-window backtest.

    For each fold:
        1. Validate the 13-week embargo.
        2. Compute per-fold fusion weights from the training window.
        3. For each MSA and horizon:
            a. Generate forecasts at every weekly origin in the test window.
            b. Compute MAPE and RMSE.
            c. Classify the regime at each origin.
        4. Log fold-to-fold weight evolution.

    Args:
        df: aligned_weekly.csv with 'date' as datetime.
        agent_runners: {agent_name: callable(df, msa, origin, horizon) -> AgentOutput}
        fusion_fn: callable(agent_outputs, weights) -> fused_forecast
        msas: List of MSAs (defaults to all 15).
        folds: Fold specs (defaults to FOLDS).
        horizons: Horizons to evaluate (defaults to [4, 8, 13]).

    Returns:
        List of BacktestResult objects.
    """
    if msas is None:
        msas = sorted(df["msa"].unique().tolist())
    if folds is None:
        folds = FOLDS
    if horizons is None:
        horizons = HORIZONS

    results = []
    weight_log = []

    for fold in folds:
        # 1. Validate embargo
        validate_embargo(fold)

        # 2. Compute fold weights
        # Use the last 20 weeks of training data as validation for weighting
        val_window_start = fold.train_end - datetime.timedelta(weeks=20)
        from facts_mas.fusion_baseline import compute_fold_weights
        weights = compute_fold_weights(
            df, agent_runners, msas,
            val_window_start, fold.train_end,
            horizon_weeks=8,  # Use 8-week horizon for weight calibration
        )
        weight_log.append({"fold": fold.fold, "weights": weights})
        print(f"Fold {fold.fold} weights: {weights}")

        for msa in msas:
            msa_data = df[df["msa"] == msa].sort_values("date")

            for horizon in horizons:
                # Get all weekly origins in the test window
                test_origins = msa_data[
                    (msa_data["date"] >= pd.Timestamp(fold.test_start))
                    & (msa_data["date"] <= pd.Timestamp(fold.test_end))
                ]["date"].values

                all_mapes = []
                all_rmses = []
                regimes = []

                for origin_ts in test_origins:
                    origin = pd.Timestamp(origin_ts).date()

                    # Get actuals
                    future = msa_data[
                        msa_data["date"] > pd.Timestamp(origin)
                    ].head(horizon)

                    if len(future) < horizon:
                        continue

                    actuals = future["inventory_count"].values.astype(np.float64)

                    # Run each agent and fuse
                    agent_outputs = {}
                    for agent_name, runner in agent_runners.items():
                        try:
                            output = runner(df, msa, origin, horizon)
                            agent_outputs[agent_name] = output
                        except Exception as e:
                            # Agent failed — skip this origin
                            break
                    else:
                        # All agents succeeded
                        fused = fusion_fn(agent_outputs, weights)
                        fused_arr = np.array(fused, dtype=np.float64)

                        mape = compute_mape(actuals, fused_arr)
                        rmse = compute_rmse(actuals, fused_arr)
                        regime = classify_regime(df, msa, origin)

                        all_mapes.append(mape)
                        all_rmses.append(rmse)
                        regimes.append(regime)

                if all_mapes:
                    # Group by regime
                    for regime in set(regimes):
                        regime_mask = [r == regime for r in regimes]
                        regime_mapes = [
                            m for m, mask in zip(all_mapes, regime_mask) if mask
                        ]
                        regime_rmses = [
                            r for r, mask in zip(all_rmses, regime_mask) if mask
                        ]

                        results.append(BacktestResult(
                            fold=fold.fold,
                            msa=msa,
                            horizon_weeks=horizon,
                            mape=float(np.mean(regime_mapes)),
                            rmse=float(np.mean(regime_rmses)),
                            regime=regime,
                            n_origins=len(regime_mapes),
                            agent_weights=weights,
                        ))

    # Confirm weights changed across folds (not frozen)
    if len(weight_log) >= 2:
        w1 = weight_log[0]["weights"]
        w_last = weight_log[-1]["weights"]
        if w1 == w_last:
            print("WARNING: Fold weights did not change between fold 1 and "
                  f"fold {weight_log[-1]['fold']}. Check if recomputation "
                  "is working correctly.")
        else:
            print("OK: Fold weights changed across folds (not frozen).")

    return results


if __name__ == "__main__":
    print("Backtest harness loaded. Run via: python -m facts_mas.run_backtest")
