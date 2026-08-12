"""
Backtest Harness — FACTS-MAS Phase 2.

6-fold expanding-window backtest per evaluation_protocol.md.

Fold table -- CORRECTED to the Jan-2026 live-eval cutoff (Prof. Pan Round-2
feedback). The table below supersedes evaluation_protocol.md's own fold
table, which still reflects a since-superseded Jan-2025 cutoff assumption;
that doc's update is deliberately deferred until all agents are validated
against these dates (see progress.md) -- this file is the actual source of
truth for backtest execution in the meantime.

Fold table (from fold_boundaries.py, Jan-2026 cutoff):
    | Fold | Train start | Train end  | Test start | Test end   |
    |------|-------------|------------|------------|------------|
    | 1    | 2019-02-02  | 2021-01-23 | 2021-05-01 | 2021-11-13 |
    | 2    | 2019-02-02  | 2021-11-13 | 2022-02-19 | 2022-09-03 |
    | 3    | 2019-02-02  | 2022-09-03 | 2022-12-10 | 2023-06-24 |
    | 4    | 2019-02-02  | 2023-06-24 | 2023-09-30 | 2024-04-13 |
    | 5    | 2019-02-02  | 2024-04-13 | 2024-07-20 | 2025-02-01 |
    | 6    | 2019-02-02  | 2025-02-01 | 2025-05-10 | 2025-11-22 |

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
             datetime.date(2021, 5, 1), datetime.date(2021, 11, 13)),
    FoldSpec(2, datetime.date(2019, 2, 2), datetime.date(2021, 11, 13),
             datetime.date(2022, 2, 19), datetime.date(2022, 9, 3)),
    FoldSpec(3, datetime.date(2019, 2, 2), datetime.date(2022, 9, 3),
             datetime.date(2022, 12, 10), datetime.date(2023, 6, 24)),
    FoldSpec(4, datetime.date(2019, 2, 2), datetime.date(2023, 6, 24),
             datetime.date(2023, 9, 30), datetime.date(2024, 4, 13)),
    FoldSpec(5, datetime.date(2019, 2, 2), datetime.date(2024, 4, 13),
             datetime.date(2024, 7, 20), datetime.date(2025, 2, 1)),
    FoldSpec(6, datetime.date(2019, 2, 2), datetime.date(2025, 2, 1),
             datetime.date(2025, 5, 10), datetime.date(2025, 11, 22)),
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

WEIGHTING_MODES = ("fold", "regime", "adaptive", "adaptive_granular")


def _select_weights(
    weighting_mode: str,
    horizon: int,
    regime: str,
    fold_weights: dict,
    regime_grid: Optional[dict],
    mode_selection: dict,
) -> dict:
    """
    Which weight set applies to one (horizon, regime) cell.

    "fold"      one set per fold, regime-blind (the original behaviour).
    "regime"    one set per (regime, horizon), looked up per origin.
    "adaptive"  per (fold, horizon), whichever of the two scored lower on
                that fold's own validation window.
    "adaptive_granular"
                same idea, decided per (fold, horizon, regime).

    Selection always happens on validation data (see run_backtest), never on
    the test window, so the choice itself cannot leak into the held-out
    evaluation.
    """
    if weighting_mode == "fold":
        return fold_weights
    if weighting_mode == "regime":
        return regime_grid[regime][f"{horizon}wk"]["weights"]
    key = horizon if weighting_mode == "adaptive" else (horizon, regime)
    if mode_selection.get(key, "fold") == "regime":
        return regime_grid[regime][f"{horizon}wk"]["weights"]
    return fold_weights


def run_backtest(
    df: pd.DataFrame,
    agent_runners: dict[str, callable],
    fusion_fn: callable,
    msas: Optional[list[str]] = None,
    folds: Optional[list[FoldSpec]] = None,
    horizons: Optional[list[int]] = None,
    weighting_mode: str = "adaptive",
) -> list[BacktestResult]:
    """
    Execute the 6-fold expanding-window backtest.

    For each fold:
        1. Validate the 13-week embargo.
        2. Compute per-fold fusion weights from the training window.
        3. If the mode needs them, compute the (regime x horizon) weight
           grid and run the mode-selection comparison -- both on the same
           104-week validation window, never on test data.
        4. For each MSA and horizon:
            a. Forecast at every weekly origin in the test window.
            b. Classify the regime, pick the weight set for that cell, fuse.
            c. Compute MAPE and RMSE, grouped by regime.
        5. Log fold-to-fold weight evolution.

    Args:
        df: aligned_weekly.csv with 'date' as datetime.
        agent_runners: {agent_name: callable(df, msa, origin, horizon) -> AgentOutput}
        fusion_fn: callable(agent_outputs, weights) -> fused_forecast
        msas: List of MSAs (defaults to all 15).
        folds: Fold specs (defaults to FOLDS).
        horizons: Horizons to evaluate (defaults to [4, 8, 13]).
        weighting_mode: One of WEIGHTING_MODES. Defaults to "adaptive", the
            production default per factor_attribution.md. Fold-level
            weighting alone loses to naive at 13wk during hiking and
            cutting, which is the failure the regime-aware work exists to
            fix. "adaptive_granular" is fully implemented but deliberately
            NOT the default: it fixes 8wk/hiking and regresses three other
            cells, because rare-regime validation slices can come back
            empty and fall back to "fold". See factor_attribution.md
            sections 3 and 9.

    Returns:
        List of BacktestResult objects.
    """
    if msas is None:
        msas = sorted(df["msa"].unique().tolist())
    if folds is None:
        folds = FOLDS
    if horizons is None:
        horizons = HORIZONS
    if weighting_mode not in WEIGHTING_MODES:
        raise ValueError(
            f"weighting_mode must be one of {WEIGHTING_MODES}, "
            f"got '{weighting_mode}'"
        )

    results = []
    weight_log = []

    for fold in folds:
        # 1. Validate embargo
        validate_embargo(fold)

        # 2. Compute fold weights
        # 104-week lookback aggregated across all 3 horizons, p=2 weighting --
        # see fusion_baseline.py's module docstring for why the original
        # 20-week/8wk-only/p=1 design couldn't separate Macro's validated
        # null from agents with real signal.
        from facts_mas.fusion_baseline import compute_fold_weights
        fold_weights = compute_fold_weights(
            df, agent_runners, msas,
            train_end=fold.train_end,
        )
        weight_log.append({"fold": fold.fold, "weights": fold_weights})
        print(f"Fold {fold.fold} weights ({weighting_mode}): {fold_weights}")

        # 3. Regime grid + mode selection, validation window only. Skipped
        #    entirely for "fold" mode so the original path pays none of it.
        regime_grid = None
        mode_selection: dict = {}
        if weighting_mode != "fold":
            from facts_mas.factor_attribution import (
                compute_all_regime_weights,
                evaluate_validation_mape,
            )
            regime_grid = compute_all_regime_weights(
                df, agent_runners, msas, fold
            )["grid"]

            if weighting_mode == "adaptive":
                for horizon in horizons:
                    fm, rm = evaluate_validation_mape(
                        df, agent_runners, msas, fold, horizon,
                        fold_weights, regime_grid,
                    )
                    # Ties and non-finite comparisons fall back to "fold".
                    mode_selection[horizon] = "fold" if fm <= rm else "regime"
                    print(f"  fold {fold.fold} h={horizon}: fold={fm:.3f} "
                          f"regime={rm:.3f} -> {mode_selection[horizon]}")

            elif weighting_mode == "adaptive_granular":
                for horizon in horizons:
                    for regime in ("hiking", "cutting", "stable"):
                        fm, rm = evaluate_validation_mape(
                            df, agent_runners, msas, fold, horizon,
                            fold_weights, regime_grid, regime_filter=regime,
                        )
                        key = (horizon, regime)
                        mode_selection[key] = "fold" if fm <= rm else "regime"
                        print(f"  fold {fold.fold} h={horizon} {regime}: "
                              f"fold={fm:.3f} regime={rm:.3f} "
                              f"-> {mode_selection[key]}")

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
                            # Agent failed - skip this origin
                            break
                    else:
                        # All agents succeeded. Regime is classified BEFORE
                        # fusing now, because every mode except "fold" needs
                        # it to choose the weight set.
                        regime = classify_regime(df, msa, origin)
                        weights = _select_weights(
                            weighting_mode, horizon, regime,
                            fold_weights, regime_grid, mode_selection,
                        )

                        fused = fusion_fn(agent_outputs, weights)
                        fused_arr = np.array(fused, dtype=np.float64)

                        mape = compute_mape(actuals, fused_arr)
                        rmse = compute_rmse(actuals, fused_arr)

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
                            # The weights actually used for this cell, which
                            # under a regime-aware mode is not fold_weights.
                            agent_weights=_select_weights(
                                weighting_mode, horizon, regime,
                                fold_weights, regime_grid, mode_selection,
                            ),
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


def _make_intrinsic_runner(static_df: pd.DataFrame) -> callable:
    """run_intrinsic_agent takes an extra static_df positional arg (weekly_df,
    static_df, msa, forecast_origin, horizon_weeks) that doesn't fit the
    (df, msa, origin, horizon) convention every other runner in agent_runners
    uses. functools.partial can't fix this by itself since static_df sits in
    the middle of the positional order, not the end -- so a small explicit
    closure is needed to reorder the call. Not a change to intrinsic_agent.py
    itself, just glue at the call site (same kind of adapter as
    _fusion_fn_adapter below, for the same reason: bridging calling
    conventions, not changing model logic).
    """
    from facts_mas.agents.intrinsic_agent import run_intrinsic_agent

    def _runner(df, msa, forecast_origin, horizon_weeks):
        return run_intrinsic_agent(df, static_df, msa, forecast_origin, horizon_weeks)

    return _runner


def _fusion_fn_adapter(agent_outputs: dict, weights: dict) -> list:
    """run_backtest() calls fusion_fn(agent_outputs, weights) with a raw
    dict[str, AgentOutput] -- but fusion_baseline.fuse_forecasts expects a
    validated FusionInput, not a raw dict. This is a real (minor) signature
    mismatch between the two modules, not something to silently paper over:
    bridged here with an adapter rather than by modifying either module, per
    "confirm they work without modification."
    """
    from facts_mas.fusion_baseline import fuse_forecasts
    from facts_mas.schema import FusionInput

    sample = next(iter(agent_outputs.values()))
    fusion_input = FusionInput(
        msa=sample.msa,
        forecast_origin=sample.forecast_origin,
        horizon_weeks=sample.horizon_weeks,
        agent_outputs=agent_outputs,
    )
    return fuse_forecasts(fusion_input, weights)


def build_agent_runners() -> dict:
    """All 5 agents, each matching the (df, msa, forecast_origin, horizon_weeks)
    -> AgentOutput calling convention agent_runners requires."""
    from facts_mas.agents.ar_agent import run_ar_agent
    from facts_mas.agents.event_agent import run_event_agent
    from facts_mas.agents.macro_agent import run_macro_agent
    from facts_mas.agents.seasonality_agent import run_seasonality_agent

    static_df = pd.read_csv("intrinsic_static.csv")

    return {
        "ar": run_ar_agent,
        "macro": run_macro_agent,
        "event": run_event_agent,
        "seasonality": run_seasonality_agent,
        "intrinsic": _make_intrinsic_runner(static_df),
    }


if __name__ == "__main__":
    import argparse
    import sys

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, ".")

    parser = argparse.ArgumentParser()
    parser.add_argument("--msas", type=str, default=None,
                         help="Comma-separated MSA subset (default: all 15)")
    parser.add_argument("--folds", type=str, default=None,
                         help="Comma-separated fold indices, e.g. '1' or '1,2' (default: all 6)")
    parser.add_argument("--weighting-mode", type=str, default="adaptive",
                         choices=list(WEIGHTING_MODES),
                         help="Fusion weighting strategy (default: adaptive)")
    args = parser.parse_args()

    weekly_df = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])

    msas = args.msas.split(",") if args.msas else None
    folds = [f for f in FOLDS if f.fold in {int(x) for x in args.folds.split(",")}] if args.folds else None

    print(f"Running backtest: msas={msas or 'all 15'}, "
           f"folds={[f.fold for f in folds] if folds else 'all 6'}, "
           f"weighting_mode={args.weighting_mode}")

    agent_runners = build_agent_runners()
    results = run_backtest(weekly_df, agent_runners, _fusion_fn_adapter,
                            msas=msas, folds=folds,
                            weighting_mode=args.weighting_mode)

    print(f"\n{len(results)} BacktestResult rows produced.")
    for r in results[:10]:
        print(f"  fold={r.fold} msa={r.msa} horizon={r.horizon_weeks} regime={r.regime} "
              f"mape={r.mape:.2f} rmse={r.rmse:.1f} n_origins={r.n_origins}")
