import numpy as np
import pandas as pd
import pytest

from ar_agent import run_ar_agent
from compare_ar_vs_naive import actual_values, mape
from fold_boundaries import build_fold_table
from macro_agent import LAG_WEEKS, assert_macro_no_lookahead, fit_and_forecast, run_macro_agent


@pytest.fixture
def sample_df():
    dates = pd.date_range("2019-01-05", periods=250, freq="W-SAT")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "date": dates,
        "msa": "A",
        "inventory_count": 10000 + np.cumsum(rng.normal(0, 50, 250)),
        "mortgage_rate": 4 + np.cumsum(rng.normal(0, 0.02, 250)),
        "fed_funds": 2 + np.cumsum(rng.normal(0, 0.02, 250)),
        "cpi": 250 + np.cumsum(rng.normal(0, 0.1, 250)),
        "unemployment": 5 + np.cumsum(rng.normal(0, 0.05, 250)),
    })


@pytest.mark.parametrize("horizon", [4, 8, 13])
def test_modifier_length_matches_horizon(sample_df, horizon):
    train_end = sample_df["date"].iloc[200]
    result = fit_and_forecast(sample_df, "A", train_end, horizon)
    assert len(result["modifier"]) == horizon


def test_lag_weeks_used_matches_documented_fixed_lags(sample_df):
    train_end = sample_df["date"].iloc[200]
    result = fit_and_forecast(sample_df, "A", train_end, 4)
    assert result["lag_weeks_used"] == {
        "mortgage_rate": 6,
        "fed_funds": 10,
        "cpi": 10,
        "unemployment": 5,
    }


def test_no_lookahead_guard_raises_on_injected_future_date():
    # source_dates deliberately includes a date only 3 weeks before
    # fold_train_end, while lag_weeks=6 requires source data to be at
    # least 6 weeks old -- this must be caught.
    fold_train_end = pd.Timestamp("2022-06-04")
    source_dates = pd.DatetimeIndex([
        pd.Timestamp("2022-01-01"),
        pd.Timestamp("2022-05-14"),  # only 3 weeks before fold_train_end -- violates lag=6
    ])
    with pytest.raises(ValueError, match="lookahead violation"):
        assert_macro_no_lookahead(source_dates, fold_train_end, lag_weeks=6, feature_name="mortgage_rate")


def test_no_lookahead_guard_passes_at_exact_boundary():
    # source date exactly at fold_train_end - lag_weeks should NOT raise
    # (the guard is "later than", not "on or after").
    fold_train_end = pd.Timestamp("2022-06-04")
    source_dates = pd.DatetimeIndex([fold_train_end - pd.Timedelta(weeks=6)])
    assert_macro_no_lookahead(source_dates, fold_train_end, lag_weeks=6, feature_name="mortgage_rate")


def macro_marginal_contribution():
    """Not a pass/fail test -- computes and prints whether adjusting the AR
    forecast with the macro modifier helps, hurts, or is neutral vs. AR
    alone, per horizon, on the same folds used in compare_ar_vs_naive.py.

    Two combination rules compared (diagnostic only, NEITHER is the real
    fusion logic -- that's owned downstream):
      - compounding_combine: adjusted[week i] = ar_forecast[i] * (1+modifier)^(i+1)
        -- a small noisy single-week modifier compounds across the horizon.
      - additive_combine: adjusted = ar_forecast * (1+modifier), ONE
        multiplicative adjustment applied uniformly across the whole
        horizon, not repeated/compounded per week.
    Run side by side to see whether compounding is artificially amplifying
    macro's apparent harm.
    """
    df = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
    folds = build_fold_table()
    horizons = [4, 8, 13]

    ar_forecasts = run_ar_agent(df, folds, horizons=horizons)
    macro_forecasts = run_macro_agent(df, folds, horizons=horizons)

    merged = ar_forecasts.merge(
        macro_forecasts[["msa", "fold_index", "horizon", "modifier"]],
        on=["msa", "fold_index", "horizon"],
    )

    def compounding_combine(ar_forecast: np.ndarray, modifier: np.ndarray) -> np.ndarray:
        week_index = np.arange(1, len(ar_forecast) + 1)
        return ar_forecast * (1 + modifier) ** week_index

    def additive_combine(ar_forecast: np.ndarray, modifier: np.ndarray) -> np.ndarray:
        factor = 1 + modifier[-1]  # modifier is constant across the horizon in this v1 design
        return ar_forecast * factor

    print("=" * 70)
    for horizon in horizons:
        h = merged[merged["horizon"] == horizon]
        ar_mapes, compound_mapes, additive_mapes = [], [], []

        for row in h.itertuples():
            actual = actual_values(df, row.msa, row.train_end, row.horizon)
            if np.isnan(actual).any():
                continue

            ar_forecast = np.array(row.forecast)
            modifier = np.array(row.modifier)

            ar_mapes.append(mape(actual, ar_forecast))
            compound_mapes.append(mape(actual, compounding_combine(ar_forecast, modifier)))
            additive_mapes.append(mape(actual, additive_combine(ar_forecast, modifier)))

        ar_mape = float(np.mean(ar_mapes))
        compound_mape = float(np.mean(compound_mapes))
        additive_mape = float(np.mean(additive_mapes))
        compound_delta = ar_mape - compound_mape
        additive_delta = ar_mape - additive_mape

        def verdict(delta):
            if delta > 0.05:
                return "HELPS"
            elif delta < -0.05:
                return "HURTS"
            return "ROUGHLY NEUTRAL"

        print(f"Horizon = {horizon} weeks")
        print(f"  AR alone MAPE:                 {ar_mape:.3f}%")
        print(f"  AR + macro (compounding):      {compound_mape:.3f}%   "
              f"delta={compound_delta:+.3f}pp  -> {verdict(compound_delta)}")
        print(f"  AR + macro (additive, 1x):     {additive_mape:.3f}%   "
              f"delta={additive_delta:+.3f}pp  -> {verdict(additive_delta)}")
    print("=" * 70)


def test_macro_marginal_contribution_runs_without_error():
    macro_marginal_contribution()


if __name__ == "__main__":
    macro_marginal_contribution()
