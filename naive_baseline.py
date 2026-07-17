"""Naive last-value baseline: the minimum bar every other forecasting agent
(AR, macro, fusion, ...) must beat. Forecast = the last observed value,
repeated for every week of the horizon.

Output format decision: `forecast` is stored as a single list/array-valued
column (length == horizon) rather than exploded into forecast_week_1..N
columns. Horizons differ (4/8/13), so exploding would leave NaN-padded
columns for the shorter-horizon rows; a list column keeps every row dense
and also matches the AR agent's point_forecast: list[float] schema
(Step 3), which matters once the fusion harness compares agents directly.
"""
import numpy as np
import pandas as pd


def forecast_naive(series: pd.Series, horizon_weeks: int) -> np.ndarray:
    """Repeat the series' last value horizon_weeks times.

    Assumes `series` has already been truncated to the correct point-in-time
    training window by the caller -- this function has no notion of fold
    boundaries, it just reads the last value it's given.
    """
    last_value = float(series.iloc[-1])
    return np.full(horizon_weeks, last_value, dtype=float)


def assert_no_lookahead(series: pd.Series, as_of) -> None:
    """Raise if series contains any date after as_of.

    Defense-in-depth: run_naive_baseline already slices each MSA's series
    down to fold.train_end before calling this, so under normal operation
    this should never trigger. It exists to fail loudly if a future change
    accidentally passes an untruncated series instead of silently
    forecasting from leaked future data.
    """
    if len(series) and series.index.max() > pd.Timestamp(as_of):
        raise ValueError(
            f"lookahead violation: series contains data as late as "
            f"{series.index.max().date()}, which is after as_of={pd.Timestamp(as_of).date()}"
        )


def run_naive_baseline(df: pd.DataFrame, fold_boundaries: pd.DataFrame,
                        horizons: list = [4, 8, 13]) -> pd.DataFrame:
    """One row per (msa, fold_index, horizon), forecasting from the last
    value observed in that fold's training window (data up to and
    including fold.train_end -- never the test window).
    """
    rows = []
    for msa, msa_df in df.groupby("msa"):
        msa_series = msa_df.sort_values("date").set_index("date")["inventory_count"]

        for fold in fold_boundaries.itertuples():
            train_end = pd.Timestamp(fold.train_end)
            train_window = msa_series[msa_series.index <= train_end]

            assert_no_lookahead(train_window, train_end)
            if train_window.empty:
                raise ValueError(f"no training data for msa={msa} on/before {train_end.date()}")

            last_training_value = float(train_window.iloc[-1])

            for horizon in horizons:
                forecast = forecast_naive(train_window, horizon)
                rows.append({
                    "msa": msa,
                    "fold_index": fold.fold,
                    "horizon": horizon,
                    "train_end": train_end.date(),
                    "last_training_value": last_training_value,
                    "forecast": forecast.tolist(),
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from fold_boundaries import build_fold_table

    df = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
    folds = build_fold_table()

    result = run_naive_baseline(df, folds)
    print(f"run_naive_baseline output: {len(result)} rows "
          f"({df['msa'].nunique()} MSAs x {len(folds)} folds x {len([4, 8, 13])} horizons)")

    sample_msa = df["msa"].iloc[0]
    print(f"\nSample MSA: {sample_msa}")
    print(result[result["msa"] == sample_msa].to_string(index=False))
