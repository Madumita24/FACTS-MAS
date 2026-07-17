"""AR Agent: per-MSA ETS (Holt-Winters, additive trend, no seasonality)
inventory forecaster. Seasonality is a separate agent owned by a teammate,
so it's deliberately left out here rather than approximated.

Output schema matches naive_baseline.py's run_naive_baseline exactly
(msa, fold_index, horizon, forecast, train_end, last_training_value) so the
two are directly comparable row-for-row -- see compare_ar_vs_naive.py.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from naive_baseline import assert_no_lookahead


def fit_and_forecast(series: pd.Series, horizon_weeks: int) -> dict:
    """Fit ETS (additive trend, no seasonal component) and forecast ahead.

    Non-negativity (clip at 0) is enforced here as a temporary local
    safeguard against ETS's additive-trend extrapolation going negative on
    a declining series -- it is NOT a substitute for the boundary-constraint
    layer owned downstream; that layer should still validate/own this
    concern for the fused output.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated")
        fitted = model.fit()
        raw_forecast = fitted.forecast(horizon_weeks)

    point_forecast = np.clip(raw_forecast.to_numpy(), 0, None).tolist()

    return {
        "model_name": "ets",
        "point_forecast": point_forecast,
        "prediction_intervals": None,
    }


def run_ar_agent(df: pd.DataFrame, fold_boundaries: pd.DataFrame,
                  horizons: list = [4, 8, 13]) -> pd.DataFrame:
    """One row per (msa, fold_index, horizon), fitting ETS only on data up
    to and including fold.train_end -- never the test window. Refits per
    horizon (per the fit_and_forecast(series, horizon_weeks) signature)
    rather than fitting once at max horizon and slicing, trading a bit of
    redundant computation for a simpler, spec-matching interface.
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
                result = fit_and_forecast(train_window, horizon)
                rows.append({
                    "msa": msa,
                    "fold_index": fold.fold,
                    "horizon": horizon,
                    "train_end": train_end.date(),
                    "last_training_value": last_training_value,
                    "forecast": result["point_forecast"],
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from fold_boundaries import build_fold_table

    df = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
    folds = build_fold_table()

    result = run_ar_agent(df, folds)
    print(f"run_ar_agent output: {len(result)} rows "
          f"({df['msa'].nunique()} MSAs x {len(folds)} folds x {len([4, 8, 13])} horizons)")

    sample_msa = df["msa"].iloc[0]
    print(f"\nSample MSA: {sample_msa}")
    print(result[result["msa"] == sample_msa].to_string(index=False))
