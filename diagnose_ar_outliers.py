"""Diagnostic only -- does not change ar_agent.py. Investigates why Miami
and Phoenix were the two MSAs where ETS underperformed the naive baseline
in compare_ar_vs_naive.py (Miami at all 3 horizons, Phoenix at 13 weeks).

Checks two distinct hypotheses:
  1. "Volatile/choppy series" -- high variance and frequent direction
     changes, where any trend-extrapolating model would struggle.
  2. "ETS overshooting a trend reversal" -- ETS's fitted trend slope points
     one way at train_end, but the series actually reverses direction
     within the forecast window.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from fold_boundaries import build_fold_table

TARGET_MSAS = ["Miami", "Phoenix"]
ROLLING_WEEKS = 13


def trend_reversal_count(series: pd.Series, window: int = ROLLING_WEEKS) -> int:
    """Number of times the rolling-mean direction flips sign over the series."""
    rolling_mean = series.rolling(window).mean().dropna()
    diffs = rolling_mean.diff().dropna()
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0
    return int((signs != signs.shift()).sum() - 1)  # first observation isn't a reversal


def ets_trend_slope(series: pd.Series) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(series, trend="add", seasonal=None, initialization_method="estimated")
        fitted = model.fit()
    return float(fitted.trend.iloc[-1])


def main():
    df = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
    df = df[df["date"] < "2026-01-01"]
    folds = build_fold_table()

    disagreements = {}

    for msa in TARGET_MSAS:
        series = df[df["msa"] == msa].sort_values("date").set_index("date")["inventory_count"]

        print("=" * 70)
        print(msa)
        print(f"  range: {series.index.min().date()} to {series.index.max().date()}  (n={len(series)})")
        print(f"  mean: {series.mean():.1f}   std: {series.std():.1f}   cv: {series.std() / series.mean():.3f}")
        print(f"  trend-reversal count (13wk rolling-mean direction changes, full series): "
              f"{trend_reversal_count(series)}")

        print(f"\n  ETS trend slope at each fold's train_end vs. what actually happened next:")
        disagreement_count = 0
        for fold in folds.itertuples():
            train_end = pd.Timestamp(fold.train_end)
            train_window = series[series.index <= train_end]
            slope = ets_trend_slope(train_window)

            future = series[(series.index > train_end) & (series.index <= train_end + pd.Timedelta(weeks=13))]
            if len(future) == 13:
                actual_change = future.iloc[-1] - train_window.iloc[-1]
                sign_agree = np.sign(slope) == np.sign(actual_change)
                if not sign_agree:
                    disagreement_count += 1
                flag = "" if sign_agree else "  <-- ETS slope sign disagrees with actual 13wk move"
                print(f"    fold {fold.fold}  train_end={train_end.date()}  "
                      f"ets_slope/wk={slope:+8.2f}   actual_13wk_change={actual_change:+8.1f}{flag}")
            else:
                print(f"    fold {fold.fold}  train_end={train_end.date()}  "
                      f"ets_slope/wk={slope:+8.2f}   actual_13wk_change=n/a (insufficient future data)")

        disagreements[msa] = disagreement_count
        print()

    print("=" * 70)
    print("Conclusion:")
    for msa in TARGET_MSAS:
        n = disagreements[msa]
        if n >= 3:
            verdict = "ETS specifically overshooting trend reversals"
        else:
            verdict = "volatile/choppy series where any trend model would struggle"
        print(f"  {msa}: sign-disagreement in {n}/6 folds -> {verdict}")


if __name__ == "__main__":
    main()
