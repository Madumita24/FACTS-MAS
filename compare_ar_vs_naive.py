"""Compare the AR Agent (ETS) against the naive last-value baseline.

Forecast origin for each (msa, fold) is that fold's train_end -- the same
point-in-time cutoff both agents trained up to. Actual values are looked
up at train_end + 1..horizon weeks directly from aligned_weekly.csv.

This is a direct comparison script, not the official 6-fold backtest from
evaluation_protocol.md: it evaluates every fold's forecast starting right
at train_end (including horizons that land inside that fold's embargo
gap), which is fine here since the embargo exists to keep training-label
construction from leaking future data, not to forbid checking a forecast
against dates that have already occurred by now.
"""
import numpy as np
import pandas as pd

from ar_agent import run_ar_agent
from fold_boundaries import build_fold_table
from naive_baseline import run_naive_baseline

HORIZONS = [4, 8, 13]


def actual_values(df: pd.DataFrame, msa: str, train_end, horizon: int) -> np.ndarray:
    msa_series = df[df["msa"] == msa].set_index("date")["inventory_count"]
    target_dates = [pd.Timestamp(train_end) + pd.Timedelta(weeks=w) for w in range(1, horizon + 1)]
    return msa_series.reindex(target_dates).to_numpy()


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast) / np.abs(actual)) * 100)


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def score_rows(rows: pd.DataFrame, df: pd.DataFrame, label: str) -> pd.DataFrame:
    scored = []
    for row in rows.itertuples():
        actual = actual_values(df, row.msa, row.train_end, row.horizon)
        forecast = np.array(row.forecast)

        if np.isnan(actual).any():
            continue  # target dates ran past available data; skip rather than fabricate a score

        scored.append({
            "msa": row.msa,
            "fold_index": row.fold_index,
            "horizon": row.horizon,
            f"mape_{label}": mape(actual, forecast),
            f"rmse_{label}": rmse(actual, forecast),
        })
    return pd.DataFrame(scored)


def main():
    df = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
    folds = build_fold_table()

    naive_forecasts = run_naive_baseline(df, folds, horizons=HORIZONS)
    ar_forecasts = run_ar_agent(df, folds, horizons=HORIZONS)

    naive_scores = score_rows(naive_forecasts, df, "naive")
    ar_scores = score_rows(ar_forecasts, df, "ar")

    merged = naive_scores.merge(ar_scores, on=["msa", "fold_index", "horizon"])

    print("=" * 70)
    for horizon in HORIZONS:
        h = merged[merged["horizon"] == horizon]
        per_msa = h.groupby("msa")[["mape_naive", "mape_ar"]].mean()
        per_msa["ar_wins"] = per_msa["mape_ar"] < per_msa["mape_naive"]
        per_msa["improvement_pp"] = per_msa["mape_naive"] - per_msa["mape_ar"]

        n_wins = int(per_msa["ar_wins"].sum())
        avg_improvement = per_msa["improvement_pp"].mean()

        print(f"\nHorizon = {horizon} weeks")
        print(f"  AR beats naive on MAPE for {n_wins}/15 MSAs")
        print(f"  Average MAPE improvement (naive - ar) across all 15 MSAs: {avg_improvement:+.3f} pp")
        print(f"  Overall MAPE  -- naive: {h['mape_naive'].mean():.3f}%   ar: {h['mape_ar'].mean():.3f}%")
        print(f"  Overall RMSE  -- naive: {h['rmse_naive'].mean():.1f}     ar: {h['rmse_ar'].mean():.1f}")
        print(per_msa.sort_values("improvement_pp", ascending=False).to_string(
            float_format=lambda x: f"{x:.3f}"))
    print("=" * 70)


if __name__ == "__main__":
    main()
