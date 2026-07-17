"""Schema-valid stub for run_ar_agent, to unblock the fusion-harness build
against the AR Agent's output shape before the real ETS model is fully
validated. Forecasts here are fabricated (last_training_value + small
noise per horizon week, clipped non-negative) -- shape/type-correct only,
not accurate. Do not use for anything but harness development.
"""
import numpy as np
import pandas as pd

from naive_baseline import assert_no_lookahead

NOISE_STD_FRACTION = 0.02  # +/- ~2% of last_training_value per horizon week
_RNG_SEED = 42  # fixed seed so stub output is reproducible run-to-run


def run_ar_agent_stub(df: pd.DataFrame, fold_boundaries: pd.DataFrame,
                       horizons: list = [4, 8, 13]) -> pd.DataFrame:
    rng = np.random.default_rng(_RNG_SEED)
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
                noise = rng.normal(0, last_training_value * NOISE_STD_FRACTION, size=horizon)
                forecast = np.clip(last_training_value + noise, 0, None).tolist()
                rows.append({
                    "msa": msa,
                    "fold_index": fold.fold,
                    "horizon": horizon,
                    "train_end": train_end.date(),
                    "last_training_value": last_training_value,
                    "forecast": forecast,
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from fold_boundaries import build_fold_table

    df = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
    folds = build_fold_table()

    result = run_ar_agent_stub(df, folds)
    print(f"run_ar_agent_stub output: {len(result)} rows")
    print(result.head(6).to_string(index=False))
