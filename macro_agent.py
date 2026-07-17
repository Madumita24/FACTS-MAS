"""Macro Agent: produces a MODIFIER (not a full forecast) representing macro
pressure on inventory, via a ridge regression of weekly inventory pct-change
on lagged FRED features. Output matches Umang's MacroModifier schema
(model_name, modifier, lag_weeks_used) and joins to the AR output by
(msa, fold_index, horizon).

Design note -- constant modifier across the horizon: the only lag-consistent
macro snapshot available at fold_train_end is each feature's value from
lag_weeks ago (that's what "still working its way through the pipeline"
means). Predicting a *different* modifier for week 2, 3, ... of the horizon
would require future macro values that don't exist yet for the shorter-lag
features (unemployment's 5wk lag < the 13wk horizon). So this v1 predicts one
pct-change "pressure" value from the train_end snapshot and repeats it across
the horizon. Decaying/varying that pressure over the horizon is a candidate
refinement, not this step.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from naive_baseline import assert_no_lookahead

# Midpoints of the transmission-lag ranges documented as comments in
# align_data.py (mortgage ~4-8wk, fed_funds ~8-12wk, cpi ~8-12wk,
# unemployment ~4-6wk). These are reasonable starting points carried over
# from Phase 1's documented priors, NOT fitted/optimized values -- lag
# optimization (e.g. grid-searching lag per MSA/feature against backtest
# error) is a candidate for later refinement, not this step.
LAG_WEEKS = {
    "mortgage_rate": 6,
    "fed_funds": 10,
    "cpi": 10,
    "unemployment": 5,
}


def assert_macro_no_lookahead(source_dates, fold_train_end, lag_weeks: int, feature_name: str) -> None:
    """Raise if any macro source date used for `feature_name` is later than
    fold_train_end - lag_weeks. This is what makes the lag-shifting from
    Phase 1 (align_data.py's documented transmission lags) actually load-
    bearing here, rather than just a comment.
    """
    source_dates = pd.DatetimeIndex(source_dates).dropna()
    if len(source_dates) == 0:
        return
    max_allowed = pd.Timestamp(fold_train_end) - pd.Timedelta(weeks=lag_weeks)
    latest = source_dates.max()
    if latest > max_allowed:
        raise ValueError(
            f"lookahead violation: {feature_name} feature uses source data as late as "
            f"{latest.date()}, which is after fold_train_end - lag_weeks = {max_allowed.date()} "
            f"(fold_train_end={pd.Timestamp(fold_train_end).date()}, lag={lag_weeks}wk)"
        )


def _lagged_feature(history: pd.DataFrame, feature: str, lag_weeks: int):
    """Returns (lagged_series indexed by row date t, source_dates = t - lag_weeks)."""
    source_dates = history.index - pd.Timedelta(weeks=lag_weeks)
    lagged = pd.Series(
        history[feature].reindex(source_dates).to_numpy(),
        index=history.index, name=feature,
    )
    return lagged, source_dates


def fit_and_forecast(df: pd.DataFrame, msa: str, fold_train_end, horizon_weeks: int) -> dict:
    train_end = pd.Timestamp(fold_train_end)
    msa_df = df[df["msa"] == msa].sort_values("date").set_index("date")
    history = msa_df[msa_df.index <= train_end]

    if history.empty:
        raise ValueError(f"no training data for msa={msa} on/before {train_end.date()}")
    assert_no_lookahead(history["inventory_count"], train_end)

    lagged_cols = {}
    for feature, lag in LAG_WEEKS.items():
        lagged, source_dates = _lagged_feature(history, feature, lag)
        assert_macro_no_lookahead(source_dates, train_end, lag, feature)
        lagged_cols[feature] = lagged

    X = pd.DataFrame(lagged_cols, index=history.index)
    y = history["inventory_count"].pct_change().rename("y")

    train_frame = pd.concat([X, y], axis=1).dropna()
    if len(train_frame) < 10:
        raise ValueError(
            f"insufficient training rows ({len(train_frame)}) for msa={msa}, "
            f"fold ending {train_end.date()}"
        )

    model = Ridge(alpha=1.0)
    model.fit(train_frame[list(LAG_WEEKS)], train_frame["y"])

    X_forecast = X.loc[[train_end]]
    if X_forecast.isna().any(axis=None):
        raise ValueError(
            f"forecast-snapshot macro features contain NaN for msa={msa}, "
            f"fold ending {train_end.date()} -- insufficient lag history"
        )

    predicted_pct_change = float(model.predict(X_forecast)[0])
    modifier = [predicted_pct_change] * horizon_weeks

    return {
        "model_name": "macro_ridge",
        "modifier": modifier,
        "lag_weeks_used": dict(LAG_WEEKS),
    }


def run_macro_agent(df: pd.DataFrame, fold_boundaries: pd.DataFrame,
                     horizons: list = [4, 8, 13]) -> pd.DataFrame:
    """One row per (msa, fold_index, horizon). Re-fits per horizon like
    run_ar_agent does (spec-matching, simple interface) even though here the
    fitted model and prediction don't actually depend on horizon_weeks at
    all -- only the repeated-list length does. Fine at this data size
    (15 MSAs x 6 folds x 3 horizons = 270 cheap ridge fits).
    """
    rows = []
    for msa in df["msa"].unique():
        for fold in fold_boundaries.itertuples():
            train_end = pd.Timestamp(fold.train_end)
            for horizon in horizons:
                result = fit_and_forecast(df, msa, train_end, horizon)
                rows.append({
                    "msa": msa,
                    "fold_index": fold.fold,
                    "horizon": horizon,
                    "train_end": train_end.date(),
                    "modifier": result["modifier"],
                    "lag_weeks_used": result["lag_weeks_used"],
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from fold_boundaries import build_fold_table

    df = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
    folds = build_fold_table()

    result = run_macro_agent(df, folds)
    print(f"run_macro_agent output: {len(result)} rows")

    sample_msa = df["msa"].iloc[0]
    print(f"\nSample MSA: {sample_msa}")
    print(result[result["msa"] == sample_msa].to_string(index=False))
