"""Macro Agent (UI-claims pilot) -- ridge regression of weekly claims
pct-change on lagged FRED features.

Adapted from (not imported from) macro_agent.py's ridge-regression core and
facts_mas/agents/macro_agent.py's wrapper -- copied so a change to the
housing Macro agent can never silently affect this domain, and vice versa.

SCOPE DIFFERENCE FROM HOUSING -- worth remembering when interpreting
results: this pilot has only 2 macro features (fed_funds, cpi) vs.
housing's 4 (mortgage_rate, fed_funds, cpi, unemployment). unemployment/
UNRATE is excluded deliberately, not by oversight: claims and unemployment
are two views of the same labor-market phenomenon, so including UNRATE as
an input while claims is the forecast target would regress the target
partly on a close proxy of itself -- the circularity flagged when this
pilot was scoped (Step 1.3). mortgage_rate was never pulled for this domain
either (see pull_macro_ui.py) -- it has no first-order relevance to UI
claims the way it does to housing inventory. A 2-feature regression is a
materially weaker model than housing's 4-feature one on its own terms: if
this agent underperforms housing's Macro agent, that difference is a real
candidate explanation, not evidence about the domain itself.

LAG_WEEKS values (fed_funds=10, cpi=10) are carried over UNCHANGED from the
housing agent's documented transmission-lag guesses (midpoint of an 8-12wk
range) -- NOT re-derived for the claims domain. Same reasoning as reusing
housing's 4/8/13wk horizon set for this pilot (see evaluation_protocol_ui.md):
the question this pilot exists to answer is whether the FACTS-MAS
architecture and its modeling choices transfer across domains, so the
modeling assumptions are held fixed rather than re-tuned per domain.
Whether a 10-week fed_funds/cpi lag is actually right for unemployment
claims is an open question this pilot does not answer -- monetary-policy
transmission to labor markets is often cited in the literature as a
many-MONTHS lag, not weeks, which would make 10 weeks too short here even
though it's a reasonable guess for housing. Flagged, not fixed.

Same lag-safety discipline as macro_agent.py, copied not imported.
"""
import datetime

import pandas as pd
from sklearn.linear_model import Ridge

import sys
sys.path.insert(0, ".")
from domains.ui_claims.naive_baseline_ui import assert_no_lookahead

LAG_WEEKS = {
    "fed_funds": 10,
    "cpi": 10,
}


def assert_macro_no_lookahead(source_dates, fold_train_end, lag_weeks: int, feature_name: str) -> None:
    """Raise if any macro source date used for `feature_name` is later than
    fold_train_end - lag_weeks. Copied from macro_agent.py, unmodified.
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


def fit_and_forecast(df: pd.DataFrame, state: str, fold_train_end, horizon_weeks: int) -> dict:
    train_end = pd.Timestamp(fold_train_end)
    state_df = df[df["state"] == state].sort_values("date").set_index("date")
    history = state_df[state_df.index <= train_end]

    if history.empty:
        raise ValueError(f"no training data for state={state} on/before {train_end.date()}")
    assert_no_lookahead(history["claims_count"], train_end)

    lagged_cols = {}
    for feature, lag in LAG_WEEKS.items():
        lagged, source_dates = _lagged_feature(history, feature, lag)
        assert_macro_no_lookahead(source_dates, train_end, lag, feature)
        lagged_cols[feature] = lagged

    X = pd.DataFrame(lagged_cols, index=history.index)
    y = history["claims_count"].pct_change().rename("y")

    train_frame = pd.concat([X, y], axis=1).dropna()
    if len(train_frame) < 10:
        raise ValueError(
            f"insufficient training rows ({len(train_frame)}) for state={state}, "
            f"fold ending {train_end.date()}"
        )

    model = Ridge(alpha=1.0)
    model.fit(train_frame[list(LAG_WEEKS)], train_frame["y"])

    X_forecast = X.loc[[train_end]]
    if X_forecast.isna().any(axis=None):
        raise ValueError(
            f"forecast-snapshot macro features contain NaN for state={state}, "
            f"fold ending {train_end.date()} -- insufficient lag history"
        )

    predicted_pct_change = float(model.predict(X_forecast)[0])
    modifier = [predicted_pct_change] * horizon_weeks

    return {
        "model_name": "macro_ridge",
        "modifier": modifier,
        "lag_weeks_used": dict(LAG_WEEKS),
    }


def run_macro_agent_ui(
    df: pd.DataFrame,
    state: str,
    forecast_origin: datetime.date,
    horizon_weeks: int,
) -> dict:
    """Anchors the ridge model's pct-change modifier to the last observed
    claims value, applied as a single multiplicative adjustment -- same
    reasoning as the housing Macro agent's wrapper: compounding a small,
    noisy weekly modifier across the horizon mechanically inflates apparent
    error rather than reflecting the signal itself.
    """
    state_series = df[df["state"] == state].sort_values("date").set_index("date")["claims_count"]
    train_end = pd.Timestamp(forecast_origin)
    train_window = state_series[state_series.index <= train_end]
    if train_window.empty:
        raise ValueError(f"no training data for state={state} on/before {forecast_origin}")
    last_claims = float(train_window.iloc[-1])

    # fit_and_forecast does an EXACT date lookup internally (X.loc[[train_end]]),
    # which assumes train_end is itself a real grid date -- snap to the last
    # real grid date <= forecast_origin, same guard the housing wrapper uses.
    snapped_train_end = train_window.index[-1]

    result = fit_and_forecast(df, state, snapped_train_end, horizon_weeks)
    modifier = result["modifier"][-1]  # constant across the horizon, same design as housing

    factor = 1.0 + modifier
    values = [max(0.0, round(last_claims * factor, 2)) for _ in range(horizon_weeks)]

    return {
        "agent_name": "macro",
        "state": state,
        "forecast_origin": forecast_origin,
        "horizon_weeks": horizon_weeks,
        "values": values,
    }
