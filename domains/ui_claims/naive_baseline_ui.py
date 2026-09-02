"""Naive last-value baseline for the UI-claims pilot. Adapted from (not
imported from) the housing pipeline's naive_baseline.py -- copied so a
change to one domain's baseline can never silently affect the other.

Forecast = the last observed claims_count, repeated for every week of the
horizon. Same output-shape reasoning as the housing version: a single
list-valued column (length == horizon) rather than exploded per-week
columns, since horizons differ (4/8/13) and a list column keeps every row
dense.

assert_no_lookahead is also defined here (not just in this file's own
forecast function) because ar_agent_ui.py and macro_agent_ui.py both need
it and import it from here -- the same intra-domain sharing pattern the
housing pipeline uses (ar_agent.py and macro_agent.py both import it from
naive_baseline.py). This is reuse WITHIN the UI-claims domain, not a
cross-domain import -- the isolation rule is about never importing from or
being imported by facts_mas/, not about avoiding code reuse inside this
pilot.
"""
import numpy as np
import pandas as pd


def forecast_naive_ui(series: pd.Series, horizon_weeks: int) -> np.ndarray:
    """Repeat the series' last value horizon_weeks times.

    Assumes `series` has already been truncated to the correct point-in-time
    training window by the caller.
    """
    last_value = float(series.iloc[-1])
    return np.full(horizon_weeks, last_value, dtype=float)


def assert_no_lookahead(series: pd.Series, as_of) -> None:
    """Raise if series contains any date after as_of. Defense-in-depth,
    same role as the housing version: callers should already have
    truncated the series, this exists to fail loudly if a future change
    accidentally doesn't.
    """
    if len(series) and series.index.max() > pd.Timestamp(as_of):
        raise ValueError(
            f"lookahead violation: series contains data as late as "
            f"{series.index.max().date()}, which is after as_of={pd.Timestamp(as_of).date()}"
        )


def run_naive_baseline_ui(
    df: pd.DataFrame,
    state: str,
    forecast_origin,
    horizon_weeks: int,
) -> dict:
    """(df, state, forecast_origin, horizon_weeks) -> dict, matching this
    pilot's agent calling convention (see ar_agent_ui.py / macro_agent_ui.py
    module docstrings for why a plain dict, not AgentOutput)."""
    state_series = df[df["state"] == state].sort_values("date").set_index("date")["claims_count"]
    train_end = pd.Timestamp(forecast_origin)
    train_window = state_series[state_series.index <= train_end]

    if train_window.empty:
        raise ValueError(f"no training data for state={state} on/before {forecast_origin}")
    assert_no_lookahead(train_window, train_end)

    forecast = forecast_naive_ui(train_window, horizon_weeks)

    return {
        "agent_name": "naive",
        "state": state,
        "forecast_origin": forecast_origin,
        "horizon_weeks": horizon_weeks,
        "values": forecast.tolist(),
    }
