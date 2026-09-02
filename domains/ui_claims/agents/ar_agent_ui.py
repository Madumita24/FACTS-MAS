"""AR Agent (UI-claims pilot) -- per-state ETS (Holt-Winters, additive
trend, no seasonality) claims forecaster.

Adapted from (not imported from) ar_agent.py's fit_and_forecast core and
facts_mas/agents/ar_agent.py's wrapper -- copied so a change to the housing
AR agent can never silently affect this domain, and vice versa. The ETS
specification itself is unmodified: additive trend, no seasonal component,
same non-negativity clip as a temporary local safeguard (claims counts
can't be negative, same reasoning as inventory counts).

No AgentOutput/pydantic schema here -- that class lives in facts_mas.schema,
built for housing's 5-agent fusion contract. Building a parallel schema for
this 2-agent minimal pilot is out of scope, so this returns a plain dict
with the same field names (agent_name, state, forecast_origin,
horizon_weeks, values) instead.

assert_no_lookahead is imported from this domain's own naive_baseline_ui.py
(intra-domain reuse, not a cross-domain import -- see that file's
docstring).
"""
import datetime
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import sys
sys.path.insert(0, ".")
from domains.ui_claims.naive_baseline_ui import assert_no_lookahead


def fit_and_forecast(series: pd.Series, horizon_weeks: int) -> dict:
    """Fit ETS (additive trend, no seasonal component) and forecast ahead.
    Unmodified from ar_agent.py's core model.
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
    }


def run_ar_agent_ui(
    df: pd.DataFrame,
    state: str,
    forecast_origin: datetime.date,
    horizon_weeks: int,
) -> dict:
    state_series = df[df["state"] == state].sort_values("date").set_index("date")["claims_count"]
    train_end = pd.Timestamp(forecast_origin)
    train_window = state_series[state_series.index <= train_end]

    if train_window.empty:
        raise ValueError(f"no training data for state={state} on/before {forecast_origin}")
    assert_no_lookahead(train_window, train_end)

    result = fit_and_forecast(train_window, horizon_weeks)

    return {
        "agent_name": "ar",
        "state": state,
        "forecast_origin": forecast_origin,
        "horizon_weeks": horizon_weeks,
        "values": result["point_forecast"],
    }
