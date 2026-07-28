"""AgentOutput wrapper around the top-level macro_agent.py's ridge-regression
modifier (see macro_agent.md for the full validated-null investigation --
this agent is expected to contribute little to no signal in fusion).

The core model (macro_agent.fit_and_forecast) returns a pct-change MODIFIER
(e.g. -0.013 = -1.3%/week pressure), not an inventory level -- AgentOutput
requires absolute, non-negative inventory counts. This wrapper anchors the
modifier to the last observed inventory value and applies it as a SINGLE
multiplicative adjustment, not compounded per week: compounding a small,
noisy weekly modifier across the horizon was shown (test_macro_agent.py's
Step 4b combiner comparison) to mechanically inflate apparent error rather
than reflect the signal itself. This keeps the Macro Agent's forecast
self-contained -- independent of any other agent's output, as a fusion
ensemble member must be -- without changing the underlying ridge regression.

Given the validated null, this agent is expected to receive a low/near-zero
fusion weight under inverse-MAPE weighting. That is the point of running it
through the real pipeline rather than hardcoding a low weight in: let the
weighting mechanism demonstrate the null on its own.
"""
import datetime

import pandas as pd

from facts_mas.schema import AgentOutput
from macro_agent import fit_and_forecast


def run_macro_agent(
    df: pd.DataFrame,
    msa: str,
    forecast_origin: datetime.date,
    horizon_weeks: int,
) -> AgentOutput:
    msa_series = df[df["msa"] == msa].sort_values("date").set_index("date")["inventory_count"]
    train_end = pd.Timestamp(forecast_origin)
    train_window = msa_series[msa_series.index <= train_end]
    if train_window.empty:
        raise ValueError(f"no training data for msa={msa} on/before {forecast_origin}")
    last_inventory = float(train_window.iloc[-1])

    # macro_agent.fit_and_forecast does an EXACT date lookup internally
    # (X.loc[[train_end]]), which assumes train_end is itself a real grid
    # date -- true for every fold-table date this project has used so far,
    # but not guaranteed for an arbitrary forecast_origin a caller might
    # pass. Snap to the last real grid date <= forecast_origin before
    # calling the core function, rather than fixing the core's exact-match
    # assumption itself (out of scope: core model logic stays untouched).
    snapped_train_end = train_window.index[-1]

    result = fit_and_forecast(df, msa, snapped_train_end, horizon_weeks)  # unmodified ridge core
    modifier = result["modifier"][-1]  # constant across the horizon in this design (see macro_agent.py)

    factor = 1.0 + modifier
    values = [max(0.0, round(last_inventory * factor, 2)) for _ in range(horizon_weeks)]

    return AgentOutput(
        agent_name="macro",
        msa=msa,
        forecast_origin=forecast_origin,
        horizon_weeks=horizon_weeks,
        values=values,
    )
