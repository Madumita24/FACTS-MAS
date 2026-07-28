"""AgentOutput wrapper around the top-level ar_agent.py's ETS model.

Naming note: this module's run_ar_agent(df, msa, forecast_origin, horizon_weeks)
-> AgentOutput is the backtest-harness-facing entry point (matches Umang's
run_seasonality_agent / run_intrinsic_agent calling convention: 4 positional
args, AgentOutput return). It is distinct from the top-level
ar_agent.run_ar_agent(df, fold_boundaries, horizons) -> pd.DataFrame, which
batches the SAME fit_and_forecast model over the original fold table for the
standalone comparison scripts (compare_ar_vs_naive.py, etc.) -- same core
model, different calling convention for a different caller. The two modules
happen to share a filename (facts_mas/agents/ar_agent.py vs top-level
ar_agent.py) but are distinct Python modules; imports below resolve the
top-level one by absolute name.

The underlying ETS model logic (ar_agent.fit_and_forecast) is unmodified --
this file only adapts its return shape to AgentOutput.
"""
import datetime

import pandas as pd

from ar_agent import fit_and_forecast
from facts_mas.schema import AgentOutput
from naive_baseline import assert_no_lookahead


def run_ar_agent(
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
    assert_no_lookahead(train_window, train_end)

    result = fit_and_forecast(train_window, horizon_weeks)  # unmodified ETS core

    return AgentOutput(
        agent_name="ar",
        msa=msa,
        forecast_origin=forecast_origin,
        horizon_weeks=horizon_weeks,
        values=result["point_forecast"],
    )
