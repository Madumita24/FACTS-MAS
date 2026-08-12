"""AgentOutput wrapper around event_agent.py's cached LLM interpretations.

Reads from event_agent_weekly.csv (Step 3's full batch output, 153 real LLM
calls already made once) rather than re-invoking interpret_event() live --
re-running the LLM per (msa, forecast_origin) query during backtesting would
be slow, costly, and would make repeated backtest runs non-deterministic;
the interpretation itself doesn't change between runs, so it's cached.

MAGNITUDE ADJUSTMENT REMOVED (see event_agent.md for the full history):
this wrapper used to apply impact_magnitude (-1..+1, already confidence-
gated) as a single multiplicative adjustment to the last observed inventory
value. Umang's calibration sweep (facts_mas/calibration.py,
calibrate_event_impact_scale.py) found IMPACT_SCALE=0.0 beats every positive
value tested, 6/6 folds, with no interior optimum -- the severity score has
no usable point-forecast magnitude against weekly inventory levels, only
ordinal/directional content. values is therefore last_inventory held flat
across the horizon, same anchoring approach as the Macro Agent wrapper.

confidence is populated here -- per the schema's Optional field, this is the
one agent in the current lineup expected to populate it, since it's the only
agent whose core model produces a real confidence signal (the LLM's own
calibrated confidence, not a fusion weight). Repeated across the horizon
since confidence is a per-declaration constant in this design, not a
per-week value. When no FEMA declaration is active for this
(msa, forecast_origin), the agent has nothing to say: it reports a neutral
(last-value-anchored) forecast with confidence=0.0, rather than fabricating
an event -- consistent with interpret_event()'s "never invent events" rule.
"""
import datetime

import pandas as pd

from facts_mas.schema import AgentOutput

EVENT_WEEKLY_PATH = "event_agent_weekly.csv"
_event_weekly_cache: pd.DataFrame = None

# DESIGN CHOICE, not a data fact. FEMA's "declaration open" status reflects
# administrative/funding status (how long assistance programs stay
# available), not ongoing market disruption. Diagnosed via a live sanity
# check, back when this wrapper still applied a magnitude adjustment:
# COVID-19 declarations stayed federally "open" for 172 weeks, and the
# wrapper was re-applying a fresh impact_magnitude discount to that WEEK's
# already-current inventory every single week for the entire span --
# double-counting an effect that (if real) had already happened, since the
# anchor already reflects however the market actually responded. Mean skill
# on COVID weeks was
# -1.88 vs. naive as a direct result. 8 weeks is chosen to roughly match
# the duration of short-lived disasters that scored WELL in that same
# diagnostic (e.g. Tropical Storm Imelda, 6 weeks) -- a reasonable-but-
# arbitrary starting assumption pending real calibration data.
DECLARATION_ACTIVE_WINDOW_WEEKS = 8


def _load_event_weekly() -> pd.DataFrame:
    global _event_weekly_cache
    if _event_weekly_cache is None:
        _event_weekly_cache = pd.read_csv(EVENT_WEEKLY_PATH, parse_dates=["week"])
    return _event_weekly_cache


def run_event_agent(
    df: pd.DataFrame,
    msa: str,
    forecast_origin: datetime.date,
    horizon_weeks: int,
) -> AgentOutput:
    events = _load_event_weekly()

    msa_series = df[df["msa"] == msa].sort_values("date").set_index("date")["inventory_count"]
    train_end = pd.Timestamp(forecast_origin)
    train_window = msa_series[msa_series.index <= train_end]
    if train_window.empty:
        raise ValueError(f"no training data for msa={msa} on/before {forecast_origin}")
    last_inventory = float(train_window.iloc[-1])

    # Same snap as the Macro Agent wrapper, same reason: event_agent_weekly.csv's
    # "week" column is only ever populated on real Saturday grid dates, so an
    # exact match against an arbitrary forecast_origin (e.g. a Thursday probe
    # date) would silently miss a genuinely-active declaration rather than
    # finding it. Snapping to the last real grid date <= forecast_origin keeps
    # this a no-op for the real backtest (which always passes grid dates) while
    # making the lookup correct for any date.
    snapped_week = train_window.index[-1]
    active = events[(events["msa"] == msa) & (events["week"] == snapped_week)]

    if active.empty:
        confidence = 0.0
    else:
        row = active.iloc[0]  # if multiple declarations overlap the same week, take the first
        incident_begin = pd.Timestamp(row["incident_begin_date"]) if pd.notna(row["incident_begin_date"]) else None
        age_weeks = (snapped_week - incident_begin).days / 7.0 if incident_begin is not None else None

        if age_weeks is not None and age_weeks > DECLARATION_ACTIVE_WINDOW_WEEKS:
            # Administratively still "open" per FEMA, but treated as expired
            # for forecasting purposes -- see DECLARATION_ACTIVE_WINDOW_WEEKS.
            confidence = 0.0
        else:
            # age_weeks is None only if incident_begin_date itself is missing
            # (data-quality edge case, not expected in practice).
            confidence = float(row["confidence"])

    # No magnitude adjustment (see module docstring): values is last_inventory
    # held flat across the horizon, regardless of event_type/confidence/decay
    # status above. Those signals still gate `confidence`, they just no
    # longer move the point forecast -- see event_agent.md.
    values = [last_inventory] * horizon_weeks

    return AgentOutput(
        agent_name="event",
        msa=msa,
        forecast_origin=forecast_origin,
        horizon_weeks=horizon_weeks,
        values=values,
        confidence=[confidence] * horizon_weeks,
    )
