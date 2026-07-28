"""AgentOutput wrapper around event_agent.py's cached LLM interpretations.

Reads from event_agent_weekly.csv (Step 3's full batch output, 153 real LLM
calls already made once) rather than re-invoking interpret_event() live --
re-running the LLM per (msa, forecast_origin) query during backtesting would
be slow, costly, and would make repeated backtest runs non-deterministic;
the interpretation itself doesn't change between runs, so it's cached.

Same anchoring approach as the Macro Agent wrapper and for the same reason:
impact_magnitude (-1..+1, already confidence-gated) is applied as a SINGLE
multiplicative adjustment to the last observed inventory value, not
compounded per week.

confidence is populated here -- per the schema's Optional field, this is the
one agent in the current lineup expected to populate it, since it's the only
agent whose core model produces a real confidence signal (the LLM's own
calibrated confidence, not a fusion weight). Repeated across the horizon
since impact_magnitude/confidence are per-declaration constants in this
design, not per-week values. When no FEMA declaration is active for this
(msa, forecast_origin), the agent has nothing to say: it reports a neutral
(last-value-anchored) forecast with confidence=0.0, rather than fabricating
an event -- consistent with interpret_event()'s "never invent events" rule.
"""
import datetime

import pandas as pd

from facts_mas.schema import AgentOutput

EVENT_WEEKLY_PATH = "event_agent_weekly.csv"
_event_weekly_cache: pd.DataFrame = None

# DESIGN CHOICE, not a data fact -- same honesty standard as IMPACT_SCALE
# below. FEMA's "declaration open" status reflects administrative/funding
# status (how long assistance programs stay available), not ongoing market
# disruption. Diagnosed via a live sanity check: COVID-19 declarations
# stayed federally "open" for 172 weeks, and the wrapper was re-applying a
# fresh impact_magnitude discount to that WEEK's already-current inventory
# every single week for the entire span -- double-counting an effect that
# (if real) had already happened, since the anchor already reflects
# however the market actually responded. Mean skill on COVID weeks was
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
        impact_magnitude = 0.0
        confidence = 0.0
    else:
        row = active.iloc[0]  # if multiple declarations overlap the same week, take the first
        incident_begin = pd.Timestamp(row["incident_begin_date"]) if pd.notna(row["incident_begin_date"]) else None
        age_weeks = (snapped_week - incident_begin).days / 7.0 if incident_begin is not None else None

        if age_weeks is not None and age_weeks > DECLARATION_ACTIVE_WINDOW_WEEKS:
            # Administratively still "open" per FEMA, but treated as expired
            # for forecasting purposes -- see DECLARATION_ACTIVE_WINDOW_WEEKS.
            impact_magnitude = 0.0
            confidence = 0.0
        else:
            # age_weeks is None only if incident_begin_date itself is missing
            # (data-quality edge case, not expected in practice) -- falls back
            # to the pre-fix behavior of applying the impact rather than
            # guessing at an age we don't have evidence for.
            impact_magnitude = float(row["impact_magnitude"])  # already confidence-gated in Step 3
            confidence = float(row["confidence"])

    # impact_magnitude is a QUALITATIVE -1..+1 severity score by design --
    # event_agent.py's own prompt explicitly tells the LLM it is "NOT a
    # precise unit/percent estimate," only a coarse directional signal.
    # Applying it directly as a pct-change (factor = 1 + impact_magnitude)
    # would mean a -0.75 score forecasts inventory crashing to 25% of its
    # prior level in the very next week and staying there for the whole
    # horizon -- caught via a live sanity check (fold-1 weight diagnostic:
    # Event's mean APE was 71.6% vs. 6-17% for every other agent, driven
    # entirely by declaration weeks). IMPACT_SCALE converts the qualitative
    # score into an approximate weekly pct-change, roughly on the same order
    # as the Macro Agent's own ridge-fitted modifiers (~1-3%/week) but
    # somewhat larger, reflecting that a real disaster declaration should
    # plausibly move inventory more than ambient macro conditions.
    # ARBITRARY, UNVALIDATED CHOICE -- there is no ground-truth data in this
    # project connecting qualitative severity scores to actual pct inventory
    # change, so 0.10 is a documented guess, not a fitted value. Revisit if
    # real post-disaster inventory data ever becomes available to calibrate
    # this properly.
    #
    # TODO (flagged, not fixed -- one overshoot example isn't enough to
    # safely change a shared constant tonight): a single fixed IMPACT_SCALE
    # for every event_type/severity combination looks inconsistent in the
    # data we already have. At impact_magnitude=-0.50 (program_count=2):
    # Tropical Storm Imelda and the Saddleridge/Tick/Getty Fires all scored
    # POSITIVE skill vs. naive (+0.2 to +0.5) -- the resulting -5% adjustment
    # tracked real outcomes reasonably well. At impact_magnitude=-0.75
    # (program_count=3): a generic "WILDFIRES" declaration scored strongly
    # NEGATIVE (-0.9 to -8.5) -- actual decline was only ~-1.5% while the
    # -7.5% adjustment overshot it roughly 5x. Needs per-event-type or
    # per-severity-tier calibration against more real outcomes than these
    # two data points before changing the constant itself.
    IMPACT_SCALE = 0.10
    factor = 1.0 + impact_magnitude * IMPACT_SCALE
    values = [max(0.0, round(last_inventory * factor, 2)) for _ in range(horizon_weeks)]

    return AgentOutput(
        agent_name="event",
        msa=msa,
        forecast_origin=forecast_origin,
        horizon_weeks=horizon_weeks,
        values=values,
        confidence=[confidence] * horizon_weeks,
    )
