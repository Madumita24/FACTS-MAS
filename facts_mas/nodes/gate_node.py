"""
NEXUS Modification A — Smart Early-Exit Gate.

Solves Limitation 2: the published architecture runs the full expensive
multi-agent loop for every single forecast, even when the market is calm and
seasonal and no text reasoning is needed at all.

Mechanism: a cheap text-blind model forecasts continuously. Before spending
anything, the gate checks two families of trigger. If neither fires, the
cheap forecast is accepted and the heavy loop never runs.

    1. Text-event signals -- a disaster declaration, a Fed announcement, a
       major employer move, a policy change.
    2. Statistical anomaly -- the cheap model's recent error, expressed as a
       z-score against its own error distribution. This is the "something is
       happening that my simple model cannot explain" detector.

The Mod A <-> Mod B connection, stated explicitly in the proposal because
without it the two modifications do not actually work together: the gate
watches its Granger neighbours' anomalies as well as its own. A Los Angeles
shock can wake the heavy loop for Riverside even when Riverside's own series
looks perfectly calm.

SUBSTITUTION, STATED PLAINLY: the proposal specifies TimesFM 2.5 as the
lightweight baseline. TimesFM is not implemented anywhere in this project.
The baseline is therefore pluggable, and defaults to the existing AR agent
(ETS), which is text-blind, cheap and already validated. Swapping in TimesFM
later is a one-argument change. Every result produced by this module should
be read as "gate with an ETS baseline", not "gate as specified".

No LLM is involved in any gate decision. All of it is NumPy.
"""
from __future__ import annotations

import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd

from facts_mas.state.gate import GateConfig, GateDecision
from facts_mas.state.ontology import GateTriggerType
from facts_mas.state.spatial import SpilloverSignal


# ═══════════════════════════════════════════════════════════════════════════
# Lightweight baseline
# ═══════════════════════════════════════════════════════════════════════════

def default_baseline(
    df: pd.DataFrame,
    msa: str,
    as_of: datetime.date,
    horizon_weeks: int,
) -> np.ndarray:
    """
    The cheap, text-blind forecast the gate falls back on.

    Uses the AR agent (ETS) as the TimesFM stand-in. If it fails for any
    reason, degrades to last-value rather than raising: the gate's entire
    purpose is to always have a usable forecast in hand, so a baseline that
    can throw would defeat it.
    """
    try:
        from facts_mas.agents.ar_agent import run_ar_agent
        return np.array(run_ar_agent(df, msa, as_of, horizon_weeks).values,
                        dtype=np.float64)
    except Exception:
        md = df[(df["msa"] == msa) & (df["date"] <= pd.Timestamp(as_of))]
        last = float(md.sort_values("date")["inventory_count"].iloc[-1])
        return np.full(horizon_weeks, last, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# Statistical anomaly detection
# ═══════════════════════════════════════════════════════════════════════════

def baseline_error_zscore(
    df: pd.DataFrame,
    msa: str,
    as_of: datetime.date,
    config: GateConfig,
    baseline_fn: Callable = default_baseline,
) -> Optional[float]:
    """
    How unusual is the baseline's most recent error?

    Walks back over `config.lookback_weeks`, asks the baseline for a
    one-week-ahead forecast at each point, and compares against what
    actually happened. The z-score of the latest error against that
    distribution is the anomaly measure.

    Errors are absolute percentage errors rather than raw differences, so
    the threshold means the same thing in New York and Seattle. Using raw
    errors would make the gate fire constantly in large metros and almost
    never in small ones.

    Returns None when there is not enough history to form a distribution.
    A gate that cannot measure should not fire on a guess.
    """
    md = df[(df["msa"] == msa) & (df["date"] <= pd.Timestamp(as_of))].sort_values("date")
    if len(md) < config.lookback_weeks + 3:
        return None

    dates = md["date"].tolist()
    values = md["inventory_count"].values.astype(np.float64)

    errors = []
    # One point per week in the lookback window; each needs its own origin.
    for i in range(len(dates) - config.lookback_weeks - 1, len(dates) - 1):
        if i < 1:
            continue
        origin = dates[i].date()
        actual = values[i + 1]
        if actual == 0:
            continue
        try:
            pred = baseline_fn(df, msa, origin, 1)[0]
        except Exception:
            continue
        errors.append(abs((actual - pred) / actual))

    if len(errors) < 4:
        return None

    arr = np.array(errors, dtype=np.float64)
    latest, history = arr[-1], arr[:-1]
    sd = float(np.std(history))
    if sd == 0:
        return 0.0
    return float((latest - float(np.mean(history))) / sd)


# ═══════════════════════════════════════════════════════════════════════════
# The gate
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_gate(
    df: pd.DataFrame,
    msa: str,
    as_of: datetime.date,
    horizon_weeks: int,
    config: Optional[GateConfig] = None,
    event_trigger: Optional[GateTriggerType] = None,
    spillover: Optional[SpilloverSignal] = None,
    baseline_fn: Callable = default_baseline,
) -> GateDecision:
    """
    Decide whether the heavy multi-agent loop should run.

    Trigger precedence, and the reason for it:
        1. Text event      -- a declared real-world cause outranks a
                              statistical hint, and is cheaper to check.
        2. Self anomaly    -- this city's own model is failing.
        3. Neighbour shock -- someone upstream moved (Mod A <-> B).

    A text event is checked first because if a disaster has been declared,
    the question "is something happening" is already answered and computing
    a z-score to confirm it would be wasted work.

    The baseline forecast is computed unconditionally and returned either
    way. It is the output when the gate stays shut, and it is Mod C's
    fallback when boundary correction runs out of retries -- so it has to
    exist even on the paths that do not use it.
    """
    config = config or GateConfig()
    baseline = baseline_fn(df, msa, as_of, horizon_weeks)
    baseline_list = [float(v) for v in baseline]

    # 1. Text-event trigger
    if event_trigger is not None and event_trigger in config.text_event_triggers:
        return GateDecision(
            should_activate_heavy_loop=True,
            trigger_type=event_trigger,
            trigger_details=f"Text event '{event_trigger.value}' is on the trigger list.",
            baseline_forecast=baseline_list,
        )

    # 2. Self anomaly
    z = baseline_error_zscore(df, msa, as_of, config, baseline_fn)
    if z is not None and z > config.anomaly_zscore_threshold:
        return GateDecision(
            should_activate_heavy_loop=True,
            trigger_type=GateTriggerType.BASELINE_ANOMALY,
            trigger_details=(
                f"Baseline error z-score {z:.2f} exceeded threshold "
                f"{config.anomaly_zscore_threshold:.2f}."
            ),
            baseline_forecast=baseline_list,
            baseline_error_zscore=z,
        )

    # 3. Neighbour shock (Mod A <-> Mod B)
    if spillover is not None and spillover.neighbor_count > 0:
        # max_neighbor_shock is a percentage move, not a z-score. Convert it
        # to comparable units using this city's own weekly volatility, so
        # "a big move for a neighbour" means big relative to normal weekly
        # movement rather than big as a raw percentage.
        # Score against the NEIGHBOUR's own volatility, not this city's.
        shock_msa = (spillover.contributing_msa_ids[0]
                     if spillover.contributing_msa_ids else msa)
        nz = _shock_to_zscore(df, shock_msa, as_of, spillover.max_neighbor_shock)
        if nz is not None and nz > config.neighbor_anomaly_zscore_threshold:
            return GateDecision(
                should_activate_heavy_loop=True,
                trigger_type=GateTriggerType.NEIGHBOR_SPILLOVER,
                trigger_details=(
                    f"Neighbour move {spillover.max_neighbor_shock:+.2%} "
                    f"(z={nz:.2f}) exceeded threshold "
                    f"{config.neighbor_anomaly_zscore_threshold:.2f}."
                ),
                baseline_forecast=baseline_list,
                baseline_error_zscore=z,
                neighbor_anomaly_msa_id=(spillover.contributing_msa_ids[0]
                                         if spillover.contributing_msa_ids else None),
                neighbor_error_zscore=nz,
            )

    # Nothing fired: stay dormant and accept the cheap forecast.
    return GateDecision(
        should_activate_heavy_loop=False,
        trigger_type=GateTriggerType.NONE,
        trigger_details="",
        baseline_forecast=baseline_list,
        baseline_error_zscore=z,
    )


def _shock_to_zscore(
    df: pd.DataFrame,
    msa: str,
    as_of: datetime.date,
    shock_pct: float,
    trend_weeks: int = 4,
    window: int = 104,
) -> Optional[float]:
    """
    Express a neighbour's move in units of that city's OWN normal movement,
    measured over the SAME number of weeks.

    Two corrections over the first version, both of which were inflating the
    score rather than merely mis-tuning it:

    1. LIKE-FOR-LIKE WINDOW. The spillover signal measures a move over
       `trend_weeks` (4 by default), but the old code divided it by the
       standard deviation of ONE-week changes. Multi-week moves are larger
       than single-week moves roughly in proportion to the square root of the
       window, so a 4-week move scored about twice as high as it should. That
       is why the neighbour trigger fired on 45% of forecasts: not because
       neighbours were unusually active, but because every score was doubled.
       Now the reference distribution is built from overlapping moves of the
       same length.

    2. THE NEIGHBOUR'S OWN VOLATILITY, not the target's. The question the
       gate is asking is "did something unusual happen over there", so the
       comparison has to be against what is normal over there. A calm move
       in a volatile city was previously scored against a quiet city's
       yardstick and looked alarming.
    """
    md = df[(df["msa"] == msa) & (df["date"] <= pd.Timestamp(as_of))].sort_values("date")
    if len(md) < window + trend_weeks + 2:
        return None
    v = md["inventory_count"].values.astype(np.float64)[-(window + trend_weeks + 1):]
    with np.errstate(divide="ignore", invalid="ignore"):
        moves = (v[trend_weeks:] - v[:-trend_weeks]) / v[:-trend_weeks]
    moves = moves[np.isfinite(moves)]
    if len(moves) < 12:
        return None
    sd = float(np.std(moves))
    return abs(shock_pct) / sd if sd > 0 else None


# ═══════════════════════════════════════════════════════════════════════════
# Measuring what the gate buys
# ═══════════════════════════════════════════════════════════════════════════

def gate_savings_report(decisions: list[GateDecision]) -> dict:
    """
    What fraction of forecasts skipped the heavy loop, and by which trigger.

    This is the number the whole modification exists to produce. The
    proposal's added-value claim depends on it: a low activation rate is
    what buys enough headroom to run the pipeline three to five times and
    report confidence intervals, which the original authors said was
    computationally infeasible.

    A gate that fires on everything has cost nothing but bought nothing.
    """
    n = len(decisions)
    if n == 0:
        return {}
    activated = sum(1 for d in decisions if d.should_activate_heavy_loop)
    by_trigger: dict[str, int] = {}
    for d in decisions:
        by_trigger[d.trigger_type.value] = by_trigger.get(d.trigger_type.value, 0) + 1
    return {
        "n_forecasts": n,
        "n_activated": activated,
        "activation_rate": activated / n,
        "skip_rate": 1.0 - activated / n,
        "heavy_loop_calls_saved": n - activated,
        "by_trigger": by_trigger,
    }
