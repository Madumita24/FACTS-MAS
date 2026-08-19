"""
NEXUS Modification B — Cross-Regional Spatial Spillover.

Solves Limitation 4: NEXUS forecasts 15 metros completely independently, as
if a sharp inventory drop in Los Angeles told you nothing about Riverside a
few weeks later.

Why Granger causality and not correlation (NEXUS proposal §3, Mod B):
    Two cities can look tightly linked purely because both react to the same
    national rate change. Correlation cannot separate that from one city
    genuinely leading another. Granger asks the sharper question -- does
    knowing city A's past improve a forecast of city B beyond B's own past?

This module turns that argument into the two artifacts the state models
describe:

    build_spatial_graph()      -> SpatialGraph   (system-level, computed once)
    compute_spillover_signal() -> SpilloverSignal (per target city, per date)

The graph is built once at startup. The signal is computed per forecast and
is what the gate (Mod A) and the forecasting agents actually consume.

Leakage discipline: `as_of` bounds every read. A spillover signal for week t
sees neighbour data up to t and no further, and the neighbour trend is
additionally shifted back by the edge's own Granger lag, so the signal
reflects what a neighbour was doing when it could plausibly have started
propagating.
"""
from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
import pandas as pd

from facts_mas.factor_scoring import DEFAULT_MAX_LAG, granger_spillover_graph
from facts_mas.state.spatial import GrangerEdge, SpatialGraph, SpilloverSignal

DEFAULT_SIGNIFICANCE = 0.05
DEFAULT_TREND_WEEKS = 4      # window over which a neighbour's recent move is measured
DEFAULT_TOP_K = 3            # proposal §8: cap active factors to top 3-5


# ═══════════════════════════════════════════════════════════════════════════
# Building the graph
# ═══════════════════════════════════════════════════════════════════════════

def build_spatial_graph(
    df: pd.DataFrame,
    msas: Optional[list[str]] = None,
    as_of: Optional[datetime.date] = None,
    max_lag: int = DEFAULT_MAX_LAG,
    lookback_weeks: int = 104,
    significance: float = DEFAULT_SIGNIFICANCE,
) -> SpatialGraph:
    """
    Run the pairwise Granger screen and package it as a SpatialGraph.

    Reuses `factor_scoring.granger_spillover_graph` rather than
    reimplementing the test, so the Phase-3 factor screen and the Mod-B
    edge weights can never drift apart -- they are literally the same
    computation, which was the point of building it there first.

    A note on `optimal_lag_weeks`: GrangerEdge requires it to be >= 1, but
    the screen returns lag 0 when no lag was significant. Those edges are
    non-significant by construction, so they are stored with lag 1 and
    is_significant=False rather than dropped -- the proposal asks for all
    tests to remain available for audit, not just the surviving ones.
    """
    msas = msas or sorted(df["msa"].unique().tolist())
    results = granger_spillover_graph(
        df, msas, as_of=as_of, max_lag=max_lag, lookback_weeks=lookback_weeks
    )

    edges: list[GrangerEdge] = []
    for r in results:
        # Use the screen's OWN significance flag, which already carries the
        # Benjamini-Hochberg correction across all 210 ordered pairs.
        #
        # This line previously re-derived significance as `p_value < 0.05`,
        # which silently threw that correction away: the screen returned 43
        # FDR-surviving edges and this function then re-admitted every edge
        # whose raw p-value cleared 0.05, giving 67. The corrected count was
        # only ever visible when calling granger_spillover_graph directly,
        # never through the graph the system actually uses. Caught by
        # Madumita reproducing the edge count independently.
        #
        # best_lag >= 1 is still required: the screen reports lag 0 when no
        # lag was significant, and GrangerEdge requires a positive lag.
        significant = r.significant and r.best_lag >= 1
        edges.append(GrangerEdge(
            source_msa_id=r.source,
            target_msa_id=r.target,
            optimal_lag_weeks=max(r.best_lag, 1),
            p_value=float(min(max(r.p_value, 0.0), 1.0)),
            # The screen reports p-values, not F-statistics. Rather than
            # invent a number for a required field, store a monotone proxy
            # (larger = stronger evidence) and say so plainly here.
            f_statistic=float(-np.log10(max(r.p_value, 1e-12))),
            is_significant=significant,
        ))

    return SpatialGraph(
        msa_ids=list(msas),
        edges=edges,
        significance_threshold=significance,
        max_lag_tested=max_lag,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Per-forecast spillover signal
# ═══════════════════════════════════════════════════════════════════════════

def _recent_pct_change(
    df: pd.DataFrame,
    msa: str,
    as_of: datetime.date,
    lag_weeks: int,
    trend_weeks: int,
) -> Optional[float]:
    """
    A neighbour's percentage move over `trend_weeks`, ending `lag_weeks` ago.

    The lag shift is the mechanism, not a detail. If Los Angeles is going to
    move Riverside 5 weeks from now, the relevant Los Angeles behaviour is
    what it was doing 5 weeks ago -- reading its *current* week would be
    asking the spillover to act instantaneously, which is exactly what the
    Granger lag says it does not do.
    """
    md = df[(df["msa"] == msa) & (df["date"] <= pd.Timestamp(as_of))].sort_values("date")
    if len(md) < lag_weeks + trend_weeks + 1:
        return None
    series = md["inventory_count"].values.astype(np.float64)
    end = len(series) - lag_weeks
    start = end - trend_weeks
    if start < 0 or end <= 0 or series[start] == 0:
        return None
    return float((series[end - 1] - series[start]) / series[start])


def compute_spillover_signal(
    df: pd.DataFrame,
    graph: SpatialGraph,
    target_msa: str,
    as_of: datetime.date,
    top_k: int = DEFAULT_TOP_K,
    trend_weeks: int = DEFAULT_TREND_WEEKS,
) -> SpilloverSignal:
    """
    Aggregate the target city's Granger neighbours into one signal.

    Neighbours are weighted by 1/p_value, so a strongly-evidenced link
    counts for more than a marginal one. Capped at `top_k` because the
    proposal's own risk table lists factor proliferation as a hazard and
    caps active factors at three to five.

    Returns a neutral signal (zeros, no contributors) when the city has no
    significant neighbours. That is a real answer, not a failure: some
    metros genuinely lead or follow nothing.
    """
    neighbors = graph.neighbors_of(target_msa, top_k=top_k)

    weighted_sum, weight_total = 0.0, 0.0
    max_shock = 0.0
    contributors: list[str] = []

    for edge in neighbors:
        change = _recent_pct_change(
            df, edge.source_msa_id, as_of, edge.optimal_lag_weeks, trend_weeks
        )
        if change is None:
            continue
        w = 1.0 / max(edge.p_value, 1e-10)
        weighted_sum += w * change
        weight_total += w
        max_shock = max(max_shock, abs(change))
        contributors.append(edge.source_msa_id)

    aggregate = (weighted_sum / weight_total) if weight_total > 0 else 0.0

    return SpilloverSignal(
        target_msa_id=target_msa,
        neighbor_count=len(contributors),
        aggregate_neighbor_trend=float(aggregate),
        max_neighbor_shock=float(max_shock),
        contributing_msa_ids=contributors,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Applying the signal
# ═══════════════════════════════════════════════════════════════════════════

def apply_spillover(
    forecast: np.ndarray,
    signal: SpilloverSignal,
    strength: float = 0.0,
) -> np.ndarray:
    """
    Nudge a forecast by the neighbour trend.

    `strength` defaults to 0.0, i.e. the adjustment is OFF unless someone
    turns it on. This is deliberate and follows directly from what happened
    to the Event Agent: an unvalidated magnitude constant applied to every
    forecast made things worse for months before calibration caught it.
    The same mistake is not worth repeating here.

    The intended path is to calibrate `strength` with
    facts_mas/calibration.py the same way IMPACT_SCALE was swept, and adopt
    a non-zero value only if it clears the fold-consistency gate. Until then
    the spillover signal is available as gate input and as context, and
    contributes nothing numerically.

    The adjustment ramps in over the horizon rather than applying in full at
    week 1, because spillover is a propagating effect, not a step change.
    """
    if strength == 0.0 or signal.neighbor_count == 0:
        return forecast
    h = len(forecast)
    ramp = np.arange(1, h + 1, dtype=np.float64) / h
    adjusted = forecast * (1.0 + strength * signal.aggregate_neighbor_trend * ramp)
    return np.maximum(adjusted, 0.0)
