"""
Factor Scoring — FACTS-MAS Phase 3 (Proposal §6, Weeks 6-7).

The Phase-3 checklist asks for factor scoring by four methods:

    rolling correlation · Granger screening · permutation importance ·
    event confidence

Only the last two had any implementation. `factor_attribution.py` scores
agents by skill (1 - agent_mape/naive_mape) and the Event Agent already
carries a confidence field, but nothing computed rolling correlation or
Granger screening, and permutation importance had no home at all.

This module supplies the three missing methods. It scores two different
things and the distinction matters:

    FACTORS are input variables (mortgage_rate, fed_funds, ...). Rolling
    correlation and Granger screening score these — they ask whether a
    variable carries information about future inventory.

    AGENTS are forecasters. Permutation importance scores these — it asks
    how much worse the fused forecast gets when one agent's contribution is
    destroyed. That is the direct measure of "does this agent earn its
    slot", and it is the cheap cousin of the Phase-4 ablation table.

Deliberately does not touch factor_attribution.py, which is Madumita's
under the one-file-one-owner rule. Skill scoring and softmax weighting are
imported from her module and this one, not reimplemented.

Granger screening note: the pairwise MSA-to-MSA screen here is the same
computation NEXUS Mod B (Cross-Regional Spatial Spillover) needs for its
edge weights. Built once, usable for both — `granger_spillover_graph`
returns edges in the shape `state/spatial.py`'s GrangerEdge expects.
"""
from __future__ import annotations

import contextlib
import datetime
import io
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from facts_mas.run_backtest import HORIZONS, compute_mape

# Judgment calls, stated rather than buried:
DEFAULT_ROLLING_WINDOW = 52      # one year of weekly observations
DEFAULT_MAX_LAG = 8              # weeks; covers the documented 4-8wk
                                 # mortgage transmission lag and the
                                 # proposal's 4-8wk spillover lag
GRANGER_ALPHA = 0.05
MIN_OBS_FOR_GRANGER = 60         # below this the F-test is not trustworthy
                                 # at maxlag=8 (8 lags x 2 series + margin)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Rolling correlation
# ═══════════════════════════════════════════════════════════════════════════

def rolling_correlation_scores(
    df: pd.DataFrame,
    msa: str,
    factors: list[str],
    as_of: datetime.date,
    horizon_weeks: int,
    window: int = DEFAULT_ROLLING_WINDOW,
    lookback_weeks: int = 104,
) -> dict[str, float]:
    """
    Correlation between each factor and FUTURE inventory change.

    Two choices worth flagging, because the naive version of this is
    misleading:

    1. Correlate against the forward CHANGE in inventory, not its level.
       Inventory and most macro series both trend over 2018-2026, so a
       level-on-level correlation mostly measures "both drift", which is
       the exact spurious-trend trap `validate_macro_lags.py` already
       caught once during the Macro Agent investigation.

    2. Score on the last `window` observations ending at `as_of`, never
       past it. Everything in this module is a point-in-time score.

    Returns {factor: pearson_r}. Sign is meaningful: negative means the
    factor rising precedes inventory falling.
    """
    md = df[(df["msa"] == msa) & (df["date"] <= pd.Timestamp(as_of))].sort_values("date")
    start = pd.Timestamp(as_of) - pd.Timedelta(weeks=lookback_weeks)
    md = md[md["date"] >= start]

    inv = md["inventory_count"].astype(float)
    # Forward change over the forecast horizon, aligned to the origin row.
    forward = inv.shift(-horizon_weeks) - inv

    out: dict[str, float] = {}
    for factor in factors:
        if factor not in md.columns:
            continue
        pair = pd.concat([md[factor].astype(float), forward], axis=1).dropna()
        if len(pair) < 10:
            out[factor] = 0.0
            continue
        tail = pair.tail(window)
        if tail.iloc[:, 0].std() == 0 or tail.iloc[:, 1].std() == 0:
            out[factor] = 0.0
            continue
        out[factor] = float(tail.iloc[:, 0].corr(tail.iloc[:, 1]))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 2. Granger screening
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GrangerResult:
    source: str
    target: str
    best_lag: int
    p_value: float
    significant: bool
    n_obs: int


def _granger_pair(
    source: np.ndarray,
    target: np.ndarray,
    max_lag: int = DEFAULT_MAX_LAG,
) -> tuple[int, float]:
    """
    Does `source`'s past help predict `target` beyond `target`'s own past?

    Returns (best_lag, min_p). Runs on first differences: Granger's F-test
    assumes stationarity, and raw weekly inventory is strongly trending, so
    testing it in levels inflates significance. Differencing is the standard
    correction and matches what `validate_macro_panel.py` concluded about
    trend contamination in this dataset.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    s = np.diff(np.asarray(source, dtype=float))
    t = np.diff(np.asarray(target, dtype=float))
    n = min(len(s), len(t))
    if n < MIN_OBS_FOR_GRANGER:
        return 0, 1.0

    data = np.column_stack([t[-n:], s[-n:]])  # [target, source] order matters
    if np.std(data[:, 0]) == 0 or np.std(data[:, 1]) == 0:
        return 0, 1.0

    best_lag, best_p = 0, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # grangercausalitytests prints a full report per lag to stdout and
            # offers no way to turn that off in current statsmodels. Across 210
            # ordered pairs x 8 lags that is thousands of blocks of text, which
            # buries whatever the caller was actually printing. Swallowed here
            # rather than at each call site.
            with contextlib.redirect_stdout(io.StringIO()):
                res = grangercausalitytests(data, maxlag=max_lag)
        except Exception:
            return 0, 1.0
        n_tested = 0
        for lag, (stats, _) in res.items():
            n_tested += 1
            p = float(stats["ssr_ftest"][1])
            if p < best_p:
                best_lag, best_p = int(lag), p

    # Correct for having searched over lags. Taking the minimum p-value across
    # `max_lag` separate tests and reporting it as if it were a single test is
    # itself an uncorrected multiple comparison: with 8 lags, even a pair with
    # no relationship at all has roughly a 1 - 0.95^8 = 34% chance that some
    # lag lands under 0.05. Bonferroni over the lags searched is the
    # conservative correction, and conservative is the right side to err on
    # when the output is a claim about causal spillover.
    if n_tested > 1:
        best_p = min(1.0, best_p * n_tested)
    return best_lag, best_p


def granger_screen_factors(
    df: pd.DataFrame,
    msa: str,
    factors: list[str],
    as_of: datetime.date,
    max_lag: int = DEFAULT_MAX_LAG,
    lookback_weeks: int = 104,
) -> dict[str, GrangerResult]:
    """
    Screen each macro factor against this MSA's inventory series.

    Point-in-time: only data at or before `as_of` is used.
    """
    md = df[(df["msa"] == msa) & (df["date"] <= pd.Timestamp(as_of))].sort_values("date")
    start = pd.Timestamp(as_of) - pd.Timedelta(weeks=lookback_weeks)
    md = md[md["date"] >= start]

    target = md["inventory_count"].values.astype(float)
    out: dict[str, GrangerResult] = {}
    for factor in factors:
        if factor not in md.columns:
            continue
        lag, p = _granger_pair(md[factor].values.astype(float), target, max_lag)
        out[factor] = GrangerResult(
            source=factor, target=f"{msa}:inventory", best_lag=lag,
            p_value=p, significant=p < GRANGER_ALPHA, n_obs=len(md),
        )
    return out


def granger_spillover_graph(
    df: pd.DataFrame,
    msas: Optional[list[str]] = None,
    as_of: Optional[datetime.date] = None,
    max_lag: int = DEFAULT_MAX_LAG,
    lookback_weeks: int = 104,
    remove_common_factor: bool = True,
    fdr_correct: bool = True,
) -> list[GrangerResult]:
    """
    Every ordered MSA pair, screened for inventory spillover.

    This is Phase-3 factor screening AND the edge computation NEXUS Mod B
    needs. The proposal's argument for Granger over plain correlation
    applies unchanged: two cities can move together purely because both
    react to the same national rate change, and correlation cannot tell
    that apart from one city actually leading the other.

    Returns one GrangerResult per ordered pair (source != target), so
    A->B and B->A are scored separately — spillover is directional.
    """
    msas = msas or sorted(df["msa"].unique().tolist())
    cutoff = pd.Timestamp(as_of) if as_of else df["date"].max()
    start = cutoff - pd.Timedelta(weeks=lookback_weeks)

    frames = {}
    for msa in msas:
        md = df[(df["msa"] == msa) & (df["date"] <= cutoff) & (df["date"] >= start)]
        md = md.sort_values("date")
        frames[msa] = pd.Series(md["inventory_count"].values.astype(float),
                                index=md["date"].values)

    series = _strip_national_factor(frames) if remove_common_factor else \
        {m: f.values for m, f in frames.items()}

    results: list[GrangerResult] = []
    for src in msas:
        for tgt in msas:
            if src == tgt:
                continue
            lag, p = _granger_pair(series[src], series[tgt], max_lag)
            results.append(GrangerResult(
                source=src, target=tgt, best_lag=lag, p_value=p,
                significant=False, n_obs=len(series[tgt]),
            ))

    return _apply_fdr(results) if fdr_correct else [
        GrangerResult(**{**r.__dict__, "significant": r.p_value < GRANGER_ALPHA})
        for r in results
    ]


def _strip_national_factor(frames: dict[str, pd.Series]) -> dict[str, np.ndarray]:
    """
    Remove the component every metro shares before testing for spillover.

    All 15 cities react to the same national mortgage rates and the same
    national news. That shared movement makes every pair look related, and
    Granger cannot tell "both followed the country" apart from "one led the
    other". Subtracting the cross-sectional mean at each week removes the
    common component and leaves each city's idiosyncratic movement, which is
    the only thing spillover could actually travel through.

    This is the same reasoning `validate_macro_panel.py` applied when it
    added time fixed effects to the macro regression, and for the same
    reason: without it, a shared trend masquerades as a relationship.
    """
    panel = pd.DataFrame(frames).sort_index()
    # Work in pct-change space so a large city's absolute swings do not
    # dominate the "national" average.
    pct = panel.pct_change()
    national = pct.mean(axis=1)
    residual = pct.sub(national, axis=0).fillna(0.0)
    # Re-integrate to a level series so _granger_pair's own differencing
    # still receives something with the shape it expects.
    return {c: (1.0 + residual[c]).cumprod().values for c in panel.columns}


def _apply_fdr(results: list[GrangerResult]) -> list[GrangerResult]:
    """
    Benjamini-Hochberg correction across the whole screen.

    With 15 metros there are 210 ordered pairs, so testing each at p<0.05
    would be expected to produce around 10 significant edges by chance alone
    even if no city influenced any other. Reporting an uncorrected count as
    a spillover finding would be overstating it. BH controls the false
    discovery rate across the family of tests rather than each test in
    isolation, and is the standard correction when the tests are this many
    and this correlated.
    """
    ordered = sorted(results, key=lambda r: r.p_value)
    n = len(ordered)
    cutoff_rank = 0
    for i, r in enumerate(ordered, start=1):
        if r.p_value <= GRANGER_ALPHA * i / n:
            cutoff_rank = i
    survivors = {id(r) for r in ordered[:cutoff_rank]}
    return [
        GrangerResult(source=r.source, target=r.target, best_lag=r.best_lag,
                      p_value=r.p_value, significant=id(r) in survivors,
                      n_obs=r.n_obs)
        for r in results
    ]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Permutation importance
# ═══════════════════════════════════════════════════════════════════════════

def permutation_importance(
    cache,
    weights: dict[str, float],
    horizon: Optional[int] = None,
    n_repeats: int = 10,
    seed: int = 0,
    regime: Optional[str] = None,
) -> dict[str, dict[str, float]]:
    """
    How much worse does fusion get when one agent's forecasts are shuffled?

    Shuffling breaks the alignment between an agent's forecast and the cell
    it belongs to while preserving that agent's marginal distribution. Unlike
    simply dropping the agent, this isolates *informational* value from the
    mere fact of contributing a number in the right ballpark: an agent whose
    output is essentially "last value plus noise" survives dropping badly but
    survives shuffling fine.

    SHUFFLING IS WITHIN (msa, horizon), NOT ACROSS THE WHOLE PANEL. That
    restriction is load-bearing, not a detail. Inventory levels differ by
    more than an order of magnitude across MSAs (New York ~8M housing units
    vs Seattle ~1.7M), so a panel-wide shuffle drops a New York forecast into
    a Seattle cell and the resulting error explosion measures scale mismatch,
    not lost information. An early version of this function did exactly that
    and reported AR's importance as ~63 MAPE points against a baseline of
    1.34, which is not a finding, it is an artifact. Permuting within an MSA
    destroys the temporal alignment (the thing whose value we want to
    measure) while keeping the magnitude plausible.

    Runs entirely over the cache: no agent is re-executed, so this is
    arithmetic and takes seconds.

    Returns {agent: {"baseline_mape", "permuted_mape", "importance",
                     "std", "n_cells"}} where importance is the MAPE
    increase caused by destroying that agent's alignment. Higher means more
    load-bearing. Negative means the agent was actively hurting.
    """
    rng = np.random.default_rng(seed)
    cells = [e for e in cache.cells(horizon)
             if regime is None or e.regime == regime]
    if not cells:
        return {}

    agents = [a for a in cache.agent_names if weights.get(a, 0.0) > 0.0]

    def fuse(entry, swap: Optional[tuple[str, np.ndarray]] = None) -> np.ndarray:
        total = np.zeros(entry.horizon, dtype=np.float64)
        for name in cache.agent_names:
            w = weights.get(name, 0.0)
            if w <= 0.0:
                continue
            vals = swap[1] if (swap and swap[0] == name) else entry.agent_values.get(name)
            if vals is None or np.isnan(vals).any():
                continue
            total += w * vals
        return np.maximum(total, 0.0)

    baseline = float(np.mean([compute_mape(e.actuals, fuse(e)) for e in cells]))

    # Group cell indices by (msa, horizon) so a permutation only ever moves a
    # forecast to another origin in the same city at the same horizon.
    groups: dict[tuple, list[int]] = {}
    for idx, e in enumerate(cells):
        groups.setdefault((e.msa, e.horizon), []).append(idx)

    out: dict[str, dict[str, float]] = {}
    for agent in agents:
        scores = []
        for _ in range(n_repeats):
            # Build a full index remap: within each group, cell i borrows the
            # agent's forecast from a randomly chosen other cell of the group.
            remap: dict[int, int] = {}
            for _key, idxs in groups.items():
                usable = [i for i in idxs
                          if cells[i].agent_values.get(agent) is not None
                          and not np.isnan(cells[i].agent_values[agent]).any()]
                if len(usable) < 2:
                    continue  # nothing to shuffle against; leave as-is
                shuffled = list(rng.permutation(usable))
                for src, dst in zip(usable, shuffled):
                    remap[src] = dst

            apes = []
            for idx, e in enumerate(cells):
                donor = remap.get(idx)
                if donor is None:
                    apes.append(compute_mape(e.actuals, fuse(e)))
                else:
                    apes.append(compute_mape(
                        e.actuals, fuse(e, (agent, cells[donor].agent_values[agent]))
                    ))
            scores.append(float(np.mean(apes)))

        out[agent] = {
            "baseline_mape": baseline,
            "permuted_mape": float(np.mean(scores)),
            "importance": float(np.mean(scores)) - baseline,
            "std": float(np.std(scores)),
            "n_cells": len(cells),
        }
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Combined Phase-3 factor report
# ═══════════════════════════════════════════════════════════════════════════

MACRO_FACTORS = ["mortgage_rate", "fed_funds", "cpi", "unemployment"]


def score_all_factors(
    df: pd.DataFrame,
    msas: list[str],
    as_of: datetime.date,
    factors: Optional[list[str]] = None,
    horizons: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Rolling correlation + Granger screening for every (msa, factor, horizon).

    One tidy frame so the results can be aggregated across MSAs, which is
    the level the Macro Agent investigation showed actually matters: a
    factor significant in 2 of 15 MSAs is noise, one significant in 12 is a
    finding.
    """
    factors = factors or MACRO_FACTORS
    horizons = horizons or HORIZONS

    rows = []
    for msa in msas:
        granger = granger_screen_factors(df, msa, factors, as_of)
        for horizon in horizons:
            corrs = rolling_correlation_scores(df, msa, factors, as_of, horizon)
            for factor in factors:
                g = granger.get(factor)
                rows.append({
                    "msa": msa,
                    "factor": factor,
                    "horizon": horizon,
                    "rolling_corr": corrs.get(factor, np.nan),
                    "granger_p": g.p_value if g else np.nan,
                    "granger_lag": g.best_lag if g else np.nan,
                    "granger_sig": g.significant if g else False,
                })
    return pd.DataFrame(rows)
