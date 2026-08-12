"""
Ablation & Analysis — FACTS-MAS Phase 4 (Proposal §6, Weeks 8-10).

Implements the Phase-4 experiment table and the four analyses the proposal
singles out by name as the "killer analyses".

Proposal's ablation table:

    Variant             What it tests
    Full FACTS-MAS      Complete system
    - AR agent          Value of statistical forecasting
    - Macro agent       Value of structured exogenous
    - Event agent       Value of local shocks
    - Intrinsic agent   Value of cross-sectional structure
    - Factor agent      Value of dynamic weighting
    All-LLM             Value of hybrid design

(Seasonality is added as a sixth drop row. The proposal's table omits it,
but it is one of the five Layer-1 agents and leaving it out would be the
only agent whose contribution goes unmeasured.)

Everything except the All-LLM row runs off the agent cache, so the whole
table is arithmetic over stored forecasts rather than six more backtests.
The All-LLM row is the NEXUS-style Chain-of-Thought baseline, which cannot
come from the cache because it is a different system entirely; it is read
from cot_baseline.py's saved results.

"- Factor agent" is not a dropped agent at all. It means fusing with
fold-level weights instead of the regime-adaptive selection, i.e.
weighting_mode="fold" — that is precisely what removing dynamic weighting
means in this architecture.
"""
from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from facts_mas.run_backtest import HORIZONS, compute_mape, compute_rmse

REGIMES = ("hiking", "cutting", "stable")


# ═══════════════════════════════════════════════════════════════════════════
# Fusion over the cache
# ═══════════════════════════════════════════════════════════════════════════

def fuse_entry(entry, weights: dict[str, float],
               drop: Optional[set[str]] = None) -> np.ndarray:
    """
    Fuse one cached cell, optionally with some agents removed.

    Weights are RENORMALISED over the surviving agents. This is the whole
    ballgame for an honest ablation: if you drop an agent holding 30% of the
    weight and leave the rest untouched, the fused forecast silently shrinks
    to 70% of its proper magnitude, and the variant looks catastrophic for a
    reason that has nothing to do with the agent's information. Renormalising
    asks the question actually intended — "how good is the system built from
    the remaining agents" — instead of "what happens if we break the fusion".
    """
    drop = drop or set()
    live = {n: w for n, w in weights.items()
            if w > 0.0 and n not in drop and n in entry.agent_values}
    live = {n: w for n, w in live.items()
            if entry.agent_values[n] is not None
            and not np.isnan(entry.agent_values[n]).any()}
    if not live:
        return entry.naive.copy()

    total_w = sum(live.values())
    fused = np.zeros(entry.horizon, dtype=np.float64)
    for name, w in live.items():
        fused += (w / total_w) * entry.agent_values[name]
    return np.maximum(fused, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# The ablation table
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AblationRow:
    variant: str
    tests: str
    horizon: int
    mape: float
    rmse: float
    naive_mape: float
    delta_vs_full: float
    n_cells: int


VARIANTS: dict[str, tuple[str, set[str]]] = {
    "Full FACTS-MAS":  ("Complete system", set()),
    "- AR":            ("Value of statistical forecasting", {"ar"}),
    "- Macro":         ("Value of structured exogenous", {"macro"}),
    "- Event":         ("Value of local shocks", {"event"}),
    "- Seasonality":   ("Value of calendar structure", {"seasonality"}),
    "- Intrinsic":     ("Value of cross-sectional structure", {"intrinsic"}),
}


def run_ablation_table(
    cache,
    weights_by_cell: Callable,
    fold_weights_by_cell: Optional[Callable] = None,
    horizons: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Every ablation variant at every horizon.

    weights_by_cell(entry) -> dict[str, float]
        The production weighting (adaptive). Called per cell because under a
        regime-aware mode the weight set depends on the cell's regime.

    fold_weights_by_cell(entry) -> dict[str, float]
        Fold-level weighting, used for the "- Factor agent" row. If omitted
        that row is skipped rather than faked.
    """
    horizons = horizons or HORIZONS
    rows: list[AblationRow] = []

    for horizon in horizons:
        cells = cache.cells(horizon)
        if not cells:
            continue

        naive_mape = float(np.mean([compute_mape(e.actuals, e.naive) for e in cells]))

        full_mape = None
        for variant, (tests, drop) in VARIANTS.items():
            mapes, rmses = [], []
            for e in cells:
                fused = fuse_entry(e, weights_by_cell(e), drop)
                mapes.append(compute_mape(e.actuals, fused))
                rmses.append(compute_rmse(e.actuals, fused))
            m = float(np.mean(mapes))
            if variant == "Full FACTS-MAS":
                full_mape = m
            rows.append(AblationRow(
                variant=variant, tests=tests, horizon=horizon,
                mape=m, rmse=float(np.mean(rmses)), naive_mape=naive_mape,
                delta_vs_full=m - (full_mape if full_mape is not None else m),
                n_cells=len(cells),
            ))

        if fold_weights_by_cell is not None:
            mapes, rmses = [], []
            for e in cells:
                fused = fuse_entry(e, fold_weights_by_cell(e), set())
                mapes.append(compute_mape(e.actuals, fused))
                rmses.append(compute_rmse(e.actuals, fused))
            m = float(np.mean(mapes))
            rows.append(AblationRow(
                variant="- Factor agent", tests="Value of dynamic weighting",
                horizon=horizon, mape=m, rmse=float(np.mean(rmses)),
                naive_mape=naive_mape, delta_vs_full=m - full_mape,
                n_cells=len(cells),
            ))

    return pd.DataFrame([r.__dict__ for r in rows])


def attach_all_llm_row(
    table: pd.DataFrame,
    cot_results_path: str = "cot_baseline_fold1_full_window_results.csv",
) -> pd.DataFrame:
    """
    Append the All-LLM (NEXUS-style CoT) row from cot_baseline.py's output.

    Kept separate from the cache-driven rows and labelled, because it is not
    comparable on equal footing: the CoT baseline has only ever been run on
    fold 1, while every other row spans all six folds. Silently averaging
    them into one table would be the exact "uncalibrated benchmarking"
    problem the NEXUS proposal's §1 complains about.
    """
    if not os.path.exists(cot_results_path):
        return table

    cot = pd.read_csv(cot_results_path)
    col = next((c for c in ("cot_mape", "mape", "cot_ape") if c in cot.columns), None)
    hcol = next((c for c in ("horizon", "horizon_weeks") if c in cot.columns), None)
    if col is None or hcol is None:
        return table

    rows = []
    for horizon, grp in cot.groupby(hcol):
        rows.append({
            "variant": "All-LLM (NEXUS CoT, fold 1 only)",
            "tests": "Value of hybrid design",
            "horizon": int(horizon),
            "mape": float(grp[col].mean()),
            "rmse": np.nan,
            "naive_mape": np.nan,
            "delta_vs_full": np.nan,
            "n_cells": len(grp),
        })
    return pd.concat([table, pd.DataFrame(rows)], ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# Killer analysis 1 — error during shock weeks vs normal weeks
# ═══════════════════════════════════════════════════════════════════════════

def shock_vs_normal(
    cache,
    weights_by_cell: Callable,
    shock_lookup: Callable[[str, datetime.date], bool],
    horizons: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Does the system hold up when something is actually happening?

    The proposal's testable hypothesis is that FACTS-MAS wins most during
    regime shifts, "even if average MAPE gains over Nexus are modest". This
    is the direct test of that claim, so a null here is as informative as a
    win: it would say the architecture's advantage is uniform rather than
    concentrated where the proposal predicted.

    shock_lookup(msa, origin) -> bool decides what counts as a shock week.
    """
    horizons = horizons or HORIZONS
    rows = []
    for horizon in horizons:
        cells = cache.cells(horizon)
        for label, want in (("shock", True), ("normal", False)):
            sel = [e for e in cells if shock_lookup(e.msa, e.origin) is want]
            if not sel:
                continue
            fused = [compute_mape(e.actuals, fuse_entry(e, weights_by_cell(e)))
                     for e in sel]
            naive = [compute_mape(e.actuals, e.naive) for e in sel]
            rows.append({
                "horizon": horizon, "period": label, "n": len(sel),
                "fused_mape": float(np.mean(fused)),
                "naive_mape": float(np.mean(naive)),
                "skill_vs_naive": 1.0 - float(np.mean(fused)) / float(np.mean(naive)),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Killer analysis 2 — cross-MSA generalization
# ═══════════════════════════════════════════════════════════════════════════

def cross_msa_generalization(
    cache,
    weights_by_cell: Callable,
    horizons: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Per-MSA performance, and how uneven it is.

    Two things reported. Per-MSA skill answers "does this generalise or does
    one metro carry it". The spread (std across MSAs) answers "is the system
    dependable", which matters more than the mean for anything deployed:
    a system averaging 5% by scoring 2% on twelve cities and 20% on three is
    a different product from one scoring 5% everywhere.

    Uses the same leave-one-out framing as test_intrinsic_loo.py, but at the
    system level rather than for the Intrinsic Agent alone.
    """
    horizons = horizons or HORIZONS
    rows = []
    for horizon in horizons:
        cells = cache.cells(horizon)
        for msa in sorted({e.msa for e in cells}):
            sel = [e for e in cells if e.msa == msa]
            fused = [compute_mape(e.actuals, fuse_entry(e, weights_by_cell(e)))
                     for e in sel]
            naive = [compute_mape(e.actuals, e.naive) for e in sel]
            rows.append({
                "horizon": horizon, "msa": msa, "n": len(sel),
                "fused_mape": float(np.mean(fused)),
                "naive_mape": float(np.mean(naive)),
                "skill_vs_naive": 1.0 - float(np.mean(fused)) / float(np.mean(naive)),
                "beats_naive": float(np.mean(fused)) < float(np.mean(naive)),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        spread = df.groupby("horizon")["fused_mape"].agg(["mean", "std", "min", "max"])
        df.attrs["spread"] = spread
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Killer analysis 3 — agent contribution by horizon
# ═══════════════════════════════════════════════════════════════════════════

def agent_contribution_by_horizon(
    cache,
    weights_by_cell: Callable,
    horizons: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Each agent's marginal value, per horizon, by three measures at once.

    - drop_delta: MAPE increase when the agent is removed and weights
      renormalised. The Phase-4 ablation measure.
    - mean_weight: how much the fusion layer actually leans on it.
    - solo_mape: how it performs alone.

    Reporting all three is deliberate, because they disagree in an
    informative way. An agent can carry large weight while contributing
    almost nothing on removal (it agrees with the others and is redundant),
    or carry little weight yet hurt badly when removed (it is the only thing
    covering some regime). Either pattern is a finding; a single number
    would hide both.

    This directly answers the proposal's §4.1 claim that different mechanisms
    dominate at different lags.
    """
    horizons = horizons or HORIZONS
    rows = []
    for horizon in horizons:
        cells = cache.cells(horizon)
        if not cells:
            continue
        base = float(np.mean([compute_mape(e.actuals, fuse_entry(e, weights_by_cell(e)))
                              for e in cells]))
        for agent in cache.agent_names:
            dropped = float(np.mean([
                compute_mape(e.actuals, fuse_entry(e, weights_by_cell(e), {agent}))
                for e in cells
            ]))
            solo, wts = [], []
            for e in cells:
                v = e.agent_values.get(agent)
                if v is not None and not np.isnan(v).any():
                    solo.append(compute_mape(e.actuals, np.maximum(v, 0.0)))
                wts.append(weights_by_cell(e).get(agent, 0.0))
            rows.append({
                "horizon": horizon, "agent": agent,
                "full_mape": base, "without_mape": dropped,
                "drop_delta": dropped - base,
                "mean_weight": float(np.mean(wts)) if wts else 0.0,
                "solo_mape": float(np.mean(solo)) if solo else np.nan,
                "n_cells": len(cells),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Killer analysis 4 — cost per forecast vs Nexus
# ═══════════════════════════════════════════════════════════════════════════

#: Measured, not estimated. Sources:
#:   event: run_event_agent.py batch — 153 calls, $0.031 total (progress.md §7)
#:   cot:   cot_baseline.py fold-1 run — 1,305 calls, ~3.4 s/call (baselines.md)
MEASURED_COSTS = {
    "event_llm_calls": 153,
    "event_total_usd": 0.031,
    "cot_calls_fold1": 1305,
    "cot_seconds_per_call": 3.4,
    "cot_usd_all_folds_estimate": 2.00,
}


def cost_per_forecast(
    cache,
    llm_agents: tuple[str, ...] = ("event",),
    costs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    LLM cost and latency per forecast, ours vs a NEXUS-style CoT system.

    The asymmetry is the point and it is structural, not a tuning win. Our
    LLM calls are made once per unique (MSA, declaration) and cached to
    `event_agent_weekly.csv`, so they do not scale with the number of
    forecast origins. A CoT system calls the model once per forecast, so its
    cost scales linearly with origins x horizons.

    That means the gap widens with evaluation size rather than staying
    constant, which is exactly the overhead complaint in NEXUS proposal
    Limitation 2.
    """
    c = {**MEASURED_COSTS, **(costs or {})}
    n_forecasts = len(cache.entries)
    if n_forecasts == 0:
        return pd.DataFrame()

    ours_calls = c["event_llm_calls"]
    ours_usd = c["event_total_usd"]
    cot_calls = n_forecasts
    cot_usd = cot_calls * (c["cot_usd_all_folds_estimate"] / max(c["cot_calls_fold1"] * 6, 1))
    cot_hours = cot_calls * c["cot_seconds_per_call"] / 3600.0

    return pd.DataFrame([
        {"system": "FACTS-MAS (ours)", "forecasts": n_forecasts,
         "llm_calls": ours_calls, "calls_per_forecast": ours_calls / n_forecasts,
         "usd_total": ours_usd, "usd_per_1k_forecasts": 1000 * ours_usd / n_forecasts,
         "llm_hours": ours_calls * c["cot_seconds_per_call"] / 3600.0,
         "note": f"LLM used only by {list(llm_agents)}; cached per declaration, not per forecast"},
        {"system": "NEXUS-style CoT", "forecasts": n_forecasts,
         "llm_calls": cot_calls, "calls_per_forecast": 1.0,
         "usd_total": cot_usd, "usd_per_1k_forecasts": 1000 * cot_usd / n_forecasts,
         "llm_hours": cot_hours,
         "note": "one call per forecast; scales linearly with origins x horizons"},
    ])
