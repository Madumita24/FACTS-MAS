"""
Factor Discovery & Attribution Agent -- FACTS-MAS Phase 2 extension.

Extends (does not replace) the existing skill_score_median + softmax fusion
mechanism from fusion_baseline.py with a regime dimension: instead of one
weight set per fold, computes one weight set per (regime, horizon) pair
within a fold, so a fast-moving hiking/cutting regime can down-weight an
agent that only looks skilled because the stable regime dominates the
fold-level validation window by volume.

Motivation: the full 15x6x3 backtest found the fused forecast beats naive
overall (3/3 horizons) but LOSES to naive at the 13-week horizon
specifically during hiking and cutting regimes (fused 8.77 vs naive 8.06 in
hiking; fused 14.31 vs naive 13.04 in cutting) -- masked in the pooled
number because "stable" origins outnumber hiking+cutting combined (5,310 vs
1,935+585 across the full backtest). AR holds 95%+ of the fold-level weight
in most folds, so its trend-smoothing behavior (already documented as a
weakness on fast-moving series -- see the Miami/Phoenix limitation in
ar_agent.md) is the prime suspect. This module tests that suspicion
directly and, if confirmed, gives the fusion layer a regime-aware lever --
WITHOUT discarding skill_score_median or softmax, just applying them within
a regime-filtered subset of the validation window instead of the whole
thing. Both are imported from fusion_baseline.py, not reimplemented.

Diagnostic-first: reports the regime x horizon weight grid and any
auto-generated gating rules. NOT wired into run_backtest()/
build_agent_runners() here -- that's a decision for after this is validated.
Does not touch intrinsic_agent.py or fusion_baseline.py.
"""
from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
import pandas as pd

from facts_mas.fusion_baseline import compute_weights_softmax, skill_score_median
from facts_mas.run_backtest import classify_regime, compute_mape

REGIMES = ["hiking", "cutting", "stable"]
HORIZONS = [4, 8, 13]

# Judgment calls, documented explicitly (same honesty standard as every
# other arbitrary-but-reasoned constant introduced tonight):
MIN_RELIABLE_N = 30          # below this, a cell can't safely generate a gating rule (Step 5)
GATING_SKILL_GAP_THRESHOLD = 0.10  # regime skill must trail the agent's OWN overall skill by
                                     # this much (not just be negative in isolation) to gate


def compute_regime_weights(
    weekly_df: pd.DataFrame,
    agent_runners: dict[str, callable],
    msas: list[str],
    val_window: tuple[datetime.date, datetime.date],
    horizon_weeks: int,
    regime: Optional[str],
    temperature: float = 0.05,
) -> tuple[dict[str, float], dict[str, list[float]], int]:
    """
    Same skill_score_median + softmax machinery compute_fold_weights uses
    (imported from fusion_baseline.py, not reimplemented), but restricted
    to origins classified in `regime` (via run_backtest.classify_regime,
    also imported not reimplemented) at a single `horizon_weeks` -- not
    pooled across all three horizons the way compute_fold_weights does,
    since the whole point here is to see each (regime, horizon) cell
    separately. regime=None means no filter (used for the "overall"
    per-horizon baseline Step 3's gating logic compares against).

    The per-point origin-walk and skill-score accumulation below IS new
    code (compute_fold_weights doesn't expose this loop as a separate,
    importable function -- it's inlined there), but the actual scoring
    math at each point (skill_score = 1 - agent_mape/naive_mape) and the
    aggregation/weighting steps are identical formulas to compute_fold_weights,
    and the aggregation/weighting calls themselves (skill_score_median,
    compute_weights_softmax) are imported and called, not duplicated.

    Returns (weights, per_agent_skill_points, n_points) -- raw per-point
    skill scores and the point count are returned alongside the weights so
    Step 4/5 can report actual skill scores and flag thin cells, not just
    the weights that came out of them.
    """
    val_start, val_end = val_window
    agent_points: dict[str, list[float]] = {name: [] for name in agent_runners}

    for msa in msas:
        msa_data = weekly_df[weekly_df["msa"] == msa].sort_values("date")
        origins = msa_data[
            (msa_data["date"] >= pd.Timestamp(val_start))
            & (msa_data["date"] <= pd.Timestamp(val_end))
        ]["date"].values

        for origin_ts in origins:
            origin = pd.Timestamp(origin_ts).date()

            if regime is not None and classify_regime(weekly_df, msa, origin) != regime:
                continue

            future_mask = msa_data["date"] > pd.Timestamp(origin)
            future_data = msa_data[future_mask].head(horizon_weeks)
            if len(future_data) < horizon_weeks:
                continue
            actuals = future_data["inventory_count"].values

            past_data = msa_data[msa_data["date"] <= pd.Timestamp(origin)]
            if past_data.empty:
                continue
            last_value = float(past_data["inventory_count"].iloc[-1])
            naive_forecast_arr = np.full(horizon_weeks, last_value)
            with np.errstate(divide="ignore", invalid="ignore"):
                naive_ape = np.abs((actuals - naive_forecast_arr) / actuals)
            naive_ape = naive_ape[np.isfinite(naive_ape)]
            if len(naive_ape) == 0:
                continue
            naive_mape = float(np.mean(naive_ape))
            if naive_mape <= 0:
                continue

            for agent_name, runner in agent_runners.items():
                try:
                    output = runner(weekly_df, msa, origin, horizon_weeks)
                    forecast = np.array(output.values)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ape = np.abs((actuals - forecast) / actuals)
                    ape = ape[np.isfinite(ape)]
                    if len(ape) == 0:
                        continue
                    agent_mape = float(np.mean(ape))
                    agent_points[agent_name].append(1.0 - agent_mape / naive_mape)
                except Exception:
                    agent_points[agent_name].append(-1.0)

    median_skills = {name: skill_score_median(points) for name, points in agent_points.items()}
    weights = compute_weights_softmax(median_skills, temperature=temperature)
    n_points = max((len(p) for p in agent_points.values()), default=0)

    return weights, agent_points, n_points


def compute_all_regime_weights(
    weekly_df: pd.DataFrame,
    agent_runners: dict[str, callable],
    msas: list[str],
    fold,
    lookback_weeks: int = 104,
    temperature: float = 0.05,
) -> dict:
    """
    3x3 grid: compute_regime_weights for every (regime, horizon) pair, PLUS
    an "overall" (non-regime-filtered) skill score per horizon -- needed by
    generate_factor_attribution_output's gating-rule logic, which flags an
    agent only when its regime-specific skill is MEANINGFULLY WORSE than
    its own overall skill at that horizon, not just negative in isolation
    (an agent that's mediocre everywhere shouldn't trigger a regime-specific
    gating rule just for being mediocre in one regime too).

    Same 104-week lookback default as compute_fold_weights, so this grid is
    a fair apples-to-apples decomposition of the same validation window
    the fold-level weights were computed from, not a different sample.

    Returns {
        "grid": {regime: {f"{h}wk": {"weights", "skill_scores", "n"}}},
        "overall": {f"{h}wk": {"skill_scores", "n"}},
    }
    """
    val_start = fold.train_end - datetime.timedelta(weeks=lookback_weeks)
    val_window = (val_start, fold.train_end)

    grid = {}
    for regime in REGIMES:
        grid[regime] = {}
        for h in HORIZONS:
            weights, points, n = compute_regime_weights(
                weekly_df, agent_runners, msas, val_window, h, regime, temperature=temperature,
            )
            skill_scores = {name: skill_score_median(p) for name, p in points.items()}
            grid[regime][f"{h}wk"] = {"weights": weights, "skill_scores": skill_scores, "n": n}

    overall = {}
    for h in HORIZONS:
        weights, points, n = compute_regime_weights(
            weekly_df, agent_runners, msas, val_window, h, regime=None, temperature=temperature,
        )
        skill_scores = {name: skill_score_median(p) for name, p in points.items()}
        overall[f"{h}wk"] = {"skill_scores": skill_scores, "n": n}

    return {"grid": grid, "overall": overall}


def generate_factor_attribution_output(regime_weights_grid: dict, fold) -> dict:
    """
    Matches the proposal's output structure. gating_rules are auto-generated
    from the data (regime skill negative AND trailing the agent's own
    overall skill at that horizon by more than GATING_SKILL_GAP_THRESHOLD),
    never hardcoded -- a rule like "downweight_ar_hiking_13wk" only appears
    here if the numbers actually produce it. Cells with fewer than
    MIN_RELIABLE_N points that WOULD otherwise qualify are reported in
    reliability_notes instead of silently generating an unreliable rule.
    """
    grid = regime_weights_grid["grid"]
    overall = regime_weights_grid["overall"]

    regime_weights_output = {
        regime: {h_label: cell["weights"] for h_label, cell in per_h.items()}
        for regime, per_h in grid.items()
    }

    gating_rules = []
    reliability_notes = []
    for regime, per_h in grid.items():
        for h_label, cell in per_h.items():
            overall_skills = overall[h_label]["skill_scores"]
            for agent, regime_skill in cell["skill_scores"].items():
                overall_skill = overall_skills.get(agent, 0.0)
                gap = overall_skill - regime_skill
                meaningfully_worse = regime_skill < 0 and gap > GATING_SKILL_GAP_THRESHOLD
                if not meaningfully_worse:
                    continue
                if cell["n"] < MIN_RELIABLE_N:
                    reliability_notes.append(
                        f"insufficient_data_{agent}_{regime}_{h_label} "
                        f"(n={cell['n']} < {MIN_RELIABLE_N}; would otherwise trigger a "
                        f"downweight rule -- NOT generated, too few points to trust)"
                    )
                    continue
                gating_rules.append(f"downweight_{agent}_{regime}_{h_label}")

    return {
        "fold": fold.fold,
        "regime_weights": regime_weights_output,
        "gating_rules": gating_rules,
        "reliability_notes": reliability_notes,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive mode selection -- per (fold, horizon), pick whichever weighting
# strategy (fold-level or regime-aware) actually scores better on THAT
# fold's own validation data, rather than hardcoding a horizon-to-mode
# mapping from fold 6's pattern alone. See run_backtest.py's
# weighting_mode="adaptive" for how this gets used.
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_validation_mape(
    weekly_df: pd.DataFrame,
    agent_runners: dict[str, callable],
    msas: list[str],
    fold,
    horizon_weeks: int,
    fold_weights: dict[str, float],
    regime_grid: dict,
    lookback_weeks: int = 104,
    regime_filter: Optional[str] = None,
) -> tuple[float, float]:
    """
    Walks the SAME validation window already used to compute fold_weights
    (compute_fold_weights) and regime_grid (compute_all_regime_weights),
    and computes fused MAPE under EACH weighting strategy at this horizon
    -- on validation data only, never the fold's test window, so the mode
    selection itself can never leak into the actual held-out evaluation.

    regime_filter: None (default) pools all 3 regimes together -- this is
    what weighting_mode="adaptive" uses, selecting one mode per (fold,
    horizon). Passing a specific regime restricts the walk to only that
    regime's origins -- this is what weighting_mode="adaptive_granular"
    uses, selecting one mode per (fold, horizon, regime), closing the gap
    "adaptive" left at 8wk/hiking (where the POOLED comparison favored
    "regime" even though "fold" was actually better within hiking
    specifically -- a granularity mismatch between the fold+horizon-level
    selection criterion and the fold+horizon+regime-level outcome).

    Calls each agent runner ONCE per validation origin, not twice: the
    agent forecasts themselves don't depend on which weights will combine
    them, so applying both weight sets to the same agent outputs halves
    what would otherwise be redundant (and already expensive) agent calls.

    Returns (fold_mode_mape, regime_mode_mape) -- lower is better; the
    caller picks whichever is lower.
    """
    val_start = fold.train_end - datetime.timedelta(weeks=lookback_weeks)
    val_end = fold.train_end

    fold_apes, regime_apes = [], []

    for msa in msas:
        msa_data = weekly_df[weekly_df["msa"] == msa].sort_values("date")
        origins = msa_data[
            (msa_data["date"] >= pd.Timestamp(val_start)) & (msa_data["date"] <= pd.Timestamp(val_end))
        ]["date"].values

        for origin_ts in origins:
            origin = pd.Timestamp(origin_ts).date()

            regime = classify_regime(weekly_df, msa, origin)
            if regime_filter is not None and regime != regime_filter:
                continue

            future = msa_data[msa_data["date"] > pd.Timestamp(origin)].head(horizon_weeks)
            if len(future) < horizon_weeks:
                continue
            actuals = future["inventory_count"].values.astype(np.float64)

            agent_outputs = {}
            ok = True
            for agent_name, runner in agent_runners.items():
                try:
                    agent_outputs[agent_name] = runner(weekly_df, msa, origin, horizon_weeks)
                except Exception:
                    ok = False
                    break
            if not ok:
                continue

            regime_weights = regime_grid[regime][f"{horizon_weeks}wk"]["weights"]

            fold_fused = np.zeros(horizon_weeks)
            regime_fused = np.zeros(horizon_weeks)
            for agent_name, output in agent_outputs.items():
                vals = np.array(output.values)
                fold_fused += fold_weights.get(agent_name, 0.0) * vals
                regime_fused += regime_weights.get(agent_name, 0.0) * vals
            fold_fused = np.maximum(fold_fused, 0.0)
            regime_fused = np.maximum(regime_fused, 0.0)

            fold_apes.append(compute_mape(actuals, fold_fused))
            regime_apes.append(compute_mape(actuals, regime_fused))

    fold_mape = float(np.mean(fold_apes)) if fold_apes else float("inf")
    regime_mape = float(np.mean(regime_apes)) if regime_apes else float("inf")
    return fold_mape, regime_mape


# ═══════════════════════════════════════════════════════════════════════════
# Production report -- Agent 6 (Factor Discovery & Attribution)'s finalized
# output format: agent_weights per (fold, horizon, regime) reflecting the
# ACTUAL adaptive_granular-selected weighting (not just the regime-aware
# grid in isolation), gating rules annotated with which mode each flagged
# cell uses in production, and a plain-language per-fold summary. This is
# what run_backtest.py's weighting_mode="adaptive_granular" computes
# internally every fold; this function exposes the same computation
# standalone (e.g. for documentation/reporting), independent of running a
# full backtest.
# ═══════════════════════════════════════════════════════════════════════════

def _summarize_horizon_selection(horizon: int, regime_choices: dict[str, str]) -> str:
    regime_mode_regimes = [r for r, m in regime_choices.items() if m == "regime"]
    fold_mode_regimes = [r for r, m in regime_choices.items() if m == "fold"]
    if not regime_mode_regimes:
        return f"{horizon}wk uses pooled fold-level weighting across all regimes"
    if not fold_mode_regimes:
        return f"{horizon}wk uses regime-conditional weighting across all regimes"
    return (f"{horizon}wk uses regime-conditional weighting during "
            f"{' and '.join(regime_mode_regimes)}, pooled fold-level weighting during "
            f"{' and '.join(fold_mode_regimes)}")


def generate_fold_summary(fold, mode_selection: dict[tuple[int, str], str]) -> str:
    """Plain-language description of the adaptive_granular pattern for one
    fold, e.g. 'Fold 6: 4wk uses pooled fold-level weighting across all
    regimes; 8wk uses regime-conditional weighting during cutting, pooled
    fold-level weighting during hiking and stable; 13wk uses regime-
    conditional weighting across all regimes.'
    """
    parts = [f"Fold {fold.fold}:"]
    for horizon in HORIZONS:
        regime_choices = {r: m for (h, r), m in mode_selection.items() if h == horizon}
        parts.append(_summarize_horizon_selection(horizon, regime_choices) + ";")
    return " ".join(parts).rstrip(";") + "."


def generate_production_report(
    weekly_df: pd.DataFrame,
    agent_runners: dict[str, callable],
    msas: list[str],
    fold,
    lookback_weeks: int = 104,
    temperature: float = 0.05,
) -> dict:
    """
    The finalized production report for one fold: agent_weights per
    (fold, horizon, regime) reflecting the weighting actually used (fold-
    level or regime-aware, whichever validated better for that specific
    cell -- i.e. the same selection weighting_mode="adaptive_granular"
    makes internally), auto-generated gating rules annotated with which
    mode each flagged cell uses, and a plain-language summary.
    """
    from facts_mas.fusion_baseline import compute_fold_weights

    fold_weights = compute_fold_weights(weekly_df, agent_runners, msas, train_end=fold.train_end)
    regime_data = compute_all_regime_weights(weekly_df, agent_runners, msas, fold, lookback_weeks, temperature)

    mode_selection: dict[tuple[int, str], str] = {}
    for horizon in HORIZONS:
        for regime in REGIMES:
            fold_val_mape, regime_val_mape = evaluate_validation_mape(
                weekly_df, agent_runners, msas, fold, horizon, fold_weights, regime_data["grid"],
                lookback_weeks=lookback_weeks, regime_filter=regime,
            )
            mode_selection[(horizon, regime)] = "fold" if fold_val_mape <= regime_val_mape else "regime"

    agent_weights_output = {}
    for regime in REGIMES:
        agent_weights_output[regime] = {}
        for horizon in HORIZONS:
            h_label = f"{horizon}wk"
            chosen = mode_selection[(horizon, regime)]
            weights_used = fold_weights if chosen == "fold" else regime_data["grid"][regime][h_label]["weights"]
            agent_weights_output[regime][h_label] = {"weights": weights_used, "mode_used": chosen}

    base_output = generate_factor_attribution_output(regime_data, fold)
    gating_rules_detail = []
    for rule in base_output["gating_rules"]:
        # rule format: downweight_{agent}_{regime}_{h}wk
        parts = rule.split("_")
        agent_name, regime_name, h_label = parts[1], parts[2], parts[3]
        horizon = int(h_label.replace("wk", ""))
        gating_rules_detail.append({
            "rule": rule, "agent": agent_name, "regime": regime_name, "horizon": h_label,
            "cell_weighting_mode": mode_selection.get((horizon, regime_name), "unknown"),
        })

    return {
        "fold": fold.fold,
        "agent_weights": agent_weights_output,
        "gating_rules": base_output["gating_rules"],
        "gating_rules_detail": gating_rules_detail,
        "reliability_notes": base_output["reliability_notes"],
        "mode_selection": {f"{h}wk_{r}": m for (h, r), m in mode_selection.items()},
        "summary": generate_fold_summary(fold, mode_selection),
    }
