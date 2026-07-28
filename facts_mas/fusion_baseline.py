"""
Rolling-Weight Fusion Baseline — FACTS-MAS Phase 2.

Combines all agents via weights recomputed per backtest fold as new weekly
data rolls in, not one static weight set for the whole test period
(per Prof. Pan's note on adjustable weighting).

Method (current, post-diagnosis -- see history below):
    Skill-relative-to-naive, softmax weighting. For each agent, at each
    (msa, origin, horizon) point in the validation window:
        skill_score = 1 - (agent_MAPE / naive_MAPE_at_that_exact_point)
    aggregated across all points via the MEDIAN (diagnosis #4 -- the mean
    was numerically broken), then converted to weights via softmax
    (diagnosis #5 -- a hard epsilon floor couldn't distinguish "barely
    negative" from "very negative"):
        weight_i = exp(median_skill_i / T) / sum_j exp(median_skill_j / T)
    normalized to sum to 1.0 by construction. skill_score ≈ 0 means "no
    better than naive"; softmax handles negative skill_scores natively,
    with no floor or clip, so the full rank order among agents (including
    negative-skill ones) is always preserved.

Diagnosis history (documented in full for Madumita/Umang -- three attempts,
not a silent one-shot change):
    1. Original design: 20-week lookback, single 8-week horizon, linear
       inverse-MAPE (weight ∝ 1/MAPE). Could not separate a validated-null
       agent (Macro -- confirmed null via a two-way fixed-effects panel
       regression across 5,850 rows in validate_macro_panel.py) from agents
       with real signal: a live sanity check showed Macro landing within a
       few points of AR's MAPE and getting a comparable fusion weight.
    2. First fix attempt: widened the validation window to 104 weeks and
       aggregated MAPE across all three horizons (4/8/13wk), and steepened
       the weighting to 1/MAPE^2. This did NOT work -- re-running the same
       sanity check showed Macro's weight going UP, not down (0.175 ->
       0.228), landing as the second-highest weight in the lineup. Root
       cause turned out to be different from what was assumed: it wasn't a
       statistical-power problem. Macro's ridge regression, having found
       essentially no real signal, produces heavily-regularized (near-zero)
       coefficients, so its forecast ends up almost indistinguishable from
       a naive last-value forecast -- and naive itself is a genuinely
       decent short-horizon forecast (established separately in the
       AR-vs-naive validation). Raw MAPE rewards "looks like naive"
       regardless of WHY an agent looks like naive, so it structurally
       cannot distinguish "near-naive because it has no signal" (Macro)
       from "genuinely skilled and happens to track close to the truth."
       Widening the window or steepening the exponent doesn't fix this --
       it's not noise obscuring a real gap, there mostly isn't a gap in raw
       MAPE terms to find.
    3. Second (current) fix: replaced the scoring criterion itself, not just
       its parameters. skill_score = 1 - agent_MAPE/naive_MAPE measures
       improvement OVER naive specifically, computed pointwise at the exact
       same (msa, origin, horizon) naive is evaluated at (not a pooled or
       averaged naive figure, so it's always apples-to-apples). An agent
       that merely matches naive scores ~0 and gets floored to a small
       epsilon weight regardless of how "accurate" its absolute forecast
       looked -- closeness-to-truth at short horizons is naive's free
       lunch, not a skill any agent should be credited for reproducing.
    4. Bug found in fix #3's own implementation: averaging the per-point
       skill_score ratios directly is numerically unstable. Whenever
       naive_MAPE happens to be near-zero at a given point (naive got that
       one point nearly exactly right), dividing by it blows the ratio up
       to an extreme value unrelated to the agent's real skill -- confirmed
       empirically: AR's MEDIAN per-point skill score was +0.227 (genuinely
       good, matching everything else known about AR), but its MEAN was
       -0.407, dragged there by outliers as extreme as -32.09 at single
       points. This flattened ALL agents (including AR) to negative mean
       skill, collapsing every fusion weight to a uniform 0.20 -- a
       different, purely numerical failure from #1/#2, not a finding about
       any agent's real skill.
       Two candidate fixes were built and compared empirically (fold 1, all
       15 MSAs, 104-week lookback, 3 horizons, ~4,724 points/agent):
       skill_score_median (median of per-point ratios, robust to outliers)
       and skill_score_pooled (1 - mean(agent_apes)/mean(naive_apes), a
       ratio of pooled means instead of a mean of ratios). Both fixed the
       collapse decisively (fold 1: AR weight ~0.998 under median vs. ~0.985
       under pooled; Macro reduced to ~0.15%-1.5% of AR's weight under
       either). Cross-checked against validate_macro_panel.py's rigorous
       null finding as ground truth (Macro should end up with a LOW weight
       relative to AR, which has demonstrated real skill): median gave
       Macro roughly 10x LOWER weight relative to AR than pooled did
       (0.0015 vs 0.0153 as a fraction of AR's weight), so median was
       selected as the production method. Re-checked across folds 1, 3,
       and 5 (spanning 2021-2024): Macro's skill score stayed near-zero or
       negative in every fold tested -- a stable pattern independently
       corroborating the panel regression's null finding via a completely
       different method (rolling forecast comparison vs. two-way FE
       regression), not an artifact of one window. skill_score_pooled is
       kept in this file as a documented alternative, not deleted --
       equally valid, marginally less aggressive on this specific
       cross-check.
    5. Bug found in fix #3/#4's weight conversion (compute_weights_from_skill):
       flooring EVERY non-positive skill_score to the same fixed epsilon
       before exponentiation destroyed ranking information among
       negative-skill agents. Confirmed empirically after fixing Event and
       Intrinsic's underlying agent bugs (see facts_mas/agents/): Intrinsic's
       median skill improved from -0.2152 to -0.1400 (a real, ~35% move
       toward zero), but its fold-1 weight was IDENTICAL before and after
       the fix, because both values floor to the same epsilon=1e-3. An
       agent that's barely-bad and one that's very-bad were indistinguishable
       to the weighting mechanism. Replaced compute_weights_from_skill with
       compute_weights_softmax: exp(skill_i/T) normalized across agents,
       which handles negative skill_scores natively (no floor/clip) and
       preserves full rank order. Preferred over a shift-and-power
       alternative (skill + shift, then power) specifically because a
       batch-dependent shift (chosen from that fold's minimum skill_score)
       would make a given agent's weight depend on how bad the WORST OTHER
       agent in that fold happened to be -- reintroducing the same kind of
       context-dependent instability diagnosis #4 already had to fix.
"""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
import pandas as pd

from facts_mas.schema import AgentOutput, FusionInput
from facts_mas.validation import validate_fusion_input


def compute_agent_weights(
    agent_errors: dict[str, float],
    weight_power: float = 2.0,
) -> dict[str, float]:
    """
    Compute fusion weights from per-agent MAPE scores.

    SUPERSEDED as of the Step E fix (see module docstring, diagnosis #2):
    raw-MAPE weighting -- even at p=2 with a widened window -- could not
    distinguish an agent that's near-naive because it has no real signal
    (Macro) from one that's near-naive while genuinely skilled, because
    naive itself is a decent short-horizon forecast. compute_fold_weights
    now calls compute_weights_from_skill instead. Left defined here for
    reference / in case another caller still wants raw-MAPE weighting for
    a different purpose -- not deleted, just no longer the default path.

    Method: inverse-MAPE^p weighting.
        weight_i = (1 / mape_i^p) / sum(1 / mape_j^p for all j)

    If an agent has MAPE = 0 (perfect), it gets a large finite weight.
    If all agents have the same MAPE, weights are uniform regardless of p.

    Args:
        agent_errors: {agent_name: mape_on_validation_data}
        weight_power: exponent p in weight ∝ 1/MAPE^p.

    Returns:
        {agent_name: weight}, normalized to sum to 1.0.
    """
    if not agent_errors:
        return {}

    # Clamp MAPEs to a small floor to avoid division by zero
    eps = 1e-6
    inverse_mapes = {
        name: 1.0 / (max(mape, eps) ** weight_power) for name, mape in agent_errors.items()
    }
    total = sum(inverse_mapes.values())

    return {name: inv / total for name, inv in inverse_mapes.items()}


def compute_weights_from_skill(
    skill_scores: dict[str, float],
    weight_power: float = 2.0,
    epsilon: float = 1e-3,
) -> dict[str, float]:
    """
    Compute fusion weights from per-agent skill-relative-to-naive scores.

    SUPERSEDED as of the Step Y fix (see module docstring, diagnosis #5):
    flooring every non-positive skill_score to the SAME epsilon meant an
    agent that improved from very-bad to barely-bad (e.g. Intrinsic moving
    from -0.2152 to -0.1400 after a real fix) showed IDENTICAL weight
    before and after, because both values floor to the same epsilon --
    the floor destroys ranking information among negative-skill agents.
    compute_fold_weights now calls compute_weights_softmax instead. Left
    defined here for reference -- not deleted.

    weight_i ∝ max(skill_i, epsilon)^p, normalized to sum to 1.0.

    Args:
        skill_scores: {agent_name: avg_skill_score_on_validation_data}
        weight_power: exponent p in weight ∝ max(skill, epsilon)^p.
        epsilon: floor for skill_score before exponentiation.

    Returns:
        {agent_name: weight}, normalized to sum to 1.0.
    """
    if not skill_scores:
        return {}

    floored = {
        name: max(score, epsilon) ** weight_power for name, score in skill_scores.items()
    }
    total = sum(floored.values())

    return {name: v / total for name, v in floored.items()}


def compute_weights_softmax(
    skill_scores: dict[str, float],
    temperature: float = 0.05,
) -> dict[str, float]:
    """
    Compute fusion weights from per-agent skill-relative-to-naive scores
    (see module docstring, diagnosis #5 -- this is the current method).

        weight_i = exp(skill_i / temperature) / sum_j exp(skill_j / temperature)

    Chosen over a shift-and-power alternative (skill + shift, then raise to
    a power) because a shift computed from the current batch's minimum
    skill_score would make a GIVEN agent's weight depend on how bad the
    WORST OTHER agent in that particular fold happened to be -- the same
    kind of context-dependent instability this investigation spent the
    whole night chasing out of the weighting mechanism (see diagnosis #4).
    Softmax has no such dependency: exp(skill_i / temperature) depends only
    on agent i's own skill_score and the fixed temperature, never on any
    other agent's value. It also handles negative skill_scores natively (no
    floor, no clip anywhere), is smooth and strictly monotonic in
    skill_score by construction, and preserves full rank order across ALL
    agents -- not just "less-bad beats very-bad" among the negative ones,
    which is exactly what a flat epsilon floor could not express.

    temperature controls how sharply skill differences translate to weight
    differences: smaller -> sharper (closer to winner-take-all), larger ->
    softer (closer to uniform). 0.05 is chosen so that AR's typical skill
    advantage over the field (~0.2-0.3 in the folds tested) still produces
    clear dominance (~90-98% of the blend), while still giving near-zero
    and negative-skill agents a small but genuinely RANKED gradient instead
    of an identical floor.

    Args:
        skill_scores: {agent_name: avg_skill_score_on_validation_data}
        temperature: softmax temperature (see above).

    Returns:
        {agent_name: weight}, normalized to sum to 1.0.
    """
    if not skill_scores:
        return {}

    exp_scores = {name: np.exp(score / temperature) for name, score in skill_scores.items()}
    total = sum(exp_scores.values())

    return {name: v / total for name, v in exp_scores.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Candidate fixes for the mean-of-ratios instability found in
# compute_fold_weights's skill_score aggregation (diagnosis #3 in the module
# docstring): averaging per-point (1 - agent_ape/naive_ape) ratios blows up
# whenever naive_ape happens to be near-zero at a given point, dominating the
# mean with outliers unrelated to real skill (confirmed empirically: AR's
# median per-point skill score was +0.227 -- genuinely good -- while its MEAN
# was -0.407, driven by a single point at -32.09). Two candidates below, NOT
# yet wired into compute_fold_weights -- under empirical comparison first.
# ═══════════════════════════════════════════════════════════════════════════

def skill_score_median(per_point_ratios: list[float]) -> float:
    """
    Median of per-point skill scores (each already computed as
    1 - agent_ape/naive_ape). Robust to the small-denominator outliers that
    broke the mean: a handful of extreme ratios can't drag a median around
    the way they drag a mean.
    """
    if not per_point_ratios:
        return -1.0
    return float(np.median(per_point_ratios))


def skill_score_pooled(agent_apes: list[float], naive_apes: list[float]) -> float:
    """
    Skill score from pooled (ratio-of-means) MAPEs rather than mean-of-
    ratios: 1 - mean(agent_apes) / mean(naive_apes). Sidesteps the
    small-denominator instability entirely, since the denominator is now an
    average over many points rather than any single point's naive error --
    no single point can dominate it the way it dominated the per-point ratio.
    """
    if not agent_apes or not naive_apes:
        return -1.0
    naive_mean = float(np.mean(naive_apes))
    if naive_mean <= 0:
        return 0.0
    agent_mean = float(np.mean(agent_apes))
    return 1.0 - (agent_mean / naive_mean)


def fuse_forecasts(
    fusion_input: FusionInput,
    weights: dict[str, float],
) -> list[float]:
    """
    Combine agent forecasts using the given weights.

    forecast[t] = sum(weight_i * agent_i.values[t]) for each timestep t.

    Args:
        fusion_input: Validated FusionInput with all agents' outputs.
        weights: {agent_name: weight}, should sum to ~1.0.

    Returns:
        Fused forecast values (one per horizon week).
    """
    # Validate before fusing
    errors = validate_fusion_input(fusion_input)
    if errors:
        raise ValueError(f"Fusion input validation failed: {errors}")

    horizon = fusion_input.horizon_weeks
    fused = np.zeros(horizon, dtype=np.float64)

    for agent_name, output in fusion_input.agent_outputs.items():
        w = weights.get(agent_name, 0.0)
        fused += w * np.array(output.values)

    # Non-negativity floor
    fused = np.maximum(fused, 0.0)

    return [round(float(v), 2) for v in fused]


def compute_fold_weights(
    weekly_df: pd.DataFrame,
    agent_runners: dict[str, callable],
    msas: list[str],
    train_end: datetime.date,
    lookback_weeks: int = 104,
    horizons: Optional[list[int]] = None,
    temperature: float = 0.05,
) -> dict[str, float]:
    """
    Compute per-fold agent weights by evaluating each agent's skill
    relative to naive on a validation window.

    This is the "rolling" part: weights are recomputed each fold
    using the most recent realized data.

    Scores each agent by skill_score = 1 - agent_MAPE/naive_MAPE (see
    module docstring, diagnosis #3), not raw MAPE (diagnosis #1/#2): raw
    MAPE couldn't distinguish an agent that's near-naive because it has no
    signal from one that's near-naive while genuinely skilled, since naive
    itself is a decent short-horizon forecast. naive_MAPE is computed at
    the EXACT SAME (msa, origin, horizon) as the agent's MAPE for every
    single point, never pooled or averaged separately, so the ratio is
    always apples-to-apples. Per-point scores are aggregated via the
    MEDIAN (diagnosis #4), then converted to weights via softmax
    (diagnosis #5), not a hard epsilon floor -- see compute_weights_softmax.

    104-week (2-year) lookback aggregated across all three horizons --
    a deliberate middle ground between full history (maximal power, but
    slow) and a narrow window (fast, but underpowered) -- this part of the
    original diagnosis (too little data) was correct even though the
    scoring criterion itself (raw MAPE) also needed to change.

    Args:
        weekly_df: aligned_weekly.csv.
        agent_runners: {agent_name: callable(df, msa, origin, horizon) -> AgentOutput}
        msas: List of MSA names to evaluate on.
        train_end: Last allowable date for the validation window (typically
            the fold's train_end -- the window is computed backward from here,
            centralizing that decision in this function rather than the caller).
        lookback_weeks: How far back from train_end the validation window
            extends. Default 104 weeks (2 years).
        horizons: Horizons to aggregate skill_score across. Defaults to
            [4, 8, 13] (all three, not just one).
        temperature: passed through to compute_weights_softmax.

    Returns:
        {agent_name: weight}
    """
    if horizons is None:
        horizons = [4, 8, 13]
    val_start = train_end - datetime.timedelta(weeks=lookback_weeks)
    val_end = train_end

    agent_skills: dict[str, list[float]] = {name: [] for name in agent_runners}

    for msa in msas:
        msa_data = weekly_df[weekly_df["msa"] == msa].sort_values("date")

        # Find forecast origins within the validation window
        origins = msa_data[
            (msa_data["date"] >= pd.Timestamp(val_start))
            & (msa_data["date"] <= pd.Timestamp(val_end))
        ]["date"].values

        for origin_ts in origins:
            origin = pd.Timestamp(origin_ts).date()

            for horizon_weeks in horizons:
                # Get actuals for this origin + horizon
                future_mask = msa_data["date"] > pd.Timestamp(origin)
                future_data = msa_data[future_mask].head(horizon_weeks)

                if len(future_data) < horizon_weeks:
                    continue  # Not enough future data for evaluation

                actuals = future_data["inventory_count"].values

                # Naive last-value baseline for this EXACT (msa, origin, horizon) --
                # never a pooled/averaged naive figure, so every skill_score below
                # is a fair apples-to-apples comparison at that specific point.
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
                    continue  # degenerate point (naive is exactly right) -- ratio undefined, skip

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

                        skill_score = 1.0 - (agent_mape / naive_mape)
                        agent_skills[agent_name].append(skill_score)
                    except Exception:
                        # Agent failure: treat as maximally unskilled (worse than naive)
                        agent_skills[agent_name].append(-1.0)

    # MEDIAN (not mean) of per-point skill_score, pooled across all origins,
    # MSAs, and horizons -- see module docstring, diagnosis #4. Mean-of-
    # ratios is numerically unstable (blows up whenever naive_MAPE is
    # near-zero at a point); median is robust to those outliers and was
    # selected over skill_score_pooled after an empirical cross-check
    # against validate_macro_panel.py's rigorous null finding.
    median_skills = {
        name: skill_score_median(skills) for name, skills in agent_skills.items()
    }

    return compute_weights_softmax(median_skills, temperature=temperature)
