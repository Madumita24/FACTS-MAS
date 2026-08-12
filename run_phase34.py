"""
Phase 3 + Phase 4 driver — FACTS-MAS.

Runs everything the Proposal (§6) lists for Weeks 6-10 that does not require
new LLM calls:

    Phase 3   factor scoring (rolling correlation, Granger screening,
              permutation importance) and the fixed-vs-dynamic weighting
              ablation, which is Phase 3's stated deliverable.

    Phase 4   the ablation table and the four "killer analyses": shock vs
              normal weeks, cross-MSA generalization, agent contribution by
              horizon, and cost per forecast vs Nexus.

Stage 1 (cache build) is the only expensive step and it is resumable. Every
later stage is arithmetic over the cache and takes seconds, which is what
makes running seven ablation variants tractable at all.

Usage:
    python run_phase34.py --build-cache        # stage 1, slow, run once
    python run_phase34.py                      # stages 2-3, fast
    python run_phase34.py --msas Phoenix,Miami --folds 6    # small smoke run
"""
import argparse
import datetime
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from facts_mas.ablation import (
    agent_contribution_by_horizon,
    attach_all_llm_row,
    cost_per_forecast,
    cross_msa_generalization,
    run_ablation_table,
    shock_vs_normal,
)
from facts_mas.agent_cache import (
    DEFAULT_CACHE_PATH,
    AgentCache,
    build_agent_cache,
    make_cached_runners,
)
from facts_mas.factor_scoring import (
    MACRO_FACTORS,
    granger_spillover_graph,
    permutation_importance,
    score_all_factors,
)
from facts_mas.run_backtest import FOLDS, HORIZONS, build_agent_runners

OUT = "phase34_results"


def _hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ═══════════════════════════════════════════════════════════════════════════
# Weight providers
# ═══════════════════════════════════════════════════════════════════════════

def build_weight_providers(df, cached_runners, msas, folds):
    """
    Per-cell weight lookups for the production ("adaptive") and fold-level
    weightings.

    Both are computed from the cache, so this reuses the real
    compute_fold_weights / compute_all_regime_weights / evaluate_validation_mape
    rather than approximating them. That matters: an ablation table built on
    a reimplementation of the weighting would not be measuring the system we
    actually ship.
    """
    from facts_mas.factor_attribution import (
        compute_all_regime_weights,
        evaluate_validation_mape,
    )
    from facts_mas.fusion_baseline import compute_fold_weights

    fold_w, regime_g, mode_sel = {}, {}, {}
    for fold in folds:
        print(f"  fold {fold.fold}: weights...", flush=True)
        fw = compute_fold_weights(df, cached_runners, msas, train_end=fold.train_end)
        fold_w[fold.fold] = fw
        grid = compute_all_regime_weights(df, cached_runners, msas, fold)["grid"]
        regime_g[fold.fold] = grid
        for h in HORIZONS:
            fm, rm = evaluate_validation_mape(
                df, cached_runners, msas, fold, h, fw, grid
            )
            mode_sel[(fold.fold, h)] = "fold" if fm <= rm else "regime"
        print(f"    modes: "
              f"{ {h: mode_sel[(fold.fold, h)] for h in HORIZONS} }", flush=True)

    # Map an origin back to the fold whose test window contains it.
    def fold_of(origin):
        for fold in folds:
            if fold.test_start <= origin <= fold.test_end:
                return fold.fold
        return folds[-1].fold

    def adaptive(entry):
        f = fold_of(entry.origin)
        if mode_sel.get((f, entry.horizon), "fold") == "regime":
            return regime_g[f][entry.regime][f"{entry.horizon}wk"]["weights"]
        return fold_w[f]

    def fold_level(entry):
        return fold_w[fold_of(entry.origin)]

    return adaptive, fold_level, fold_w, mode_sel


# ═══════════════════════════════════════════════════════════════════════════
# Shock lookup (FEMA declaration active)
# ═══════════════════════════════════════════════════════════════════════════

def build_shock_lookup():
    """A week is a 'shock week' if a FEMA declaration is active in that MSA."""
    from facts_mas.agents import event_agent

    ev = pd.read_csv(event_agent.EVENT_WEEKLY_PATH, parse_dates=["week"])
    window = event_agent.DECLARATION_ACTIVE_WINDOW_WEEKS
    active = set()
    for (msa, week), grp in ev.groupby(["msa", "week"], sort=False):
        begin = grp.iloc[0]["incident_begin_date"]
        if pd.isna(begin):
            continue
        if (week - pd.Timestamp(begin)).days / 7.0 <= window:
            active.add((msa, week.date()))
    return lambda msa, origin: (msa, origin) in active


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", action="store_true",
                    help="Stage 1: run every agent once and store the results (slow)")
    ap.add_argument("--cache-path", default=DEFAULT_CACHE_PATH)
    ap.add_argument("--msas", default=None, help="Comma-separated subset")
    ap.add_argument("--folds", default=None, help="Comma-separated fold indices")
    ap.add_argument("--no-rebuild", action="store_true",
                    help="Fail instead of building if the cache is missing")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
    msas = args.msas.split(",") if args.msas else sorted(df["msa"].unique().tolist())
    folds = ([f for f in FOLDS if f.fold in {int(x) for x in args.folds.split(",")}]
             if args.folds else FOLDS)

    # ── Stage 1: cache ────────────────────────────────────────────────────
    if args.build_cache or not os.path.exists(args.cache_path):
        if args.no_rebuild and not os.path.exists(args.cache_path):
            sys.exit(f"no cache at {args.cache_path} and --no-rebuild was set")
        _hr("STAGE 1 — building agent output cache (the only slow step)")
        cache = build_agent_cache(df, build_agent_runners(), msas, folds,
                                  path=args.cache_path)
    else:
        cache = AgentCache.load(args.cache_path)
        print(f"loaded cache: {cache.summary()}")

    cached_runners = make_cached_runners(cache)

    # ── Weights ───────────────────────────────────────────────────────────
    _hr("Computing production and fold-level weights (from cache)")
    adaptive_w, fold_w_fn, fold_w, mode_sel = build_weight_providers(
        df, cached_runners, msas, folds
    )

    # ══ PHASE 3 ══════════════════════════════════════════════════════════
    _hr("PHASE 3.1 — factor scoring: rolling correlation + Granger screening")
    as_of = max(f.train_end for f in folds)
    factors = score_all_factors(df, msas, as_of, MACRO_FACTORS)
    factors.to_csv(f"{OUT}/phase3_factor_scores.csv", index=False)
    agg = (factors.groupby(["factor", "horizon"])
           .agg(mean_corr=("rolling_corr", "mean"),
                n_msas_granger_sig=("granger_sig", "sum"),
                median_granger_p=("granger_p", "median"))
           .reset_index())
    print(agg.to_string(index=False))
    print("\n(n_msas_granger_sig out of "
          f"{len(msas)} — a factor significant in 2 of 15 is noise, 12 of 15 is a finding)")

    _hr("PHASE 3.2 — Granger spillover screen (also NEXUS Mod B edge weights)")
    edges = granger_spillover_graph(df, msas, as_of=as_of)
    edf = pd.DataFrame([e.__dict__ for e in edges])
    edf.to_csv(f"{OUT}/phase3_granger_spillover.csv", index=False)
    sig = edf[edf["significant"]].sort_values("p_value")
    print(f"{len(sig)} significant directed edges out of {len(edf)} ordered pairs")
    print(sig.head(15).to_string(index=False))

    _hr("PHASE 3.3 — permutation importance (agents)")
    perm_rows = []
    for h in HORIZONS:
        cells = cache.cells(h)
        if not cells:
            continue
        w = adaptive_w(cells[0])
        for agent, stats in permutation_importance(cache, w, horizon=h).items():
            perm_rows.append({"horizon": h, "agent": agent, **stats})
    perm = pd.DataFrame(perm_rows)
    perm.to_csv(f"{OUT}/phase3_permutation_importance.csv", index=False)
    if not perm.empty:
        print(perm[["horizon", "agent", "baseline_mape", "permuted_mape",
                    "importance", "std"]].to_string(index=False))

    # ══ PHASE 4 ══════════════════════════════════════════════════════════
    _hr("PHASE 4.1 — ablation table")
    table = run_ablation_table(cache, adaptive_w, fold_w_fn)
    table = attach_all_llm_row(table)
    table.to_csv(f"{OUT}/phase4_ablation_table.csv", index=False)
    for h in HORIZONS:
        sub = table[table["horizon"] == h]
        if sub.empty:
            continue
        print(f"\n--- horizon {h}wk ---")
        print(sub[["variant", "tests", "mape", "rmse", "delta_vs_full", "n_cells"]]
              .to_string(index=False))

    _hr("PHASE 4.2 — error during shock weeks vs normal weeks")
    shock = shock_vs_normal(cache, adaptive_w, build_shock_lookup())
    shock.to_csv(f"{OUT}/phase4_shock_vs_normal.csv", index=False)
    print(shock.to_string(index=False))

    _hr("PHASE 4.3 — cross-MSA generalization")
    gen = cross_msa_generalization(cache, adaptive_w)
    gen.to_csv(f"{OUT}/phase4_cross_msa.csv", index=False)
    print(gen.to_string(index=False))
    if "spread" in gen.attrs:
        print("\nspread across MSAs (dependability, not just average):")
        print(gen.attrs["spread"].round(3).to_string())

    _hr("PHASE 4.4 — agent contribution by horizon")
    contrib = agent_contribution_by_horizon(cache, adaptive_w)
    contrib.to_csv(f"{OUT}/phase4_agent_contribution.csv", index=False)
    print(contrib[["horizon", "agent", "full_mape", "without_mape",
                   "drop_delta", "mean_weight", "solo_mape"]].to_string(index=False))

    _hr("PHASE 4.5 — cost per forecast vs Nexus")
    cost = cost_per_forecast(cache)
    cost.to_csv(f"{OUT}/phase4_cost.csv", index=False)
    print(cost.to_string(index=False))

    _hr(f"done — all outputs written to {OUT}/")


if __name__ == "__main__":
    main()
