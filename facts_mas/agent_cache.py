"""
Agent Output Cache — FACTS-MAS Phase 3/4 enabler.

Every Phase-4 ablation row re-fuses the SAME agent forecasts under a
different weight set. Dropping Macro from the fusion does not change what
AR predicts. So running the backtest once per ablation row recomputes
identical agent output six times over, and the agent calls (ETS refits in
particular) are essentially the entire cost of a run.

This module computes each agent's forecast once per (msa, origin, horizon),
stores it, and hands back *runner callables that read from the store*. That
last part is the important design choice: because the cached runners honour
the same `(df, msa, origin, horizon) -> AgentOutput` contract every other
module already expects, `compute_fold_weights`, `compute_all_regime_weights`,
`evaluate_validation_mape` and `run_backtest` all work unchanged and simply
get fast. Nothing downstream had to learn about caching.

Correctness notes:
    - The cache is keyed by (msa, origin, horizon) and stores the agent's
      forecast values only. It never stores actuals in a way an agent could
      read, so it cannot become a leakage path.
    - A miss raises rather than silently recomputing. A silent recompute
      would make a partially-built cache look complete while quietly costing
      hours, and would hide the bug where an origin was never cached.
    - Cache entries also record the regime and the naive forecast, since
      every analysis in Phase 4 needs both and neither depends on weights.
"""
from __future__ import annotations

import datetime
import os
import pickle
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from facts_mas.run_backtest import FOLDS, HORIZONS, FoldSpec, classify_regime
from facts_mas.schema import AgentOutput

DEFAULT_CACHE_PATH = "agent_output_cache.pkl"
LOOKBACK_WEEKS = 104


# ═══════════════════════════════════════════════════════════════════════════
# Cache structure
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CacheEntry:
    """Everything Phase 3/4 needs about one (msa, origin, horizon) cell."""

    msa: str
    origin: datetime.date
    horizon: int
    regime: str
    actuals: np.ndarray
    naive: np.ndarray
    agent_values: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class AgentCache:
    """
    Forecasts for every agent at every cached cell.

    `agent_names` is stored explicitly rather than inferred from entries so
    a cache built before an agent existed reports the gap instead of quietly
    looking complete.
    """

    entries: dict[tuple, CacheEntry] = field(default_factory=dict)
    agent_names: list[str] = field(default_factory=list)
    built_at: Optional[str] = None

    # ── Lookup ────────────────────────────────────────────────────────────

    def get(self, msa: str, origin: datetime.date, horizon: int) -> CacheEntry:
        key = (msa, origin, horizon)
        if key not in self.entries:
            raise KeyError(
                f"agent cache miss for {key}. The cache is built over a fixed "
                f"set of origins; either rebuild it covering this one, or the "
                f"caller is walking a window the cache was not built for."
            )
        return self.entries[key]

    def has(self, msa: str, origin: datetime.date, horizon: int) -> bool:
        return (msa, origin, horizon) in self.entries

    # ── Views ─────────────────────────────────────────────────────────────

    def cells(self, horizon: Optional[int] = None) -> list[CacheEntry]:
        vals = self.entries.values()
        if horizon is None:
            return list(vals)
        return [e for e in vals if e.horizon == horizon]

    def msas(self) -> list[str]:
        return sorted({e.msa for e in self.entries.values()})

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str = DEFAULT_CACHE_PATH) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str = DEFAULT_CACHE_PATH) -> "AgentCache":
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def summary(self) -> str:
        if not self.entries:
            return "empty cache"
        hs = sorted({e.horizon for e in self.entries.values()})
        os_ = sorted({e.origin for e in self.entries.values()})
        return (
            f"{len(self.entries)} cells | {len(self.msas())} MSAs | "
            f"horizons {hs} | {len(os_)} origins "
            f"({os_[0]} .. {os_[-1]}) | agents {self.agent_names} | "
            f"built {self.built_at}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Which origins to cache
# ═══════════════════════════════════════════════════════════════════════════

def origins_for_folds(
    df: pd.DataFrame,
    msa: str,
    folds: Optional[list[FoldSpec]] = None,
    lookback_weeks: int = LOOKBACK_WEEKS,
) -> list[datetime.date]:
    """
    Union of every origin any Phase-3/4 computation will ask for.

    Two families, and both are needed:
        validation windows  (train_end - lookback .. train_end)
            used by compute_fold_weights, compute_all_regime_weights and
            evaluate_validation_mape to choose weights and modes.
        test windows        (test_start .. test_end)
            used by run_backtest and every ablation row for scoring.

    They are unioned and de-duplicated because the folds expand, so later
    folds' validation windows overlap earlier folds' test windows heavily.
    Caching the union once is far cheaper than caching each separately.
    """
    folds = folds or FOLDS
    md = df[df["msa"] == msa].sort_values("date")
    dates = md["date"]

    keep: set[datetime.date] = set()
    for fold in folds:
        val_start = fold.train_end - datetime.timedelta(weeks=lookback_weeks)
        in_val = dates[(dates >= pd.Timestamp(val_start))
                       & (dates <= pd.Timestamp(fold.train_end))]
        in_test = dates[(dates >= pd.Timestamp(fold.test_start))
                        & (dates <= pd.Timestamp(fold.test_end))]
        keep.update(d.date() for d in in_val)
        keep.update(d.date() for d in in_test)

    return sorted(keep)


# ═══════════════════════════════════════════════════════════════════════════
# Building
# ═══════════════════════════════════════════════════════════════════════════

def build_agent_cache(
    df: pd.DataFrame,
    agent_runners: dict[str, Callable],
    msas: Optional[list[str]] = None,
    folds: Optional[list[FoldSpec]] = None,
    horizons: Optional[list[int]] = None,
    path: str = DEFAULT_CACHE_PATH,
    resume: bool = True,
    verbose: bool = True,
) -> AgentCache:
    """
    Run every agent once per (msa, origin, horizon) and store the results.

    This is the expensive step, and the only expensive step: after it, every
    ablation row and every Phase-3 scoring pass is arithmetic over arrays.

    resume=True reloads an existing cache and only fills gaps, so a run that
    dies partway (or a newly added agent) does not force a full rebuild.
    """
    msas = msas or sorted(df["msa"].unique().tolist())
    folds = folds or FOLDS
    horizons = horizons or HORIZONS

    cache = AgentCache()
    if resume and os.path.exists(path):
        try:
            cache = AgentCache.load(path)
            if verbose:
                print(f"resuming from {path}: {cache.summary()}")
        except Exception as exc:
            if verbose:
                print(f"could not reuse {path} ({exc}); starting fresh")
            cache = AgentCache()

    cache.agent_names = sorted(agent_runners.keys())
    computed = 0
    reused = 0

    for msa in msas:
        md = df[df["msa"] == msa].sort_values("date")
        origins = origins_for_folds(df, msa, folds)

        for origin in origins:
            ots = pd.Timestamp(origin)
            past = md[md["date"] <= ots]
            if past.empty:
                continue
            last_value = float(past["inventory_count"].iloc[-1])
            regime = classify_regime(df, msa, origin)

            for horizon in horizons:
                fut = md[md["date"] > ots].head(horizon)
                if len(fut) < horizon:
                    continue  # not enough future to score against

                key = (msa, origin, horizon)
                entry = cache.entries.get(key)
                if entry is None:
                    entry = CacheEntry(
                        msa=msa, origin=origin, horizon=horizon, regime=regime,
                        actuals=fut["inventory_count"].values.astype(np.float64),
                        naive=np.full(horizon, last_value, dtype=np.float64),
                    )
                    cache.entries[key] = entry

                for name, runner in agent_runners.items():
                    if name in entry.agent_values:
                        reused += 1
                        continue
                    try:
                        out = runner(df, msa, origin, horizon)
                        vals = np.array(out.values, dtype=np.float64)
                    except Exception:
                        # A failed agent is recorded as NaN rather than
                        # dropped, so an ablation can tell "this agent could
                        # not run here" apart from "this cell was never
                        # cached".
                        vals = np.full(horizon, np.nan, dtype=np.float64)
                    entry.agent_values[name] = vals
                    computed += 1

        if verbose:
            print(f"  {msa:<16} cached ({computed} computed, {reused} reused)",
                  flush=True)
        cache.built_at = datetime.datetime.now().isoformat(timespec="seconds")
        cache.save(path)  # checkpoint per MSA so a crash loses at most one

    if verbose:
        print(f"\ncache complete: {cache.summary()}")
    return cache


# ═══════════════════════════════════════════════════════════════════════════
# Cached runners — the reuse hook
# ═══════════════════════════════════════════════════════════════════════════

def make_cached_runners(
    cache: AgentCache,
    agent_names: Optional[list[str]] = None,
    strict: bool = True,
) -> dict[str, Callable]:
    """
    Runner callables that read the cache instead of recomputing.

    Signature-compatible with the real runners, so this can be dropped into
    `compute_fold_weights`, `compute_all_regime_weights`,
    `evaluate_validation_mape` or `run_backtest` with no other change and
    those functions simply stop being slow.

    strict=True raises on a cache miss. That is deliberate: a miss means the
    caller is walking origins the cache was not built over, and silently
    recomputing would hide that while costing exactly the hours the cache
    exists to avoid.
    """
    names = agent_names or cache.agent_names

    def _make(name: str) -> Callable:
        def _runner(df, msa, forecast_origin, horizon_weeks):
            origin = (forecast_origin.date()
                      if isinstance(forecast_origin, (pd.Timestamp, datetime.datetime))
                      else forecast_origin)
            if not cache.has(msa, origin, horizon_weeks):
                if strict:
                    cache.get(msa, origin, horizon_weeks)  # raises with context
                raise KeyError((msa, origin, horizon_weeks))
            vals = cache.get(msa, origin, horizon_weeks).agent_values.get(name)
            if vals is None or np.isnan(vals).any():
                raise ValueError(f"agent '{name}' has no usable cached output")
            return AgentOutput(
                agent_name=name,
                msa=msa,
                forecast_origin=origin,
                horizon_weeks=horizon_weeks,
                values=[float(v) for v in vals],
            )
        return _runner

    return {name: _make(name) for name in names}
