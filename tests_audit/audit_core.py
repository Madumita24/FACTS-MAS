"""
Audit core: guarded data access and a fast mirror of the production weighting.

Nothing here edits or replaces production code. Two jobs:

1. GUARDS.
   - load_panel() reads aligned_weekly.csv and drops every row after
     DATA_CAP (2026-02-21) at read time. Phase 7 holdout = origins after
     DATA_CAP, so nothing after it may be loaded before Phase 7.
   - assert_scoring_allowed() raises if any scored target is on or after
     RESERVED_START (2026-01-01), unless the caller passes disclosure=True.
     Every scoring helper in this module routes through it.

2. FAST MIRROR of the production adaptive weighting
   (fusion_baseline.compute_fold_weights, factor_attribution.
   compute_all_regime_weights / evaluate_validation_mape, run_phase34.
   build_weight_providers, ablation.fuse_entry). The production path walks
   pandas frames per origin and takes ~10 minutes per configuration; this
   mirror works on precomputed arrays and takes seconds. It is pinned
   against production outputs in test_regression_pin.py. Options that
   production does not have (bounded validation targets, per-fold agent
   values) are explicit keyword arguments that default to production
   behaviour.
"""
from __future__ import annotations

import datetime
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from facts_mas.run_backtest import FOLDS, HORIZONS  # noqa: E402

RESERVED_START = pd.Timestamp("2026-01-01")
DATA_CAP = pd.Timestamp("2026-02-21")
AGENTS = ["ar", "event", "intrinsic", "macro", "seasonality"]
REGIMES = ("hiking", "cutting", "stable")
LOOKBACK_WEEKS = 104
TEMPERATURE = 0.05


class ReservedWindowError(RuntimeError):
    """A scored target falls in the reserved window without a disclosure flag."""


class DataCapError(RuntimeError):
    """Something dated after DATA_CAP was about to be used."""


# ═══════════════════════════════════════════════════════════════════════════
# Guards
# ═══════════════════════════════════════════════════════════════════════════

def load_panel(path: Optional[str] = None) -> pd.DataFrame:
    """aligned_weekly.csv truncated at DATA_CAP on read."""
    path = path or os.path.join(REPO, "aligned_weekly.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"] <= DATA_CAP].reset_index(drop=True)
    if df["date"].max() > DATA_CAP:  # pragma: no cover - defensive
        raise DataCapError("panel truncation failed")
    return df


def assert_scoring_allowed(target_dates, disclosure: bool = False) -> None:
    """Raise unless every scored target is before RESERVED_START (or disclosure=True).

    Dates after DATA_CAP always raise, disclosure or not.
    """
    td = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(target_dates))))
    if len(td) == 0:
        return
    if td.max() > DATA_CAP:
        raise DataCapError(f"target {td.max().date()} is after DATA_CAP {DATA_CAP.date()}")
    n_res = int((td >= RESERVED_START).sum())
    if n_res and not disclosure:
        raise ReservedWindowError(
            f"{n_res} scored target(s) on or after {RESERVED_START.date()} "
            f"(latest {td.max().date()}). Pass disclosure=True only for a "
            f"labelled disclosure run."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Cell table
# ═══════════════════════════════════════════════════════════════════════════

def fold_of_test(origin: datetime.date) -> Optional[int]:
    for f in FOLDS:
        if f.test_start <= origin <= f.test_end:
            return f.fold
    return None


@dataclass
class Cell:
    msa: str
    origin: datetime.date
    h: int
    regime: str
    actuals: np.ndarray
    naive: np.ndarray
    last_target: pd.Timestamp
    target_dates: tuple
    values: dict = field(default_factory=dict)   # agent -> forecast (production values)
    test_fold: Optional[int] = None


def cells_from_cache(cache, panel: pd.DataFrame) -> list[Cell]:
    """Every cached cell with its target dates attached (from the capped panel)."""
    dates_by_msa = {m: g.sort_values("date")["date"].reset_index(drop=True)
                    for m, g in panel.groupby("msa")}
    out = []
    for e in cache.entries.values():
        d = dates_by_msa[e.msa]
        pos = int(np.searchsorted(d.values, np.datetime64(pd.Timestamp(e.origin)), side="right"))
        tds = tuple(d.iloc[pos:pos + e.horizon])
        if len(tds) < e.horizon:
            raise DataCapError(f"cell {e.msa} {e.origin} h={e.horizon} needs data past DATA_CAP")
        out.append(Cell(e.msa, e.origin, e.horizon, e.regime, e.actuals, e.naive,
                        tds[-1], tds, dict(e.agent_values), fold_of_test(e.origin)))
    return out


def mape_pct(actuals: np.ndarray, fc: np.ndarray) -> float:
    """Same as run_backtest.compute_mape."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs((actuals - fc) / actuals)
    ape = ape[np.isfinite(ape)]
    return float(100.0 * np.mean(ape)) if len(ape) else float("inf")


def _mape_frac(actuals, fc):
    """Same as the inline computation in fusion_baseline.compute_fold_weights."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs((actuals - fc) / actuals)
    ape = ape[np.isfinite(ape)]
    return float(np.mean(ape)) if len(ape) else None


def softmax(skills: dict, temperature: float = TEMPERATURE) -> dict:
    ex = {k: np.exp(v / temperature) for k, v in skills.items()}
    t = sum(ex.values())
    return {k: v / t for k, v in ex.items()}


def median_skill(points: list) -> float:
    return float(np.median(points)) if points else -1.0


# ═══════════════════════════════════════════════════════════════════════════
# Fast mirror of the adaptive weighting
# ═══════════════════════════════════════════════════════════════════════════

ValuesFn = Callable[[Cell, int], dict]   # (cell, fold whose weights are being built) -> agent values


def production_values(cell: Cell, fold: int) -> dict:
    return cell.values


@dataclass
class FoldModel:
    fold: int
    fold_weights: dict
    regime_grid: dict          # regime -> h -> {"weights", "skills", "n"}
    mode: dict                 # h -> "fold" | "regime"
    val_mape: dict             # h -> (fold_mode_mape, regime_mode_mape)
    horizon_weights: dict = field(default_factory=dict)   # h -> weights (horizon-only, regime-blind)


def _in_val(cell: Cell, fold, bound_targets: bool) -> bool:
    start = fold.train_end - datetime.timedelta(weeks=LOOKBACK_WEEKS)
    if not (start <= cell.origin <= fold.train_end):
        return False
    if bound_targets and cell.last_target > pd.Timestamp(fold.train_end):
        return False
    return True


def build_fold_model(cells: list[Cell], fold, values_fn: ValuesFn = production_values,
                     bound_targets: bool = False, temperature: float = TEMPERATURE,
                     exclude: Optional[Callable[[Cell], bool]] = None,
                     agents: Optional[list] = None,
                     min_support_weeks: Optional[int] = None,
                     reference_fn: Optional[Callable[[Cell, int], np.ndarray]] = None) -> FoldModel:
    """min_support_weeks (audit fix 2, default None = production behaviour):
    a (regime, horizon) cell whose validation window holds fewer than this many
    DISTINCT origin weeks of that regime uses the fold weights instead of its
    own softmax weights, both in the mode-selection comparison and on test.
    Support is counted in weeks, not cells, because the regime label is national."""
    agents = agents or AGENTS
    val = [c for c in cells if _in_val(c, fold, bound_targets) and not (exclude and exclude(c))]

    pooled = {a: [] for a in agents}
    by_h = {h: {a: [] for a in agents} for h in HORIZONS}
    by_rh = {(r, h): {a: [] for a in agents} for r in REGIMES for h in HORIZONS}
    for c in val:
        # skill reference: naive (production) unless reference_fn is given (Phase 5: FA-ETS-d)
        ref = reference_fn(c, fold.fold) if reference_fn is not None else c.naive
        nm = _mape_frac(c.actuals, np.asarray(ref, dtype=np.float64))
        if nm is None or nm <= 0:
            continue
        vals = values_fn(c, fold.fold)
        for a in agents:
            am = _mape_frac(c.actuals, np.asarray(vals[a]))
            if am is None:
                continue
            s = 1.0 - am / nm
            pooled[a].append(s)
            by_h[c.h][a].append(s)
            by_rh[(c.regime, c.h)][a].append(s)

    fold_w = softmax({a: median_skill(pooled[a]) for a in agents}, temperature)
    hz_w = {h: softmax({a: median_skill(by_h[h][a]) for a in agents}, temperature) for h in HORIZONS}
    weeks = {(r, h): set() for r in REGIMES for h in HORIZONS}
    for c in val:
        weeks[(c.regime, c.h)].add(c.origin)

    grid = {r: {} for r in REGIMES}
    for r in REGIMES:
        for h in HORIZONS:
            pts = by_rh[(r, h)]
            sk = {a: median_skill(pts[a]) for a in agents}
            w = softmax(sk, temperature)
            sup = len(weeks[(r, h)])
            fb = min_support_weeks is not None and sup < min_support_weeks
            grid[r][h] = {"weights": w, "skills": sk,
                          "n": max(len(p) for p in pts.values()),
                          "support_weeks": sup, "fallback": fb,
                          "weights_eff": fold_w if fb else w}

    mode, vm = {}, {}
    for h in HORIZONS:
        fa, ra = [], []
        for c in val:
            if c.h != h:
                continue
            vals = values_fn(c, fold.fold)
            rw = grid[c.regime][h]["weights_eff"]
            ff = np.zeros(h)
            rf = np.zeros(h)
            for a in agents:
                v = np.asarray(vals[a], dtype=np.float64)
                ff += fold_w.get(a, 0.0) * v
                rf += rw.get(a, 0.0) * v
            fa.append(mape_pct(c.actuals, np.maximum(ff, 0.0)))
            ra.append(mape_pct(c.actuals, np.maximum(rf, 0.0)))
        fm = float(np.mean(fa)) if fa else float("inf")
        rm = float(np.mean(ra)) if ra else float("inf")
        vm[h] = (fm, rm)
        mode[h] = "fold" if fm <= rm else "regime"
    return FoldModel(fold.fold, fold_w, grid, mode, vm, hz_w)


def build_models(cells, values_fn: ValuesFn = production_values, **kw) -> dict:
    return {f.fold: build_fold_model(cells, f, values_fn, **kw) for f in FOLDS}


def weights_for(cell: Cell, models: dict, scheme: str = "adaptive") -> dict:
    m = models[cell.test_fold]
    if scheme == "fold":
        return m.fold_weights
    if scheme == "horizon":
        return m.horizon_weights[cell.h]
    if scheme == "regime" or (scheme == "adaptive" and m.mode[cell.h] == "regime"):
        return m.regime_grid[cell.regime][cell.h]["weights_eff"]
    return m.fold_weights


def load_r1_seasonal(path: Optional[str] = None) -> dict:
    """R1 baseline seasonal outputs (window 7, chosen on validation in every fold, Phase 1).
    Built by audit_reports/phase1_scripts/p1_build_variants.py; hash checked."""
    import hashlib
    import pickle
    path = path or os.path.join(REPO, "audit_reports", "artifacts", "seasonal_w7_r1.pkl")
    blob = open(path, "rb").read()
    if hashlib.sha256(blob).hexdigest() != R1_SHA256:
        raise RuntimeError(f"{path} does not match the Phase 1 R1 artifact hash")
    return pickle.loads(blob)


R1_SHA256 = "3b530099abfe5bedcb994ca809ce8ae4b8e8e5a28bdf94c0c9dd43e560557017"


def r1_values_fn(w7: dict) -> ValuesFn:
    """values_fn for the R1 baseline: production values with seasonality := window 7."""
    def f(cell: Cell, fold: int) -> dict:
        v = dict(cell.values)
        v["seasonality"] = w7[(cell.msa, cell.origin, cell.h)]
        return v
    return f


def fuse(values: dict, weights: dict, h: int, drop: frozenset = frozenset()) -> np.ndarray:
    """Same as ablation.fuse_entry (renormalised over surviving positive-weight agents)."""
    live = {a: w for a, w in weights.items() if w > 0 and a not in drop and a in values}
    tot = sum(live.values())
    out = np.zeros(h)
    for a, w in live.items():
        out += (w / tot) * np.asarray(values[a], dtype=np.float64)
    return np.maximum(out, 0.0)


def test_cells(cells, horizon: Optional[int] = None, exclude_bleed: bool = False):
    out = [c for c in cells if c.test_fold is not None and (horizon is None or c.h == horizon)]
    if exclude_bleed:
        out = [c for c in out if c.last_target < RESERVED_START]
    return out


def score(cells, models, values_fn: ValuesFn = production_values, scheme: str = "adaptive",
          exclude_bleed: bool = False, disclosure: bool = False,
          weights_fn: Optional[Callable[[Cell], dict]] = None) -> pd.DataFrame:
    """Per-cell MAPE for fused, naive and every standalone agent, test cells only."""
    tc = test_cells(cells, exclude_bleed=exclude_bleed)
    assert_scoring_allowed([d for c in tc for d in c.target_dates], disclosure=disclosure)
    rows = []
    for c in tc:
        vals = values_fn(c, c.test_fold)
        w = weights_fn(c) if weights_fn else weights_for(c, models, scheme)
        fz = fuse(vals, w, c.h)
        r = dict(h=c.h, fold=c.test_fold, msa=c.msa, origin=c.origin, regime=c.regime,
                 bleed=bool(c.last_target >= RESERVED_START),
                 fused=mape_pct(c.actuals, fz), naive=mape_pct(c.actuals, c.naive))
        for a in AGENTS:
            r[f"solo_{a}"] = mape_pct(c.actuals, np.maximum(np.asarray(vals[a]), 0.0))
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, cols=("naive", "fused")) -> pd.DataFrame:
    return df.groupby("h")[list(cols)].mean()


def val_fused_mape(cells, fold, model: FoldModel, values_fn: ValuesFn, h: int,
                   scheme: str = "adaptive", weights_fn: Optional[Callable[[Cell], dict]] = None,
                   bound_targets: bool = True) -> float:
    """Mean fused MAPE over fold's own validation cells at horizon h, with the
    weighting that fold would use on test (adaptive: its chosen mode with the
    effective regime weights; horizon: horizon-only; or a fixed weights_fn)."""
    out = []
    for c in cells:
        if c.h != h or not _in_val(c, fold, bound_targets):
            continue
        if weights_fn:
            w = weights_fn(c)
        elif scheme == "horizon":
            w = model.horizon_weights[h]
        elif scheme == "fold" or model.mode[h] == "fold":
            w = model.fold_weights
        else:
            w = model.regime_grid[c.regime][h]["weights_eff"]
        out.append(mape_pct(c.actuals, fuse(values_fn(c, fold.fold), w, h)))
    return float(np.mean(out)) if out else float("inf")
