"""
Phase 4: regime placebo on R1c (5 channels, as in Phases 2-3). Pre-registered in phase4_log.md.

  true       production regime labels (fed funds 13-week change)
  permuted   national label sequence over the 344 cached origin dates cut into 13-week blocks,
             blocks shuffled (200 permutations, seed 20261016), full R1c pipeline re-run each time
  period     origin < 2023-01-01 vs >= 2023-01-01 as the two "regimes"
  recency    no regime; weighted median skill per (fold, h), weights 0.5^(age/26 weeks)
  horizon    HZ-weights (no regime, no recency)

Usage (repo root): python audit_reports/phase4_scripts/p4_placebo.py
"""
import dataclasses
import datetime
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.getcwd(), "tests_audit"))
import audit_core as ac  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R1C = dict(min_support_weeks=13, bound_targets=True)
N_PERM, BLOCK, SEED = 200, 13, 20261016
_CELLS = None


def cells_and_fn():
    global _CELLS
    if _CELLS is None:
        from facts_mas.agent_cache import AgentCache
        _CELLS = ac.cells_from_cache(AgentCache.load("agent_output_cache.pkl"), ac.load_panel())
    return _CELLS, ac.r1_values_fn(ac.load_r1_seasonal())


def relabel(cells, mapping):
    return [dataclasses.replace(c, regime=mapping[c.origin]) for c in cells]


def score_labels(mapping):
    cells, fn = cells_and_fn()
    cc = relabel(cells, mapping) if mapping is not None else cells
    m = ac.build_models(cc, fn, **R1C)
    d = ac.score(cc, m, fn, exclude_bleed=True)
    return d.groupby("h").fused.mean().to_dict()


def perm_job(seed):
    cells, _ = cells_and_fn()
    lab = {}
    for c in cells:
        lab[c.origin] = c.regime
    dates = sorted(lab)
    seq = [lab[d] for d in dates]
    blocks = [seq[i:i + BLOCK] for i in range(0, len(seq), BLOCK)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(blocks))
    new = [x for i in order for x in blocks[i]]
    return seed, score_labels(dict(zip(dates, new)))


def recency_score():
    cells, fn = cells_and_fn()
    from facts_mas.run_backtest import FOLDS
    W = {}
    for f in FOLDS:
        val = [c for c in cells if ac._in_val(c, f, True)]
        for h in (4, 8, 13):
            pts = {a: ([], []) for a in ac.AGENTS}
            for c in val:
                if c.h != h:
                    continue
                nm = ac._mape_frac(c.actuals, c.naive)
                if not nm or nm <= 0:
                    continue
                age = (f.train_end - c.origin).days / 7.0
                wt = 0.5 ** (age / 26.0)
                v = fn(c, f.fold)
                for a in ac.AGENTS:
                    pts[a][0].append(1 - ac._mape_frac(c.actuals, np.asarray(v[a])) / nm)
                    pts[a][1].append(wt)
            sk = {}
            for a, (x, w) in pts.items():
                x, w = np.asarray(x), np.asarray(w)
                o = np.argsort(x)
                cw = np.cumsum(w[o]) / w.sum()
                sk[a] = float(x[o][np.searchsorted(cw, 0.5)])
            W[(f.fold, h)] = ac.softmax(sk)
    m = ac.build_models(cells, fn, **R1C)
    d = ac.score(cells, m, fn, exclude_bleed=True, weights_fn=lambda c: W[(c.test_fold, c.h)])
    return d.groupby("h").fused.mean().to_dict(), d


def main():
    cells, fn = cells_and_fn()
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    true = score_labels(None)
    period = score_labels({c.origin: ("stable" if c.origin < datetime.date(2023, 1, 1) else "hiking") for c in cells})
    rec, _ = recency_score()
    m = ac.build_models(cells, fn, **R1C)
    hz = ac.score(cells, m, fn, "horizon", exclude_bleed=True).groupby("h").fused.mean().to_dict()
    with ProcessPoolExecutor(max_workers=14) as ex:
        perms = dict(ex.map(perm_job, [SEED + i for i in range(N_PERM)]))
    P = pd.DataFrame(perms).T
    rows = []
    for h in (4, 8, 13):
        dist = P[h].to_numpy()
        rows.append(dict(h=h, true_regime=true[h], period_labels=period[h], recency_skill=rec[h], horizon_only=hz[h],
                         placebo_median=np.median(dist), placebo_p05=np.percentile(dist, 5), placebo_p95=np.percentile(dist, 95),
                         share_placebo_better_than_true=float((dist < true[h]).mean()), n_perm=len(dist)))
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(R.round(4).to_string(index=False))
    out = os.path.join("audit_reports", "results_tables", "phase4_regime_placebo.csv")
    with open(out, "w", newline="") as fh:
        fh.write(f"# CONFIG: commit={commit} | R1c (5 channels) TEST CLEANBLEED fused MAPE | placebo = 13-week block "
                 f"permutation of national labels, {N_PERM} perms, seed {SEED} | period = pre/post 2023-01-01 | "
                 f"recency = weighted median skill, half-life 26 weeks, no regime\n")
        R.to_csv(fh, index=False)
    with open(out.replace(".csv", "_distribution.csv"), "w", newline="") as fh:
        fh.write("# CONFIG: per-permutation pooled test MAPE (CLEANBLEED), R1c pipeline re-run per permutation\n")
        P.rename_axis("seed").reset_index().to_csv(fh, index=False)


if __name__ == "__main__":
    main()
