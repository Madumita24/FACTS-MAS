"""
Statistical helpers used from Phase 5 on. Same definitions as p4_stats.py (pre-registered in phase4_log.md):
panel DM (Newey-West Bartlett lag h-1), per-MSA DM with HLN, BH, moving block bootstrap over whole weeks
(L = 13), paired equivalence CI, MCS (T_max) with exact duplicates merged.
"""
import numpy as np
import pandas as pd
from scipy import stats

B, L, SEED = 5000, 13, 20261016


def nw_var(x, lag):
    x = np.asarray(x, float) - np.mean(x)
    T = len(x)
    v = x @ x / T
    for j in range(1, lag + 1):
        v += 2 * (1 - j / (lag + 1)) * (x[j:] @ x[:-j]) / T
    return v


def dm(d, h, hln=False):
    d = np.asarray(d, float)
    T = len(d)
    t = d.mean() / np.sqrt(nw_var(d, h - 1) / T)
    if hln:
        t *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return float(t), float(2 * stats.t.sf(abs(t), T - 1))


def bh(p, q=0.05):
    p = np.asarray(p)
    o = np.argsort(p)
    m = len(p)
    passed = p[o] <= q * np.arange(1, m + 1) / m
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    sig = np.zeros(m, bool)
    sig[o[:k]] = True
    return sig


def block_indices(T, seed, block_len=None):
    """Moving block bootstrap indices over T weeks. block_len defaults to L = 13 (Phases 4-5)."""
    Lb = L if block_len is None else int(block_len)
    rng = np.random.default_rng(seed)
    n = int(np.ceil(T / Lb))
    out = np.empty((B, T), dtype=int)
    for b in range(B):
        starts = rng.integers(0, T - Lb + 1, size=n)
        out[b] = np.concatenate([np.arange(s, s + Lb) for s in starts])[:T]
    return out


def weekly(sub, cols):
    assert (sub.groupby("origin").size() == 15).all()
    return sub.groupby("origin")[cols].mean().sort_index()


def pair_stats(sub, W, idx, a, b, h):
    d = (W[a] - W[b]).to_numpy()
    t, p = dm(d, h)
    dd = W[a].to_numpy()[idx].mean(1) - W[b].to_numpy()[idx].mean(1)
    lo, hi = np.percentile(dd, [2.5, 97.5])
    lo90, hi90 = np.percentile(dd, [5, 95])
    ps, ts = [], []
    for _, g in sub.groupby("msa"):
        g = g.sort_values("origin")
        tt, pp = dm((g[a] - g[b]).to_numpy(), h, hln=True)
        ts.append(tt)
        ps.append(pp)
    sig, ts = bh(ps), np.array(ts)
    return dict(h=h, A=a, B=b, mean_diff=float(d.mean()), panel_dm_t=t, panel_dm_p=p, ci95_lo=lo, ci95_hi=hi,
                ci90_lo=lo90, ci90_hi=hi90, BH_A_better=int((sig & (ts < 0)).sum()), BH_B_better=int((sig & (ts > 0)).sum()),
                msas_A_lower=int((ts < 0).sum()), T_weeks=len(d))


def level_ci(W, idx, m):
    lo, hi = np.percentile(W[m].to_numpy()[idx].mean(1), [2.5, 97.5])
    return float(W[m].mean()), float(lo), float(hi)


def mcs(W, idx, methods):
    M, dup = [], {}
    for m_ in methods:
        twin = next((k for k in M if np.allclose(W[m_].to_numpy(), W[k].to_numpy(), rtol=0, atol=1e-12)), None)
        (dup.__setitem__(m_, twin) if twin else M.append(m_))
    MSET = list(M)
    Lw = W[MSET].to_numpy()
    Lb = Lw[idx]
    pvals, running = {}, 0.0
    while len(M) > 1:
        cols = [MSET.index(m) for m in M]
        l, lb = Lw[:, cols], Lb[:, :, cols]
        dbar = l.mean(0) - l.mean()
        dboot = lb.mean(1) - lb.mean(axis=(1, 2))[:, None]
        se = dboot.std(0, ddof=1)
        tstat = dbar / se
        p = float((((dboot - dbar) / se).max(1) >= tstat.max()).mean())
        running = max(running, p)
        worst = M[int(np.argmax(tstat))]
        pvals[worst] = running
        M.remove(worst)
    pvals[M[0]] = 1.0
    return {m: pvals[dup.get(m, m)] for m in methods}, dup
