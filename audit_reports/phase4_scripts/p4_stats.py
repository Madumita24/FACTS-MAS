"""
Phase 4 step 2: statistical evidence on the per-cell loss table (pre-registered in phase4_log.md).

  panel DM (Qu-Timmermann-Zhu equal average performance; NW Bartlett lag h-1)
  per-MSA DM with HLN correction, lag h-1, BH at 5%
  moving block bootstrap over weekly cross-sections (L=13, B=5000) vs iid over cells
  C2 vs FA-ETS-d (and P vs FA-ETS-d) equivalence: 90% paired block CI within +/- 2% of FA-ETS-d MAPE
  Model Confidence Set (T_max, block bootstrap), alpha 0.10 and 0.25
  TimesFM, fold 1 only

Usage (repo root): python audit_reports/phase4_scripts/p4_stats.py
"""
import os
import subprocess

import numpy as np
import pandas as pd
from scipy import stats

COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
OUT = os.path.join("audit_reports", "results_tables")
D = pd.read_csv(os.path.join(OUT, "phase4_cell_losses.csv"), comment="#", parse_dates=["origin"])
METHODS = ["P", "P_5ch", "naive", "snaive", "naive2", "fa_ets_d", "selbest", "C2", "R1c", "ew_ts", "hz"]
MCS_SET = ["P", "naive", "fa_ets_d", "selbest", "C2", "ew_ts", "hz", "R1c", "naive2"]
B, L, SEED = 5000, 13, 20261016
pd.set_option("display.width", 250)


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
    v = nw_var(d, h - 1)
    t = d.mean() / np.sqrt(v / T)
    if hln:
        t *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return t, 2 * stats.t.sf(abs(t), T - 1)


def bh(p, q=0.05):
    p = np.asarray(p)
    o = np.argsort(p)
    m = len(p)
    thr = q * np.arange(1, m + 1) / m
    passed = p[o] <= thr
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    sig = np.zeros(m, bool)
    sig[o[:k]] = True
    return sig


def block_idx(T, rng):
    n = int(np.ceil(T / L))
    starts = rng.integers(0, T - L + 1, size=n)
    return np.concatenate([np.arange(s, s + L) for s in starts])[:T]


def weekly(sub, cols):
    w = sub.groupby("origin")[cols].mean().sort_index()
    assert (sub.groupby("origin").size() == 15).all()
    return w


results = {}
for label, data in (("CLEANBLEED", D[~D.bleed]), ("DISCLOSURE", D)):
    rows_dm, rows_msa, rows_ci, rows_eq, rows_mcs = [], [], [], [], []
    for h in (4, 8, 13):
        sub = data[data.h == h]
        W = weekly(sub, METHODS)
        T = len(W)
        rng = np.random.default_rng(SEED + h)
        idx = np.stack([block_idx(T, rng) for _ in range(B)])
        boot = {m: W[m].to_numpy()[idx].mean(axis=1) for m in METHODS}
        rng2 = np.random.default_rng(SEED + 100 + h)
        n = len(sub)
        iidx = rng2.integers(0, n, size=(B, n))
        iid = {m: sub[m].to_numpy()[iidx].mean(axis=1) for m in ("P", "fa_ets_d", "naive", "C2")}

        # levels CI
        for m in METHODS:
            lo, hi = np.percentile(boot[m], [2.5, 97.5])
            rec = dict(h=h, method=m, mape=W[m].mean(), block_lo=lo, block_hi=hi, block_width=hi - lo)
            if m in iid:
                a, b = np.percentile(iid[m], [2.5, 97.5])
                rec.update(iid_lo=a, iid_hi=b, iid_width=b - a, width_ratio=(hi - lo) / (b - a))
            rows_ci.append(rec)
        # pairwise DM + difference CIs
        pairs = [("P", "naive"), ("P", "snaive"), ("P", "naive2"), ("P", "fa_ets_d"), ("P", "selbest"), ("P", "R1c"),
                 ("P", "ew_ts"), ("P", "hz"), ("P", "P_5ch"), ("C2", "fa_ets_d"), ("fa_ets_d", "naive"), ("R1c", "fa_ets_d")]
        for a, b in pairs:
            d = (W[a] - W[b]).to_numpy()
            t, p = dm(d, h)
            dd = boot[a] - boot[b]
            lo, hi = np.percentile(dd, [2.5, 97.5])
            rec = dict(h=h, A=a, B=b, mean_diff=d.mean(), panel_dm_t=t, panel_dm_p=p, block_lo=lo, block_hi=hi, T_weeks=T)
            if a in iid and b in iid:
                ia, ib = np.percentile(iid[a] - iid[b], [2.5, 97.5])
                rec.update(iid_lo=ia, iid_hi=ib)
            rows_dm.append(rec)
            # per-MSA DM
            ps, ts = [], []
            for msa, g in sub.groupby("msa"):
                g = g.sort_values("origin")
                tt, pp = dm((g[a] - g[b]).to_numpy(), h, hln=True)
                ts.append(tt)
                ps.append(pp)
            sig = bh(ps)
            ts = np.array(ts)
            rows_msa.append(dict(h=h, A=a, B=b, n_msa=len(ps), raw_p05_A_better=int(((np.array(ps) < .05) & (ts < 0)).sum()),
                                 raw_p05_B_better=int(((np.array(ps) < .05) & (ts > 0)).sum()),
                                 BH_A_better=int((sig & (ts < 0)).sum()), BH_B_better=int((sig & (ts > 0)).sum()),
                                 A_lower_mape_msas=int((ts < 0).sum())))
        # equivalence
        for a in ("C2", "P"):
            dd = boot[a] - boot["fa_ets_d"]
            lo, hi = np.percentile(dd, [5, 95])
            delta = 0.02 * W["fa_ets_d"].mean()
            rows_eq.append(dict(h=h, A=a, B="fa_ets_d", mean_diff=(W[a] - W["fa_ets_d"]).mean(), ci90_lo=lo, ci90_hi=hi,
                                delta=delta, equivalent=bool(lo > -delta and hi < delta),
                                A_significantly_worse=bool(lo > 0), A_significantly_better=bool(hi < 0)))
        # MCS (T_max). Exact duplicates (e.g. SelBest-model == FA-ETS-d when validation picks FA-ETS-d in
        # every fold) have zero-variance differences and break the t-statistic, so they are merged first.
        M, dup = [], {}
        for m_ in MCS_SET:
            twin = next((k for k in M if np.allclose(W[m_].to_numpy(), W[k].to_numpy(), rtol=0, atol=1e-12)), None)
            if twin is None:
                M.append(m_)
            else:
                dup[m_] = twin
        MSET = list(M)
        Lw = W[MSET].to_numpy()
        Lb = Lw[idx]  # B x T x m
        pvals, elim = {}, []
        running = 0.0
        while len(M) > 1:
            cols = [MSET.index(m) for m in M]
            l = Lw[:, cols]
            lb = Lb[:, :, cols]
            dbar = l.mean(0) - l.mean()             # d_i. relative to set average
            dboot = lb.mean(1) - lb.mean(axis=(1, 2))[:, None]
            se = dboot.std(0, ddof=1)
            tstat = dbar / se
            tb = ((dboot - dbar) / se).max(1)
            p = float((tb >= tstat.max()).mean())
            running = max(running, p)
            worst = M[int(np.argmax(tstat))]
            pvals[worst] = running
            elim.append(worst)
            M.remove(worst)
        pvals[M[0]] = 1.0
        for m in MCS_SET:
            k = dup.get(m, m)
            rows_mcs.append(dict(h=h, method=m, identical_to=dup.get(m, ""), mcs_p=pvals[k], in_MCS90=pvals[k] >= 0.10,
                                 in_MCS75=pvals[k] >= 0.25,
                                 elimination_order=elim.index(k) + 1 if k in elim else len(MSET)))
    results[label] = dict(dm=pd.DataFrame(rows_dm), msa=pd.DataFrame(rows_msa), ci=pd.DataFrame(rows_ci),
                          eq=pd.DataFrame(rows_eq), mcs=pd.DataFrame(rows_mcs))

for label, r in results.items():
    print(f"\n################ {label}")
    print("\n-- panel DM and block-bootstrap difference CIs (negative = A better)")
    print(r["dm"].round(4).to_string(index=False))
    print("\n-- per-MSA DM (HLN, lag h-1), counts")
    print(r["msa"].to_string(index=False))
    print("\n-- MAPE level CIs: block (L=13) vs iid over cells")
    print(r["ci"].round(3).to_string(index=False))
    print("\n-- equivalence vs FA-ETS-d (90% paired block CI, margin 2%)")
    print(r["eq"].round(4).to_string(index=False))
    print("\n-- MCS (T_max)")
    print(r["mcs"].pivot(index="method", columns="h", values="mcs_p").round(3).to_string())
    print(r["mcs"].pivot(index="method", columns="h", values="elimination_order").to_string())
    print(r["mcs"][r["mcs"].identical_to != ""][["h", "method", "identical_to"]].to_string(index=False))
    cfgline = (f"# CONFIG: commit={COMMIT} | {label} | TEST weekly cross-section mean MAPE | NW Bartlett lag h-1 | "
               f"block bootstrap L={L} B={B} seed={SEED} | R1c protocol, T=0.05 PROVISIONAL\n")
    for k, df in r.items():
        with open(os.path.join(OUT, f"phase4_{k}_{label}.csv"), "w", newline="") as fh:
            fh.write(cfgline)
            df.to_csv(fh, index=False)

# TimesFM, fold 1 only
tf = pd.read_csv("timesfm_baseline_fold1_full_window_results.csv", parse_dates=["origin"])
print("\n################ TimesFM fold 1 (failed flags:", int(tf.timesfm_failed.sum()), ")")
m = D[D.fold == 1].merge(tf.rename(columns={"horizon": "h"}), on=["msa", "origin", "h"], how="inner")
assert np.allclose(m.naive, m.naive_mape), "naive mismatch with TimesFM file"
rows = []
for h in (4, 8, 13):
    sub = m[m.h == h]
    W = sub.groupby("origin")[["P", "fa_ets_d", "timesfm_mape", "naive"]].mean()
    for a in ("P", "fa_ets_d"):
        t, p = dm((W[a] - W["timesfm_mape"]).to_numpy(), h)
        rows.append(dict(h=h, A=a, B="timesfm", n_cells=len(sub), mape_A=W[a].mean(), mape_timesfm=W["timesfm_mape"].mean(),
                         panel_dm_t=t, panel_dm_p=p, msas_A_better=int((sub.groupby("msa")[a].mean() < sub.groupby("msa").timesfm_mape.mean()).sum())))
TF = pd.DataFrame(rows)
print(TF.round(4).to_string(index=False))
with open(os.path.join(OUT, "phase4_timesfm_fold1.csv"), "w", newline="") as fh:
    fh.write(f"# CONFIG: commit={COMMIT} | fold 1 TEST only (29 weeks x 15 MSAs) | TimesFM 2.5 from timesfm_baseline_fold1_full_window_results.csv | panel DM NW lag h-1\n")
    TF.to_csv(fh, index=False)
