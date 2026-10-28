"""
Phase 2 analysis. Baseline R1 (seasonal window 7). Fast mirror pinned to production.

Run from repo root:  python audit_reports/phase2_scripts/p2_analysis.py
Writes audit_reports/results_tables/phase2_*.csv (config header on each).
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "tests_audit"))
import audit_core as ac  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from facts_mas.agent_cache import AgentCache  # noqa: E402
from facts_mas.run_backtest import FOLDS, HORIZONS  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
OUT = os.path.join("audit_reports", "results_tables")

panel = ac.load_panel()
cache = AgentCache.load("agent_output_cache.pkl")
cells = ac.cells_from_cache(cache, panel)
R1 = ac.r1_values_fn(ac.load_r1_seasonal())
NAIVE_LIKE = ("event", "macro", "intrinsic")


def cfg(**kw):
    base = dict(commit=COMMIT, baseline="R1 (seasonal w7 nested)", split="TEST", folds="1-6", msas=15,
                horizons="4,8,13", zillow="smoothed", fusion="adaptive T=0.05 104wk (PROVISIONAL: T, Router, adaptive choice)",
                rho_max=0.30, primary="CLEANBLEED")
    base.update(kw)
    return " | ".join(f"{k}={v}" for k, v in base.items())


def write(name, df, config):
    p = os.path.join(OUT, f"phase2_{name}.csv")
    with open(p, "w", newline="") as fh:
        fh.write(f"# CONFIG: {config}\n")
        df.to_csv(fh, index=False)


def source(c, M):
    m = M[c.test_fold]
    if m.mode[c.h] == "fold":
        return "fold"
    g = m.regime_grid[c.regime][c.h]
    if g["fallback"]:
        return "fold(fallback)"
    return "regime(n=0 uniform)" if g["n"] == 0 else "regime"


def scored(M, fn=R1, exclude_bleed=True, disclosure=False):
    d = ac.score(cells, M, fn, exclude_bleed=exclude_bleed, disclosure=disclosure)
    tc = {(c.msa, c.origin, c.h): c for c in ac.test_cells(cells, exclude_bleed=exclude_bleed)}
    d["source"] = [source(tc[(r.msa, r.origin, r.h)], M) for r in d.itertuples()]
    return d


# ═══ A. Fix 2 and F0.10 ═══════════════════════════════════════════════════
print("=== A. fix 2 (support >= 13 distinct validation weeks) and F0.10")
configs = {
    "R1": dict(),
    "R1 + fix2": dict(min_support_weeks=13),
    "R1 + F0.10": dict(bound_targets=True),
    "R1 + fix2 + F0.10 (combined)": dict(min_support_weeks=13, bound_targets=True),
    "DISCLOSURE sens: combined, threshold 4": dict(min_support_weeks=4, bound_targets=True),
    "DISCLOSURE sens: combined, threshold 26": dict(min_support_weeks=26, bound_targets=True),
}
models, res_c, res_d, rows_h, rows_sup, rows_pf = {}, {}, {}, [], [], []
for lab, kw in configs.items():
    M = ac.build_models(cells, R1, **kw)
    models[lab] = M
    dc = scored(M)
    dd = scored(M, exclude_bleed=False, disclosure=True)
    res_c[lab], res_d[lab] = dc, dd
    for h in HORIZONS:
        rows_h.append(dict(config=lab, h=h, fused_CLEANBLEED=dc[dc.h == h].fused.mean(),
                           naive_CLEANBLEED=dc[dc.h == h].naive.mean(),
                           fused_DISCLOSURE=dd[dd.h == h].fused.mean(), naive_DISCLOSURE=dd[dd.h == h].naive.mean(),
                           n_CLEANBLEED=int((dc.h == h).sum())))
    for f, m in M.items():
        for r in ac.REGIMES:
            for h in HORIZONS:
                g = m.regime_grid[r][h]
                rows_sup.append(dict(config=lab, fold=f, regime=r, h=h, support_weeks=g["support_weeks"],
                                     n_points=g["n"], fallback=g["fallback"], mode=m.mode[h]))
    pf = dc.groupby(["h", "fold"])[["fused", "naive"]].mean().reset_index()
    pf.insert(0, "config", lab)
    rows_pf.append(pf)
H = pd.DataFrame(rows_h)
print(H.round(4).to_string(index=False))
write("A_headlines", H, cfg(note="fix2 = fold weights when regime support < threshold distinct validation weeks"))
PF = pd.concat(rows_pf)
write("A_per_fold", PF, cfg())
SUP = pd.DataFrame(rows_sup)
write("A_support", SUP, cfg(split="VALIDATION support counts"))
print("\nfallback cells (fold, regime, h) by config:")
for lab in configs:
    s = SUP[(SUP.config == lab) & SUP.fallback]
    print(f"  {lab}: {sorted(zip(s.fold, s.regime, s.h))}")
print("\nper fold CLEANBLEED fused:")
print(PF.pivot_table(index=["h", "fold"], columns="config", values="fused").round(3).to_string())

base = models["R1"]
print("\nmode flips vs R1 and test cells whose weight set changes:")
rows = []
for lab in list(configs)[1:]:
    M = models[lab]
    flips = [(f, h, base[f].mode[h], M[f].mode[h]) for f in base for h in HORIZONS if base[f].mode[h] != M[f].mode[h]]
    a, b = res_c["R1"], res_c[lab]
    j = a.merge(b, on=["msa", "origin", "h"], suffixes=("_r1", "_new"))
    ch = j[j.source_r1 != j.source_new]
    wchg = []
    for c in ac.test_cells(cells, exclude_bleed=True):
        w0, w1 = ac.weights_for(c, base), ac.weights_for(c, M)
        wchg.append(max(abs(w0[k] - w1[k]) for k in ac.AGENTS) > 1e-12)
    print(f"  {lab}: mode flips {flips}; cells changing weight SOURCE {len(ch)} "
          f"{ch.groupby(['h', 'source_r1', 'source_new']).size().to_dict()}; cells with any weight change {sum(wchg)}")
    rows.append(dict(config=lab, mode_flips=str(flips), cells_source_change=len(ch), cells_any_weight_change=sum(wchg)))
    wins = {h: int((b[b.h == h].groupby("msa").fused.mean() < a[a.h == h].groupby("msa").fused.mean()).sum()) for h in HORIZONS}
    pfw = {h: int((PF[(PF.config == lab) & (PF.h == h)].set_index("fold").fused
                   < PF[(PF.config == "R1") & (PF.h == h)].set_index("fold").fused).sum()) for h in HORIZONS}
    print(f"      per-MSA wins vs R1 {wins}; per-fold wins vs R1 {pfw}")
    rows[-1].update(msa_wins_vs_r1=str(wins), fold_wins_vs_r1=str(pfw))
write("A_changes", pd.DataFrame(rows), cfg())
print("\nweight source of CLEANBLEED test cells:")
for lab in ("R1", "R1 + fix2", "R1 + fix2 + F0.10 (combined)"):
    print(f"  {lab}:", res_c[lab].groupby(["h", "source"]).size().to_dict())

# ═══ B. Weight ordering anomaly ═══════════════════════════════════════════
print("\n=== B. weight ordering (R1 and PROVISIONAL), mean test weight per channel")
Mp = ac.build_models(cells)  # PROVISIONAL
rows = []
for lab, M, fn in (("PROVISIONAL", Mp, ac.production_values), ("R1", base, R1),
                   ("R1 combined", models["R1 + fix2 + F0.10 (combined)"], R1)):
    for c in ac.test_cells(cells, exclude_bleed=True):
        w = ac.weights_for(c, M)
        rows.append(dict(config=lab, h=c.h, fold=c.test_fold, regime=c.regime, source=source(c, M), **w))
WT = pd.DataFrame(rows)
mw = WT.groupby(["config", "h"])[ac.AGENTS].mean()
print(mw.round(3).to_string())
write("B_mean_weights", mw.reset_index(), cfg(note="mean fusion weight over CLEANBLEED test cells"))
N = WT.groupby(["config", "h"]).size().rename("N_h").reset_index()
grp = WT.groupby(["config", "h", "fold", "regime", "source"])
dec = grp[ac.AGENTS].sum().reset_index()
dec["n_cells"] = grp.size().values
dec = dec.merge(N, on=["config", "h"])
dec[ac.AGENTS] = dec[ac.AGENTS].div(dec["N_h"], axis=0)
print("\n13wk decomposition of mean weight (contribution to the mean), PROVISIONAL and R1:")
print(dec[(dec.h == 13) & dec.config.isin(["PROVISIONAL", "R1"])].round(3).to_string(index=False))
write("B_weight_decomposition", dec, cfg(note="sum of weights in group / all CLEANBLEED test cells at h"))

print("\n--- validation skill three ways (R1 values, production validation window, unbounded)")
rows = []
for f in FOLDS:
    val = [c for c in cells if ac._in_val(c, f, False)]
    for h in HORIZONS + ["pooled"]:
        sub = [c for c in val if h == "pooled" or c.h == h]
        rec = dict(fold=f.fold, h=h)
        for a in ac.AGENTS:
            pts, ae_k, ae_n, mk, mn = [], 0.0, 0.0, [], []
            for c in sub:
                v = np.asarray(R1(c, f.fold)[a])
                nm = ac._mape_frac(c.actuals, c.naive)
                am = ac._mape_frac(c.actuals, v)
                if nm and nm > 0:
                    pts.append(1 - am / nm)
                ae_k += np.abs(c.actuals - v).sum()
                ae_n += np.abs(c.actuals - c.naive).sum()
                mk.append(am)
                mn.append(nm)
            rec[f"{a}_median"] = float(np.median(pts))
            rec[f"{a}_maeratio"] = 1 - ae_k / ae_n
            rec[f"{a}_maperatio"] = 1 - np.mean(mk) / np.mean(mn)
        rows.append(rec)
SK = pd.DataFrame(rows)
cols = ["fold", "h"] + [f"{a}_{k}" for k in ("median", "maeratio", "maperatio") for a in ("ar", "macro", "seasonality", "event", "intrinsic")]
print(SK[SK.h == 13][cols].round(3).to_string(index=False))
write("B_validation_skill_3ways", SK, cfg(split="VALIDATION (production window, unbounded)"))
rank = SK[SK.h == 13]
print("folds where AR > macro at 13wk: median", int((rank.ar_median > rank.macro_median).sum()),
      "| MAE-ratio", int((rank.ar_maeratio > rank.macro_maeratio).sum()),
      "| MAPE-ratio", int((rank.ar_maperatio > rank.macro_maperatio).sum()))

print("\n--- regime-cell median skills at 13wk (R1) for cells that carry test cells")
rows = []
for f, m in base.items():
    for r in ac.REGIMES:
        g = m.regime_grid[r][13]
        ntest = sum(1 for c in ac.test_cells(cells, 13, True) if c.test_fold == f and c.regime == r)
        if ntest:
            rows.append(dict(fold=f, regime=r, n_test=ntest, support_weeks=g["support_weeks"], n=g["n"], mode=m.mode[13],
                             **{f"skill_{a}": g["skills"][a] for a in ac.AGENTS},
                             **{f"w_{a}": g["weights"][a] for a in ac.AGENTS}))
RS = pd.DataFrame(rows)
print(RS.round(3).to_string(index=False))
write("B_regime_cells_13wk", RS, cfg(split="VALIDATION skills; n_test = CLEANBLEED test cells"))

print("\n--- standalone TEST MAPE per fold, 13wk, CLEANBLEED (R1)")
st = res_c["R1"][res_c["R1"].h == 13].groupby("fold")[["naive", "solo_ar", "solo_macro", "solo_seasonality", "solo_event", "solo_intrinsic"]].mean()
print(st.round(3).to_string())
write("B_standalone_test_13wk", st.reset_index(), cfg())

# ═══ C. Persistence clones ═════════════════════════════════════════════════
print("\n=== C. persistence clones")
dev = {a: float(np.mean([np.mean(np.abs(np.asarray(R1(c, 1)[a]) - c.naive) / c.naive) for c in cells])) * 100
       for a in ac.AGENTS}
print("mean |channel - naive| / naive over all 15,480 cells (%):", {a: round(v, 3) for a, v in dev.items()})
print("cells where event == naive exactly:", sum(np.array_equal(c.values["event"], c.naive) for c in cells), "of", len(cells))
nl = WT.assign(naive_like=WT[list(NAIVE_LIKE)].sum(axis=1)).groupby(["config", "h"]).naive_like.mean()
print("mean total weight on event + macro + intrinsic:", nl.round(3).to_dict())
write("C_naive_like_weight", nl.reset_index(), cfg(note=f"deviation from naive % {dev}"))

# ═══ D. Attribution ════════════════════════════════════════════════════════
print("\n=== D. attribution in levels (eq. 11) vs in forecasted change, final step H, R1 weights")
def contrib(c, M):
    w = ac.weights_for(c, M)
    tot = sum(v for v in w.values() if v > 0)
    vals = R1(c, c.test_fold)
    yt = c.naive[-1]
    lvl = {a: (w[a] / tot) * vals[a][-1] for a in ac.AGENTS}
    chg = {a: (w[a] / tot) * (vals[a][-1] - yt) for a in ac.AGENTS}
    return w, yt, lvl, chg

ex = [("New York", "2021-05-01", 4), ("Miami", "2024-11-09", 8), ("Phoenix", "2022-12-10", 13)]
by = {(c.msa, c.origin.isoformat(), c.h): c for c in ac.test_cells(cells)}
rows = []
for k in ex:
    c = by[k]
    w, yt, lvl, chg = contrib(c, base)
    fused = sum(lvl.values())
    for a in ac.AGENTS:
        rows.append(dict(msa=k[0], origin=k[1], h=k[2], y_t=yt, fused_final=fused, actual_final=c.actuals[-1], agent=a,
                         weight=w[a], level_contrib=lvl[a], level_share=lvl[a] / fused,
                         change_contrib=chg[a], change_share_abs=abs(chg[a]) / sum(abs(x) for x in chg.values())))
EX = pd.DataFrame(rows)
print(EX.round(3).to_string(index=False))
write("D_attribution_examples", EX, cfg(note="final step H; weights renormalised as in fusion"))
rows = []
for c in ac.test_cells(cells, exclude_bleed=True):
    w, yt, lvl, chg = contrib(c, base)
    sl, sc = sum(lvl.values()), sum(abs(x) for x in chg.values())
    for a in ac.AGENTS:
        rows.append(dict(h=c.h, agent=a, level_share=lvl[a] / sl, change_share=abs(chg[a]) / sc if sc > 0 else np.nan))
AG = pd.DataFrame(rows).groupby(["h", "agent"])[["level_share", "change_share"]].mean().reset_index()
print(AG.pivot(index="agent", columns="h", values=["level_share", "change_share"]).round(3).to_string())
write("D_attribution_aggregate", AG, cfg(note="mean over CLEANBLEED test cells of each channel's share"))

# ═══ E. Regime threshold sensitivity ══════════════════════════════════════
print("\n=== E. regime labels: fed funds 13-week change near +/-0.25")
md = panel[panel.msa == "Phoenix"].sort_values("date").reset_index(drop=True)
md["delta"] = md["fed_funds"] - md["fed_funds"].shift(13)
lab = {c.origin: c.regime for c in cells if c.h == 4}
md = md[md["date"].dt.date.isin(lab)].copy()
md["label"] = md["date"].dt.date.map(lab)
recon = np.where(md.delta > 0.25, "hiking", np.where(md.delta < -0.25, "cutting", "stable"))
print("reconstructed label == cached label on", int((recon == md.label).sum()), "of", len(md), "dates")
near = (md.delta.abs() - 0.25).abs() <= 0.05
print("dates with |delta| within 0.05 of 0.25:", int(near.sum()), "of", len(md), f"({near.mean():.1%})")
for thr in (0.20, 0.30):
    alt = np.where(md.delta > thr, "hiking", np.where(md.delta < -thr, "cutting", "stable"))
    print(f"  threshold {thr}: labels that flip = {int((alt != md.label).sum())}")
md["test_fold"] = md["date"].dt.date.map(ac.fold_of_test)
print("near-threshold dates by test fold:", md[near].groupby("test_fold", dropna=False).size().to_dict())
write("E_regime_threshold", md[["date", "fed_funds", "delta", "label", "test_fold"]].assign(near_threshold=near.values),
      cfg(split="all cached origins", note="delta = fed_funds[t] - fed_funds[t-13 rows]"))
zero = SUP[(SUP.config == "R1") & (SUP.support_weeks == 0) & (SUP.h == 13) & (SUP.regime != "stable")]
print("R1: folds with zero validation weeks in hiking/cutting:", sorted(zip(zero.fold, zero.regime)))

# ═══ F. Ablation numbers referenced in the definitions ═════════════════════
print("\n=== F. '- Router' = fold weights pooled over horizons and regimes (ablation.py run_ablation_table)")
for lab, M in (("R1", base), ("R1 combined", models["R1 + fix2 + F0.10 (combined)"])):
    a = ac.score(cells, M, R1, "adaptive", exclude_bleed=True).groupby("h").fused.mean()
    b = ac.score(cells, M, R1, "fold", exclude_bleed=True).groupby("h").fused.mean()
    print(f"  {lab}: adaptive {a.round(3).to_dict()} | Router removed (fold weights) {b.round(3).to_dict()} | "
          f"Router effect {(b - a).round(3).to_dict()}")
