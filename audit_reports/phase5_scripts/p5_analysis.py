"""
Phase 5 analysis (pre-registered in phase5_preregistration.md, incl. Addendum A).

B5   base: channels {FA-ETS-d (damped choice per fold), seasonal w7, macro}, skill reference FA-ETS-d,
     R1c protocol (fix 2, F0.10, adaptive Router, T = 0.05)
P5a  B5 + listing-flow channel
P5b  delta fusion on FA-ETS-d: y = y_FA + sum beta_k (y_k - y_FA), beta >= 0, NNLS on relative errors, per (fold, h)
P5c  B5 with temperature T and shrinkage lambda toward equal weights, chosen per (fold, h) on validation

Usage (repo root): python audit_reports/phase5_scripts/p5_analysis.py <phase3 baselines.pkl> <flow.pkl>
"""
import itertools
import os
import pickle
import subprocess
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "tests_audit"))
sys.path.insert(0, os.path.join(os.getcwd(), "audit_reports", "phase4_scripts"))
sys.path.insert(0, os.path.join(os.getcwd(), "audit_reports", "phase5_scripts"))
import audit_core as ac  # noqa: E402
import stats_lib as sl  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

from facts_mas.agent_cache import AgentCache  # noqa: E402
from facts_mas.run_backtest import FOLDS, HORIZONS  # noqa: E402

pd.set_option("display.width", 250)
COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
OUT = os.path.join("audit_reports", "results_tables")
R1C = dict(min_support_weeks=13, bound_targets=True)
KEEP = ["ar", "seasonality", "macro"]
FOLD = {f.fold: f for f in FOLDS}

cells = ac.cells_from_cache(AgentCache.load("agent_output_cache.pkl"), ac.load_panel())
B = pickle.load(open(sys.argv[1], "rb"))
FLOW = pickle.load(open(sys.argv[2], "rb"))
R1 = ac.r1_values_fn(ac.load_r1_seasonal())
key = lambda c: (c.msa, c.origin, c.h)
val = {f.fold: [c for c in cells if ac._in_val(c, f, True)] for f in FOLDS}

ch = {}
for f in FOLDS:
    mu = np.mean([ac.mape_pct(c.actuals, B[key(c)]["fa_ets"]) for c in val[f.fold]])
    md = np.mean([ac.mape_pct(c.actuals, B[key(c)]["fa_ets_d"]) for c in val[f.fold]])
    ch[f.fold] = "fa_ets_d" if md < mu else "fa_ets"
print("damped choice:", ch)


def b5_fn(c, f):
    v = dict(R1(c, f))
    v["ar"] = B[key(c)][ch[f]]
    return v


def p5a_fn(c, f):
    v = b5_fn(c, f)
    v["flow"] = FLOW[key(c)]
    return v


ref = lambda c, f: B[key(c)]["fa_ets_d"]
M5 = ac.build_models(cells, b5_fn, agents=KEEP, reference_fn=ref, **R1C)
MA = ac.build_models(cells, p5a_fn, agents=KEEP + ["flow"], reference_fn=ref, **R1C)

# ── P5b: NNLS betas per (fold, h) ─────────────────────────────────────────
DELTA = ["seasonality", "macro"]
betas = {}
for f in FOLDS:
    for h in HORIZONS:
        X, y = [], []
        for c in val[f.fold]:
            if c.h != h:
                continue
            v = b5_fn(c, f.fold)
            fa = np.asarray(v["ar"])
            X.append(np.column_stack([(np.asarray(v[k]) - fa) / c.actuals for k in DELTA]))
            y.append((c.actuals - fa) / c.actuals)
        betas[(f.fold, h)] = nnls(np.vstack(X), np.concatenate(y))[0]


def p5b_fc(c, f):
    v = b5_fn(c, f)
    fa = np.asarray(v["ar"])
    return np.maximum(fa + sum(b * (np.asarray(v[k]) - fa) for b, k in zip(betas[(f, c.h)], DELTA)), 0.0)


# ── P5c: temperature x shrinkage, chosen per (fold, h) on validation ─────
TS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
LAMS = [0.0, 0.25, 0.5, 0.75, 1.0]
MT = {T: ac.build_models(cells, b5_fn, agents=KEEP, reference_fn=ref, temperature=T, **R1C) for T in TS}


def router_w(m, c):
    return m.fold_weights if m.mode[c.h] == "fold" else m.regime_grid[c.regime][c.h]["weights_eff"]


def shrink(w, lam):
    return {a: (1 - lam) * w[a] + lam / len(KEEP) for a in KEEP}


p5c_choice, p5c_val = {}, {}
for f in FOLDS:
    for h in HORIZONS:
        best = None
        for T, lam in itertools.product(TS, LAMS):
            m = MT[T][f.fold]
            vm = ac.val_fused_mape(cells, f, m, b5_fn, h, weights_fn=lambda c, m=m, lam=lam: shrink(router_w(m, c), lam))
            cand = (round(vm, 12), -lam, -T)
            if best is None or cand < best[0]:
                best = (cand, T, lam, vm)
        p5c_choice[(f.fold, h)] = (best[1], best[2])
        p5c_val[(f.fold, h)] = best[3]

# ── validation MAPE per candidate (mean over folds of each fold's own validation) ──
VAL = []
for h in HORIZONS:
    for f in FOLDS:
        vb5 = ac.val_fused_mape(cells, f, M5[f.fold], b5_fn, h)
        vpa = ac.val_fused_mape(cells, f, MA[f.fold], p5a_fn, h)
        vpb = float(np.mean([ac.mape_pct(c.actuals, p5b_fc(c, f.fold)) for c in val[f.fold] if c.h == h]))
        vfa = float(np.mean([ac.mape_pct(c.actuals, B[key(c)]["fa_ets_d"]) for c in val[f.fold] if c.h == h]))
        VAL.append(dict(h=h, fold=f.fold, B5=vb5, P5a=vpa, P5b=vpb, P5c=p5c_val[(f.fold, h)], fa_ets_d=vfa))
VAL = pd.DataFrame(VAL)

# ── test per-cell losses ──────────────────────────────────────────────────
tc = ac.test_cells(cells)
ac.assert_scoring_allowed([d for c in tc for d in c.target_dates], disclosure=True)  # bleed rows flagged
rows = []
for c in tc:
    f = c.test_fold
    T, lam = p5c_choice[(f, c.h)]
    fc = {"fa_ets_d": B[key(c)]["fa_ets_d"],
          "B5": ac.fuse(b5_fn(c, f), ac.weights_for(c, M5), c.h),
          "P5a": ac.fuse(p5a_fn(c, f), ac.weights_for(c, MA), c.h),
          "P5b": p5b_fc(c, f),
          "P5c": ac.fuse(b5_fn(c, f), shrink(router_w(MT[T][f], c), lam), c.h),
          "flow_solo": FLOW[key(c)], "naive": c.naive}
    rows.append(dict(h=c.h, fold=f, msa=c.msa, origin=c.origin, bleed=bool(c.last_target >= ac.RESERVED_START),
                     **{k: ac.mape_pct(c.actuals, v) for k, v in fc.items()}))
D = pd.DataFrame(rows)
P4 = pd.read_csv(os.path.join(OUT, "phase4_cell_losses.csv"), comment="#", parse_dates=["origin"])[["h", "msa", "origin", "P"]]
D["origin"] = pd.to_datetime(D["origin"])
D = D.merge(P4, on=["h", "msa", "origin"], how="left")
assert D["P"].notna().all()
METH = ["fa_ets_d", "B5", "P5a", "P5b", "P5c", "P", "flow_solo", "naive"]
CAND = ["P5a", "P5b", "P5c"]

print("\n=== pooled TEST MAPE (CLEANBLEED | DISCLOSURE)")
C = D[~D.bleed]
print(pd.concat([C.groupby("h")[METH].mean().add_suffix("_CB"), D.groupby("h")[METH].mean().add_suffix("_DI")], axis=1)[
    [m + s for m in METH for s in ("_CB",)] + [m + "_DI" for m in ("fa_ets_d", "B5", "P5a", "P5b", "P5c")]].round(4).to_string())
PF = C.groupby(["h", "fold"])[METH].mean()
print("\n=== per fold (CLEANBLEED)")
print(PF.round(3).to_string())
print("\n=== validation MAPE per fold")
print(VAL.round(3).to_string(index=False))

# ── promotion and comparisons ─────────────────────────────────────────────
PR, ST = [], []
for h in HORIZONS:
    sub = C[C.h == h]
    W = sl.weekly(sub, METH)
    idx = sl.block_indices(len(W), sl.SEED + h)
    vmean = VAL[VAL.h == h].mean()
    pf = PF.loc[h]
    pm = sub.groupby("msa")[METH].mean()
    for k in CAND:
        fb = int((pf[k] < pf["B5"]).sum())
        PR.append(dict(h=h, candidate=k, val=vmean[k], val_B5=vmean["B5"], val_win=bool(vmean[k] < vmean["B5"]),
                       test=W[k].mean(), test_B5=W["B5"].mean(), folds_better_than_B5=fb,
                       PROMOTE=bool(vmean[k] < vmean["B5"] and fb >= 5),
                       test_fa=W["fa_ets_d"].mean(), folds_better_than_fa=int((pf[k] < pf["fa_ets_d"]).sum()),
                       msas_better_than_B5=int((pm[k] < pm["B5"]).sum()), msas_better_than_fa=int((pm[k] < pm["fa_ets_d"]).sum())))
    for a, b in [(k, "B5") for k in CAND] + [(k, "fa_ets_d") for k in CAND + ["B5"]] + [("B5", "P"), ("flow_solo", "fa_ets_d")]:
        ST.append(sl.pair_stats(sub, W, idx, a, b, h))
    mp, dup = sl.mcs(W, idx, ["fa_ets_d", "B5", "P5a", "P5b", "P5c", "P"])
    for m_, p in mp.items():
        ST.append(dict(h=h, A=m_, B="MCS", mcs_p=p, identical_to=dup.get(m_, "")))
PR, ST = pd.DataFrame(PR), pd.DataFrame(ST)
print("\n=== promotion vs B5 (val win AND test better in >= 5/6 folds), and vs FA-ETS-d")
print(PR.round(4).to_string(index=False))
print("\n=== paired statistics (negative mean_diff = A better)")
print(ST[ST.B != "MCS"][["h", "A", "B", "mean_diff", "panel_dm_t", "panel_dm_p", "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi",
                         "BH_A_better", "BH_B_better"]].round(4).to_string(index=False))
print("\n=== MCS p-values (90% set = p >= 0.10)")
print(ST[ST.B == "MCS"].pivot(index="A", columns="h", values="mcs_p").round(3).to_string())

# ── weights, betas, choices ───────────────────────────────────────────────
wr = []
for lab, M, agents in (("B5", M5, KEEP), ("P5a", MA, KEEP + ["flow"])):
    for c in ac.test_cells(cells, exclude_bleed=True):
        w = ac.weights_for(c, M)
        wr.append(dict(cand=lab, h=c.h, **{a: w.get(a, 0.0) for a in KEEP + ["flow"]}))
for c in ac.test_cells(cells, exclude_bleed=True):
    T, lam = p5c_choice[(c.test_fold, c.h)]
    w = shrink(router_w(MT[T][c.test_fold], c), lam)
    wr.append(dict(cand="P5c", h=c.h, **w, flow=0.0))
WT = pd.DataFrame(wr).groupby(["cand", "h"])[KEEP + ["flow"]].mean()
print("\n=== mean weights over CLEANBLEED test cells ('ar' = FA-ETS-d)")
print(WT.round(3).to_string())
BT = pd.DataFrame([dict(fold=f, h=h, beta_seasonal=b[0], beta_macro=b[1]) for (f, h), b in betas.items()])
CHC = pd.DataFrame([dict(fold=f, h=h, T=t, lam=l) for (f, h), (t, l) in p5c_choice.items()])
print("\n=== P5b betas"); print(BT.round(4).to_string(index=False))
print("\n=== P5c choices"); print(CHC.to_string(index=False))
print("\nmodes B5:", {f: M5[f].mode for f in M5}, "\nmodes P5a:", {f: MA[f].mode for f in MA})

hdr = (f"# CONFIG: commit={COMMIT} | Phase 5 | base B5 = {{FA-ETS-d, seasonal w7, macro}}, skill reference FA-ETS-d, R1c "
       f"protocol (fix2 13 weeks, F0.10, adaptive Router, T=0.05) | TEST unless stated | CLEANBLEED primary | "
       f"stats as Phase 4 (NW lag h-1, block L=13, B=5000)\n")
for name, df in (("cell_losses", D), ("validation", VAL), ("promotion", PR), ("stats", ST), ("weights", WT.reset_index()),
                 ("p5b_betas", BT), ("p5c_choices", CHC), ("per_fold", PF.reset_index())):
    with open(os.path.join(OUT, f"phase5_{name}.csv"), "w", newline="") as fh:
        fh.write(hdr)
        df.to_csv(fh, index=False)
