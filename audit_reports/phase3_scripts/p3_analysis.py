"""
Phase 3 analysis: baselines, candidates C1-C3, promotion rule, main question.

Usage (repo root):  python audit_reports/phase3_scripts/p3_analysis.py <baselines.pkl>
Writes audit_reports/results_tables/phase3_*.csv. All selections on validation (nested, targets <= train_end).
"""
import os
import pickle
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
R1C = dict(min_support_weeks=13, bound_targets=True)

cells = ac.cells_from_cache(AgentCache.load("agent_output_cache.pkl"), ac.load_panel())
W7 = ac.load_r1_seasonal()
B = pickle.load(open(sys.argv[1], "rb"))
key = lambda c: (c.msa, c.origin, c.h)
R1 = ac.r1_values_fn(W7)
FOLD = {f.fold: f for f in FOLDS}


def cfg(**kw):
    base = dict(commit=COMMIT, protocol="R1c (w7, fix2 13 weeks, F0.10)", split="TEST", folds="1-6", msas=15,
                horizons="4,8,13", zillow="smoothed", T="0.05 PROVISIONAL", primary="CLEANBLEED",
                note_unconfirmed="improvements UNCONFIRMED until Phase 4")
    base.update(kw)
    return " | ".join(f"{k}={v}" for k, v in base.items())


def write(name, df, config):
    with open(os.path.join(OUT, f"phase3_{name}.csv"), "w", newline="") as fh:
        fh.write(f"# CONFIG: {config}\n")
        df.to_csv(fh, index=False)


# ── standalone forecasts per cell ─────────────────────────────────────────
STAND = ["naive", "snaive", "naive2", "holt", "holt_d", "fa_ets", "fa_ets_d", "theta", "macro", "intrinsic"]


def stand(c, name):
    if name == "naive":
        return c.naive
    if name == "holt":
        return np.maximum(c.values["ar"], 0)
    if name in ("macro", "intrinsic"):
        return np.maximum(c.values[name], 0)
    return B[key(c)][name]


val_cells = {f.fold: [c for c in cells if ac._in_val(c, f, True)] for f in FOLDS}


def val_mape_standalone(name, fold, h=None):
    v = [ac.mape_pct(c.actuals, stand(c, name)) for c in val_cells[fold] if h is None or c.h == h]
    return float(np.mean(v))


print("Naive2 == R1 seasonal channel:", max(float(np.max(np.abs(B[key(c)]["naive2"] - W7[key(c)]))) for c in cells))

# ── candidate temporal choices (damped vs undamped), nested by fold ──────
choice = {}
rows = []
for f in FOLDS:
    for cand, (u, d) in (("C1", ("holt", "holt_d")), ("C2", ("fa_ets", "fa_ets_d"))):
        mu, md = val_mape_standalone(u, f.fold), val_mape_standalone(d, f.fold)
        choice[(cand, f.fold)] = d if md < mu else u
        rows.append(dict(candidate=cand, fold=f.fold, val_undamped=mu, val_damped=md, chosen=choice[(cand, f.fold)]))
CH = pd.DataFrame(rows)
print("\n=== damped vs undamped choice (validation, standalone, h pooled)")
print(CH.round(4).to_string(index=False))
write("damped_choice", CH, cfg(split="VALIDATION"))


def temporal_fn(cand):
    def f(c, fold):
        v = R1(c, fold)
        v = dict(v)
        v["ar"] = stand(c, choice[(cand, fold)])
        return v
    return f


VF = {"R1c": R1, "C1": temporal_fn("C1"), "C2": temporal_fn("C2"), "C3": temporal_fn("C2")}
MODELS = {k: ac.build_models(cells, fn, **R1C) for k, fn in VF.items()}
SCHEME = {"R1c": "adaptive", "C1": "adaptive", "C2": "adaptive", "C3": "horizon"}

# ── selections on validation ──────────────────────────────────────────────
CHANNELS = ["ar", "seasonality", "macro", "event", "intrinsic"]
chan_name = {"ar": "holt", "seasonality": "naive2", "macro": "macro", "event": "naive", "intrinsic": "intrinsic"}
sel_ch, sel_m, rows = {}, {}, []
for f in FOLDS:
    for h in HORIZONS:
        vm = {n: val_mape_standalone(n, f.fold, h) for n in STAND}
        sel_m[(f.fold, h)] = min(vm, key=vm.get)
        vc = {ch: vm[chan_name[ch]] for ch in CHANNELS}
        sel_ch[(f.fold, h)] = min(vc, key=vc.get)
        rows.append(dict(fold=f.fold, h=h, selbest_model=sel_m[(f.fold, h)], selbest_channel=sel_ch[(f.fold, h)],
                         **{f"val_{n}": vm[n] for n in STAND}))
SEL = pd.DataFrame(rows)
print("\n=== validation selections (standalone validation MAPE, per fold x h)")
print(SEL[["fold", "h", "selbest_model", "selbest_channel"] + [f"val_{n}" for n in STAND]].round(3).to_string(index=False))
write("validation_selection", SEL, cfg(split="VALIDATION"))

# ── per-cell test results for every method ────────────────────────────────
tc_all = ac.test_cells(cells)
ac.assert_scoring_allowed([d for c in tc_all for d in c.target_dates], disclosure=True)  # DISCLOSURE rows included, flagged
EW5 = {a: 0.2 for a in ac.AGENTS}
EWTS = {"ar": 0.5, "seasonality": 0.5, "event": 0.0, "intrinsic": 0.0, "macro": 0.0}
rows = []
for c in tc_all:
    f = c.test_fold
    r = dict(h=c.h, fold=f, msa=c.msa, origin=c.origin, regime=c.regime, bleed=bool(c.last_target >= ac.RESERVED_START))
    fc = {n: stand(c, n) for n in STAND}
    fc["SelBest-model"] = fc[sel_m[(f, c.h)]]
    fc["SelBest-channel"] = fc[chan_name[sel_ch[(f, c.h)]]]
    v = R1(c, f)
    fc["EW5"] = ac.fuse(v, EW5, c.h)
    fc["EW-TS"] = ac.fuse(v, EWTS, c.h)
    fc["HZ-weights"] = ac.fuse(v, MODELS["R1c"][f].horizon_weights[c.h], c.h)
    for k in ("R1c", "C1", "C2", "C3"):
        fc[k] = ac.fuse(VF[k](c, f), ac.weights_for(c, MODELS[k], SCHEME[k]), c.h)
    for n, x in fc.items():
        r[f"mape|{n}"] = ac.mape_pct(c.actuals, x)
        r[f"rmse|{n}"] = float(np.sqrt(np.mean((c.actuals - x) ** 2)))
        r[f"mae|{n}"] = float(np.mean(np.abs(c.actuals - x)))
    rows.append(r)
D = pd.DataFrame(rows)
METHODS = [c.split("|")[1] for c in D.columns if c.startswith("mape|")]


def table(Dsub):
    out = []
    for h in HORIZONS:
        d = Dsub[Dsub.h == h]
        per_msa = d.groupby("msa")
        for m in METHODS:
            mae_m = per_msa[f"mae|{m}"].mean()
            rel_n = np.exp(np.mean(np.log(mae_m / per_msa["mae|naive"].mean())))
            rel_s = np.exp(np.mean(np.log(mae_m / per_msa["mae|snaive"].mean())))
            mm = per_msa[f"mape|{m}"].mean()
            rec = dict(h=h, method=m, mape=d[f"mape|{m}"].mean(), rmse=d[f"rmse|{m}"].mean(),
                       avgrelmae_vs_naive=rel_n, avgrelmae_vs_snaive=rel_s,
                       msa_wins_vs_naive=int((mm < per_msa["mape|naive"].mean()).sum()),
                       msa_wins_vs_R1c=int((mm < per_msa["mape|R1c"].mean()).sum()))
            pf = d.groupby("fold")[f"mape|{m}"].mean()
            rec.update({f"fold{k}": v for k, v in pf.items()})
            out.append(rec)
    return pd.DataFrame(out)


TC = table(D[~D.bleed])
TD = table(D)
for h in HORIZONS:
    print(f"\n=== TEST, CLEANBLEED, h={h}")
    print(TC[TC.h == h].sort_values("mape").round(3).to_string(index=False))
write("comparison_CLEANBLEED", TC, cfg())
write("comparison_DISCLOSURE", TD, cfg(label="DISCLOSURE: includes 165 fold-6 reserved-window cells"))

# ORACLE
print("\n=== ORACLE (best on TEST, labelled; not a competitor), CLEANBLEED")
for h in HORIZONS:
    t = TC[TC.h == h].set_index("method").mape
    ch = t[[chan_name[x] for x in CHANNELS]]
    st = t[STAND]
    print(f"  h={h}: best channel {ch.idxmin()} {ch.min():.3f} | best standalone {st.idxmin()} {st.min():.3f}")

# ── candidates: validation, weights, promotion ────────────────────────────
print("\n=== candidates: validation fused MAPE (mean over folds of each fold's own validation), test per fold, promotion")
rows = []
for h in HORIZONS:
    base_val = np.mean([ac.val_fused_mape(cells, FOLD[f], MODELS["R1c"][f], R1, h, "adaptive") for f in FOLD])
    tc = TC[TC.h == h].set_index("method")
    for k in ("C1", "C2", "C3"):
        v = np.mean([ac.val_fused_mape(cells, FOLD[f], MODELS[k][f], VF[k], h, SCHEME[k]) for f in FOLD])
        folds_better = int(sum(tc.loc[k, f"fold{f}"] < tc.loc["R1c", f"fold{f}"] for f in FOLD))
        promote = (v < base_val) and folds_better >= 5
        rows.append(dict(h=h, candidate=k, val_mape=v, val_mape_R1c=base_val, val_win=v < base_val,
                         test_mape=tc.loc[k, "mape"], test_mape_R1c=tc.loc["R1c", "mape"],
                         test_folds_better=folds_better, msa_wins_vs_R1c=int(tc.loc[k, "msa_wins_vs_R1c"]),
                         PROMOTE=promote))
PR = pd.DataFrame(rows)
print(PR.round(4).to_string(index=False))
write("promotion", PR, cfg(rule="val win AND test better in >= 5 of 6 folds (CLEANBLEED)"))

rows = []
for k in ("R1c", "C1", "C2", "C3"):
    for c in ac.test_cells(cells, exclude_bleed=True):
        rows.append(dict(cand=k, h=c.h, **ac.weights_for(c, MODELS[k], SCHEME[k])))
MW = pd.DataFrame(rows).groupby(["cand", "h"])[ac.AGENTS].mean()
print("\n=== mean channel weights over CLEANBLEED test cells ('ar' = temporal slot)")
print(MW.round(3).to_string())
write("candidate_weights", MW.reset_index(), cfg())
print("\nmode selections:", {k: {f: MODELS[k][f].mode for f in FOLD} for k in ("R1c", "C1", "C2")})

# ── main question ─────────────────────────────────────────────────────────
print("\n=== MAIN QUESTION: FACTS-MAS variants vs SelBest-model (validation-selected), CLEANBLEED")
for h in HORIZONS:
    t = TC[TC.h == h].set_index("method")
    sb = t.loc["SelBest-model"]
    for k in ("R1c", "C1", "C2", "C3", "HZ-weights"):
        fb = int(sum(t.loc[k, f"fold{f}"] < sb[f"fold{f}"] for f in FOLD))
        print(f"  h={h:<2} {k:<10} {t.loc[k, 'mape']:.3f} vs SelBest-model {sb['mape']:.3f} -> "
              f"{'BEATS' if t.loc[k, 'mape'] < sb['mape'] else 'does NOT beat'} (folds better {fb}/6)")
