"""
Phase 1 main analysis. Uses the fast mirror (tests_audit/audit_core.py,
pinned to production in tests_audit/test_regression_pin.py) and the variant
outputs from p1_build_variants.py.

Usage (repo root):  python audit_reports/phase1_scripts/p1_analysis.py <variants.pkl>
Writes CSVs to audit_reports/results_tables/phase1_*.csv (each with a config header).

Labels:
  PROVISIONAL  current-cache system (seasonal window tuned on fold 6 test)
  DISCLOSURE   run that scores fold-6 targets in Jan-Feb 2026 (guard bypassed on purpose)
  CLEAN-BLEED  same, with the 165 bleed cells excluded (guard enforced)
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

COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
OUT = os.path.join("audit_reports", "results_tables")
pd.set_option("display.width", 220)

panel = ac.load_panel()
cache = AgentCache.load("agent_output_cache.pkl")
cells = ac.cells_from_cache(cache, panel)
VAR = pickle.load(open(sys.argv[1], "rb"))
key = lambda c: (c.msa, c.origin, c.h)


def vfn(seasonal="prod", event="prod", intrinsic="prod", per_fold_window=None):
    """values_fn factory. seasonal: 'prod' | 'w1'|'w3'|'w5'|'w7'; event: 'prod'|'july';
    intrinsic: 'prod'|'naive'; per_fold_window: {fold: w} overrides seasonal per fold."""
    def f(c, fold):
        v = dict(c.values)
        s = f"w{per_fold_window[fold]}" if per_fold_window else seasonal
        if s != "prod":
            v["seasonality"] = VAR[key(c)][f"seasonal_{s}"]
        if event == "july":
            v["event"] = VAR[key(c)]["event_july"]
        if intrinsic == "naive":
            v["intrinsic"] = c.naive
        return v
    return f


def cfg(**kw):
    base = dict(commit=COMMIT, split="TEST", folds="1-6", msas=15, horizons="4,8,13",
                zillow="smoothed", fusion="adaptive T=0.05 104wk", rho_max=0.30)
    base.update(kw)
    return " | ".join(f"{k}={v}" for k, v in base.items())


def write(name, df, config):
    p = os.path.join(OUT, f"phase1_{name}.csv")
    with open(p, "w", newline="") as fh:
        fh.write(f"# CONFIG: {config}\n")
        df.to_csv(fh, index=False)
    return p


def run(values_fn=ac.production_values, bound=False, exclude=None, scheme="adaptive"):
    m = ac.build_models(cells, values_fn, bound_targets=bound, exclude=exclude)
    with_b = ac.score(cells, m, values_fn, scheme, exclude_bleed=False, disclosure=True)
    no_b = ac.score(cells, m, values_fn, scheme, exclude_bleed=True)
    return m, with_b, no_b


def headline(with_b, no_b, label):
    a = with_b.groupby("h")[["naive", "fused"]].mean().add_suffix("_DISCLOSURE")
    b = no_b.groupby("h")[["naive", "fused"]].mean().add_suffix("_CLEANBLEED")
    r = pd.concat([a, b], axis=1).reset_index()
    r.insert(0, "config", label)
    return r


def per_fold(d, label):
    t = d.groupby(["h", "fold"])[["naive", "fused"]].mean().reset_index()
    t.insert(0, "config", label)
    return t


# ── A. PROVISIONAL baseline ───────────────────────────────────────────────
print("\n=== A. PROVISIONAL baseline (current cache), with / without 165 bleed cells")
M0, B0w, B0n = run()
H = [headline(B0w, B0n, "PROVISIONAL current cache")]
print(H[0].round(4).to_string(index=False))
f6 = pd.concat([B0w[B0w.fold == 6].groupby("h")[["naive", "fused"]].mean().add_suffix("_fold6_DISCL"),
                B0n[B0n.fold == 6].groupby("h")[["naive", "fused"]].mean().add_suffix("_fold6_CLEAN"),
                B0w[B0w.fold == 6].groupby("h").size().rename("n_fold6_all"),
                B0n[B0n.fold == 6].groupby("h").size().rename("n_fold6_clean")], axis=1)
print(f6.round(4).to_string())
write("A_fold6_bleed", f6.reset_index(), cfg(label="PROVISIONAL", note="fold 6 only"))
PF = [per_fold(B0w, "PROVISIONAL DISCLOSURE"), per_fold(B0n, "PROVISIONAL CLEANBLEED")]

# ── B. F0.10: bound validation targets to train_end ──────────────────────
print("\n=== B. F0.10 bounded validation targets")
M1, B1w, B1n = run(bound=True)
H.append(headline(B1w, B1n, "PROVISIONAL + F0.10 bound"))
rows = []
for f in FOLDS:
    a, b = M0[f.fold], M1[f.fold]
    for ag in ac.AGENTS:
        rows.append(dict(fold=f.fold, set="fold", regime="", h="", agent=ag,
                         w_prod=a.fold_weights[ag], w_bound=b.fold_weights[ag]))
        for r in ac.REGIMES:
            for h in HORIZONS:
                rows.append(dict(fold=f.fold, set="regime", regime=r, h=h, agent=ag,
                                 w_prod=a.regime_grid[r][h]["weights"][ag],
                                 w_bound=b.regime_grid[r][h]["weights"][ag],
                                 n_prod=a.regime_grid[r][h]["n"], n_bound=b.regime_grid[r][h]["n"]))
W = pd.DataFrame(rows)
W["abs_change"] = (W.w_bound - W.w_prod).abs()
print("max |weight change| fold-level per fold:", W[W.set == "fold"].groupby("fold").abs_change.max().round(4).to_dict())
print("max |weight change| regime grid (cells with n>0 both):",
      W[(W.set == "regime") & (W.n_prod > 0) & (W.n_bound > 0)].groupby("fold").abs_change.max().round(4).to_dict())
flips = [(f.fold, h, M0[f.fold].mode[h], M1[f.fold].mode[h]) for f in FOLDS for h in HORIZONS
         if M0[f.fold].mode[h] != M1[f.fold].mode[h]]
print("mode flips:", flips)
print(H[-1].round(4).to_string(index=False))
write("B_F010_weights", W, cfg(label="PROVISIONAL vs PROVISIONAL+F0.10", split="VALIDATION weights"))
PF += [per_fold(B1w, "F0.10 DISCLOSURE"), per_fold(B1n, "F0.10 CLEANBLEED")]

# ── C. Imputed values ─────────────────────────────────────────────────────
print("\n=== C. Imputed values")
fl = pd.read_csv("filled_rows_log.csv", parse_dates=["date"])
filled = set(zip(fl.msa, fl.date))
touch = lambda c: any((c.msa, d) in filled for d in c.target_dates) or (c.msa, pd.Timestamp(c.origin)) in filled
rows = []
for f in FOLDS:
    val = [c for c in cells if ac._in_val(c, f, False)]
    t = [c for c in val if touch(c)]
    z = [c for c in val if (ac._mape_frac(c.actuals, c.naive) or 0) <= 0]
    rows.append(dict(fold=f.fold, val_cells=len(val), touching_imputed=len(t),
                     naive_mape_zero_skipped=len(z),
                     naive_zero_and_imputed=len([c for c in z if touch(c)])))
IMP = pd.DataFrame(rows)
print(IMP.to_string(index=False))
M2, B2w, B2n = run(exclude=touch)
dw = max(abs(M2[f].fold_weights[a] - M0[f].fold_weights[a]) for f in M0 for a in ac.AGENTS)
print("max |fold weight change| excluding imputed-touching validation cells:", round(dw, 5),
      "| mode flips:", [(f, h) for f in M0 for h in HORIZONS if M0[f].mode[h] != M2[f].mode[h]])
n_test_touch = len([c for c in ac.test_cells(cells) if touch(c)])
print("test cells touching imputed:", n_test_touch)
H.append(headline(B2w, B2n, "PROVISIONAL, imputed-touching validation cells excluded"))
write("C_imputed", IMP, cfg(split="VALIDATION", note=f"test cells touching imputed = {n_test_touch}"))

# ── D. Version decomposition (disclosure) ────────────────────────────────
print("\n=== D. Version decomposition July -> +smoothing -> +event removal")
versions = {"V0 July agents (no smoothing, event x0.10)": vfn("w1", "july"),
            "V1 + smoothing w5 (event x0.10)": vfn("w5", "july"),
            "V1b + event removal only (no smoothing)": vfn("w1", "prod"),
            "V2 + both = current": vfn("w5", "prod")}
for lab, fn in versions.items():
    _, w_, n_ = run(fn)
    H.append(headline(w_, n_, lab))
    PF += [per_fold(w_, lab + " DISCLOSURE")]
    print(H[-1].round(4).to_string(index=False, header=False))

# ── E. Nested window re-selection ─────────────────────────────────────────
print("\n=== E. Nested seasonal window re-selection (validation only, targets bounded)")
rows, chosen = [], {}
for f in FOLDS:
    val = [c for c in cells if ac._in_val(c, f, True)]
    rec = {"fold": f.fold, "n_val_cells": len(val)}
    for w in (1, 3, 5, 7):
        rec[f"val_mape_w{w}"] = float(np.mean([ac.mape_pct(c.actuals, VAR[key(c)][f"seasonal_w{w}"]) for c in val]))
    best = min((1, 3, 5, 7), key=lambda w: (round(rec[f"val_mape_w{w}"], 12), w))
    rec["chosen"] = chosen[f.fold] = best
    rows.append(rec)
NW = pd.DataFrame(rows)
print(NW.round(4).to_string(index=False))
write("E_nested_window_selection", NW, cfg(split="VALIDATION (targets<=train_end)", criterion="standalone seasonal MAPE, 4/8/13 pooled"))
fn = vfn(per_fold_window=chosen)
_, R1w, R1n = run(fn)
H.append(headline(R1w, R1n, f"R1 nested window {chosen}"))
_, R2w, R2n = run(fn, bound=True)
H.append(headline(R2w, R2n, f"R2 nested window + F0.10 bound"))
PF += [per_fold(R1w, "R1 DISCLOSURE"), per_fold(R1n, "R1 CLEANBLEED"),
       per_fold(R2w, "R2 DISCLOSURE"), per_fold(R2n, "R2 CLEANBLEED")]
for d, lab in ((R1n, "R1"), (R2n, "R2")):
    print(lab, "per-MSA wins vs PROVISIONAL (CLEANBLEED):",
          {h: int((d[d.h == h].groupby("msa").fused.mean() < B0n[B0n.h == h].groupby("msa").fused.mean()).sum()) for h in HORIZONS})
print(pd.concat(H[-2:]).round(4).to_string(index=False))

# ── F. Original seasonal sweep rebuild (fold 6 test, DISCLOSURE) ─────────
print("\n=== F. Rebuild of the original smoothing sweep (fold 6 test window)")
f6c = [c for c in ac.test_cells(cells) if c.test_fold == 6]
ac.assert_scoring_allowed([d for c in f6c for d in c.target_dates], disclosure=True)
rows = []
for w in (1, 3, 5, 7):
    for bleed_ok in (True, False):
        sel = [c for c in f6c if bleed_ok or c.last_target < ac.RESERVED_START]
        r = {"def": "a standalone seasonal", "window": w, "bleed_cells": "included" if bleed_ok else "excluded"}
        for h in HORIZONS:
            r[f"h{h}"] = float(np.mean([ac.mape_pct(c.actuals, VAR[key(c)][f"seasonal_w{w}"]) for c in sel if c.h == h]))
        rows.append(r)
SW = pd.DataFrame(rows)
target = {1: (1.373, 2.195, 3.052), 3: (1.223, 2.090, 2.979), 5: (1.183, 2.062, 2.980), 7: (1.213, 2.107, 3.069)}
match_a = all(np.allclose(SW[(SW.window == w) & (SW.bleed_cells == "included")][["h4", "h8", "h13"]].values, target[w], atol=5e-4) for w in target)
print(SW.round(3).to_string(index=False))
print("definition (a) matches original table:", match_a)
if not match_a:
    for lab, scheme in (("b fused adaptive", "adaptive"), ("c fused fold-level", "fold")):
        for w in (1, 3, 5, 7):
            fnw = vfn(f"w{w}", "july")
            m = ac.build_models(cells, fnw)
            d = ac.score(cells, m, fnw, scheme, disclosure=True)
            d6 = d[d.fold == 6]
            r = {"def": lab, "window": w, "bleed_cells": "included"}
            r.update({f"h{h}": d6[d6.h == h].fused.mean() for h in HORIZONS})
            rows.append(r)
        SW = pd.DataFrame(rows)
        sub = SW[SW["def"] == lab]
        ok = all(np.allclose(sub[sub.window == w][["h4", "h8", "h13"]].values, target[w], atol=5e-4) for w in target)
        print(sub.round(3).to_string(index=False))
        print(f"definition ({lab}) matches:", ok)
        if ok:
            break
write("F_seasonal_sweep_rebuild", SW, cfg(label="DISCLOSURE", split="TEST fold 6 only", note="rebuild of f56fe8a sweep"))

# ── G. COVID declarations ────────────────────────────────────────────────
print("\n=== G. COVID (Biological) declarations")
ev = pd.read_csv("event_agent_weekly.csv", parse_dates=["week"])
first = ev.groupby(["msa", "week"], sort=False).first().reset_index()
itype = {(r.msa, r.week.date()): r.incident_type for r in first.itertuples()}
diff = [c for c in cells if not np.allclose(VAR[key(c)]["event_july"], c.naive)]
g = pd.DataFrame([dict(msa=c.msa, origin=c.origin, h=c.h, type=itype.get((c.msa, c.origin)),
                       test=c.test_fold is not None) for c in diff])
print("cells where July event != naive:", len(g), "| by type:", g.groupby("type").size().to_dict())
cov = g[g.type == "Biological"]
print("COVID-driven cells:", len(cov), "| origins", cov.origin.min() if len(cov) else None, "..", cov.origin.max() if len(cov) else None,
      "| in test windows:", int(cov.test.sum()) if len(cov) else 0)
print("current event agent == naive in all cells:",
      all(np.allclose(c.values["event"], c.naive) for c in cells))
write("G_covid_event_cells", g, cfg(split="ALL cached cells", note="July event agent vs naive"))

# ── H. Static vintage bound ───────────────────────────────────────────────
print("\n=== H. Intrinsic channel replaced by naive (same pipeline)")
_, Hw, Hn = run(vfn(intrinsic="naive"))
H.append(headline(Hw, Hn, "PROVISIONAL, intrinsic output := naive"))
print(H[-1].round(4).to_string(index=False))

# ── I. Cells fused with weights estimated from zero validation points ────
print("\n=== I. Test cells using regime weights with n=0 validation points")
rows = []
for c in ac.test_cells(cells):
    m = M0[c.test_fold]
    if m.mode[c.h] == "regime" and m.regime_grid[c.regime][c.h]["n"] == 0:
        rows.append(dict(fold=c.test_fold, h=c.h, regime=c.regime, bleed=bool(c.last_target >= ac.RESERVED_START),
                         fused_prod=ac.mape_pct(c.actuals, ac.fuse(c.values, ac.weights_for(c, M0), c.h)),
                         fused_foldw=ac.mape_pct(c.actuals, ac.fuse(c.values, M0[c.test_fold].fold_weights, c.h)),
                         naive=ac.mape_pct(c.actuals, c.naive)))
Z = pd.DataFrame(rows)
print(Z.groupby(["fold", "h", "regime"])[["fused_prod", "fused_foldw", "naive"]].agg(["mean", "size"]).round(3).to_string())
write("I_zero_support_cells", Z.groupby(["fold", "h", "regime"])[["fused_prod", "fused_foldw", "naive"]].mean().reset_index(),
      cfg(label="PROVISIONAL DISCLOSURE", note="descriptive only; not a selection"))

# ── Save summary tables ───────────────────────────────────────────────────
HH = pd.concat(H, ignore_index=True)
print("\n=== HEADLINE SUMMARY (MAPE; DISCLOSURE includes 165 bleed cells, CLEANBLEED excludes them)")
print(HH.round(4).to_string(index=False))
write("headlines", HH, cfg(label="see config column; all PROVISIONAL except where stated"))
write("per_fold", pd.concat(PF, ignore_index=True), cfg(label="see config column"))
