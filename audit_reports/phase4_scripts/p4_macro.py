"""
Phase 4: macro panel regression re-check (pre-registered in phase4_log.md). Run in the facts-audit env:
  C:\\Anaconda3\\envs\\facts-audit\\python.exe audit_reports/phase4_scripts/p4_macro.py

  orig   validate_macro_panel.py spec: 13wk inventory % change on lagged levels, MSA FE + year-quarter
         dummies, SE clustered by MSA (expect FFR p = 0.042)
  (a)    MSA FE only, Driscoll-Kraay (kernel, Bartlett, bandwidth 13)
  (b)    orig spec, wild cluster restricted bootstrap, Webb six-point, 15 clusters, B = 9,999
  (c)    13-week changes of each macro variable (same lags), under orig and under (a)
  (d)    FFR distributed lag 0..26 weeks (levels), MSA FE, Driscoll-Kraay; sums over lags 0-12 and 13-26
  (e)    Holm across the 4 variables within each spec
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

sys.path.insert(0, os.path.join(os.getcwd(), "tests_audit"))
import audit_core as ac  # noqa: E402

LAGS = {"mortgage_rate": 6, "fed_funds": 10, "cpi": 10, "unemployment": 5}
H = 13
HELD = pd.Timestamp("2026-01-01")
B_WILD, SEED = 9999, 20261016
COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def build(df, mode="level", dl=False):
    parts = []
    for msa, g in df.sort_values(["msa", "date"]).groupby("msa"):
        g = g.reset_index(drop=True)
        g["y"] = (g["inventory_count"].shift(-H) - g["inventory_count"]) / g["inventory_count"]
        for f, lag in LAGS.items():
            base = g[f] if mode == "level" else g[f].diff(13)
            g[f"{f}_x"] = base.shift(lag)
        if dl:
            for l in range(27):
                g[f"ff_l{l}"] = g["fed_funds"].shift(l)
        parts.append(g)
    p = pd.concat(parts, ignore_index=True)
    p["target_date"] = p["date"] + pd.Timedelta(weeks=H)
    p = p[(p["date"] < HELD) & (p["target_date"] < HELD)]
    cols = [f"{f}_x" for f in LAGS] + ([f"ff_l{l}" for l in range(27)] if dl else [])
    p = p.dropna(subset=["y"] + cols).copy()
    p["yq"] = p["date"].dt.to_period("Q").astype(str)
    return p


def holm(pvals):
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def panel_fit(p, xcols, yq, cov):
    X = p[xcols].copy()
    if yq:
        X = pd.concat([X, pd.get_dummies(p["yq"], drop_first=True, dtype=float)], axis=1)
    idx = pd.MultiIndex.from_frame(p[["msa", "date"]])
    X.index = idx
    y = pd.Series(p["y"].to_numpy(), index=idx)
    mod = PanelOLS(y, X, entity_effects=True, drop_absorbed=True)
    if cov == "cluster":
        return mod.fit(cov_type="clustered", cluster_entity=True)
    return mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=13)


def wild_cluster(p, xcols):
    """Restricted wild cluster bootstrap (Webb six-point) for each of xcols, MSA + year-quarter dummies."""
    D = pd.concat([p[xcols], pd.get_dummies(p["yq"], drop_first=True, dtype=float),
                   pd.get_dummies(p["msa"], drop_first=True, dtype=float)], axis=1)
    D.insert(0, "const", 1.0)
    X = D.to_numpy(float)
    y = p["y"].to_numpy(float)
    g = pd.factorize(p["msa"])[0]
    G, n, K = g.max() + 1, len(y), X.shape[1]
    c = G / (G - 1) * (n - 1) / (n - K)
    webb = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
    rng = np.random.default_rng(SEED)
    out = {}
    XtXi = np.linalg.pinv(X.T @ X)
    for name in xcols:
        j = list(D.columns).index(name)
        beta = XtXi @ X.T @ y
        u = y - X @ beta
        z = X @ XtXi[j]
        se = np.sqrt(c * sum((z[g == k] @ u[g == k]) ** 2 for k in range(G)))
        t0 = beta[j] / se
        Xr = np.delete(X, j, axis=1)
        br = np.linalg.lstsq(Xr, y, rcond=None)[0]
        yr, ur = Xr @ br, y - Xr @ br
        ts = []
        for start in range(0, B_WILD, 1000):
            b = min(1000, B_WILD - start)
            v = webb[rng.integers(0, 6, size=(G, b))]          # G x b
            ystar = yr[:, None] + ur[:, None] * v[g]           # n x b
            bstar = XtXi @ (X.T @ ystar)                       # K x b
            ustar = ystar - X @ bstar
            s = np.stack([z[g == k] @ ustar[g == k] for k in range(G)])   # G x b
            ts.append(bstar[j] / np.sqrt(c * (s ** 2).sum(0)))
        ts = np.concatenate(ts)
        out[name] = (float(beta[j]), float(t0), float((np.abs(ts) >= abs(t0)).mean()))
    return out


df = ac.load_panel()
rows = []
xc = [f"{f}_x" for f in LAGS]
specs = [("orig: levels, MSA+YQ FE, cluster", "level", True, "cluster"),
         ("(a) levels, MSA FE, Driscoll-Kraay", "level", False, "dk"),
         ("(c1) 13wk changes, MSA+YQ FE, cluster", "diff", True, "cluster"),
         ("(c2) 13wk changes, MSA FE, Driscoll-Kraay", "diff", False, "dk")]
for lab, mode, yq, cov in specs:
    p = build(df, mode)
    r = panel_fit(p, xc, yq, cov)
    pv = [float(r.pvalues[c]) for c in xc]
    hp = holm(pv)
    for c, pp, hh in zip(xc, pv, hp):
        rows.append(dict(spec=lab, variable=c.replace("_x", ""), coef=float(r.params[c]), se=float(r.std_errors[c]),
                         t=float(r.tstats[c]), p=pp, p_holm=hh, n=int(r.nobs)))
    print(f"{lab}: n={r.nobs}", {c.replace('_x', ''): round(v, 4) for c, v in zip(xc, pv)})
    if lab.startswith("orig"):
        wb = wild_cluster(p, xc)
        wp = [wb[c][2] for c in xc]
        for c, (bb, tt, pp), hh in zip(xc, [wb[c] for c in xc], holm(wp)):
            rows.append(dict(spec="(b) orig spec, wild cluster bootstrap (Webb, B=9999)", variable=c.replace("_x", ""),
                             coef=bb, se=np.nan, t=tt, p=pp, p_holm=hh, n=len(p)))
        print("(b) wild cluster p:", {c.replace('_x', ''): round(v, 4) for c, v in zip(xc, wp)})

R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(R.round(5).to_string(index=False))

# (d) distributed lag
p = build(df, "level", dl=True)
lc = [f"ff_l{l}" for l in range(27)]
r = panel_fit(p, lc, False, "dk")
cov = r.cov.loc[lc, lc].to_numpy()
bvec = r.params[lc].to_numpy()
dl_rows = [dict(lag=l, coef=bvec[l], se=float(np.sqrt(cov[l, l])), t=float(bvec[l] / np.sqrt(cov[l, l]))) for l in range(27)]
sums = []
for lab, rng_ in (("lags 0-12", range(0, 13)), ("lags 13-26", range(13, 27))):
    w = np.zeros(27)
    w[list(rng_)] = 1
    s, se = float(w @ bvec), float(np.sqrt(w @ cov @ w))
    sums.append(dict(lag=lab, coef=s, se=se, t=s / se))
DL = pd.DataFrame(dl_rows + sums)
print("\n(d) FFR distributed lag 0..26, MSA FE, Driscoll-Kraay (n=%d)" % r.nobs)
print(DL.round(5).to_string(index=False))

hdr = (f"# CONFIG: commit={COMMIT} | env=facts-audit (linearmodels 7.0) | panel capped 2026-02-21; sample dates and "
       f"13wk targets < 2026-01-01 | y = 13-week inventory % change | lags mortgage 6, FFR 10, CPI 10, unemployment 5\n")
out = os.path.join("audit_reports", "results_tables")
with open(os.path.join(out, "phase4_macro_specs.csv"), "w", newline="") as fh:
    fh.write(hdr)
    R.to_csv(fh, index=False)
with open(os.path.join(out, "phase4_macro_distributed_lag.csv"), "w", newline="") as fh:
    fh.write(hdr)
    DL.to_csv(fh, index=False)
