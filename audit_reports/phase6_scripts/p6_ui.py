"""
Phase 6d: unemployment-claims domain under the housing protocol (pre-registered in audit_reports/ui_protocol.md).

Usage (repo root): python audit_reports/phase6_scripts/p6_ui.py
"""
import datetime
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

REPO = os.getcwd()
for p in ("tests_audit", os.path.join("audit_reports", "phase3_scripts"), os.path.join("audit_reports", "phase4_scripts")):
    sys.path.insert(0, os.path.join(REPO, p))
sys.path.insert(0, REPO)
import audit_core as ac  # noqa: E402
import stats_lib as sl  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from facts_mas.run_backtest import FOLDS, classify_regime  # noqa: E402

UI = os.path.join("domains", "ui_claims", "aligned_weekly_ui_claims.csv")
COVID = (datetime.date(2020, 3, 14), datetime.date(2021, 9, 4))
KEEP = ["ar", "seasonality", "macro"]
R1C = dict(min_support_weeks=13, bound_targets=True)
H_MAX = 13
COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def load_ui():
    d = pd.read_csv(UI, parse_dates=["date"])
    d = d[d["date"] <= ac.DATA_CAP].reset_index(drop=True)
    assert d["date"].max() <= ac.DATA_CAP
    return d


def as_housing(d):
    return d.rename(columns={"state": "msa", "claims_count": "inventory_count"})


def worker(args):
    state, origins = args
    import p3_build_baselines as pb
    from ar_agent import fit_and_forecast
    from domains.ui_claims.agents.macro_agent_ui import run_macro_agent_ui
    from facts_mas.agents import seasonality_agent as sa
    from facts_mas.calibration import override_constant
    d = load_ui()
    ph = as_housing(d)
    md = ph[ph.msa == state].sort_values("date").reset_index(drop=True)
    dates = list(md["date"])
    y = md["inventory_count"].to_numpy(float)
    out = {}
    for o in origins:
        i = dates.index(pd.Timestamp(o))
        hist = md[md["date"] <= pd.Timestamp(o)].set_index("date")["inventory_count"]
        with override_constant(sa, "SEASONAL_SMOOTHING_WINDOW", 7):
            f = sa.compute_seasonal_factors(ph, state, o)
        fa = pb.forecasts_at(ph, state, o)
        out[o] = {"holt": np.maximum(np.array(fit_and_forecast(hist, H_MAX)["point_forecast"]), 0),
                  "seasonality": np.array(sa.run_seasonality_agent(ph, state, o, H_MAX, seasonal_factors=f).values),
                  "macro": np.array(run_macro_agent_ui(d, state, o, H_MAX)["values"]),
                  "fa_ets": fa["fa_ets"], "fa_ets_d": fa["fa_ets_d"],
                  "snaive": np.array([y[i + k - 52] for k in range(1, H_MAX + 1)]), "naive": np.full(H_MAX, y[i])}
    return state, out


def in_covid(c):
    if COVID[0] <= c.origin <= COVID[1]:
        return True
    return any(COVID[0] <= t.date() <= COVID[1] for t in c.target_dates)


def main():
    d = load_ui()
    ph = as_housing(d)
    print("UI panel sha256", hashlib.sha256(open(UI, "rb").read()).hexdigest(), "| rows <= cap", len(d))
    states = sorted(ph.msa.unique())
    by = {}
    for s in states:
        dd = ph[ph.msa == s].sort_values("date")["date"]
        keep = set()
        for f in FOLDS:
            vs = f.train_end - datetime.timedelta(weeks=104)
            keep |= {x.date() for x in dd if vs <= x.date() <= f.train_end or f.test_start <= x.date() <= f.test_end}
        by[s] = sorted(keep)
    fc = {}
    with ProcessPoolExecutor(max_workers=14) as ex:
        for s, out in ex.map(worker, list(by.items())):
            fc[s] = out
    cells = []
    for s in states:
        md = ph[ph.msa == s].sort_values("date").reset_index(drop=True)
        dates, y = list(md["date"]), md["inventory_count"].to_numpy(float)
        for o in by[s]:
            i = dates.index(pd.Timestamp(o))
            tf = next((f.fold for f in FOLDS if f.test_start <= o <= f.test_end), None)
            reg = classify_regime(ph, s, o)
            for h in (4, 8, 13):
                if i + h >= len(dates):
                    continue
                tds = tuple(dates[i + 1:i + 1 + h])
                if tf is not None and tds[-1] >= ac.RESERVED_START:
                    continue  # reserved-window targets are never built, so never scored
                v = {k: x[:h].copy() for k, x in fc[s][o].items()}
                cells.append(ac.Cell(s, o, h, reg, y[i + 1:i + 1 + h], v["naive"], tds[-1], tds, v, tf))

    results, weights, pf_rows, stat_rows, counts = [], [], [], [], []
    for label, excl in (("PRIMARY covid-excluded", in_covid), ("SECONDARY covid-included", None)):
        choice = {}
        for f in FOLDS:
            val = [c for c in cells if ac._in_val(c, f, True) and not (excl and excl(c))]
            mu = np.mean([ac.mape_pct(c.actuals, c.values["fa_ets"]) for c in val])
            md_ = np.mean([ac.mape_pct(c.actuals, c.values["fa_ets_d"]) for c in val])
            choice[f.fold] = "fa_ets_d" if md_ < mu else "fa_ets"
        v4 = lambda c, f: {"ar": c.values["holt"], "seasonality": c.values["seasonality"], "macro": c.values["macro"]}
        v8 = lambda c, f: {"ar": c.values[choice[f]], "seasonality": c.values["seasonality"], "macro": c.values["macro"]}
        ref = lambda c, f: c.values["fa_ets_d"]
        kw = dict(agents=KEEP, exclude=excl, **R1C)
        M4 = {f.fold: ac.build_fold_model(cells, f, v4, **kw) for f in FOLDS}
        M8 = {f.fold: ac.build_fold_model(cells, f, v8, **kw) for f in FOLDS}
        MB = {f.fold: ac.build_fold_model(cells, f, v8, reference_fn=ref, **kw) for f in FOLDS}
        rows = []
        for c in cells:
            if c.test_fold is None or (excl and excl(c)):
                continue
            f = c.test_fold
            wP = ac.weights_for(c, M4) if c.h == 4 else ac.weights_for(c, M8)
            fcs = {"P_UI": ac.fuse(v4(c, f) if c.h == 4 else v8(c, f), wP, c.h),
                   "B5_UI": ac.fuse(v8(c, f), ac.weights_for(c, MB), c.h),
                   "R1c_UI": ac.fuse(v4(c, f), ac.weights_for(c, M4), c.h),
                   "EW_TS": ac.fuse(v4(c, f), {"ar": 0.5, "seasonality": 0.5}, c.h),
                   "fa_ets_d": c.values["fa_ets_d"], "holt": c.values["holt"], "seasonal": c.values["seasonality"],
                   "macro": c.values["macro"], "snaive": c.values["snaive"], "naive": c.naive}
            rows.append(dict(analysis=label, h=c.h, fold=f, msa=c.msa, origin=c.origin,
                             **{k: ac.mape_pct(c.actuals, x) for k, x in fcs.items()}))
            weights.append(dict(analysis=label, h=c.h, system="R1c_UI", **ac.weights_for(c, M4)))
            weights.append(dict(analysis=label, h=c.h, system="P_UI", **wP))
        D = pd.DataFrame(rows)
        results.append(D)
        meth = ["P_UI", "B5_UI", "R1c_UI", "EW_TS", "fa_ets_d", "holt", "seasonal", "macro", "snaive", "naive"]
        pf = D.groupby(["h", "fold"])[meth].mean().reset_index()
        pf.insert(0, "analysis", label)
        pf_rows.append(pf)
        counts.append(D.groupby(["h", "fold"]).size().rename("n_cells").reset_index().assign(analysis=label))
        for h in (4, 8, 13):
            sub = D[D.h == h]
            W = sl.weekly(sub, meth)
            idx = sl.block_indices(len(W), sl.SEED + h)
            for a, b in (("P_UI", "naive"), ("fa_ets_d", "naive"), ("seasonal", "naive"), ("holt", "naive"),
                         ("P_UI", "fa_ets_d"), ("B5_UI", "fa_ets_d"), ("R1c_UI", "fa_ets_d"), ("R1c_UI", "naive")):
                r = sl.pair_stats(sub, W, idx, a, b, h)
                r["analysis"] = label
                stat_rows.append(r)
        print(f"\n=== {label}: damped choice {choice}")
        print(D.groupby("h")[meth].mean().round(3).to_string())
    R = pd.concat(results)
    WT = pd.DataFrame(weights).groupby(["analysis", "system", "h"])[KEEP].mean()
    ST = pd.DataFrame(stat_rows)
    PF = pd.concat(pf_rows)
    CN = pd.concat(counts)
    pd.set_option("display.width", 250)
    print("\n=== mean weights (handoff check)"); print(WT.round(3).to_string())
    print("\n=== paired statistics"); print(ST[["analysis", "h", "A", "B", "mean_diff", "panel_dm_p", "ci95_lo", "ci95_hi",
                                              "BH_A_better", "BH_B_better"]].round(4).to_string(index=False))
    print("\n=== cells per fold"); print(CN.pivot_table(index=["analysis", "fold"], columns="h", values="n_cells").to_string())
    hdr = (f"# CONFIG: commit={COMMIT} | UI claims domain (ui_protocol.md) | data capped 2026-02-21; reserved-target test "
           f"cells excluded | COVID window {COVID[0]}..{COVID[1]} | housing folds | frozen recipes, no re-tuning | "
           f"stats as Phase 4\n")
    for name, df in (("cell_losses", R), ("weights", WT.reset_index()), ("stats", ST), ("per_fold", PF), ("cell_counts", CN)):
        with open(os.path.join("audit_reports", "results_tables", f"phase6_ui_{name}.csv"), "w", newline="") as fh:
            fh.write(hdr)
            df.to_csv(fh, index=False)


if __name__ == "__main__":
    main()
