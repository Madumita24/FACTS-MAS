"""
UI addendum A (ui_protocol.md, added after the Phase 6 results; disclosed): validation-selected single model
(SelBest-UI) vs P-UI and B5-UI. Candidates: housing SelBest set minus intrinsic.

Usage (repo root): python audit_reports/phase6_scripts/p6_ui_selbest.py
"""
import datetime
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

REPO = os.getcwd()
for p in ("tests_audit", os.path.join("audit_reports", "phase3_scripts"), os.path.join("audit_reports", "phase4_scripts"),
          os.path.join("audit_reports", "phase6_scripts")):
    sys.path.insert(0, os.path.join(REPO, p))
sys.path.insert(0, REPO)
import audit_core as ac  # noqa: E402
import stats_lib as sl  # noqa: E402
from p6_ui import COVID, as_housing, in_covid, load_ui  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from facts_mas.run_backtest import FOLDS  # noqa: E402

CANDS = ["naive", "snaive", "naive2", "holt", "holt_d", "fa_ets", "fa_ets_d", "theta", "macro"]
COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def worker(args):
    state, origins = args
    import p3_build_baselines as pb
    from ar_agent import fit_and_forecast
    from domains.ui_claims.agents.macro_agent_ui import run_macro_agent_ui
    d = load_ui()
    ph = as_housing(d)
    md = ph[ph.msa == state].sort_values("date").reset_index(drop=True)
    dates = list(md["date"])
    y = md["inventory_count"].to_numpy(float)
    out = {}
    for o in origins:
        i = dates.index(pd.Timestamp(o))
        hist = md[md["date"] <= pd.Timestamp(o)].set_index("date")["inventory_count"]
        fa = pb.forecasts_at(ph, state, o)
        out[o] = {"naive": np.full(13, y[i]), "snaive": fa["snaive"], "naive2": fa["naive2"],
                  "holt": np.maximum(np.array(fit_and_forecast(hist, 13)["point_forecast"]), 0), "holt_d": fa["holt_d"],
                  "fa_ets": fa["fa_ets"], "fa_ets_d": fa["fa_ets_d"], "theta": fa["theta"],
                  "macro": np.array(run_macro_agent_ui(d, state, o, 13)["values"])}
    return state, out


def main():
    ph = as_housing(load_ui())
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
            for h in (4, 8, 13):
                if i + h >= len(dates):
                    continue
                tds = tuple(dates[i + 1:i + 1 + h])
                if tf is not None and tds[-1] >= ac.RESERVED_START:
                    continue
                v = {k: x[:h].copy() for k, x in fc[s][o].items()}
                cells.append(ac.Cell(s, o, h, "stable", y[i + 1:i + 1 + h], v["naive"], tds[-1], tds, v, tf))
    P6 = pd.read_csv(os.path.join("audit_reports", "results_tables", "phase6_ui_cell_losses.csv"), comment="#",
                     parse_dates=["origin"])
    P6["origin"] = P6.origin.dt.date
    sel_rows, out_rows, st_rows = [], [], []
    for label, excl in (("PRIMARY covid-excluded", in_covid), ("SECONDARY covid-included", None)):
        sel = {}
        for f in FOLDS:
            val = [c for c in cells if ac._in_val(c, f, True) and not (excl and excl(c))]
            for h in (4, 8, 13):
                vm = {k: float(np.mean([ac.mape_pct(c.actuals, c.values[k]) for c in val if c.h == h])) for k in CANDS}
                best = min(CANDS, key=lambda k: (vm[k], CANDS.index(k)))
                sel[(f.fold, h)] = best
                sel_rows.append(dict(analysis=label, fold=f.fold, h=h, selected=best, **{f"val_{k}": vm[k] for k in CANDS}))
        rows = []
        for c in cells:
            if c.test_fold is None or (excl and excl(c)):
                continue
            rows.append(dict(analysis=label, h=c.h, fold=c.test_fold, msa=c.msa, origin=c.origin,
                             selbest=ac.mape_pct(c.actuals, c.values[sel[(c.test_fold, c.h)]]),
                             holt_d=ac.mape_pct(c.actuals, c.values["holt_d"]), theta=ac.mape_pct(c.actuals, c.values["theta"]),
                             fa_ets=ac.mape_pct(c.actuals, c.values["fa_ets"])))
        D = pd.DataFrame(rows).merge(P6[P6.analysis == label][["h", "msa", "origin", "P_UI", "B5_UI", "seasonal", "fa_ets_d", "naive"]],
                                     on=["h", "msa", "origin"], how="left")
        assert D.P_UI.notna().all(), "P-UI cell missing"
        out_rows.append(D)
        meth = ["P_UI", "B5_UI", "selbest", "seasonal", "fa_ets_d", "fa_ets", "holt_d", "theta", "naive"]
        print(f"\n=== {label}: selections (fold x h)")
        print(pd.Series(sel).unstack().to_string())
        print(D.groupby("h")[meth].mean().round(3).to_string())
        pf = D.groupby(["h", "fold"])[["P_UI", "B5_UI", "selbest"]].mean()
        print(pf.round(3).to_string())
        for h in (4, 8, 13):
            sub = D[D.h == h]
            W = sl.weekly(sub, meth)
            idx = sl.block_indices(len(W), sl.SEED + h)
            for a, b in (("P_UI", "selbest"), ("B5_UI", "selbest"), ("selbest", "naive")):
                r = sl.pair_stats(sub, W, idx, a, b, h)
                r["analysis"] = label
                r["folds_A_better"] = int((pf.loc[h][a] < pf.loc[h][b]).sum()) if b in pf.columns else np.nan
                st_rows.append(r)
    ST = pd.DataFrame(st_rows)
    pd.set_option("display.width", 250)
    print(ST[["analysis", "h", "A", "B", "mean_diff", "panel_dm_p", "ci95_lo", "ci95_hi", "BH_A_better", "BH_B_better",
              "folds_A_better"]].round(4).to_string(index=False))
    hdr = (f"# CONFIG: commit={COMMIT} | UI addendum A (added after Phase 6 results; disclosed) | SelBest-UI = validation-selected "
           f"single model per (fold, h) from the housing SelBest set minus intrinsic | COVID window {COVID[0]}..{COVID[1]} | "
           f"stats as Phase 4\n")
    for name, df in (("selbest_selection", pd.DataFrame(sel_rows)), ("selbest_cells", pd.concat(out_rows)), ("selbest_stats", ST)):
        with open(os.path.join("audit_reports", "results_tables", f"phase6_ui_{name}.csv"), "w", newline="") as fh:
            fh.write(hdr)
            df.to_csv(fh, index=False)


if __name__ == "__main__":
    main()
