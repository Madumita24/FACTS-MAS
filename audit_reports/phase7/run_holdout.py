"""
Phase 7 runner for the FROZEN configuration (FINAL_CONFIG.md, freeze commit a27a23a).

Every channel forecast is computed from the frozen agent code (not the agent cache):
  Holt        ar_agent.fit_and_forecast (fitted once per (msa, info origin) and sliced)
  seasonal    seasonality_agent.run_seasonality_agent with window-7 factors (SEASONAL_SMOOTHING_WINDOW = 7)
  macro       facts_mas.agents.macro_agent.run_macro_agent
  FA-ETS(-d), snaive   p3_build_baselines.forecasts_at
Weights: tests_audit/audit_core.build_fold_model with agents {ar, seasonality, macro}, fix 2 (13 weeks), F0.10.
P = Holt+seasonal+macro at 4 weeks; FA-ETS-d(choice)+seasonal+macro at 8 and 13 weeks. B5 = the latter with an
FA-ETS-d skill reference at every horizon.

Modes
  --gate             folds 1-6 on the capped panel (<= 2026-02-21); must reproduce Phase 4 P (CLEANBLEED).
                     Writes audit_reports/phase7/gate_record.json (includes this file's sha256).
  --dry-run          pseudo-holdout made only of pre-cap data (train_end 2025-02-01, origins 2025-02-08 ..
                     last origin whose targets end before 2026-01-01). Exercises the holdout code path,
                     incl. --pub-lag 2 and --cut raw. Labelled DRY RUN; no conclusions.
  --holdout          the real Phase 7 run. Refuses unless: --phase7-go, a gate record matching this file's
                     sha256, a clean freeze check, and explicit refreshed panel paths. NOT to be run before
                     co-author sign-off.
Options
  --pub-lag {0,2}    information set at origin t = data dated <= t - lag weeks (secondary analysis uses 2)
  --cut {smoothed,raw}
  --timesfm-csv / --cot-csv   external per-step forecasts (msa, origin, horizon, step, forecast) merged into scoring
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tests_audit"))
sys.path.insert(0, os.path.join(REPO, "audit_reports", "phase3_scripts"))
sys.path.insert(0, os.path.join(REPO, "audit_reports", "phase4_scripts"))
sys.path.insert(0, HERE)
os.chdir(REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import audit_core as ac  # noqa: E402
from facts_mas.run_backtest import FOLDS, FoldSpec, classify_regime  # noqa: E402

KEEP = ["ar", "seasonality", "macro"]
R1C = dict(min_support_weeks=13, bound_targets=True)
H_MAX = 13
GATE_TARGET = {4: 2.1176, 8: 3.6218, 13: 5.7309}   # Phase 4 P, CLEANBLEED (reported, see gate criterion v2)
GATE_FILE = os.path.join(HERE, "gate_record.json")
RAW_CAPPED = os.path.join("audit_reports", "artifacts", "zillow_raw_vintage20261015_capped20260221.csv")


def self_sha():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()


# ═══ channel forecasts ════════════════════════════════════════════════════

def _load(panel_path):
    return ac.load_panel() if panel_path is None else pd.read_csv(panel_path, parse_dates=["date"])


def forecasts_for_msa(args):
    """{origin: {channel: 13-step array for targets origin+1..origin+13}} using data dated <= origin - lag."""
    panel_path, msa, origins, lag = args
    import p3_build_baselines as pb
    from ar_agent import fit_and_forecast
    from facts_mas.agents import seasonality_agent as sa
    from facts_mas.agents.macro_agent import run_macro_agent
    from facts_mas.calibration import override_constant
    panel = _load(panel_path)
    md = panel[panel.msa == msa].sort_values("date").reset_index(drop=True)
    dates = list(md["date"])
    y = md["inventory_count"].to_numpy(float)
    out = {}
    hh = H_MAX + lag
    for o in origins:
        i = dates.index(pd.Timestamp(o))
        oi = dates[i - lag].date()                       # information origin
        hist = md[md["date"] <= pd.Timestamp(oi)].set_index("date")["inventory_count"]
        holt = np.array(fit_and_forecast(hist, hh)["point_forecast"])
        with override_constant(sa, "SEASONAL_SMOOTHING_WINDOW", 7):
            f = sa.compute_seasonal_factors(panel, msa, oi)
        seas = np.array(sa.run_seasonality_agent(panel, msa, oi, hh, seasonal_factors=f).values)
        macro = np.array(run_macro_agent(panel, msa, oi, hh).values)
        with override_constant(pb, "H_MAX", hh):
            fa = pb.forecasts_at(panel, msa, oi)
        sl_ = slice(lag, lag + H_MAX)
        out[o] = {"holt": np.maximum(holt[sl_], 0), "seasonality": seas[sl_], "macro": macro[sl_],
                  "fa_ets": fa["fa_ets"][sl_], "fa_ets_d": fa["fa_ets_d"][sl_],
                  "snaive": np.array([y[i + k - 52] for k in range(1, H_MAX + 1)]),
                  "naive": np.full(H_MAX, y[i - lag])}
    return msa, out


def build_cells(panel, panel_path, folds, lag, workers=14, exclude_reserved=True):
    """Validation + test cells for the given folds. Cells whose targets reach the reserved window are
    never built when exclude_reserved=True (so they cannot be scored)."""
    msas = sorted(panel.msa.unique())
    by_msa = {}
    for m in msas:
        d = panel[panel.msa == m].sort_values("date")["date"]
        keep = set()
        for f in folds:
            vs = f.train_end - datetime.timedelta(weeks=104)
            keep |= {x.date() for x in d if vs <= x.date() <= f.train_end or f.test_start <= x.date() <= f.test_end}
        by_msa[m] = sorted(keep)
    fc = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for m, out in ex.map(forecasts_for_msa, [(panel_path, m, by_msa[m], lag) for m in msas]):
            fc[m] = out
    cells = []
    for m in msas:
        md = panel[panel.msa == m].sort_values("date").reset_index(drop=True)
        dates = list(md["date"])
        y = md["inventory_count"].to_numpy(float)
        for o in by_msa[m]:
            i = dates.index(pd.Timestamp(o))
            regime = classify_regime(panel, m, o)
            tf = next((f.fold for f in folds if f.test_start <= o <= f.test_end), None)
            for h in (4, 8, 13):
                if i + h >= len(dates):
                    continue
                tds = tuple(dates[i + 1:i + 1 + h])
                if exclude_reserved and tf is not None and tds[-1] >= ac.RESERVED_START:
                    continue
                if tds[-1] > ac.DATA_CAP and not getattr(build_cells, "allow_post_cap", False):
                    raise ac.DataCapError(f"target {tds[-1].date()} after cap outside --holdout")
                r = fc[m][o]
                vals = {k: v[:h].copy() for k, v in r.items()}
                c = ac.Cell(m, o, h, regime, y[i + 1:i + 1 + h], vals["naive"], tds[-1], tds, vals, tf)
                cells.append(c)
    return cells


# ═══ weighting and scoring ════════════════════════════════════════════════

def run_system(cells, folds):
    fold_map = {f.fold: f for f in folds}
    choice = {}
    for f in folds:
        val = [c for c in cells if ac._in_val(c, f, True)]
        mu = np.mean([ac.mape_pct(c.actuals, c.values["fa_ets"]) for c in val])
        md = np.mean([ac.mape_pct(c.actuals, c.values["fa_ets_d"]) for c in val])
        choice[f.fold] = "fa_ets_d" if md < mu else "fa_ets"
    v4 = lambda c, f: {"ar": c.values["holt"], "seasonality": c.values["seasonality"], "macro": c.values["macro"]}
    v8 = lambda c, f: {"ar": c.values[choice[f]], "seasonality": c.values["seasonality"], "macro": c.values["macro"]}
    ref = lambda c, f: c.values["fa_ets_d"]
    M4 = {f.fold: ac.build_fold_model(cells, f, v4, agents=KEEP, **R1C) for f in folds}
    M8 = {f.fold: ac.build_fold_model(cells, f, v8, agents=KEEP, **R1C) for f in folds}
    MB = {f.fold: ac.build_fold_model(cells, f, v8, agents=KEEP, reference_fn=ref, **R1C) for f in folds}
    rows = []
    for c in cells:
        if c.test_fold is None:
            continue
        f = c.test_fold
        P = ac.fuse(v4(c, f), ac.weights_for(c, M4), c.h) if c.h == 4 else ac.fuse(v8(c, f), ac.weights_for(c, M8), c.h)
        B5 = ac.fuse(v8(c, f), ac.weights_for(c, MB), c.h)
        fcs = {"P": P, "B5": B5, "fa_ets_d": c.values["fa_ets_d"], "naive": c.naive, "snaive": c.values["snaive"],
               "naive2": c.values["seasonality"]}
        rows.append(dict(h=c.h, fold=f, msa=c.msa, origin=c.origin, last_target=c.last_target,
                         **{k: ac.mape_pct(c.actuals, v) for k, v in fcs.items()}))
    return pd.DataFrame(rows), choice, {k: {f: (M4[f].mode, M8[f].mode, MB[f].mode) for f in fold_map} for k in ["modes"]}


def merge_external(D, cells, path, name):
    ext = pd.read_csv(path, parse_dates=["origin"])
    ext["origin"] = ext["origin"].dt.date
    act = {(c.msa, c.origin, c.h): c.actuals for c in cells if c.test_fold is not None}
    vals = []
    for r in D.itertuples():
        e = ext[(ext.msa == r.msa) & (ext.origin == r.origin) & (ext.horizon == r.h)].sort_values("step")
        vals.append(ac.mape_pct(act[(r.msa, r.origin, r.h)], e.forecast.to_numpy()) if len(e) == r.h else np.nan)
    D[name] = vals
    return D


def holdout_stats(D, methods, label):
    import stats_lib as sl
    rows = []
    for h in (4, 8, 13):
        sub = D[D.h == h].dropna(subset=methods)
        W = sl.weekly(sub, methods)
        T = len(W)
        Lb = h if h <= T / 2 else int(np.floor(T / 2))
        idx = sl.block_indices(T, sl.SEED + h, block_len=Lb)
        for a, b in [("P", "naive"), ("fa_ets_d", "naive"), ("P", "fa_ets_d"), ("B5", "fa_ets_d"), ("P", "snaive"),
                     ("P", "naive2")] + [(m, "fa_ets_d") for m in methods if m in ("timesfm", "cot")]:
            if a in methods and b in methods:
                r = sl.pair_stats(sub, W, idx, a, b, h)
                r.update(block_len=Lb, ci_flag_unreliable=bool(h > T / 2), label=label,
                         margin_2pct=0.02 * W["fa_ets_d"].mean())
                rows.append(r)
    return pd.DataFrame(rows)


# ═══ modes ════════════════════════════════════════════════════════════════

def gate(args):
    import check_freeze
    check_freeze.check()
    panel = ac.load_panel()
    cells = build_cells(panel, None, FOLDS, lag=0)
    D, choice, modes = run_system(cells, FOLDS)
    ref = pd.read_csv(os.path.join("audit_reports", "results_tables", "phase4_cell_losses.csv"), comment="#",
                      parse_dates=["origin"])
    ref = ref[~ref.bleed][["h", "msa", "origin", "P", "fa_ets_d", "naive"]]
    ref["origin"] = ref["origin"].dt.date
    j = D.merge(ref, on=["h", "msa", "origin"], suffixes=("", "_p4"))
    assert len(j) == len(ref) == len(D), (len(j), len(ref), len(D))
    pooled = D.groupby("h")[["P", "B5", "fa_ets_d", "naive"]].mean()
    rec = {"runner_sha256": self_sha(), "freeze_commit": "a27a23ab7b1a136c860b110e6871ad1bb4c7dfa8",
           "run_at": datetime.datetime.now().isoformat(timespec="seconds"), "damped_choice": choice,
           "n_cells": {int(h): int((D.h == h).sum()) for h in (4, 8, 13)}, "pooled": {}, "max_abs_cell_diff": {},
           "naive_max_abs_cell_diff": float((j.naive - j.naive_p4).abs().max())}
    # Gate criterion (v2, after the v1 failure documented in phase6_log.md): the tolerance is AR optimizer
    # noise. (1) pooled P within 1e-3 of Phase 4 at every horizon; (2) naive identical; (3) attribution: with
    # the cached AR values substituted for the refits, P must match Phase 4 on every cell to 1e-9, so any
    # remaining gap is attributable to the AR (Holt) optimizer alone.
    from facts_mas.agent_cache import AgentCache
    cache = AgentCache.load("agent_output_cache.pkl")
    cells_ar = []
    for c in cells:
        v = dict(c.values)
        v["holt"] = np.maximum(np.asarray(cache.entries[(c.msa, c.origin, c.h)].agent_values["ar"], float), 0)
        cells_ar.append(ac.Cell(c.msa, c.origin, c.h, c.regime, c.actuals, c.naive, c.last_target, c.target_dates, v, c.test_fold))
    D2, _, _ = run_system(cells_ar, FOLDS)
    j2 = D2.merge(ref, on=["h", "msa", "origin"], suffixes=("", "_p4"))
    rec["attribution_cached_AR_max_abs_cell_diff_P"] = float((j2.P - j2.P_p4).abs().max())
    rec["attribution_cached_AR_pooled_diff_P"] = {int(h): float(j2[j2.h == h].P.mean() - j2[j2.h == h].P_p4.mean()) for h in (4, 8, 13)}
    ar_rel = [float(np.max(np.abs(c.values["holt"] - ca.values["holt"]) / np.maximum(ca.values["holt"], 1)))
              for c, ca in zip(cells, cells_ar)]
    rec["AR_refit_vs_cache_rel_diff"] = {"median": float(np.median(ar_rel)), "p99": float(np.percentile(ar_rel, 99)),
                                         "max": float(np.max(ar_rel))}
    ok = True
    for h in (4, 8, 13):
        p = float(pooled.loc[h, "P"])
        jd = j[j.h == h]
        rec["pooled"][int(h)] = {"P": p, "P_phase4": float(jd.P_p4.mean()), "diff": p - float(jd.P_p4.mean()),
                                 "B5": float(pooled.loc[h, "B5"]), "fa_ets_d": float(pooled.loc[h, "fa_ets_d"]),
                                 "naive": float(pooled.loc[h, "naive"])}
        rec["max_abs_cell_diff"][int(h)] = {"P": float((jd.P - jd.P_p4).abs().max()),
                                            "fa_ets_d": float((jd.fa_ets_d - jd.fa_ets_d_p4).abs().max())}
        ok &= abs(p - jd.P_p4.mean()) < 1e-3
    ok &= rec["naive_max_abs_cell_diff"] < 1e-12
    ok &= rec["attribution_cached_AR_max_abs_cell_diff_P"] < 1e-9
    rec["PASS"] = bool(ok)
    print(json.dumps(rec, indent=1, default=str))
    if ok:
        json.dump(rec, open(GATE_FILE, "w"), indent=1, default=str)
        print("GATE PASSED; record written to", GATE_FILE)
    else:
        raise SystemExit("GATE FAILED; holdout mode stays locked")


def dry_run(args):
    print("DRY RUN: pseudo-holdout on pre-cap data only. No conclusions.")
    panel = pd.read_csv(RAW_CAPPED, parse_dates=["date"]) if args.cut == "raw" else ac.load_panel()
    last = max(d.date() for d in panel.date if d + pd.Timedelta(weeks=13) < ac.RESERVED_START)
    pseudo = FoldSpec(7, datetime.date(2019, 2, 2), datetime.date(2025, 2, 1), datetime.date(2025, 2, 8), last)
    cells = build_cells(panel, RAW_CAPPED if args.cut == "raw" else None, [pseudo], lag=args.pub_lag)
    D, choice, modes = run_system(cells, [pseudo])
    print("damped choice", choice, "| modes (P4, P813, B5)", modes)
    print(D.groupby("h")[["P", "B5", "fa_ets_d", "naive", "snaive", "naive2"]].mean().round(4).to_string())
    S = holdout_stats(D, ["P", "B5", "fa_ets_d", "naive", "snaive", "naive2"], f"DRY RUN cut={args.cut} lag={args.pub_lag}")
    print(S[["h", "A", "B", "mean_diff", "panel_dm_p", "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi", "block_len",
             "ci_flag_unreliable"]].round(4).to_string(index=False))


def holdout(args):
    import check_freeze
    if not args.phase7_go:
        raise SystemExit("--holdout requires --phase7-go (Phase 7 needs Umang's confirmation of co-author sign-off)")
    check_freeze.check()
    rec = json.load(open(GATE_FILE))
    if not rec.get("PASS") or rec.get("runner_sha256") != self_sha():
        raise SystemExit("gate record missing, failed, or for a different runner version")
    if not args.panel:
        raise SystemExit("--holdout needs --panel <refreshed long panel CSV> (never aligned_weekly.csv)")
    build_cells.allow_post_cap = True
    panel = pd.read_csv(args.panel, parse_dates=["date"])
    last = panel.date.max().date()
    fold = FoldSpec(7, datetime.date(2019, 2, 2), datetime.date(2026, 2, 21), datetime.date(2026, 2, 28), last)
    cells = build_cells(panel, args.panel, [fold], lag=args.pub_lag, exclude_reserved=False)
    D, choice, modes = run_system(cells, [fold])
    methods = ["P", "B5", "fa_ets_d", "naive", "snaive", "naive2"]
    for name, p in (("timesfm", args.timesfm_csv), ("cot", args.cot_csv)):
        if p:
            D = merge_external(D, cells, p, name)
            methods.append(name)
    outdir = os.path.join(HERE, f"results_{datetime.date.today():%Y%m%d}_{args.cut}_lag{args.pub_lag}")
    os.makedirs(outdir, exist_ok=False)
    D.to_csv(os.path.join(outdir, "cell_losses.csv"), index=False)
    holdout_stats(D, methods, f"HOLDOUT cut={args.cut} lag={args.pub_lag}").to_csv(os.path.join(outdir, "stats.csv"), index=False)
    json.dump({"choice": choice, "modes": modes, "runner_sha256": self_sha(), "panel": args.panel,
               "panel_sha256": hashlib.sha256(open(args.panel, "rb").read()).hexdigest()},
              open(os.path.join(outdir, "run_meta.json"), "w"), indent=1, default=str)
    print("written", outdir)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gate", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--holdout", action="store_true")
    ap.add_argument("--pub-lag", type=int, choices=[0, 2], default=0)
    ap.add_argument("--cut", choices=["smoothed", "raw"], default="smoothed")
    ap.add_argument("--panel", default=None)
    ap.add_argument("--timesfm-csv", default=None)
    ap.add_argument("--cot-csv", default=None)
    ap.add_argument("--phase7-go", action="store_true")
    a = ap.parse_args()
    (gate if a.gate else dry_run if a.dry_run else holdout)(a)


if __name__ == "__main__":
    main()
