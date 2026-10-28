"""
Phase 6c: score TimesFM forecasts (from p6_timesfm.py) on the backtest test cells and compare with
P, FA-ETS-d and naive (Phase 4 per-cell losses). Parity check against the existing fold-1 results.

Usage (repo root): python audit_reports/phase6_scripts/p6_timesfm_score.py <forecasts.csv> [--parity-only]
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "tests_audit"))
sys.path.insert(0, os.path.join(os.getcwd(), "audit_reports", "phase4_scripts"))
import audit_core as ac  # noqa: E402
import stats_lib as sl  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COMMIT = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
F = pd.read_csv(sys.argv[1], parse_dates=["origin"])
F["origin"] = F.origin.dt.date
panel = ac.load_panel()
rows = []
for (msa, o, H), g in F.groupby(["msa", "origin", "horizon"]):
    s = panel[panel.msa == msa].sort_values("date").reset_index(drop=True)
    i = s.index[s.date == pd.Timestamp(o)][0]
    fc = g.sort_values("step").forecast.to_numpy()
    for h in ((4, 8, 13) if H == 13 else (H,)):
        if i + h >= len(s):
            continue
        tds = s.date.iloc[i + 1:i + 1 + h]
        rows.append(dict(msa=msa, origin=o, h=h, call_h=H, bleed=bool(tds.iloc[-1] >= ac.RESERVED_START),
                         timesfm=ac.mape_pct(s.inventory_count.iloc[i + 1:i + 1 + h].to_numpy(float), fc[:h])))
T = pd.DataFrame(rows)
ref = pd.read_csv("timesfm_baseline_fold1_full_window_results.csv", parse_dates=["origin"])
ref["origin"] = ref.origin.dt.date
par = T.merge(ref.rename(columns={"horizon": "h"}), on=["msa", "origin", "h"])
if len(par):
    print(f"parity vs timesfm_baseline_fold1 file: {len(par)} cells, max |diff| = {np.abs(par.timesfm - par.timesfm_mape).max():.2e}, "
          f"mean diff = {(par.timesfm - par.timesfm_mape).mean():+.2e}")
if "--parity-only" in sys.argv:
    sys.exit(0)

P4 = pd.read_csv(os.path.join("audit_reports", "results_tables", "phase4_cell_losses.csv"), comment="#", parse_dates=["origin"])
P4["origin"] = P4.origin.dt.date
D = P4.merge(T[["msa", "origin", "h", "timesfm"]], on=["msa", "origin", "h"], how="left")
assert D.timesfm.notna().all(), "missing TimesFM cells"
C = D[~D.bleed]
meth = ["timesfm", "P", "fa_ets_d", "naive"]
print(C.groupby("h")[meth].mean().round(3).to_string())
print(C.groupby(["h", "fold"])[meth].mean().round(3).to_string())
st = []
for h in (4, 8, 13):
    sub = C[C.h == h]
    W = sl.weekly(sub, meth)
    idx = sl.block_indices(len(W), sl.SEED + h)
    for a, b in (("timesfm", "naive"), ("P", "timesfm"), ("fa_ets_d", "timesfm")):
        st.append(sl.pair_stats(sub, W, idx, a, b, h))
ST = pd.DataFrame(st)
print(ST[["h", "A", "B", "mean_diff", "panel_dm_p", "ci95_lo", "ci95_hi", "BH_A_better", "BH_B_better"]].round(4).to_string(index=False))
hdr = (f"# CONFIG: commit={COMMIT} | TimesFM 2.5 (google/timesfm-2.5-200m-pytorch, 52-week context) on folds 1-6 TEST, "
       f"CLEANBLEED primary | env facts-timesfm | stats as Phase 4\n")
for name, df in (("timesfm_cells", D), ("timesfm_stats", ST)):
    with open(os.path.join("audit_reports", "results_tables", f"phase6_{name}.csv"), "w", newline="") as fh:
        fh.write(hdr)
        df.to_csv(fh, index=False)
