"""
Post-hoc statistics for the smoothed lag-2 holdout run (FINAL_CONFIG 11.5). The frozen runner wrote
cell_losses.csv and then failed in holdout_stats: 27 cells have no CoT forecast (connection failures), and
stats_lib.weekly needs all 15 MSAs in every week. The runner is not modified. This script calls the runner's own
holdout_stats on that cell_losses.csv, twice:
  1. without CoT: all cells, the pre-registered methods minus cot;
  2. with CoT: only (h, origin) weeks in which all 15 MSAs have a CoT forecast, every method.
It also runs the filled-target flag (filled_target_check.flag_cells) on the same cells.

Usage: python audit_reports/phase7/posthoc_lag2_stats.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import filled_target_check as ftc  # noqa: E402
import run_holdout as rh  # noqa: E402

RES = os.path.join(HERE, "results_20261022_smoothed_lag2")
FILLED = os.path.join(HERE, "..", "..", "..", "facts_mas_phase7_archive", "archive_20261020b", "filled_rows_smoothed.csv")
BASE = ["P", "B5", "fa_ets_d", "naive", "snaive", "naive2", "timesfm"]


def main():
    D = pd.read_csv(os.path.join(RES, "cell_losses.csv"))
    assert D[BASE].notna().all().all(), "only CoT may be missing"
    miss = D[D.cot.isna()]
    print("cells without CoT:", len(miss), "| by h:", miss.groupby("h").size().to_dict())

    s1 = rh.holdout_stats(D.copy(), BASE, "HOLDOUT cut=smoothed lag=2 POSTHOC without CoT (all cells)")
    s1.to_csv(os.path.join(RES, "stats_posthoc_without_cot.csv"), index=False)

    bad = miss[["h", "origin"]].drop_duplicates()
    keep = D.merge(bad.assign(_drop=1), on=["h", "origin"], how="left")
    keep = keep[keep._drop.isna()].drop(columns="_drop")
    dropped = D.groupby("h").size() - keep.groupby("h").size()
    print("with-CoT version: weeks dropped by h:", (dropped // 15).to_dict(), "| cells kept:", keep.groupby("h").size().to_dict())
    s2 = rh.holdout_stats(keep.copy(), BASE + ["cot"], "HOLDOUT cut=smoothed lag=2 POSTHOC with CoT (complete weeks)")
    s2.to_csv(os.path.join(RES, "stats_posthoc_with_cot_complete_weeks.csv"), index=False)

    flags = ftc.flag_cells(D, pd.read_csv(FILLED))
    out = pd.DataFrame({"n_flagged": flags.groupby(D.h).sum(), "n_cells": D.groupby("h").size()})
    out.to_csv(os.path.join(RES, "filled_targets_count.csv"))
    print("filled-target flags:\n", out.to_string())


if __name__ == "__main__":
    main()
