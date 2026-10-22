"""Collect the Phase 7 holdout results into audit_reports/results_tables/phase7_*.csv (config header first line)."""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results_tables")
RUNS = [("smoothed", 0, "stats.csv", "pre-registered"),
        ("smoothed", 2, "stats_posthoc_without_cot.csv", "post hoc 11.5: without CoT, all cells"),
        ("smoothed", 2, "stats_posthoc_with_cot_complete_weeks.csv", "post hoc 11.5: with CoT, complete weeks only"),
        ("raw", 0, "stats.csv", "pre-registered"),
        ("raw", 2, "stats.csv", "pre-registered")]
METHODS = ["P", "B5", "fa_ets_d", "naive", "snaive", "naive2", "timesfm", "cot"]
CONFIG = ("# CONFIG: Phase 7 HOLDOUT | frozen P (FINAL_CONFIG, frozen a27a23a; runner sha256 06dee8fd...) | one fold: "
          "train_end 2026-02-21, origins 2026-02-28..2026-09-12, data to 2026-10-10 (archive_20261020b) | 15 MSAs | "
          "MAPE per cell over steps 1..H | stats: panel DM NW lag h-1; block bootstrap whole weeks B=5000 seed "
          "20261016, L=h or floor(T/2) (flagged); equivalence margin 2% of FA-ETS-d | 13wk descriptive (FINAL_CONFIG 11.1) "
          "| CoT gpt-5.6-luna (11.4), smoothed only | TimesFM 2.5 | lag = publication lag on inventory (weeks)")


def write(df, name):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(CONFIG + "\n")
        df.to_csv(f, index=False)
    print("wrote", p, len(df))


def main():
    mape, stats = [], []
    for cut, lag, sfile, status in RUNS:
        d = os.path.join(HERE, f"results_20261022_{cut}_lag{lag}")
        D = pd.read_csv(os.path.join(d, "cell_losses.csv"))
        if sfile == "stats.csv" or "without" in sfile:
            m = D.groupby("h")[[c for c in METHODS if c in D]].mean()
            n = D.groupby("h").size().rename("n_cells")
            mape.append(m.join(n).reset_index().assign(cut=cut, lag=lag))
        S = pd.read_csv(os.path.join(d, sfile))
        stats.append(S.assign(cut=cut, lag=lag, status=status))
    M = pd.concat(mape)
    M = M[["cut", "lag", "h", "n_cells"] + [c for c in METHODS if c in M]]
    write(M.round(4), "phase7_holdout_mape.csv")
    S = pd.concat(stats)
    write(S[["cut", "lag", "status"] + [c for c in S.columns if c not in ("cut", "lag", "status")]], "phase7_holdout_stats.csv")


if __name__ == "__main__":
    main()
