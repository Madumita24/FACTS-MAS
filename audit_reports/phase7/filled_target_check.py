"""
Phase 7 post-hoc filled-target check (FINAL_CONFIG section 11, subsection B). Scoring code is unchanged:
this filters the runner's cell_losses.csv after the fact.

The cut is read from the run's run_meta.json (panel path) and must match the filled-rows file
(filled_rows_smoothed.csv / filled_rows_raw.csv); a mismatch or an unknown cut refuses to run.
A holdout cell is flagged if any of its target dates (origin, last_target] carries a forward-filled
inventory value for that MSA (filled_rows_{smoothed,raw}.csv from build_holdout_panel.py). Reports the count
per horizon and, if any are flagged, the primary statistics with and without those cells (the runner's own
holdout_stats function, imported unchanged).

Usage (repo root): python audit_reports/phase7/filled_target_check.py <results_dir> <filled_rows.csv>
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pandas as pd  # noqa: E402

METHODS = ["P", "B5", "fa_ets_d", "naive", "snaive", "naive2", "timesfm", "cot"]


def flag_cells(D, filled):
    """Boolean Series: cell's target window (origin, last_target] contains a filled date for its MSA."""
    f = filled.copy()
    f["date"] = pd.to_datetime(f["date"])
    by = {m: set(g["date"]) for m, g in f.groupby("msa")}
    o = pd.to_datetime(D["origin"])
    lt = pd.to_datetime(D["last_target"])
    return pd.Series([any(o_ < d <= l_ for d in by.get(m, ())) for m, o_, l_ in zip(D["msa"], o, lt)], index=D.index)


def cut_of(path):
    """'smoothed' or 'raw' from a panel / filled-rows file name (build_holdout_panel.py naming), else None."""
    name = os.path.basename(str(path)).lower()
    if "smoothed" in name:
        return "smoothed"
    if "raw" in name:
        return "raw"
    return None


def check_cut(res, filled_path):
    """Cut of the run (from run_meta.json's panel path) must equal the cut of the filled-rows file."""
    import json
    meta = json.load(open(os.path.join(res, "run_meta.json")))
    run_cut, filled_cut = cut_of(meta.get("panel")), cut_of(filled_path)
    if run_cut is None or filled_cut is None:
        raise SystemExit(f"cannot determine the cut (run panel: {meta.get('panel')!r}, filled rows: {filled_path!r})")
    if run_cut != filled_cut:
        raise SystemExit(f"refusing: run is on the {run_cut} cut but the filled-rows file is for the {filled_cut} cut")
    return run_cut


def main():
    res, filled_path = sys.argv[1], sys.argv[2]
    cut = check_cut(res, filled_path)
    print("cut:", cut)
    D = pd.read_csv(os.path.join(res, "cell_losses.csv"))
    filled = pd.read_csv(filled_path)
    D["filled_target"] = flag_cells(D, filled)
    counts = D.groupby("h").filled_target.agg(["sum", "size"]).rename(columns={"sum": "n_flagged", "size": "n_cells"})
    print("cells whose targets include a forward-filled value:\n", counts.to_string())
    counts.reset_index().to_csv(os.path.join(res, "filled_targets_count.csv"), index=False)
    if D.filled_target.any():
        import run_holdout as rh
        methods = [m for m in METHODS if m in D.columns]
        with_ = rh.holdout_stats(D, methods, "all cells")
        without = rh.holdout_stats(D[~D.filled_target], methods, "filled-target cells excluded")
        pd.concat([with_, without]).to_csv(os.path.join(res, "stats_with_without_filled.csv"), index=False)
        print("written stats_with_without_filled.csv")


if __name__ == "__main__":
    main()
