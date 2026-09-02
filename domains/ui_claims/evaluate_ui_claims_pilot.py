"""Step 3 -- the real single train/test split evaluation for the UI-claims
pilot. Matches the housing pipeline's evaluation convention (MAPE/RMSE per
state per horizon, horizons never averaged) but over ONE split, not a
6-fold backtest -- see evaluation_protocol_ui.md for why.

Split (confirmed): train_end=2023-06-24, embargo=91d, test_start=2023-09-23,
test_end=2026-08-22 (last available date). Forecasts are generated at every
weekly origin in the test window, not a single snapshot -- same lesson the
housing pipeline's CoT-baseline work already demonstrated the hard way
(single-origin evaluation is misleading; averaging over the full test
window is the only methodology that produced trustworthy conclusions).

Also computes a naive equal-weight AR+macro blend -- NOT the full adaptive
fusion mechanism (regime grids, skill-based softmax weighting) used in the
housing pipeline; that machinery is out of scope for this minimal 2-agent
pilot. This is a simple average of the two, to get a first read on whether
combining even naively helps at all.
"""
import sys
sys.path.insert(0, ".")

import datetime

import numpy as np
import pandas as pd

from domains.ui_claims.agents.ar_agent_ui import run_ar_agent_ui
from domains.ui_claims.agents.macro_agent_ui import run_macro_agent_ui
from domains.ui_claims.naive_baseline_ui import run_naive_baseline_ui

HORIZONS = [4, 8, 13]
TEST_START = datetime.date(2023, 9, 23)
TEST_END = datetime.date(2026, 8, 22)

DATA_PATH = "domains/ui_claims/aligned_weekly_ui_claims.csv"
OUT_PER_ORIGIN = "domains/ui_claims/ui_claims_eval_per_origin.csv"
OUT_PER_STATE = "domains/ui_claims/ui_claims_eval_per_state.csv"
OUT_SUMMARY = "domains/ui_claims/ui_claims_eval_summary.csv"


def compute_mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs((actual - forecast) / actual)) * 100)


def compute_rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    states = sorted(df["state"].unique().tolist())

    origins = df[(df["date"] >= pd.Timestamp(TEST_START))
                 & (df["date"] <= pd.Timestamp(TEST_END))]["date"].sort_values().unique()
    print(f"Test window: {len(origins)} candidate origins/state, {len(states)} states, "
          f"{len(HORIZONS)} horizons")

    rows = []
    n_errors = 0
    for state in states:
        sdf = df[df["state"] == state].sort_values("date")

        for origin_ts in origins:
            origin = pd.Timestamp(origin_ts).date()

            for horizon in HORIZONS:
                future = sdf[sdf["date"] > pd.Timestamp(origin)].head(horizon)
                if len(future) < horizon:
                    continue  # not enough future data at the tail of the file
                actuals = future["claims_count"].values.astype(np.float64)

                try:
                    naive_v = np.array(run_naive_baseline_ui(df, state, origin, horizon)["values"])
                    ar_v = np.array(run_ar_agent_ui(df, state, origin, horizon)["values"])
                    macro_v = np.array(run_macro_agent_ui(df, state, origin, horizon)["values"])
                    blend_v = (ar_v + macro_v) / 2.0
                except Exception as e:
                    n_errors += 1
                    print(f"  ERROR state={state} origin={origin} h={horizon}: {type(e).__name__}: {e}")
                    continue

                rows.append({
                    "state": state, "origin": origin, "horizon": horizon,
                    "naive_mape": compute_mape(actuals, naive_v),
                    "naive_rmse": compute_rmse(actuals, naive_v),
                    "ar_mape": compute_mape(actuals, ar_v),
                    "ar_rmse": compute_rmse(actuals, ar_v),
                    "macro_mape": compute_mape(actuals, macro_v),
                    "macro_rmse": compute_rmse(actuals, macro_v),
                    "blend_mape": compute_mape(actuals, blend_v),
                    "blend_rmse": compute_rmse(actuals, blend_v),
                })

        print(f"  {state} done ({len(rows)} cells so far, {n_errors} errors)", flush=True)

    pdf = pd.DataFrame(rows)
    pdf.to_csv(OUT_PER_ORIGIN, index=False)
    print(f"\n{OUT_PER_ORIGIN}: {len(pdf)} rows, {n_errors} errors")

    # Per-(state, horizon) aggregation
    agg = (pdf.groupby(["state", "horizon"])
           .agg(n=("naive_mape", "size"),
                naive_mape=("naive_mape", "mean"), naive_rmse=("naive_rmse", "mean"),
                ar_mape=("ar_mape", "mean"), ar_rmse=("ar_rmse", "mean"),
                macro_mape=("macro_mape", "mean"), macro_rmse=("macro_rmse", "mean"),
                blend_mape=("blend_mape", "mean"), blend_rmse=("blend_rmse", "mean"))
           .reset_index())
    agg.to_csv(OUT_PER_STATE, index=False)

    print("\n" + "=" * 100)
    print("PER-STATE, PER-HORIZON (MAPE)")
    print("=" * 100)
    for h in HORIZONS:
        print(f"\n--- horizon {h}wk ---")
        print(agg[agg.horizon == h][["state", "n", "naive_mape", "ar_mape", "macro_mape", "blend_mape"]]
              .sort_values("state").to_string(index=False))

    # Aggregate summary + win counts
    summary_rows = []
    print("\n" + "=" * 100)
    print("AGGREGATE SUMMARY (mean/median MAPE across 15 states) + WIN COUNTS")
    print("=" * 100)
    for h in HORIZONS:
        s = agg[agg.horizon == h]
        ar_beats_naive = int((s.ar_mape < s.naive_mape).sum())
        macro_beats_naive = int((s.macro_mape < s.naive_mape).sum())
        blend_beats_naive = int((s.blend_mape < s.naive_mape).sum())
        blend_beats_ar = int((s.blend_mape < s.ar_mape).sum())
        row = {
            "horizon": h, "n_states": len(s),
            "naive_mean": s.naive_mape.mean(), "naive_median": s.naive_mape.median(),
            "ar_mean": s.ar_mape.mean(), "ar_median": s.ar_mape.median(),
            "macro_mean": s.macro_mape.mean(), "macro_median": s.macro_mape.median(),
            "blend_mean": s.blend_mape.mean(), "blend_median": s.blend_mape.median(),
            "ar_beats_naive": ar_beats_naive, "macro_beats_naive": macro_beats_naive,
            "blend_beats_naive": blend_beats_naive, "blend_beats_ar": blend_beats_ar,
        }
        summary_rows.append(row)
        print(f"\nh={h}wk (n={len(s)} states):")
        print(f"  mean:   naive={row['naive_mean']:.3f}  ar={row['ar_mean']:.3f}  "
              f"macro={row['macro_mean']:.3f}  blend={row['blend_mean']:.3f}")
        print(f"  median: naive={row['naive_median']:.3f}  ar={row['ar_median']:.3f}  "
              f"macro={row['macro_median']:.3f}  blend={row['blend_median']:.3f}")
        print(f"  AR beats naive:    {ar_beats_naive}/{len(s)}")
        print(f"  Macro beats naive: {macro_beats_naive}/{len(s)}")
        print(f"  Blend beats naive: {blend_beats_naive}/{len(s)}")
        print(f"  Blend beats AR:    {blend_beats_ar}/{len(s)}")

    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    print(f"\nSaved: {OUT_PER_ORIGIN}, {OUT_PER_STATE}, {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
