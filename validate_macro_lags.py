"""Bounded follow-up to validate_macro_signal.py's negative result. Tests
two specific alternative hypotheses before concluding macro has no usable
signal -- diagnostic only, no changes to macro_agent.py.

STEP A: for each of the 4 macro features independently, sweep lag across
the full documented range from align_data.py (not just the midpoint) and
correlate the raw lagged feature value against realized 13-week inventory
pct-change. Isolates lag choice from feature choice.

STEP B: instead of regressing a single-week pct-change and combining it
across the horizon (the design that caused the earlier compounding
artifact), regress DIRECTLY on the realized horizon-total pct-change, one
model per horizon, using each feature's best lag from Step A.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

from fold_boundaries import build_fold_table
from validate_macro_signal import actual_pct_change

LAG_RANGES = {
    "mortgage_rate": range(4, 9),   # documented 4-8wk
    "fed_funds": range(8, 13),      # documented 8-12wk
    "cpi": range(8, 13),            # documented 8-12wk
    "unemployment": range(4, 7),    # documented 4-6wk
}

HORIZONS = [4, 8, 13]


def lagged_feature_value(feature_dict: dict, as_of: pd.Timestamp, lag_weeks: int):
    return feature_dict.get(as_of - pd.Timedelta(weeks=lag_weeks), np.nan)


def step_a(df: pd.DataFrame, folds: pd.DataFrame) -> dict:
    msas = df["msa"].unique()
    train_ends = [pd.Timestamp(f.train_end) for f in folds.itertuples()]

    actual_13wk = {}
    for msa in msas:
        for train_end in train_ends:
            actual_13wk[(msa, train_end)] = actual_pct_change(df, msa, train_end, 13)

    feature_dicts = {
        msa: {feat: df[df["msa"] == msa].set_index("date")[feat].to_dict() for feat in LAG_RANGES}
        for msa in msas
    }

    print("=" * 70)
    print("STEP A -- lag grid search (target: realized 13wk pct-change)")
    best_per_feature = {}
    for feature, lag_range in LAG_RANGES.items():
        print(f"\n{feature}  (documented range: {lag_range.start}-{lag_range.stop - 1}wk)")
        grid = []
        for lag in lag_range:
            xs, ys = [], []
            for msa in msas:
                for train_end in train_ends:
                    y = actual_13wk[(msa, train_end)]
                    if np.isnan(y):
                        continue
                    x = lagged_feature_value(feature_dicts[msa][feature], train_end, lag)
                    if np.isnan(x):
                        continue
                    xs.append(x)
                    ys.append(y)
            r, p = pearsonr(xs, ys)
            grid.append((lag, r, p, len(xs)))
            print(f"    lag={lag:2d}wk   r={r:+.3f}   p={p:.4f}   n={len(xs)}")

        best = max(grid, key=lambda row: abs(row[1]))
        best_per_feature[feature] = {"lag": best[0], "r": best[1], "p": best[2], "n": best[3]}
        midpoint_lag = sorted(lag_range)[len(lag_range) // 2]
        midpoint = next(row for row in grid if row[0] == midpoint_lag)
        print(f"  BEST: lag={best[0]}wk  r={best[1]:+.3f}  p={best[2]:.4f}   "
              f"(midpoint lag={midpoint_lag}wk was r={midpoint[1]:+.3f}, p={midpoint[2]:.4f})")

    return best_per_feature


def step_b(df: pd.DataFrame, folds: pd.DataFrame, best_lags: dict) -> dict:
    msas = df["msa"].unique()
    results = {}

    print("\n" + "=" * 70)
    print("STEP B -- horizon-level regression target (using Step A's best lags)")
    lag_summary = ", ".join(f"{f}={v['lag']}wk" for f, v in best_lags.items())
    print(f"Lags used: {lag_summary}")

    for horizon in HORIZONS:
        preds, actuals = [], []

        for msa in msas:
            msa_df = df[df["msa"] == msa].sort_values("date").set_index("date")
            feature_dicts = {feat: msa_df[feat].to_dict() for feat in LAG_RANGES}
            inv_dict = msa_df["inventory_count"].to_dict()

            for fold in folds.itertuples():
                train_end = pd.Timestamp(fold.train_end)
                history = msa_df[msa_df.index <= train_end]
                cutoff = train_end - pd.Timedelta(weeks=horizon)
                candidate_t = history.index[history.index <= cutoff]

                X_rows, y_rows = [], []
                for t in candidate_t:
                    feats = [lagged_feature_value(feature_dicts[feat], t, best_lags[feat]["lag"])
                             for feat in LAG_RANGES]
                    if any(np.isnan(v) for v in feats):
                        continue
                    target_date = t + pd.Timedelta(weeks=horizon)
                    y_target = inv_dict.get(target_date, np.nan)
                    y_t0 = inv_dict.get(t, np.nan)
                    if np.isnan(y_target) or np.isnan(y_t0) or y_t0 == 0:
                        continue
                    X_rows.append(feats)
                    y_rows.append((y_target - y_t0) / y_t0)

                if len(X_rows) < 15:
                    continue

                model = Ridge(alpha=1.0)
                model.fit(np.array(X_rows), np.array(y_rows))

                feats_forecast = [lagged_feature_value(feature_dicts[feat], train_end, best_lags[feat]["lag"])
                                   for feat in LAG_RANGES]
                if any(np.isnan(v) for v in feats_forecast):
                    continue

                pred = float(model.predict([feats_forecast])[0])
                actual = actual_pct_change(df, msa, train_end, horizon)
                if np.isnan(actual):
                    continue

                preds.append(pred)
                actuals.append(actual)

        r, p = pearsonr(preds, actuals)
        results[horizon] = {"r": r, "p": p, "n": len(preds)}
        print(f"\nHorizon = {horizon} weeks  (n={len(preds)})")
        print(f"  Pearson r (horizon-level prediction vs. actual pct-change): {r:+.3f}")
        print(f"  p-value: {p:.4f}  "
              f"{'(distinguishable from noise)' if p < 0.05 else '(NOT distinguishable from noise)'}")

    return results


def main():
    df = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
    folds = build_fold_table()

    best_lags = step_a(df, folds)
    horizon_results = step_b(df, folds, best_lags)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n1. Best lag per feature vs. documented midpoint:")
    lag_improved = []
    for feature, res in best_lags.items():
        lag_range = LAG_RANGES[feature]
        midpoint_lag = sorted(lag_range)[len(lag_range) // 2]
        meaningfully_better = res["p"] < 0.05 and abs(res["r"]) >= 0.2
        print(f"   {feature}: best lag={res['lag']}wk (r={res['r']:+.3f}, p={res['p']:.4f}) "
              f"vs. midpoint={midpoint_lag}wk -- "
              f"{'MEANINGFUL IMPROVEMENT' if meaningfully_better else 'still weak/not significant'}")
        if meaningfully_better:
            lag_improved.append(feature)

    print("\n2. Horizon-level regression vs. single-week target:")
    horizon_signal = []
    for horizon, res in horizon_results.items():
        found_signal = res["p"] < 0.05 and abs(res["r"]) >= 0.2
        print(f"   {horizon}wk: r={res['r']:+.3f}, p={res['p']:.4f} -- "
              f"{'SIGNAL FOUND' if found_signal else 'no meaningful signal'}")
        if found_signal:
            horizon_signal.append(horizon)

    print("\n3. Recommendation:")
    if lag_improved or horizon_signal:
        parts = []
        if lag_improved:
            parts.append(f"features {lag_improved} show meaningful signal at a non-midpoint lag")
        if horizon_signal:
            parts.append(f"horizon-level regression finds signal at {horizon_signal}")
        print(f"   A revised Macro Agent MAY be worth building: {'; '.join(parts)}. "
              f"Recommend rebuilding macro_agent.py with the specific winning lag(s)/target only -- "
              f"not a wholesale redesign -- and re-validating against fresh folds before trusting it.")
    else:
        print("   CONFIRMS macro genuinely has no usable signal in this dataset at any tested "
              "configuration (documented-range lag sweep x 4 features x raw single-week target, "
              "and horizon-level regression x 3 horizons using each feature's own best lag). "
              "Recommend deprioritizing the Macro Agent rather than continuing to tune it -- "
              "the null result is now well-triangulated across lag choice, feature choice, and "
              "regression target, not an artifact of any one of those three.")


if __name__ == "__main__":
    main()
