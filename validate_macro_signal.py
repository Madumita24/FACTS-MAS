"""Decisive macro-signal diagnostic: does the macro modifier carry real
predictive signal about future inventory change, independent of any
combination rule with the AR forecast?

For each (msa, fold, horizon), compares the macro_agent's predicted
modifier (a single weekly pct-change "pressure" value, constant across the
horizon in this v1 design -- see macro_agent.py) against the ACTUAL
realized total pct-change in inventory over that same horizon, computed
directly from aligned_weekly.csv:
    actual_pct_change = (actual_value_at_horizon_end - last_training_value) / last_training_value

Correlating modifier against total pct-change (rather than an average
weekly rate) is equivalent for this purpose: within a fixed horizon, total
change is just weekly-rate * horizon, a constant rescaling that doesn't
affect Pearson r.

Report-only. No changes to macro_agent.py, no combiners.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from fold_boundaries import build_fold_table
from macro_agent import run_macro_agent

HORIZONS = [4, 8, 13]


def actual_pct_change(df: pd.DataFrame, msa: str, train_end, horizon: int) -> float:
    msa_series = df[df["msa"] == msa].sort_values("date").set_index("date")["inventory_count"]
    train_end = pd.Timestamp(train_end)

    last_training_value = msa_series[msa_series.index <= train_end].iloc[-1]
    target_date = train_end + pd.Timedelta(weeks=horizon)
    actual_end = msa_series.reindex([target_date]).iloc[0]

    if np.isnan(actual_end):
        return np.nan
    return (actual_end - last_training_value) / last_training_value


def main():
    df = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
    folds = build_fold_table()

    macro_forecasts = run_macro_agent(df, folds, horizons=HORIZONS)

    print("=" * 70)
    results = {}
    for horizon in HORIZONS:
        h = macro_forecasts[macro_forecasts["horizon"] == horizon]

        modifiers, actuals = [], []
        for row in h.itertuples():
            actual = actual_pct_change(df, row.msa, row.train_end, row.horizon)
            if np.isnan(actual):
                continue
            modifiers.append(row.modifier[0])  # constant across the horizon in this v1 design
            actuals.append(actual)

        r, p = pearsonr(modifiers, actuals)
        results[horizon] = {"r": r, "p": p, "n": len(modifiers)}

        print(f"Horizon = {horizon} weeks  (n={len(modifiers)} msa/fold pairs)")
        print(f"  Pearson r (modifier vs. actual pct-change): {r:+.3f}")
        print(f"  p-value: {p:.4f}  "
              f"{'(distinguishable from noise at alpha=0.05)' if p < 0.05 else '(NOT distinguishable from noise at alpha=0.05)'}")
    print("=" * 70)

    significant = [h for h, res in results.items() if res["p"] < 0.05]
    meaningful = [h for h, res in results.items() if res["p"] < 0.05 and abs(res["r"]) >= 0.2]

    print("\nConclusion:")
    if meaningful:
        horizons_str = ", ".join(f"{h}wk (r={results[h]['r']:+.3f}, p={results[h]['p']:.4f})" for h in meaningful)
        print(f"  The macro modifier shows a statistically significant AND non-trivial correlation "
              f"with actual realized inventory pct-change at: {horizons_str}. This suggests real, if "
              f"modest, predictive signal at those horizons, independent of how it might later be combined "
              f"with AR's forecast -- worth keeping in the pipeline, though the effect size is small enough "
              f"that it likely needs a gentler combination rule than either compounding or crude additive "
              f"blending to actually improve accuracy in practice.")
    elif significant:
        horizons_str = ", ".join(f"{h}wk (r={results[h]['r']:+.3f}, p={results[h]['p']:.4f})" for h in significant)
        print(f"  The macro modifier is statistically distinguishable from noise at: {horizons_str}, but the "
              f"correlation magnitude is weak (|r|<0.2) at every horizon. There is a real but very thin signal "
              f"-- not nothing, but not strong enough on its own to expect a combination rule to meaningfully "
              f"move the needle.")
    else:
        print(f"  No horizon shows a statistically significant correlation between the macro modifier and "
              f"actual realized inventory pct-change (all p >= 0.05). This is consistent with the macro "
              f"modifier carrying little to no usable predictive signal in its current form -- the earlier "
              f"combiner results (roughly neutral-to-mildly-negative under additive combination) are best "
              f"explained by the features/lags/regression not capturing real signal, not by a bad combination "
              f"rule. Revisiting the lag assumptions or feature set is a more promising next step than further "
              f"combiner tuning.")


if __name__ == "__main__":
    main()
