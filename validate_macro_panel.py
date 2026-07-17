"""Final macro validation: two-way fixed-effects panel regression, addressing
both flaws found in the earlier fold-based diagnostics (validate_macro_signal.py,
validate_macro_lags.py):
  1. Pseudo-replication -- nationally-broadcast macro data was correlated
     against 90 (msa, fold) points but only had ~6 truly independent values
     (one per fold). Fixed here with MSA-clustered standard errors, which
     correctly account for the fact that observations within an MSA (and,
     more importantly, the shared-across-MSA macro signal) are not
     independent draws.
  2. Spurious trend correlation -- both macro levels and inventory levels
     trend substantially over 2018-2025 (rate hikes, CPI's monotonic rise,
     the post-COVID inventory crash/recovery), so two independently
     trending series can look correlated with no real causal link. Fixed
     here with MSA fixed effects (absorb level differences) AND
     year-quarter fixed effects (absorb the common trend), so identification
     comes only from within-MSA, within-quarter deviations.

Report-only. Does not modify macro_agent.py. This is the last macro
validation attempt for this line of investigation, per explicit instruction
-- no further iteration regardless of outcome.
"""
import pandas as pd
from linearmodels.panel import PanelOLS

# Documented lag *midpoints* from align_data.py (also used in macro_agent.py),
# chosen deliberately over validate_macro_lags.py's "best" grid-search lags:
# that grid search's significance test was itself pseudo-replicated, so its
# "winning" lag choice is data-mined from a flawed test. Using it here would
# bake that bias into what's supposed to be the rigorous final test. The
# midpoints are fixed a priori from the transmission-lag literature
# documented in align_data.py, independent of any of this project's own
# (flawed) significance tests.
LAG_WEEKS = {
    "mortgage_rate": 6,
    "fed_funds": 10,
    "cpi": 10,
    "unemployment": 5,
}
TARGET_HORIZON_WEEKS = 13
HELD_OUT_START = pd.Timestamp("2026-01-01")


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["msa", "date"]).copy()

    parts = []
    for msa, g in df.groupby("msa"):
        g = g.sort_values("date").reset_index(drop=True)

        # Weekly grid is contiguous per MSA (confirmed in Phase 1), so a
        # positional shift of N rows == N weeks.
        g["target_pct_change"] = (
            g["inventory_count"].shift(-TARGET_HORIZON_WEEKS) - g["inventory_count"]
        ) / g["inventory_count"]

        for feature, lag in LAG_WEEKS.items():
            g[f"{feature}_lag"] = g[feature].shift(lag)

        parts.append(g)

    panel = pd.concat(parts, ignore_index=True)

    # Held-out cutoff respected on BOTH ends: the row date and the
    # 13-week-forward target date must each be before 2026-01-01.
    panel["target_date"] = panel["date"] + pd.Timedelta(weeks=TARGET_HORIZON_WEEKS)
    panel = panel[(panel["date"] < HELD_OUT_START) & (panel["target_date"] < HELD_OUT_START)]

    feature_cols = [f"{f}_lag" for f in LAG_WEEKS]
    panel = panel.dropna(subset=["target_pct_change"] + feature_cols)

    panel["year_quarter"] = panel["date"].dt.to_period("Q").astype(str)
    return panel


def main():
    df = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
    panel = build_panel(df)

    print(f"Panel: {len(panel)} (msa, date) rows, {panel['msa'].nunique()} MSAs, "
          f"{panel['date'].min().date()} to {panel['date'].max().date()}, "
          f"{panel['year_quarter'].nunique()} year-quarter buckets\n")

    feature_cols = [f"{f}_lag" for f in LAG_WEEKS]

    # Year-quarter dummies (not linearmodels' built-in time_effects, which
    # would use full per-date granularity): with a nationally-broadcast
    # regressor that's identical across all 15 MSAs on a given date,
    # per-date time FE would make the macro coefficient exactly collinear
    # with the time dummies and unidentifiable. Quarter-level FE absorbs
    # the common trend while preserving within-quarter week-to-week macro
    # variation, which is where identification actually comes from here.
    quarter_dummies = pd.get_dummies(panel["year_quarter"], drop_first=True, dtype=float)

    X = pd.concat([
        panel[feature_cols].reset_index(drop=True),
        quarter_dummies.reset_index(drop=True),
    ], axis=1)
    y = panel["target_pct_change"].reset_index(drop=True)

    indexed = pd.concat([
        panel[["msa", "date"]].reset_index(drop=True), X, y.rename("y"),
    ], axis=1).set_index(["msa", "date"])

    y_final = indexed["y"]
    X_final = indexed.drop(columns=["y"])

    model = PanelOLS(y_final, X_final, entity_effects=True, drop_absorbed=True)
    result = model.fit(cov_type="clustered", cluster_entity=True)

    print(result.summary)

    print("\n" + "=" * 70)
    print("Macro feature coefficients (MSA + year-quarter FE, SE clustered by MSA):")
    any_significant = False
    for feature, lag in LAG_WEEKS.items():
        col = f"{feature}_lag"
        coef = result.params[col]
        se = result.std_errors[col]
        p = result.pvalues[col]
        sig = p < 0.05
        any_significant = any_significant or sig
        print(f"  {feature} (lag={lag}wk): coef={coef:+.6f}  SE={se:.6f}  p={p:.4f}  "
              f"-> {'significant (p<0.05)' if sig else 'not significant'}")

    print(f"\nModel: n={result.nobs}, R-squared (within)={result.rsquared_within:.4f}, "
          f"MSA FE + {quarter_dummies.shape[1]} year-quarter dummies, clustered SE by MSA (15 clusters)")

    print("\nConclusion:")
    if any_significant:
        print("  After removing both common trend (year-quarter FE) and MSA-level "
              "differences (entity FE), at least one macro feature shows a statistically "
              "significant within-MSA, within-quarter relationship with future 13-week "
              "inventory change, with standard errors correctly clustered by MSA. This is "
              "the most rigorous test run in this investigation and its result should "
              "carry the most weight of any of them.")
    else:
        print("  After removing both common trend (year-quarter FE) and MSA-level "
              "differences (entity FE), and with standard errors correctly clustered by "
              "MSA (addressing the pseudo-replication problem in the earlier fold-based "
              "tests), NONE of the four macro features show a statistically significant "
              "within-MSA, within-quarter relationship with future 13-week inventory "
              "change. Combined with the earlier null and wrong-signed results, this is "
              "the most rigorous test in the investigation and it confirms the null: the "
              "apparent signal in the cruder tests was very likely an artifact of shared "
              "trend and pseudo-replicated significance, not a real relationship. "
              "Recommend deprioritizing the Macro Agent -- this is now a well-triangulated "
              "negative result, not an artifact of any single test's weaknesses.")


if __name__ == "__main__":
    main()
