"""Lag grid search for the UI-claims Macro agent -- resolves the two-
hypothesis ambiguity left open in pilot_results_ui.md Section 5: is
Macro's 0/15 null a lag-specification problem (fixable, some lag shows
real signal) or a genuine no-signal domain fact (matches housing)?

Adapted from (not imported from) validate_macro_panel.py's methodology --
the same rigor, not the naive pooled-correlation approach that project's
OWN earlier rounds (validate_macro_signal.py, validate_macro_lags.py) used
and then had to retract. Nationally-broadcast fed_funds/cpi are identical
across all 15 states on a given date, so a naive pooled correlation across
15 states x ~700 weeks would face the exact pseudo-replication problem
housing's Round 2 found: true n is ~weeks, not states x weeks. Fixed here
the same way housing's Round 3 fixed it:
    - STATE fixed effects (entity_effects=True) absorb level differences
      between states (California's ~60K/week mean vs. Colorado's ~3.6K).
    - YEAR-QUARTER dummies (not per-date time effects) absorb the common
      trend without making the broadcast macro regressor collinear with
      per-date dummies -- identical reasoning to validate_macro_panel.py.
    - Standard errors CLUSTERED BY STATE (15 clusters) correct for the
      non-independence a naive OLS SE would understate.

Unlike housing's fixed-midpoint validation, this genuinely sweeps lag --
4 to 26 weeks by 2, well beyond the fixed 10wk currently in
macro_agent_ui.py -- because the hypothesis under test is specifically
that monetary-policy transmission to labor markets may take months, not
weeks, longer than housing's own 8-12wk assumption.

Each (horizon, feature, lag) cell is its own single-feature panel
regression (not fed_funds and cpi jointly) -- the question here is which
lag, if any, shows a real relationship for EACH feature independently, not
a combined model's overall fit.

Uses the full available date range (2013-2026), not restricted to the
pilot's train-window split -- same choice validate_macro_panel.py made:
this is a standalone signal-existence investigation, not part of the
operational forecast pipeline that must respect the train/test embargo.

COVID ROBUSTNESS CHECK -- added after the first full sweep, not planned
upfront. The first pass over the full 2013-2026 panel found 55/72 cells
significant at p<0.05, coefficients 10-100x larger than anything housing's
investigation ever found, AND fed_funds's sign flipping between adjacent
lags within the same horizon -- a pattern that should not appear in front
of a genuine, stable causal relationship. Unlike housing inventory, which
never moved more than a few percent in any week even during COVID, this
domain's claims target swung 20-46x its baseline in April 2020 (Step 1
sanity check). A target with an outlier that extreme can dominate an OLS-
type fit even with fixed effects, independent of whether any real
relationship exists. Every cell is therefore fit twice: once on the full
panel, once excluding rows where the origin date OR the horizon-forward
target date falls inside 2020-02-01 to 2021-12-31 (the acute spike plus
the extended-benefits wind-down window, matching evaluation_protocol_ui.md's
own COVID discussion). The COVID-excluded fit is the one the verdict is
actually based on; the with-COVID numbers are reported alongside for
transparency about how much of the apparent effect they explain.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

DATA_PATH = "domains/ui_claims/aligned_weekly_ui_claims.csv"
OUT_PATH = "domains/ui_claims/lag_grid_search_results_ui.csv"

HORIZONS = [4, 8, 13]
FEATURES = ["fed_funds", "cpi"]
LAGS = list(range(4, 27, 2))  # 4, 6, 8, ..., 26 weeks

COVID_START = pd.Timestamp("2020-02-01")
COVID_END = pd.Timestamp("2021-12-31")


def build_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """One row per (state, date) with that horizon's realized forward
    pct-change in claims_count. Positional shift == weeks since the grid
    is contiguous per state (confirmed at alignment time: 0 rows
    forward-filled)."""
    parts = []
    for state, g in df.groupby("state"):
        g = g.sort_values("date").reset_index(drop=True)
        g["target_pct_change"] = (
            g["claims_count"].shift(-horizon) - g["claims_count"]
        ) / g["claims_count"]
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def drop_covid(target_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Drop rows where the origin date OR the horizon-forward target date
    falls inside the COVID window -- see module docstring."""
    target_date = target_df["date"] + pd.Timedelta(weeks=horizon)
    in_covid = (
        ((target_df["date"] >= COVID_START) & (target_df["date"] <= COVID_END))
        | ((target_date >= COVID_START) & (target_date <= COVID_END))
    )
    return target_df[~in_covid]


def fit_one(panel: pd.DataFrame, feature: str, lag: int) -> dict:
    p = panel.copy()
    p[f"{feature}_lag"] = p.groupby("state")[feature].shift(lag)
    p = p.dropna(subset=["target_pct_change", f"{feature}_lag"])

    p["year_quarter"] = p["date"].dt.to_period("Q").astype(str)
    quarter_dummies = pd.get_dummies(p["year_quarter"], drop_first=True, dtype=float)

    X = pd.concat([
        p[[f"{feature}_lag"]].reset_index(drop=True),
        quarter_dummies.reset_index(drop=True),
    ], axis=1)
    y = p["target_pct_change"].reset_index(drop=True)

    indexed = pd.concat([
        p[["state", "date"]].reset_index(drop=True), X, y.rename("y"),
    ], axis=1).set_index(["state", "date"])

    y_final = indexed["y"]
    X_final = indexed.drop(columns=["y"])

    model = PanelOLS(y_final, X_final, entity_effects=True, drop_absorbed=True)
    result = model.fit(cov_type="clustered", cluster_entity=True)

    col = f"{feature}_lag"
    return {
        "coef": float(result.params[col]),
        "se": float(result.std_errors[col]),
        "p": float(result.pvalues[col]),
        "n": int(result.nobs),
        "r2_within": float(result.rsquared_within),
    }


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"Loaded: {len(df)} rows, {df['state'].nunique()} states, "
          f"{df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Sweeping lags {LAGS} for features {FEATURES}, horizons {HORIZONS}\n")

    rows = []
    for horizon in HORIZONS:
        target_df = build_target(df, horizon)
        target_nc = drop_covid(target_df, horizon)
        for feature in FEATURES:
            for lag in LAGS:
                try:
                    res_full = fit_one(target_df, feature, lag)
                    res_nc = fit_one(target_nc, feature, lag)
                except Exception as e:
                    print(f"  h={horizon} {feature} lag={lag}: FAILED ({type(e).__name__}: {e})")
                    continue
                row = {"horizon": horizon, "feature": feature, "lag": lag}
                row.update({f"{k}_with_covid": v for k, v in res_full.items()})
                row.update({f"{k}_no_covid": v for k, v in res_nc.items()})
                rows.append(row)
                print(f"  h={horizon:2d}wk {feature:10s} lag={lag:2d}wk  "
                      f"WITH-COVID coef={res_full['coef']:+.4f} p={res_full['p']:.4f}   "
                      f"NO-COVID coef={res_nc['coef']:+.4f} p={res_nc['p']:.4f}")

    rdf = pd.DataFrame(rows)
    rdf.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH}: {len(rdf)} rows (columns suffixed _with_covid / _no_covid)")

    print("\n" + "=" * 90)
    print("BEST LAG PER (horizon, feature), by lowest NO-COVID p-value")
    print("=" * 90)
    for horizon in HORIZONS:
        for feature in FEATURES:
            sub = rdf[(rdf.horizon == horizon) & (rdf.feature == feature)]
            if sub.empty:
                continue
            best = sub.loc[sub["p_no_covid"].idxmin()]
            n_sig_nc = int((sub["p_no_covid"] < 0.05).sum())
            n_sig_wc = int((sub["p_with_covid"] < 0.05).sum())
            print(f"h={horizon:2d}wk {feature:10s}: best lag={int(best.lag):2d}wk  "
                  f"NO-COVID coef={best.coef_no_covid:+.4f} p={best.p_no_covid:.4f}  "
                  f"({n_sig_nc}/{len(sub)} lags significant NO-COVID vs. "
                  f"{n_sig_wc}/{len(sub)} WITH-COVID)")

    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)
    total_tests = len(rdf)
    n_sig_wc = int((rdf["p_with_covid"] < 0.05).sum())
    n_sig_nc = int((rdf["p_no_covid"] < 0.05).sum())
    expected_false_positives = total_tests * 0.05
    # A cell counts as a stable signal only if it's significant with COVID
    # excluded AND the coefficient didn't flip sign when COVID was removed
    # -- a real relationship shouldn't reverse direction depending on
    # whether a handful of extreme weeks are in the sample.
    same_sign = np.sign(rdf["coef_with_covid"]) == np.sign(rdf["coef_no_covid"])
    stable_signal = rdf["p_no_covid"] < 0.05
    n_stable = int((stable_signal & same_sign).sum())

    print(f"Total (horizon, feature, lag) cells tested: {total_tests}")
    print(f"Significant WITH COVID included:  {n_sig_wc}/{total_tests} "
          f"(expected by chance: ~{expected_false_positives:.1f})")
    print(f"Significant with COVID EXCLUDED:  {n_sig_nc}/{total_tests}")
    print(f"Significant AND sign-stable after excluding COVID: {n_stable}/{total_tests}")

    if n_stable <= expected_false_positives * 1.5:
        print("\nThe large with-COVID signal (55/72 cells significant in the first pass) "
              "does NOT survive removing the 2020-2021 shock window, and what little "
              "remains is not sign-stable between the with/without-COVID fits -- the "
              "same instability pattern (a coefficient reversing sign depending on "
              "which nearby lag or which few extreme weeks are included) that housing's "
              "own investigation treated as disqualifying, not confirmatory. This "
              "resolves the Section 5 ambiguity toward hypothesis (b): a genuine "
              "no-signal domain fact for fed_funds/cpi against claims, not a lag-"
              "specification problem. The apparent 'signal' in the naive full-history "
              "sweep was COVID acting as a small number of extreme-leverage "
              "observations, not evidence of a real relationship at any lag. Widening "
              "the lag range did not surface real signal that a mis-specified 10wk lag "
              "was hiding.")
    else:
        print("\nA meaningful number of cells remain significant AND sign-stable even "
              "after excluding the COVID window -- this favors hypothesis (a): a real "
              "relationship exists at a lag other than the fixed 10wk currently used. "
              "Worth updating macro_agent_ui.py's LAG_WEEKS deliberately, with the same "
              "theoretical-plausibility check housing's Round 3 applied before trusting "
              "any single significant cell.")

    apply_bh_correction(rdf, same_sign)


# ═══════════════════════════════════════════════════════════════════════════
# Multiple-testing correction -- reuses facts_mas.factor_scoring._apply_fdr
# ═══════════════════════════════════════════════════════════════════════════

def apply_bh_correction(rdf: pd.DataFrame, same_sign: pd.Series) -> None:
    """
    Benjamini-Hochberg correction on the COVID-excluded cpi cells, reusing
    the exact BH implementation already built, tested, and fixed for a real
    bug in facts_mas/factor_scoring.py (Mod B's Granger screen) -- not a
    reimplementation. That function operates on GrangerResult objects, so
    each cpi (horizon, lag) cell is wrapped in one: source/target carry the
    (horizon, lag) identity since they're otherwise unused for this
    purpose, p_value is the NO-COVID p-value, and the family being
    corrected across is all 36 cpi cells (12 lags x 3 horizons) -- not just
    the 24 that were pre-filtered as significant+stable, since BH must see
    the whole tested family or the correction is invalid.

    fed_funds is not run through this: its raw surviving signal (2/36
    cells) was already at the level chance alone predicts, so a correction
    would not change the reading of it.
    """
    import sys
    sys.path.insert(0, ".")
    from facts_mas.factor_scoring import GrangerResult, _apply_fdr

    cpi = rdf[rdf.feature == "cpi"].copy()
    wrapped = [
        GrangerResult(
            source=f"h{int(r.horizon)}wk", target=f"lag{int(r.lag)}wk",
            best_lag=int(r.lag), p_value=float(r.p_no_covid),
            significant=False, n_obs=0,
        )
        for r in cpi.itertuples()
    ]
    corrected = _apply_fdr(wrapped)

    cpi["bh_significant"] = [c.significant for c in corrected]
    cpi["stable"] = same_sign[rdf.feature == "cpi"].values

    n_before = int((cpi["p_no_covid"] < 0.05).sum())
    n_before_stable = int(((cpi["p_no_covid"] < 0.05) & cpi["stable"]).sum())
    n_after = int(cpi["bh_significant"].sum())
    n_after_stable = int((cpi["bh_significant"] & cpi["stable"]).sum())

    print("\n" + "=" * 90)
    print("BENJAMINI-HOCHBERG CORRECTION on cpi cells (reusing facts_mas.factor_scoring._apply_fdr)")
    print("=" * 90)
    print(f"Family size (all cpi lag/horizon cells tested): {len(cpi)}")
    print(f"Raw significant (p<0.05), COVID excluded:            {n_before}/{len(cpi)}")
    print(f"Raw significant AND sign-stable (the '24' from before): {n_before_stable}/{len(cpi)}")
    print(f"BH-corrected significant:                            {n_after}/{len(cpi)}")
    print(f"BH-corrected significant AND sign-stable:            {n_after_stable}/{len(cpi)}")

    print("\nSurviving cells after BH correction:")
    surv = cpi[cpi["bh_significant"]][["horizon", "lag", "coef_no_covid", "p_no_covid", "stable"]]
    print(surv.to_string(index=False) if not surv.empty else "  (none)")

    print("\nFINAL VERDICT ON cpi:")
    if n_after_stable == 0:
        print("  NONE of the previously-significant, sign-stable cpi cells survive "
              "Benjamini-Hochberg correction across the 36-cell cpi family. The "
              "uncorrected '24 of 36' count was multiple-testing inflation from "
              "highly autocorrelated adjacent lags, not 24 independent confirmations. "
              "This resolves cpi to hypothesis (b): null, matching fed_funds and "
              "matching housing's own well-triangulated macro null. Neither macro "
              "feature shows a real, correctable relationship with claims at any "
              "lag tested -- the pilot's Macro agent null is a genuine domain fact, "
              "not a fixable lag-specification problem.")
    elif n_after_stable < n_before_stable * 0.5:
        print(f"  Most (not all) of the previously-significant cpi cells collapse under "
              f"correction ({n_before_stable} -> {n_after_stable}). Leans toward hypothesis "
              f"(b) -- the uncorrected count substantially overstated the evidence -- but "
              f"a small residual signal remains and isn't fully ruled out.")
    else:
        print(f"  Most of the previously-significant, sign-stable cpi cells "
              f"({n_before_stable} -> {n_after_stable}) survive Benjamini-Hochberg "
              f"correction. This is real evidence for hypothesis (a): a genuine, modest "
              f"cpi relationship, not just multiple-testing noise.")


if __name__ == "__main__":
    main()
