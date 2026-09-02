"""Steps 6-9: serial-correlation-robust follow-up on the cpi lag-grid
result from lag_grid_search_ui.py.

Reuses build_target/drop_covid from that file (intra-domain reuse, not a
cross-domain import) so the exact same COVID-excluded target construction
is used here, not a re-derived approximation.

STEP 6 -- diagnose first, don't skip to the fix. Two complementary checks
on the fitted entity+quarter-FE model's own residuals, not an assumption
from the p-value pattern alone:
  (a) AR(1)-on-residuals test: regress resid_t on resid_{t-1} (pooled
      across states, clustered by state), test H0: coefficient == 0. This
      asks directly "are the residuals of THIS fitted model serially
      correlated" -- the exact condition Driscoll-Kraay exists to correct
      for.
  (b) Per-state lag-1 residual autocorrelation, treating the 15 states as
      independent draws (a legitimate cross-sectional sample -- one number
      per state, not reusing the same time-series residuals twice) to
      characterize how strong and how consistent the effect is, not just
      whether it's statistically present.

STEP 7 -- refit each of the 24 surviving (COVID-excluded, BH-corrected)
cpi cells with Driscoll-Kraay standard errors (linearmodels' cov_type=
"kernel", the panel-appropriate analog of Newey-West for combined
cross-sectional + serial dependence -- NOT generic HAC, which assumes a
single time series).

STEP 8/9 -- comparison table and plain verdict. Per the explicit
instruction: no model changes, no alternative specifications, no lag
search to restore significance if this collapses the result.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

from domains.ui_claims.lag_grid_search_ui import build_target, drop_covid

DATA_PATH = "domains/ui_claims/aligned_weekly_ui_claims.csv"
OUT_PATH = "domains/ui_claims/serial_correlation_check_results_ui.csv"

# The 24 cells identified in Step 7/8 of the prior investigation: COVID-
# excluded, significant, sign-stable, and BH-survived.
SURVIVING_CELLS = (
    [(4, lag) for lag in [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]]
    + [(8, lag) for lag in [4, 6, 8, 10, 12, 14, 16, 18]]
    + [(13, lag) for lag in [16, 18, 20, 22]]
)


def build_design(df: pd.DataFrame, horizon: int, lag: int) -> pd.DataFrame:
    target = build_target(df, horizon)
    nc = drop_covid(target, horizon)
    p = nc.copy()
    p["cpi_lag"] = p.groupby("state")["cpi"].shift(lag)
    p = p.dropna(subset=["target_pct_change", "cpi_lag"])
    p["year_quarter"] = p["date"].dt.to_period("Q").astype(str)
    return p


def fit_panel(p: pd.DataFrame, cov_type: str):
    quarter_dummies = pd.get_dummies(p["year_quarter"], drop_first=True, dtype=float)
    X = pd.concat([p[["cpi_lag"]].reset_index(drop=True), quarter_dummies.reset_index(drop=True)], axis=1)
    y = p["target_pct_change"].reset_index(drop=True)
    indexed = pd.concat([p[["state", "date"]].reset_index(drop=True), X, y.rename("y")], axis=1).set_index(["state", "date"])
    y_final = indexed["y"]
    X_final = indexed.drop(columns=["y"])
    model = PanelOLS(y_final, X_final, entity_effects=True, drop_absorbed=True)
    if cov_type == "clustered":
        return model.fit(cov_type="clustered", cluster_entity=True)
    return model.fit(cov_type="kernel")  # Driscoll-Kraay


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 -- diagnose serial correlation in the residuals
# ═══════════════════════════════════════════════════════════════════════════

def diagnose_serial_correlation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, lag in SURVIVING_CELLS:
        p = build_design(df, horizon, lag)
        result = fit_panel(p, "clustered")

        resid = result.resids.reset_index()
        resid.columns = ["state", "date", "resid"]
        resid = resid.sort_values(["state", "date"])
        resid["resid_lag1"] = resid.groupby("state")["resid"].shift(1)
        rr = resid.dropna(subset=["resid_lag1"])

        # (a) AR(1)-on-residuals test, pooled, clustered by state
        X = sm.add_constant(rr["resid_lag1"])
        m = sm.OLS(rr["resid"], X).fit(cov_type="cluster", cov_kwds={"groups": rr["state"]})
        ar1_coef = float(m.params["resid_lag1"])
        ar1_p = float(m.pvalues["resid_lag1"])

        # (b) per-state lag-1 ACF, states as independent draws
        state_acfs = []
        for state, g in resid.groupby("state"):
            g = g.sort_values("date")
            s = g["resid"].values
            if len(s) > 5:
                state_acfs.append(float(np.corrcoef(s[:-1], s[1:])[0, 1]))
        mean_acf = float(np.mean(state_acfs))
        # one-sample t-test: is the mean per-state ACF significantly != 0,
        # treating the 15 states as independent cross-sectional draws
        t_stat, t_p = sm_ttest_1samp(state_acfs)

        rows.append({
            "horizon": horizon, "lag": lag,
            "ar1_resid_coef": ar1_coef, "ar1_resid_p": ar1_p,
            "mean_state_lag1_acf": mean_acf, "acf_ttest_p": t_p,
            "n_states": len(state_acfs),
        })
    return pd.DataFrame(rows)


def sm_ttest_1samp(values):
    from scipy import stats
    t_stat, p = stats.ttest_1samp(values, 0.0)
    return float(t_stat), float(p)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 -- refit the 24 cells with Driscoll-Kraay SEs
# ═══════════════════════════════════════════════════════════════════════════

def refit_with_driscoll_kraay(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, lag in SURVIVING_CELLS:
        p = build_design(df, horizon, lag)
        r_clustered = fit_panel(p, "clustered")
        r_dk = fit_panel(p, "kernel")
        rows.append({
            "horizon": horizon, "lag": lag,
            "coef": float(r_clustered.params["cpi_lag"]),
            "p_clustered": float(r_clustered.pvalues["cpi_lag"]),
            "se_clustered": float(r_clustered.std_errors["cpi_lag"]),
            "p_dk": float(r_dk.pvalues["cpi_lag"]),
            "se_dk": float(r_dk.std_errors["cpi_lag"]),
            "sig_dk": bool(r_dk.pvalues["cpi_lag"] < 0.05),
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"Loaded: {len(df)} rows, {df['state'].nunique()} states")
    print(f"Testing the {len(SURVIVING_CELLS)} surviving cpi cells "
          f"(COVID-excluded, significant, sign-stable, BH-survived)\n")

    # ── STEP 6 ──────────────────────────────────────────────────────────
    print("=" * 90)
    print("STEP 6 -- diagnosing serial correlation in fitted-model residuals")
    print("=" * 90)
    diag = diagnose_serial_correlation(df)
    print(diag.to_string(index=False))
    n_ar1_sig = int((diag["ar1_resid_p"] < 0.05).sum())
    n_acf_sig = int((diag["acf_ttest_p"] < 0.05).sum())
    mean_acf_overall = float(diag["mean_state_lag1_acf"].mean())
    print(f"\nAR(1)-on-residuals significant (p<0.05): {n_ar1_sig}/{len(diag)} cells")
    print(f"Cross-state mean-ACF t-test significant (p<0.05): {n_acf_sig}/{len(diag)} cells")
    print(f"Average lag-1 residual autocorrelation across all cells: {mean_acf_overall:+.3f}")

    # ── STEP 7 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("STEP 7 -- refitting with Driscoll-Kraay standard errors")
    print("=" * 90)
    dk = refit_with_driscoll_kraay(df)
    print(dk.to_string(index=False))
    dk.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH}")

    # ── STEP 8 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("STEP 8 -- comparison table")
    print("=" * 90)
    n_survive_dk = int(dk["sig_dk"].sum())
    print(f"| CPI test              | Original (clustered) | BH corrected | Driscoll-Kraay |")
    print(f"|---|---|---|---|")
    print(f"| Significant results   | 24/24                 | 24/24        | {n_survive_dk}/24          |")
    print(f"| Median p-value        | {dk['p_clustered'].median():.2e}              | (unchanged)  | {dk['p_dk'].median():.2e}       |")
    print(f"| Smallest p-value      | {dk['p_clustered'].min():.2e}              | (unchanged)  | {dk['p_dk'].min():.2e}       |")
    print(f"| Largest p-value       | {dk['p_clustered'].max():.2e}              | (unchanged)  | {dk['p_dk'].max():.2e}       |")

    # ── STEP 9 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("STEP 9 -- verdict")
    print("=" * 90)
    frac_survive = n_survive_dk / len(dk)
    if frac_survive >= 0.8:
        verdict = "(a) significance survives -- strong evidence for a genuine cpi relationship"
    elif frac_survive >= 0.3:
        verdict = "(b) partial survival -- evidence varies by cell, not a clean confirmation or collapse"
    else:
        verdict = ("(c) mostly/fully collapses -- the clustered-SE result was a serial-correlation "
                   "artifact, a caught false positive (same category as the COVID artifact)")
    print(f"{n_survive_dk}/{len(dk)} cells ({frac_survive:.0%}) remain significant under Driscoll-Kraay SEs.")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
