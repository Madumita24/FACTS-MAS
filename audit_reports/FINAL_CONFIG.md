# FINAL_CONFIG (PROPOSED, not frozen)

Status: **PROPOSED** on 2026-10-16 at the end of Phase 5. It becomes frozen only when Umang confirms in chat. At that point the commit hash of the confirming commit is written below, and nothing in the forecasting path may change before the Phase 7 holdout run.

Frozen commit: _(to be filled at confirmation)_

## 1. What the production system is, and why

The production system **P** is the outcome of the pre-registered selection rules, nothing else:

- Phase 3 rule: C2 promoted at 8 and 13 weeks; R1c kept at 4 weeks (C2 lost on validation there).
- Authors' decision after Phase 3: event and intrinsic channels retired from the forecast path.
- Phase 5 rule: no candidate (listing flows, delta fusion, temperature and shrinkage) beat the base. The base itself (B5) did not replace P at any horizon (4 weeks: validation loss; 8 and 13 weeks: test better in only 2 of 6 folds).

Reported performance (TEST, CLEANBLEED, smoothed cut): P = 2.118 / 3.622 / 5.731. FA-ETS-d = 1.974 / 3.608 / 5.664. P is not better than FA-ETS-d at any horizon (Phase 4). **FA-ETS-d is reported as the strongest baseline.**

## 2. Data

| Item | Specification |
|---|---|
| Target | Zillow `Metro_invt_fs_uc_sfrcondo_sm_week.csv` (smoothed, a four-week trailing average per Zillow's notes), 15 MSAs as in `pull_zillow.py`, Saturday dates |
| Panel | `align_data.py` logic: macro joined by backward `merge_asof`, FRED publication lags (FFR +32 d, UNRATE +35 d, CPI +42 d), inventory forward-filled within MSA only (filled rows logged) |
| Availability convention | Inventory at date t treated as known at origin t (as in all backtests). Zillow actually posts it 7-13 days later; this is disclosed, not changed (claims C27). |
| Robustness cut | Raw `Metro_invt_fs_uc_sfrcondo_week.csv`, same processing, reported in the main results table |

## 3. Channels

| Horizon | Temporal | Seasonal | Exogenous | Retired |
|---|---|---|---|---|
| 4 weeks | **Holt**: `ar_agent.fit_and_forecast` (statsmodels ETS, additive trend, undamped, on levels), clipped at 0 | **Naive2**: y_t * f(w_{t+k}) / f(w_t) with window-7 ratio-to-moving-average factors computed at the origin (`seasonality_agent` with `SEASONAL_SMOOTHING_WINDOW = 7`) | `macro_agent` (Ridge on lagged macro levels, lags 6 / 10 / 10 / 5 weeks) | event, intrinsic |
| 8, 13 weeks | **FA-ETS-d**: divide history by the window-7 factors, ETS additive damped trend, multiply forecasts back (damped vs undamped chosen on the validation window: lower standalone validation MAPE, h 4/8/13 pooled; damped won in every fold) | same | same | event, intrinsic |

Future week index for all factor-based forecasts: ((w_t - 1 + k) mod 53) + 1, as in the seasonal agent.

## 4. Weighting (per fold; for the holdout, one fold, section 6)

1. Validation window: origins in [train_end - 104 weeks, train_end]; **a cell is used only if its last target is on or before train_end** (F0.10).
2. Skill per validation cell: 1 - MAPE_k / MAPE_naive (MAPE over the 1..H path); cells with naive MAPE 0 skipped; median per channel.
3. Softmax with **T = 0.05** (hand-set originally; Phase 5's validation-chosen temperature was not promoted).
4. Fold weights: skill pooled over all three horizons and all regimes.
5. Regime grid: national fed funds 13-week change, hiking > +0.25, cutting < -0.25, else stable; weights per (regime, horizon). **Fix 2:** a (regime, horizon) cell with fewer than 13 distinct validation weeks of that regime uses the fold weights.
6. Mode selection per horizon: fold weights vs regime weights by fused validation MAPE on the same validation cells; ties go to fold weights.
7. Fusion: weighted sum renormalised over the live channels, clipped at 0.

Note: the authors withdrew the regime-conditioning *claim* (C6). The Router mechanism is still part of P because P is the rule outcome; removing it now would be a post-hoc change. Phase 4 showed the regime labels carry no detectable information, so the paper should describe the Router only as part of the selected configuration.

## 5. Reference implementation

- Weighting and fusion: `tests_audit/audit_core.py`, pinned bit-for-bit against the original production code (`tests_audit/test_regression_pin.py`), with the options `bound_targets=True`, `min_support_weeks=13`, `agents=["ar", "seasonality", "macro"]`.
- Channel forecasts: `facts_mas/agents/*` (Holt, seasonal, macro) and `audit_reports/phase3_scripts/p3_build_baselines.py::forecasts_at` (FA-ETS-d).
- **Gap:** P currently runs only on cached cells. The project package `facts_mas/` does not implement window 7, fix 2, F0.10, FA-ETS-d or the channel retirement. Phase 7 needs an executable runner (section 7).

## 6. Proposed Phase 7 holdout protocol (for confirmation)

- Refresh the Zillow pull (inventory smoothed and raw), archive the raw downloads with date stamps and sha256, and check that refreshed values on or before 2026-02-21 equal the archived capped panels (any revision is reported, not hidden).
- **One holdout fold:** train_end = **2026-02-21**. Validation window as in section 4 (targets <= 2026-02-21). Weights, damped choice and modes computed once from that window.
- **Holdout origins: 2026-02-28 onward**, every week with a complete h-week target in the refreshed data. With data to 2026-10-03 that gives about 28 / 24 / 19 origins x 15 MSAs at 4 / 8 / 13 weeks. Option B: a 13-week embargo (origins from 2026-05-30), consistent with the backtest folds but leaving only about 15 / 11 / 6 origins. I recommend option A: the F0.10 bound already keeps validation targets before train_end, so the embargo is not needed to prevent leakage.
- Methods: naive, seasonal naive, FA-ETS-d, P; B5 as the documented equivalent alternative; TimesFM; the CoT baseline (state the LLM and its knowledge cutoff). No Phase 5 winner exists.
- Report with Phase 4 statistics (block bootstrap CIs, panel DM), both cuts. **Nothing may be changed after seeing results**; bugs are reported, not silently fixed.

## 7. Decisions needed before freezing

1. Confirm P as the frozen configuration (or state a change before freezing).
2. Holdout option A (no embargo) or B (13-week embargo).
3. Runner: (a) the audit builds `audit_reports/phase7/run_holdout.py` from the existing agent code plus `audit_core`. It must first reproduce the Phase 4 P numbers on folds 1-6 from raw agent code (not the cache) before it touches holdout data. Or (b) the authors port the configuration into `facts_mas/` (owner: Umang).
4. Environment: pinned `requirements.txt` stack (Python 3.11, numpy 2.4.6, pandas 3.0.6, statsmodels 0.15.0, scikit-learn 1.9.1). TimesFM in its own environment.
