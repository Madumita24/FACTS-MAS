# Phase 4 log: pre-registration and running notes

Written and committed BEFORE any Phase 4 number is computed (together with `phase5_preregistration.md`).

## Methods compared (per horizon)

- **P (production)** = the Phase 3 rule outcome, with event and intrinsic retired (authors, 2026-10-16): R1c protocol with channels {temporal = Holt, seasonal, macro} at 4 weeks; C2 with channels {temporal = FA-ETS-d, seasonal, macro} at 8 and 13 weeks. Naive skill reference, as in Phase 3. The effect of retiring the two channels is reported (P vs P with all five channels).
- Also: naive, FA-ETS-d, SelBest-model, C2 (5 channels, as in Phase 3), EW-TS, HZ-weights (5 channels), R1c (5 channels). For the original Phase 4 list, also seasonal naive and Naive2 (the seasonal channel).
- TimesFM: fold 1 only (the only fold it was run on), 435 cells per horizon, compared with P and FA-ETS-d on those cells.

## Loss and data

- Loss per cell = MAPE over steps 1..H. Time index = origin week. CLEANBLEED is primary (whole origin weeks are removed, because bleed is defined by date), DISCLOSURE alongside for the headline.
- All test origins are concatenated in time order: 174 weeks at 4wk; fewer at 8 and 13wk after the bleed exclusion.

## Tests (fixed now)

1. **Panel DM (Qu, Timmermann and Zhu 2024, test of equal average performance).** d_bar_t = mean over MSAs of d_{i,t} = L_A - L_B. t = mean(d_bar) / sqrt(HAC var / T), with Newey-West Bartlett and lag h - 1, compared with t_{T-1}, two-sided.
2. **Per-MSA DM:** same statistic on each MSA's d_{i,t}, with the Harvey-Leybourne-Newbold small-sample factor and lag h - 1. Benjamini-Hochberg at 5% across the 15 MSAs. Reported as counts significant in favour of A and in favour of B.
3. **Block bootstrap:** moving blocks of whole weekly cross-sections (all MSAs together), block length L = 13 weeks at every horizon (>= h), B = 5,000, seed 20261016. 95% percentile CIs for each method's MAPE and for differences vs P and vs FA-ETS-d. The iid bootstrap over cells is reported alongside to show it is too narrow. Blocks may straddle fold boundaries; noted, not corrected.
4. **Equivalence of C2 and FA-ETS-d (per horizon):** paired block bootstrap (as in 3) 90% CI of MAPE(C2) - MAPE(FA-ETS-d). Equivalence margin delta = 2% of FA-ETS-d's pooled MAPE at that horizon. Equivalent if the 90% CI lies inside (-delta, +delta) (TOST at 5%). The CI is reported so other margins can be applied. The same check is run for P vs FA-ETS-d.
5. **Model Confidence Set (Hansen, Lunde and Nason 2011):** T_max statistic on the weekly cross-section mean losses, the moving block bootstrap above (L = 13, B = 5,000). alpha = 0.10 primary, 0.25 also reported. Methods: P, naive, FA-ETS-d, SelBest-model, C2, EW-TS, HZ-weights, R1c, Naive2. Seasonal naive excluded as obviously dominated. Implemented in the audit code (`arch` is not installed; no new packages).

## Macro panel regression (facts-audit env)

- First reproduce the original spec (`validate_macro_panel.py`): 13-week inventory % change on lagged levels (mortgage 6, FFR 10, CPI 10, unemployment 5 weeks); MSA FE + year-quarter dummies; SEs clustered by MSA; sample dates and targets before 2026-01-01. Expect FFR p = 0.042. If it does not reproduce, report and continue.
- (a) MSA FE only (no year-quarter dummies), Driscoll-Kraay SEs (kernel, bandwidth = 13).
- (b) Original spec with a wild cluster bootstrap p-value (restricted, Webb six-point weights, 15 clusters, B = 9,999) for each coefficient.
- (c) 13-week changes of each macro variable (same lags) instead of levels; run under both the original FE structure and (a).
- (d) Distributed lag: FFR at lags 0..26 weeks (levels), MSA FE, Driscoll-Kraay. Report each coefficient and the sums over lags 0-12 and 13-26, with SEs. Sign-flip hypothesis: first sum > 0, second < 0.
- (e) Holm correction across the 4 variables within each of the original, (a), (b) and (c) specs.
- All specifications reported; no re-specification after seeing results.

## Regime placebo (R1c, 5 channels, as in Phases 2-3)

- (a) Block-permuted labels: the national label sequence over the 344 cached origin dates is cut into consecutive 13-week blocks, the blocks are shuffled (200 permutations, seed 20261016), and labels are re-assigned. The full R1c pipeline (fix 2, F0.10) is re-run per permutation. Reported: where the true-label 8- and 13-week MAPE falls in the placebo distribution.
- (b) Period labels: origin < 2023-01-01 = one label, otherwise another (two "regimes").
- (c) Recency-weighted skill, no regime: weighted median of pointwise skill with weights 0.5^(age / 26 weeks), per (fold, horizon); half-life 26 fixed, not tuned.
- Compared with true regimes (R1c) and HZ-weights.

## Predictions (written before running)

- D1. P beats naive at every horizon: panel DM p < 0.01; per-MSA DM significant (BH) in >= 12 of 15 MSAs.
- D2. At 4 weeks P (= R1c channels minus two) is significantly worse than FA-ETS-d (panel DM p < 0.05).
- D3. At 8 and 13 weeks C2 vs FA-ETS-d: not significantly different; equivalence holds at 8 weeks and is inconclusive at 13 weeks.
- D4. Retiring event and intrinsic changes P by under 0.02 at every horizon.
- D5. The iid bootstrap CIs are at least 2x narrower than the block bootstrap CIs.
- D6. The 90% MCS contains FA-ETS-d at every horizon; it also contains C2 and SelBest-model at 8 and 13 weeks; it excludes R1c at 8 and 13 weeks.
- D7. The original FFR p = 0.042 does not survive Holm, and the wild bootstrap p-value is above 0.05.
- D8. The distributed-lag sign-flip pattern is not significant (at least one of the two lag sums has |t| < 2).
- D9. True regime labels are not better than 90% of the placebo permutations at 13 weeks; period labels and recency weighting are within 0.3 of true regimes at 13 weeks.
- D10. TimesFM on fold 1 is significantly worse than FA-ETS-d at 8 and 13 weeks.

## Running notes

(appended as runs complete)
