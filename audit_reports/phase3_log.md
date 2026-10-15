# Phase 3 log: pre-registration and running notes

Written and committed BEFORE any Phase 3 number is computed.

## Protocol (fixed)

- Baseline **R1c** = R1 channels (seasonal window 7) + fix 2 (regime cell uses fold weights if < 13 distinct validation weeks) + F0.10 (validation cells kept only if last target <= train_end). Adaptive Router, median skill, softmax T = 0.05 (PROVISIONAL).
- Primary column CLEANBLEED (165 fold-6 bleed cells excluded); DISCLOSURE alongside.
- Every selection is made on validation only, nested by fold: fold f uses validation cells with origin in [train_end - 104 weeks, train_end] and last target <= train_end.
- Seasonal factors for every factor-based method are the window-7 factors computed at the forecast origin from data on or before it (identical to the R1 seasonal channel). Future week index follows the seasonal agent's convention ((week_t - 1 + k) mod 53) + 1.

## A. Baselines (definitions fixed now)

| Name | Definition |
|---|---|
| naive | last value (cache) |
| snaive | y at t + k - 52 rows (always <= t for k <= 13) |
| Naive2 | y_t / f(week_t) x f(week_{t+k}) with window-7 factors. **Identical to the R1 seasonal channel by construction**; reported to confirm. |
| Holt | ETS additive trend, undamped, on levels (= the temporal channel; cached AR values used, parity-checked on a sample) |
| Holt-d | ETS additive damped trend, on levels |
| FA-ETS | divide history by window-7 factors, ETS additive trend undamped, multiply forecasts back |
| FA-ETS-d | same with damped trend |
| Theta | statsmodels ThetaModel (deseasonalize=False) on the factor-adjusted series, reseasonalized (the standard M4 Theta deseasonalizes first; we use our leakage-safe factors) |
| EW5 | equal weights over the five R1 channels |
| EW-TS | equal weights over temporal + seasonal |
| SelBest-channel | per (fold, h): the one R1 channel with the lowest standalone validation MAPE |
| SelBest-model | per (fold, h): the one standalone model with the lowest validation MAPE among {naive, snaive, Naive2, Holt, Holt-d, FA-ETS, FA-ETS-d, Theta, macro, intrinsic}. **This is the "best single standalone model (validation-selected)" in the report's main question.** |
| HZ-weights | R1 channels; per (fold, h) median skill over validation cells at h pooled over regimes, softmax T = 0.05; no Router |
| ORACLE | best R1 channel and best standalone model by pooled test MAPE per horizon. Labelled ORACLE; not a competitor. |

ETS / Theta fits are horizon-independent, so each is fitted once per (MSA, origin) and sliced to 4 / 8 / 13. All forecasts clipped at 0.

## A2. Raw vs smoothed Zillow robustness

- Download `Metro_invt_fs_uc_sfrcondo_week.csv` (raw) and, for a same-vintage comparison, the current `..._sm_week.csv` (smoothed), each into its own new empty directory outside the repo. Log sha256 and download time.
- Read only the header first, then load only date columns <= 2026-02-21 (`usecols`), so nothing after the cap is parsed. Assert the cap.
- Same 15 MSAs, forward-fill within MSA as `align_data.py` does, log filled rows.
- Run naive, temporal (Holt), seasonal (window 7, not re-selected), and FACTS-MAS R1c with all five channels recomputed on the raw series (macro on raw pct changes, intrinsic on raw averages, event = naive), same origins and folds.
- Prediction: every method's MAPE is higher on raw (noisier targets); naive suffers most at 4 weeks; FACTS-MAS still beats naive at all horizons; method ranking unchanged.

## B. Candidate improvements (max 3, no 4th)

- **C1**: temporal channel := Holt-d or Holt, chosen per fold on validation.
- **C2**: temporal channel := FA-ETS-d or FA-ETS, chosen per fold on validation. Seasonal channel unchanged.
- **C3**: C2 + HZ-weights (no Router).
- **Damped choice rule**: per fold, the variant with lower standalone validation MAPE of the new temporal channel, horizons 4 / 8 / 13 pooled, targets <= train_end. Tie: undamped.
- Then rerun the full R1c pipeline (fix 2, F0.10, T = 0.05, adaptive Router except C3) with the new temporal values in that fold's validation and test cells.
- Report per candidate: test MAPE per fold and pooled (CLEANBLEED + DISCLOSURE), per-MSA wins vs R1c, mean channel weights, validation MAPE.
- **Promotion rule (per horizon)**: promoted only if (a) it wins on validation, defined as: the mean over folds of each fold's fused validation MAPE (with that fold's chosen weighting) is lower than R1c's at that horizon; AND (b) its test MAPE is lower than R1c's in at least 5 of 6 folds at that horizon (CLEANBLEED). Otherwise reported and R1c kept. Every improvement labelled UNCONFIRMED until Phase 4.
- If two candidates qualify at a horizon, the one with lower validation MAPE is promoted (decided on validation, not test).

## Expectations (written before running)

Umang's: C2 gives the biggest gain at 8 / 13 weeks; C1 fixes most of the 13-week temporal loss in folds 3-6; FA-ETS (STL-ETS) standalone is close to or better than R1c at 13 weeks.

Mine:
- E1. Naive2 equals the seasonal channel exactly.
- E2. snaive is worse than naive at every horizon (large level shifts 2021-2023 make year-ago values poor).
- E3. Holt-d beats Holt at 13 weeks pooled, and validation chooses damped in at least 4 of 6 folds. Holt-d standalone at 13 weeks beats naive in more folds than Holt (Holt beats naive only in folds 1-2).
- E4. FA-ETS-d is the best standalone model at 8 and 13 weeks; at 13 weeks it is within 0.3 of R1c (either side). At 4 weeks Holt or FA-ETS beats Naive2.
- E5. Theta lies between Naive2 and FA-ETS-d at 8 and 13 weeks.
- E6. EW5 is worse than R1c at 4 weeks by more than 0.5 (three naive-like channels drag it); at 13 weeks it is within 0.5 of R1c.
- E7. EW-TS beats R1c at 8 weeks (R1c's 8-week temporal weight is 0.59 while the seasonal channel is the better 8-week channel).
- E8. HZ-weights are within 0.15 of R1c at 13 weeks (Phase 2 suggested the Router's 13-week value is mostly horizon conditioning); at 8 weeks within 0.15 as well.
- E9. C1 improves the fused 13-week MAPE by under 0.2 (temporal weight at 13 weeks is only 0.26), and improves the temporal channel standalone in at least 3 of folds 3-6.
- E10. C2 improves fused 4- and 8-week MAPE; at 13 weeks the factor-adjusted temporal channel is highly correlated with the seasonal channel, so the gain is smaller. I expect C2 to fail the 5-of-6 promotion rule at 13 weeks and possibly pass at 4 or 8 weeks.
- E11. C3 within 0.1 of C2 at every horizon.
- E12. Main question: at 4 weeks some FACTS-MAS variant beats SelBest-model; at 13 weeks I am unsure, leaning "no" (FA-ETS-d close).

## Running notes

(appended as runs complete)
- 2026-10-15 downloads (A2): raw `Metro_invt_fs_uc_sfrcondo_week.csv` sha256 18dd108cf82a928efed130039b6a5bed77a9cd5c24f5470c203744f9a0f82e42; smoothed `..._sm_week.csv` sha256 93fdb5266281be0f05fde3f148440c0bfc6cad5cec49b9e736ce8d858d74a548; downloaded 2026-10-15T23:15:26Z into new empty dirs outside the repo. 32 post-cap date columns never parsed. Current smoothed == July smoothed on all 6,315 capped rows (no revisions). Raw vs smoothed: median |diff| 2.0%, p95 8.8%, max 34.6%. Raw has 20 grid gaps, forward-filled and logged.
- 2026-10-15 builder self-checks: Naive2 == R1 seasonal to 0.005 units (2-dp rounding in the agent) (E1 CONFIRMED). Holt refit == cached AR (1.3e-5).
- 2026-10-15 `p3_analysis.py` outcomes (TEST CLEANBLEED):
  - FA-ETS-d is the best method at every horizon: 1.974 / 3.608 / 5.664 vs R1c 2.122 / 4.468 / 6.987. Validation selects it as SelBest-model in 4/6 folds at 4wk and 6/6 at 8 and 13wk. Leakage test (`tests_audit/test_baselines_leakage.py`: truncation, negative control, parity with stored output) passes.
  - E2 CONFIRMED (snaive 20.8 / 20.5 / 20.6, far worse than naive). E3 CONFIRMED (damped chosen 6/6 folds; Holt-d beats naive at 13wk in 5 folds vs 2 for Holt). E4 PARTLY WRONG: FA-ETS-d best at 8 and 13 (yes) but 1.32 better than R1c at 13wk, not within 0.3. E5 WRONG: Theta slightly worse than Naive2 at 8 and 13. E6 PARTLY WRONG: EW5 1.15 worse at 4wk (yes) but 1.32 worse at 13wk (predicted within 0.5). E7 CONFIRMED (EW-TS 4.120 < R1c 4.468 at 8wk). E8 PARTLY: HZ-weights equal R1c at 8wk (4.468) and are 0.161 better at 13wk (just outside the 0.15 band); 2.356 vs 2.122 at 4wk. E9 WRONG: C1 improves fused 13wk by 0.48, not < 0.2. E10 WRONG: C2's largest gain is at 13wk and it passes the promotion rule at 8 and 13; it fails at 4wk on validation. E11 CONFIRMED. E12 CONFIRMED (4wk: C2/C3 beat SelBest-model, R1c does not; 13wk: no variant beats it).
  - Umang's expectations: C2 biggest gain at 8/13 CONFIRMED. C1 fixes most of the 13wk temporal loss in folds 3-6: standalone Holt-d beats naive at 13wk in folds 3, 5, 6 but not 4 (3 of 4). FA-ETS standalone close to or better than R1c at 13wk: better by 1.32.
  - Promotion: C2 promoted at 8 and 13 weeks (val win, 6/6 folds, lower validation MAPE than C3 and C1). C1 also qualified at 13wk (5/6 folds) but C2 has lower validation MAPE. No candidate promoted at 4wk (all fail the validation condition). UNCONFIRMED until Phase 4.
- Addition beyond the pre-registration (decided before running, robustness only, no selection): FA-ETS-d standalone and C2 also run on the raw cut, because they are the leading methods. They are reported in the A2 section and do not feed any choice.
