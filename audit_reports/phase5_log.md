# Phase 5 log: running notes

Pre-registration: `phase5_preregistration.md` (main text committed in `fc0c7c8` before any Phase 4 result; Addendum A for P5a committed in `4025af4` before any flow data was downloaded or used).

## Step 0: raw-cut statistics (reporting only, no selection): Umang's instruction after Phase 4

- `p5_raw_stats.py`, Phase 4 statistics on the raw cut (CLEANBLEED):
  - P vs naive: significant at every horizon (DM p <= 0.0001; BH 13 / 14 / 14 MSAs).
  - P vs FA-ETS-d: 4wk +0.240 (p = 0.003, FA-ETS-d better in 4 MSAs after BH); 8wk +0.055 (p = 0.33); 13wk +0.025 (p = 0.75). Not equivalent within 2% at any horizon (CIs too wide at 8 / 13).
  - P vs seasonal channel (Naive2): 4wk +0.015 (p = 0.83); 8wk -0.249 (p = 0.003); 13wk -0.300 (p = 0.011).
  - R1c vs Naive2: no significant difference at any horizon. Holt vs naive: no significant difference at any horizon.
  - 90% MCS: 4wk {FA-ETS-d, C2, P, Naive2} (the last three at p = 0.100); 8wk {FA-ETS-d, P, C2}; 13wk {FA-ETS-d, P, C2, R1c, Naive2}.

## P5a data (Addendum A)

- Files (smoothed weekly metro), downloaded 2026-10-16T06:44:45Z into new empty directories outside the repo:
  - `Metro_new_listings_uc_sfrcondo_sm_week.csv` sha256 2ffc894fac9514e277550509cf9674392d33e2fa8ba456e478771c067ed7eac6
  - `Metro_new_pending_uc_sfrcondo_sm_week.csv` sha256 66875b652ad6405b6e6d9ebe1db07cd924acce8a56c602918e57ea7580778718
  - `Metro_perc_listings_price_cut_uc_sfrcondo_sm_week.csv` sha256 edb3b68de61d60dff78d50673de2e7903ea70d179a2c34eaee5e826e7156c344 (exists at weekly metro level, so it is used; item 8)
  - Capped output `audit_reports/artifacts/zillow_flows_sm_vintage20261016_capped20260221.csv` sha256 e4fa3a76f8bdfd3e34849c30478f6147f2c7cfc71dea853aa3ccbc117ee2b6e5. 32 post-cap columns never parsed in each file.
- History start: every series has data from 2018-01-27 for all 15 MSAs, so none starts after 2019-02-02 and **all 6 folds are eligible**. Interior gaps forward-filled from past values only (new listings up to 29 weeks for Riverside, new pending up to 30 for San Francisco, price cut up to 5 for Miami); counts per MSA in the capping log.
- **Publication timing and lag L.** Zillow's current release schedule could not be confirmed (zillow.com/research/data returns 403 to automated fetch; the only documentation found is from 2020 weekly press releases, which were issued the Friday after the Saturday week-end, about 6 days, and which state that history is revised regularly). The downloaded files end on 2026-10-03, the same date as the inventory file, and were downloaded 12-13 days later, so a week ending Saturday S is posted somewhere between S + 7 and S + 13 days. A 1-week lag is therefore not reliably safe in real time. Per Addendum A item 2 (timing not confirmed): **L = 2 weeks**.
- Same 2020 documentation states that total for-sale listings (the target) is a **four-week trailing average**, and new listings / newly pending are seven-day trailing averages of daily counts. This confirms the earlier inference about the smoothed target (F0.13).
- Observation for all methods (not changed): the backtest treats inventory at date t as known at origin t, but Zillow posts it 7-13 days later. Every method, including naive, gets this advantage equally.
- Leakage test `tests_audit/test_flow_leakage.py`: 3 passed (truncation identical; corrupting flows dated after origin - 2 weeks changes nothing; corrupting the flow dated origin - 2 weeks changes the forecast).

## Outcomes (`p5_analysis.py`, `p5_b5_vs_p_4wk.py`; TEST CLEANBLEED unless stated)

- B5 = 1.977 / 3.617 / 5.700; FA-ETS-d = 1.974 / 3.608 / 5.664. B5 gives FA-ETS-d 0.996 / 0.987 / 0.960 of the weight. B5 vs FA-ETS-d: not significant (DM p = 0.22 / 0.35 / 0.18). Expectation "B5 within 0.05 of FA-ETS-d" CONFIRMED.
- **P5a (listing flows): not promoted.** 1.982 / 3.648 / 5.755. Validation win at 8 and 13 weeks (by 0.004 and 0.016), test better than B5 in 3 / 3 / 2 folds. Flow channel alone: 2.931 / 4.861 / 7.122, worse than FA-ETS-d by about 1 point (DM p <= 0.0001). Weight on the flow channel 0.008 / 0.023 / 0.041. Expectation "small gain at 4 / 8 weeks" WRONG (small loss); "low probability of promotion" CONFIRMED. Null reported, no extra flow variants (Addendum A item 6).
- **P5b (delta fusion): not promoted.** 2.009 / 3.702 / 5.887, significantly worse than B5 at every horizon (DM p = 0.013 / 0.045 / 0.039). Betas: seasonal non-zero in 4 of 6 folds at 8 weeks (0.10-0.14) and 5 of 6 at 13 weeks (0.04-0.46), 0 in fold 4; macro mostly 0. Expectation "betas near 0 in most folds, gain < 0.05" WRONG (betas often non-zero; it loses).
- **P5c (temperature + shrinkage): not promoted.** 1.976 / 3.639 / 5.753. Validation win at every horizon, test better than B5 in 2 / 3 / 1 folds. **Validation chose lambda = 0 (no shrinkage) in all 18 (fold, horizon) cells**; T ranged 0.01-0.20. Expectation "lambda > 0 in at least half" WRONG.
- **Nothing beats FA-ETS-d** pooled at any horizon (best: P5c at 4 weeks, +0.002, p = 0.42). Expectation "most likely nothing beats FA-ETS-d" CONFIRMED.
- 90% MCS among {FA-ETS-d, B5, P5a, P5b, P5c, P}: everything at 4 weeks; everything except P5b at 8 and 13 weeks.
- **4-week decision (Umang: "Phase 5's rule with base C2 at all horizons decides it"):** same rule applied to B5 vs P at 4 weeks: B5 better on test in 5 / 6 folds (1.977 vs 2.118) but loses on validation (2.386 vs 2.363), so **B5 does not replace P at 4 weeks**. Also at 8 and 13 weeks B5 does not replace P (validation win by 0.001 but test better in 2 / 6 folds). **Under the pre-registered rules, production P stands at all horizons.**
