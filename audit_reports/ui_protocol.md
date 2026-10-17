# Unemployment-claims domain: protocol, reserved window and leakage rules (Phase 6d)

Written and committed **before any UI result is loaded or computed** (Umang's instruction). Before this commit, only the protocol document (`domains/ui_claims/evaluation_protocol_ui.md`), the agent code and the data schema were read. The pilot's result files (`pilot_results_ui.md`, `ui_claims_eval_*.csv`, `ui_claims_fusion_*.csv`, `lag_grid_search_results_ui.csv`, `serial_correlation_check_results_ui.csv`) and the README were **not** read, and will not be used to choose anything.

## Purpose

Test whether the housing protocol transfers to a second domain without re-tuning: channels tested against the strongest endogenous reference (FA-ETS-d), fusion with the frozen recipe, and whether the temporal-seasonal handoff replicates.

## Data and leakage rules

- Panel: `domains/ui_claims/aligned_weekly_ui_claims.csv` (state initial claims NSA, weekly Saturday dates, 15 states, from 2013-01-05; macro: fed_funds, cpi with the pilot's publication-lag alignment). sha256 recorded at run time.
- **Cap: loaded only up to 2026-02-21** (same as housing). Nothing after the cap is loaded or summarized.
- **Reserved window:** test cells with any target on or after 2026-01-01 are excluded from scoring (CLEANBLEED primary). A future UI holdout would be origins after 2026-02-21. **Disclosure:** the existing pilot (single split, test window 2023-09-23 to 2026-08-22) already scored data through 2026-08-22, so a UI holdout would not be pristine for the authors. It is pristine only for decisions made in this audit, which never read those results.
- Every forecast uses only data dated on or before the origin. Validation cells are bounded: last target <= train_end (F0.10).

## Folds

The housing fold table exactly (`facts_mas/run_backtest.py` FOLDS: 6 expanding folds, 29-week test windows from 2021-05-01 to 2025-11-22, 13-week embargo, 104-week validation windows), for comparability. Horizons 4, 8, 13.

## COVID-2020 handling (pre-registered)

- **COVID window = 2020-03-14 to 2021-09-04.** It runs from the start of the national claims surge to the last Saturday before the federal pandemic unemployment programs (PUA, PEUC, FPUC) expired on 2021-09-06. These are policy dates, not chosen from the data.
- **Primary analysis:** any validation or test cell whose origin or any target date lies in the COVID window is excluded from:
  - skill estimation and Router mode selection;
  - the damped vs undamped choice;
  - scoring.

  Model fitting still uses the full history (series continuity, as the pilot argued).
- **Secondary analysis (reporting only):** COVID cells included.
- Consequence known in advance: fold 1's test window (origins 2021-05-01 to 2021-11-13) loses every origin whose targets end by 2021-09-04, and folds 1-3 lose part of their validation windows. Cell counts are reported per fold.

## Channels (no re-tuning)

| Channel | Implementation |
|---|---|
| naive | last value |
| seasonal naive | y at t + k - 52 |
| seasonal (Naive2) | `seasonality_agent` with window 7 (the housing value; **not** re-selected) |
| temporal (Holt) | `ar_agent.fit_and_forecast` (the UI pilot's AR agent uses the same specification) |
| FA-ETS / FA-ETS-d | `p3_build_baselines.forecasts_at` (window-7 factors) |
| exogenous | `domains/ui_claims/agents/macro_agent_ui.py` core as written (fed_funds and cpi, lags 10 / 10, carried over from housing priors) |

Columns are mapped `state -> msa`, `claims_count -> inventory_count` so the housing code runs unchanged.

## Systems (recipes applied as frozen; nothing tuned on UI)

- **P-UI**: the frozen P recipe. 4 weeks {Holt, seasonal, macro}; 8 and 13 weeks {FA-ETS-d (damped choice on validation), seasonal, macro}. Fix 2 (13 weeks), F0.10, adaptive Router with the national fed-funds regimes, T = 0.05, naive skill reference.
- **B5-UI**: {FA-ETS-d, seasonal, macro}, FA-ETS-d skill reference, same protocol.
- **R1c-UI**: {Holt, seasonal, macro} at every horizon, to test whether the temporal-seasonal handoff replicates.
- **EW-TS-UI**: equal weights on Holt and seasonal.

## Statistics

Phase 4 methods (`stats_lib.py`): panel DM (Newey-West lag h-1), paired moving block bootstrap (L = 13, B = 5,000), per-state DM (HLN, BH). Primary = COVID-excluded, CLEANBLEED.

## Questions and predictions (written now)

- U1. FA-ETS-d beats naive at every horizon (prediction: yes).
- U2. The seasonal channel beats naive at 8 and 13 weeks (prediction: yes; NSA claims are strongly seasonal).
- U3. P-UI beats naive at every horizon (prediction: yes) but is not better than FA-ETS-d at any horizon (prediction: not better).
- U4. Handoff: under R1c-UI, the temporal weight falls and the seasonal weight rises from 4 to 13 weeks (prediction: yes).
- U5. Including COVID cells raises every method's MAPE substantially and can reverse rankings (prediction: yes).

## Addendum A (2026-10-16): validation-selected single model (SelBest-UI), added after the Phase 6 UI results

**Disclosure:** this check was requested by Umang and added **after** the Phase 6 UI results were seen (we already knew P-UI beat FA-ETS-d at 8 and 13 weeks and nearly tied the seasonal channel). It is the housing SelBest-model analogue, with the same candidate set and rule, fixed here before it is run.

- **Candidates:** the housing SelBest-model set {naive, seasonal naive, Naive2 (seasonal channel), Holt, Holt-d, FA-ETS, FA-ETS-d, Theta (factor-adjusted), macro, intrinsic}. The UI domain has no intrinsic channel, so the set is the other nine. Definitions as in `phase3_log.md` (window-7 factors computed at the origin; ETS / Theta fitted once per origin and sliced).
- **Rule:** per (fold, horizon), the single candidate with the lowest mean standalone MAPE over that fold's validation cells at that horizon (origins in [train_end - 104 weeks, train_end], last target <= train_end). Primary analysis excludes COVID-window cells from the selection and from scoring; secondary includes them. Ties: first in the list order above.
- **Comparison:** P-UI and B5-UI vs SelBest-UI on the same test cells (CLEANBLEED): pooled MAPE, per-fold MAPE, paired block-bootstrap CI (L = 13, B = 5,000), panel DM (Newey-West lag h-1), per-state DM (HLN, BH). "Beats" means lower pooled test MAPE; significance is reported separately.
- **Predictions:** SelBest-UI picks Naive2 or an FA-ETS variant in most (fold, horizon) cells; P-UI is not significantly different from SelBest-UI at any horizon.
