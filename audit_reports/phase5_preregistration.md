# Phase 5 pre-registration (re-scoped)

Written 2026-10-16, **before any Phase 4 result was computed or read** (see git history: this file is committed before the first Phase 4 script runs). Scope set by Umang on 2026-10-16. Nothing in this file may be changed after Phase 4 results are seen, except to record that a step could not be executed (with the reason).

## Base system (B5)

- Protocol: R1c (seasonal window 7; fix 2 with 13 distinct validation weeks; validation targets <= train_end; adaptive Router; median skill; softmax T = 0.05; CLEANBLEED primary, DISCLOSURE alongside).
- Channels: **temporal = FA-ETS-d** (factor-adjusted ETS; damped vs undamped chosen per fold on validation exactly as in Phase 3; damped was chosen in all folds), **seasonal** (Naive2, window 7), **macro**. Event and intrinsic are retired from the forecast path (authors' decision, 2026-10-16).
- **Skill reference = FA-ETS-d** instead of naive: per validation cell, s_k = 1 - MAPE_k / MAPE_FA. Cells where MAPE_FA = 0 are skipped. FA-ETS-d's own skill is 0 by construction. Median per channel, softmax T = 0.05, as before.
- B5 is applied at all three horizons (base = C2, per the re-scope). Its numbers are reported as the Phase 5 base, not as a candidate.

## Candidates (max 3; no others)

### P5a: listing-flow channel

- **Data:** Zillow weekly metro files, smoothed cut (same cut as the target): new listings `Metro_new_listings_uc_sfrcondo_sm_week.csv`, newly pending `Metro_new_pending_uc_sfrcondo_sm_week.csv`, and price-cut share `Metro_perc_listings_price_cut_uc_sfrcondo_sm_week.csv` if it exists at weekly metro level (if not, the channel uses the first two only; recorded, not substituted).
- **Leakage rules:**
  - each file goes into its own new empty directory outside the repo, sha256 logged;
  - header read first and only date columns <= 2026-02-21 loaded;
  - same 15 MSAs, forward-fill within MSA, filled rows logged.
- **Publication timing:** compare each flow file's last available date with the inventory file from the same download. If a flow series ends later than inventory, it is used with the same timing as inventory. If it ends k weeks earlier, it is lagged k weeks (value at t taken from t - k).
- **Model** (fitted per fold on training data only, i.e. pairs whose target date <= train_end):
  - x_t = log(NL_t / NP_t), and p_t = price-cut share (if available);
  - deseasonalized change z_{t,k} = log(I_{t+k} / I_t) - log(f(w_{t+k}) / f(w_t)), with window-7 factors f computed from data <= train_end;
  - pooled OLS across MSAs, one regression per step k = 1..13: z_{t,k} = a_k + b_k x_t (+ c_k p_t);
  - forecast at origin t: I_t x f(w_{t+k}) / f(w_t) x exp(a_k + b_k x_t (+ c_k p_t)), with f the window-7 factors computed at the origin.
- **P5a system** = B5 + this channel (four channels), same protocol.

### P5b: delta fusion on top of FA-ETS-d

- y_hat = y_hat_FA + sum_k beta_k (y_hat_k - y_hat_FA), k in {seasonal, macro}, beta_k >= 0.
- Per (fold, horizon), betas fitted on that fold's validation cells (targets <= train_end) by non-negative least squares on relative errors: minimise sum over cells and steps of ((y - y_hat_FA - sum_k beta_k (y_hat_k - y_hat_FA)) / y)^2 (`scipy.optimize.nnls`). No regime conditioning, no Router.
- Naive-like channels contribute nothing unless their deviation from FA-ETS-d helps.

### P5c: temperature and shrinkage toward equal weights

- On B5: w = (1 - lambda) x w_softmax(T) + lambda x (1/3, 1/3, 1/3), where w_softmax is whatever weight set B5's Router would use for that cell.
- Grid: T in {0.01, 0.02, 0.05, 0.10, 0.20, 0.50}, lambda in {0, 0.25, 0.5, 0.75, 1}. 30 points; no other values.
- Per (fold, horizon): pick the grid point with the lowest fused validation MAPE on that fold's validation cells (targets <= train_end). Ties: the larger lambda, then the larger T (prefer the less aggressive weighting).
- This carries the Phase 1 finding (equal weights beat skill weights at 13 weeks in zero-support cells) into a validation-chosen form.

## Promotion rule (per horizon)

- A candidate replaces B5 only if (a) its mean over folds of each fold's validation fused MAPE is lower than B5's, AND (b) its test MAPE is lower than B5's in at least 5 of 6 folds (CLEANBLEED).
- If several qualify, the one with the lowest validation MAPE is promoted.
- **Separately, every candidate and B5 are compared with FA-ETS-d standalone** (pooled, per fold, per MSA). If nothing beats FA-ETS-d, the report says so plainly.
- Losers are reported in full.
- Phase 4 statistical tools (paired block bootstrap CI vs B5 and vs FA-ETS-d; panel DM) are applied to every candidate. All improvements are reported with those intervals.

## Items not run in Phase 5 unless Umang confirms

- The seasonal window grid {7, 9, 11, 13} pre-registered after Phase 1 (it would change FA-ETS-d's factors). The re-scope caps Phase 5 at three candidates, so it is parked, not run.

## Expectations (written now, before Phase 4 or 5 results)

- B5 within 0.05 of FA-ETS-d standalone at every horizon (FA-ETS-d already holds 0.92-0.99 of the weight in C2).
- P5a: the flow channel carries information about short-run inventory changes; I expect at most a small gain (under 0.1 MAPE) at 4 and 8 weeks and none at 13. Probability it passes the promotion rule at any horizon: low.
- P5b: betas near 0 in most folds; gain under 0.05.
- P5c: validation picks lambda > 0 in at least half of the (fold, horizon) cells; gain under 0.05.
- Most likely outcome: nothing beats FA-ETS-d by a margin the Phase 4 bootstrap would call real.
