# FINAL_CONFIG (FROZEN)

Status: **FROZEN** on 2026-10-16. Umang confirmed in chat: freeze P exactly as written in section 1; holdout option A. Nothing in the forecast path changes after this. The configuration is frozen by the commit that introduces this text; its hash is recorded in section 9 by the next commit (a commit cannot contain its own hash). The file hashes in section 9 make any later change to the forecast path detectable.

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

## 6. Phase 7 holdout protocol (FROZEN, option A)

- **Do not run until Umang confirms co-author sign-off.**
- Refresh the Zillow pull (inventory smoothed and raw), archive each raw download with a date stamp and sha256 in its own directory outside the repo, and check that refreshed values on or before 2026-02-21 equal the archived capped panels (`audit_reports/artifacts/zillow_{sm,raw}_vintage20261015_capped20260221.csv`). Any revision is reported, not hidden. Refresh FRED macro with the same publication-lag rules.
- **One holdout fold:** train_end = **2026-02-21**. Validation window as in section 4 (origins in [train_end - 104 weeks, train_end], targets <= train_end). Weights, the damped choice and the Router modes are computed once from that window.
- **Holdout origins: 2026-02-28 onward** (no embargo), every week with a complete h-week target in the refreshed data. With data to 2026-10-03 that is about 28 / 24 / 19 origins x 15 MSAs at 4 / 8 / 13 weeks.
- **Methods:** naive, seasonal naive, Naive2 (seasonal channel), FA-ETS-d, P, B5, TimesFM, CoT. No Phase 5 winner exists.
- **Cuts:** smoothed (primary) and raw, both in the main table.
- Which methods run on which cut and lag, and further execution rules: see section 11.2 (CoT runs on the smoothed cut only).
- **Statistics:** panel DM (Newey-West lag h-1); per-MSA DM (HLN, BH); paired moving block bootstrap over whole weeks, B = 5,000, seed 20261016, block length L = h, except where h > T/2, where L = floor(T/2) and the CI is flagged as unreliable; MCS (T_max) as Phase 4.
- **Pre-registered holdout hypotheses (smoothed cut, primary):**
  - H7.1 P beats naive at every horizon: pooled MAPE lower and the 95% paired block CI of P - naive below 0.
  - H7.2 FA-ETS-d beats naive at every horizon (same test).
  - H7.3 P is approximately equal to FA-ETS-d at 8 and 13 weeks: 90% paired block CI of P - FA-ETS-d inside +/- 2% of FA-ETS-d's MAPE (equivalence). If the CI is wider than the margin but contains 0, the result is "inconclusive", not "equal".
  - H7.4 FA-ETS-d <= P at 4 weeks: point estimate of P - FA-ETS-d >= 0, reported with its 95% CI.
- **Secondary analysis (reporting only):** every method re-run with a **2-week publication lag on inventory**: the information set at origin t is data dated <= t - 2 weeks (forecasts made from t - 2 for targets t + 1 .. t + h; the naive reference is the value at t - 2; weights and validation computed the same way). Same statistics, both cuts.
- **CoT:** holdout only, not the backtest folds. Script prepared by the audit without any API key; run by Umang or Madumita. It must log the model name, its stated knowledge cutoff and the run date. Since the holdout starts 2026-02-28, CoT results count as out-of-sample only if the stated cutoff is before the earliest holdout target date.
- **TimesFM:** run in its own environment on holdout origins (context = data <= origin, or <= origin - 2 weeks in the secondary analysis).
- Report everything. **Nothing may be changed after seeing results**; bugs are reported, not silently fixed.

## 7. Runner and gate (FROZEN)

- Runner: `audit_reports/phase7/run_holdout.py`, built by the audit from the frozen components (section 5).
- **Gate:** before reading anything dated after 2026-02-21, the runner must reproduce P's Phase 4 CLEANBLEED numbers on folds 1-6 (2.1176 / 3.6218 / 5.7309) from agent code, not the cache. Tolerance: AR optimizer noise. The maximum per-cell and pooled differences are reported. The runner's own sha256 at gate pass is recorded in section 9.
- Porting to `facts_mas/` happens after Phase 7.

## 8. Environment

Pinned `requirements.txt` stack (Python 3.11.13, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, statsmodels 0.15.0, scikit-learn 1.9.1, pydantic 2.13.5), run with `C:\Anaconda3\python.exe`. Macro inference only: `facts-audit` env. TimesFM: its own environment (recorded when built).

## 9. Freeze record

Frozen commit: `a27a23ab7b1a136c860b110e6871ad1bb4c7dfa8` (2026-10-16)

sha256 of forecast-path files at freeze (working copy):

| File | sha256 |
|---|---|
| `ar_agent.py` | cc53e5f8d7b042a393916ba074c443c470ecff75c5867544af18cea0c04b9d46 |
| `macro_agent.py` | b4838e64399928505ec995ce140c455131739e22bf32b58846df866b8d186776 |
| `naive_baseline.py` | 337bba94fff4d1f9dcf2dc58703731df9fb21a248f7e2f866ac09fc2bd2a1e1d |
| `align_data.py` | c197043c121538f861588f416f5d1c36b637be8ce0e1d78ce8777a370dc492f0 |
| `pull_zillow.py` | c097693344ae9d5fea8b6ec7c3ef31904f31daafc21742dffe0fff8ffbc0223c |
| `requirements.txt` | 9f4c5464f042356736fd347507296e6abdd46fb09de6819daa7aa1672779cf1a |
| `facts_mas/agents/ar_agent.py` | 4ec27900c7e0db9a342f085e8f35a6b1b3936c8e39be842a7941700cca1c5da7 |
| `facts_mas/agents/seasonality_agent.py` | 5afcc2d24e7c1f815ce22455d8e50dcb6b88d3f2c7750e8168cf52a7e4969a41 |
| `facts_mas/agents/macro_agent.py` | 12a5ee3142ee309e0bba08d41abf21361f7fb08c7aceed000f035cba2f07b57d |
| `facts_mas/run_backtest.py` | b33afa0552cd803ad538611b33f4ecc9e049ddf2f85ba1a8bbb3e64af52b7bb3 |
| `facts_mas/schema.py` | 76144c365af7d07c4df7aa696fcd58dfc1135dfe1f8f74371c96920720480e9f |
| `facts_mas/calibration.py` | 9641e3459d6b0cf910148135e2fcc1741fb519757e2561490e139c9a49906b89 |
| `tests_audit/audit_core.py` | 6481826eb0640c7d604092fbc79d30382bd68d520b084131d1ba15a62db89131 |
| `audit_reports/phase3_scripts/p3_build_baselines.py` | b5094e271603accf977f43955531682998ee7042397e1858c00c3c1dc19e6d99 |
| `audit_reports/artifacts/seasonal_w7_r1.pkl` | 3b530099abfe5bedcb994ca809ce8ae4b8e8e5a28bdf94c0c9dd43e560557017 |
| `aligned_weekly.csv` | 3ab9fa305ad42ad20cbffc83c996cf851644af555c5fd75a2889bd57e2a09836 |
| `macro_weekly.csv` | bc7ff7fb30a0e4131259184e2fb7d02de9dac310d99fbe47d91f1ef5802685a8 |

Runner sha256 at gate pass: `06dee8fd40831734adbdc706a75ed160494c07a2d236ee3694d612485c519dce` (`audit_reports/phase7/run_holdout.py`, gate criterion v2, passed 2026-10-16; record in `audit_reports/phase7/gate_record.json`)

## 10. Decisions taken at freeze (2026-10-16)

1. P frozen exactly as in section 1.
2. Holdout option A (no embargo).
3. Runner option (a): audit-built, gated; porting to `facts_mas/` after Phase 7.
4. Phase 7 runs only after Umang confirms co-author sign-off.

## 11. Phase 7 pre-run clarifications (Umang, 2026-10-16; committed before any Phase 7 step is run)

These clarify how section 6 is executed and reported. They do not change the forecast path (section 9 hashes unchanged).

1. **Confirmatory vs descriptive.** With about 19 origins at 13 weeks, every 13-week holdout result (including H7.1, H7.2 and H7.3 at 13 weeks) is **descriptive only** and carries the unreliable-CI flag. Confirmatory weight sits on 4 and 8 weeks.
2. **Data build.** The holdout panels are built by a separate script, `audit_reports/phase7/build_holdout_panel.py` (the frozen runner is not touched): fresh Zillow smoothed and raw inventory and fresh FRED pulls, assembled with the `align_data.py` / `pull_macro.py` logic and the same publication lags (FFR +32 d, UNRATE +35 d, CPI +42 d, mortgage weekly, as-of join backward). Every download is archived with sha256. **Revision check:** every value dated on or before 2026-02-21 must equal the capped artifacts (inventory: `zillow_{sm,raw}_vintage20261015_capped20260221.csv`; macro: `aligned_weekly.csv` rows <= 2026-02-21). Any difference is reported in full, never hidden or patched.
3. **Freeze check.** `git config core.autocrlf false` is set in this clone (2026-10-16); no checkouts before Phase 7; `check_freeze.py` is run immediately before Phase 7.
4. **TimesFM.** Holdout forecasts for lag 0 and lag 2 are produced as separate files by `audit_reports/phase7/timesfm_holdout.py`, with the 52-week context ending at the information origin (origin, or origin - 2 weeks).
5. **CoT.** gpt-4o-mini is the primary (and only) model; no second model unless Umang says so before any call. `response.model` is logged for every call and asserted equal to `--model` (pass the exact snapshot name the API reports, or the run stops). The CoT failure count is reported. If more than 5% of holdout cells fail, the paired statistics are reported both with CoT (cells without a CoT forecast dropped for every method) and without CoT (all cells).

### 11.2 Additional Phase 7 rules (Umang, 2026-10-16; committed before any post-cap data is loaded)

A. **Grid start.** Zillow rows dated before 2018-02-03 (the frozen panel's first date) are dropped when building the holdout panels (`build_holdout_panel.py`; recorded in its manifest).

B. **Filled targets.** Report the number of holdout cells whose targets include a forward-filled inventory value (from `filled_rows_{smoothed,raw}.csv`). If there are any, report the primary statistics with and without those cells. Scoring code is unchanged: this is a post-hoc filter on the runner's `cell_losses.csv` (`audit_reports/phase7/filled_target_check.py`). The script reads the run's cut from `run_meta.json` (panel path) and refuses if the filled-rows file is for a different cut.

C. **Revisions.** The fresh vintage is used everywhere. Macro differences (values <= 2026-02-21 vs `aligned_weekly.csv`): reported only. Inventory differences (vs the capped artifacts): count and size reported; flagged **"material"** if more than 1% of pre-cap inventory cells change by more than 0.5%. Never patched, never a reason to stop.

D. **CoT.** The operator passes the exact dated snapshot as `--model` and the provider-documented knowledge cutoff for that snapshot as `--stated-cutoff`. Transport errors only (HTTP 429, 5xx, timeouts, and connection errors, i.e. `APIConnectionError`) are retried, at most 3 attempts with exponential backoff (1 s, 2 s); if a numeric `Retry-After` header (seconds) is present the wait is at least that long, capped at 60 s; an HTTP-date or unparseable header falls back to the normal backoff; a request for more than 60 s is not waited for and the cell is recorded as a transport failure (eligible for the single re-run pass in item I); each attempt is logged; parsing and the single conversational correction are unchanged. (Fake-client tests in `tests_audit/test_phase7_prep.py`.)

E. **FRED key.** The audit may run `build_holdout_panel.py --phase7-go`. The key is read only via `pull_macro.load_api_key`; it is never printed, logged, archived or committed. Error messages are scrubbed of it before re-raising, and an assertion checks that no output file contains the key string.

F. **Scope.** TimesFM: both cuts (smoothed, raw) x lag 0 and lag 2 (4 files). CoT: smoothed cut only, lag 0 and lag 2. **No raw CoT.** (This narrows section 6's general "methods x both cuts" wording for CoT; the narrowing stands, Umang 2026-10-16.) In the raw-cut table write: "CoT: not run on the raw cut (scope fixed in FINAL_CONFIG 11.2 before holdout data was loaded)".

G. **Phase 7 order** (after Umang confirms co-author sign-off): `check_freeze.py` -> `build_holdout_panel.py --phase7-go` -> TimesFM holdout (4 files) -> CoT run by Umang or Madumita -> `run_holdout.py --holdout --phase7-go` for {smoothed, raw} x {lag 0, lag 2} -> filled-target check -> report H7.1-H7.4 and the secondary analyses -> `SUMMARY.md`.

H. **Machine:** on AC power with sleep disabled for the whole of Phase 7.

I. **CoT re-run pass.** If the CoT failure share is above 5%, exactly one logged re-run pass is allowed over the cells that failed on transport errors only (never parse failures): `cot_holdout.py --rerun-transport-failures` with the same `--out`, `--model` and `--panel`; it refuses a second pass. Then report as already specified: the original and post-re-run failure counts, and the paired statistics with and without CoT if the post-re-run failure share is still above 5%.

### 11.3 Phase 7 authorisation

Authorisation: Umang Bid, 2026-10-20; co-author sign-off waived by the operator.

### 11.4 CoT model change (Umang, 2026-10-21; committed before any CoT API call)

Replaces the model named in section 11 item 5. Umang asked the audit to choose a current, low-cost model; the choice was made from OpenAI's public documentation only (no API call):
- `gpt-4o-mini-2024-07-18` is deprecated (shutdown 2027-01-14).
- The GPT-6 family (`gpt-6-luna`, `gpt-6.1-sol`, `gpt-6-astra`) is excluded: documented knowledge cutoffs (2026-04-30 to 2026-05-18) fall inside the holdout (first target 2026-03-07), so the results could not count as out-of-sample.

1. **Model:** `gpt-5.6-luna` (OpenAI docs: knowledge cutoff "Feb 16, 2026"; $0.20 / $1.20 per million input / output tokens; Chat Completions supported). `--stated-cutoff 2026-02-16`, which is before the earliest holdout target, so the run is out-of-sample under section 6.
2. **Parameters:** `gpt-5.6-luna` is a reasoning model. Its documentation lists `reasoning_effort` but not `temperature`. The run uses `--reasoning-effort low` and sends no temperature: `cot_holdout.py` drops the `temperature=0.1` that `cot_baseline.py` passes (FINAL_CONFIG 11 had no temperature rule for reasoning models). Prompt, parsing and the single correction are unchanged. The NEXUS system prompt still says "Your knowledge cutoff date is January 2025" (disclosed, D2b).
3. **`--model` and `response.model`:** `--model` is `gpt-5.6-luna`. If the API reports a different (dated) name, the run stops by design. It is then re-run with exactly that name and a new `--out` directory, and the stopped attempt is reported.
4. **Comparability:** the paper's fold-1 CoT used gpt-4o-mini, so the holdout CoT is a different model. The paper says so. No other model is run.
5. **Transport fixes after the first attempt (2026-10-22, before any forecast call).** The first attempt (`cot_20261022_0029_lag0`, empty folder; log `cot_logs/cot_20261022_0029_lag0.txt`) stopped on its first call, the cutoff question, with `TypeError: process() takes no keyword arguments`. Cause: httpx2 requests brotli, and the installed brotli 1.0.9 cannot decode it. A second issue was found while fixing it: the OpenAI SDK retried silently (max_retries = 2) on top of the logged wrapper, against item D's "at most 3 attempts". Fixes in `cot_holdout.harden_client` (no environment change):
   - Accept-Encoding gzip/deflate only;
   - SDK `max_retries=0`;
   - 180 s request timeout (a timeout is a transport error).

   Prompt, parsing and the forecast path are unchanged.
6. **Run safety.**
   - `--preflight`: 2 real calls on pre-cap data (Phoenix 2025-06-07). No result file is written.
   - Every finished cell is appended and fsynced to `cot_progress_lag{L}.jsonl`. The final CSVs are built from that file.
   - `--resume` continues an interrupted run in the same folder without calling finished cells again. A cell in flight at a crash is redone. Resumes are logged.
   - These are not re-runs in the sense of item I: no finished cell, and no failure, is called twice.
   - Tested offline against a simulated OpenAI server, including a forced mid-run kill and a resume (`tests_audit/test_phase7_prep.py`; full `run_cot.ps1` path checked by hand).
