# Phase 6 addendum: pre-Phase-7 preparation and the UI single-model check

Date: 2026-10-16. Branch `audit-oct2026`. Frozen commit `a27a23a` (unchanged; freeze check passes). Audit commits this step:
- `33e83d3`: FINAL_CONFIG section 11, mismatch resolutions, C17 withdrawn
- `4a920d8`: UI single-model check pre-registered before running
- `91dc983`: UI check results, `build_holdout_panel.py`, CoT model checks, tests
- this commit: TimesFM holdout script, report

**Phase 7 has not been run, and nothing dated after 2026-02-21 was loaded.**

## Summary

1. **Housekeeping is done.**
   - `core.autocrlf false` is set in this clone and the freeze check passes.
   - FINAL_CONFIG section 11 makes 13-week holdout results descriptive only (confirmatory weight on 4 and 8 weeks) and fixes the data-build, TimesFM and CoT rules.
   - The mismatch list has the ±0.001 methods note (B15) and your resolutions; C17 is withdrawn.
2. **UI single-model check: the mechanical rule is not a fair benchmark, and the honest UI answer is "no evidence that fusion beats the best single model".**
   - The rule picks seasonal naive in 6 of 18 fold-horizon cells, and it fails across the COVID break (MAPE about 276-286 in fold 1). That inflates the pooled gap in P-UI's favour. Even so, P-UI is not significantly better (DM p = 0.087 / 0.176 / 0.064) and is better in only 2-3 of 6 folds.
   - Where the rule picks undamped FA-ETS, that single model is slightly better than P-UI. Undamped FA-ETS alone is below P-UI at every horizon (descriptive).
3. **`build_holdout_panel.py` is ready and self-tested** with zero revision differences on pre-cap data. A negative-control test shows the revision check catches a single changed value.
4. **The CoT script now enforces the model.**
   - Every API call must request exactly `--model`, and `response.model` must equal `--model`. Both are logged per call; a mismatch stops the run.
   - Unit tests with a fake client confirm the override reaches the call.
   - The failure count and share are logged, with the 5% flag.
5. **TimesFM holdout script is ready** (separate lag 0 and lag 2 files). Dry runs on pre-cap data:
   - lag-0 forecasts are bit-identical to the backtest TimesFM forecasts;
   - lag-2 forecasts are bit-identical to lag-0 forecasts made 2 weeks earlier;
   - the output merges through the frozen runner with no missing cells.

## What I did

- Ran `git config core.autocrlf false` (`.git/config`); freeze check OK.
- Committed FINAL_CONFIG section 11 before any Phase 7 step.
- Updated `paper_mismatch_list.md` (B15; section G resolutions: B9 and A4 removed, A6 grid B dropped, E4 included, C17 withdrawn) and `paper_claims_status.md` (C17 withdrawn; C21 updated).
- Pre-registered the UI single-model check (`ui_protocol.md` Addendum A; disclosed as added after the Phase 6 results), then ran `p6_ui_selbest.py`.
- Wrote `audit_reports/phase7/build_holdout_panel.py`: it reuses `pull_macro.py` functions and `pull_zillow.py` constants and replicates `align_data.py` without writing repo files. Ran `--self-test` on the 2026-10-15 downloads (columns up to the cap only) with the repo's macro file standing in for FRED.
- Added `ModelCheckedClient` to `audit_reports/phase7/cot_holdout.py`.
- Wrote `audit_reports/phase7/timesfm_holdout.py`.
- Added `tests_audit/test_phase7_prep.py` (3 tests).

## Findings

| # | Finding | Evidence | Confidence | Severity | Changes paper? |
|---|---|---|---|---|---|
| F6A.1 | UI single-model rule (primary): seasonal naive in 6 / 18 cells (fold 1 all horizons; fold 2 13wk; fold 6 8 and 13wk), undamped FA-ETS in 12 / 18. Pooled SelBest-UI 34.03 / 36.04 / 57.89 vs P-UI 17.86 / 19.23 / 20.58; DM p = 0.087 / 0.176 / 0.064; P-UI better in 2 / 2 / 3 folds. | `phase6_ui_selbest_*.csv` | High | Major | Yes |
| F6A.2 | Where the rule picks FA-ETS, the single model beats P-UI slightly (fold 2 4wk 16.90 vs 17.79; folds 3-5 by 0.02-0.17). Undamped FA-ETS alone: 17.709 / 19.182 / 20.552 vs P-UI 17.863 / 19.233 / 20.578 (descriptive, not a pre-registered comparison). | same | Medium-high | Major | Yes (C21) |
| F6A.3 | Seasonal naive is fragile across regime breaks: validation in calm years selects it, then year-ago values from the COVID surge destroy it on test. This is a general caution for SelBest-style rules. | same | High | Minor | Possibly (methods) |
| F6A.4 | `build_holdout_panel.py` self-test: 0 revision differences (smoothed and raw, inventory and macro). Negative control: one changed inventory value gives 1 difference; a one-date CPI revision gives 15 (one per MSA). | self-test manifest; `test_phase7_prep.py` | High | n/a | No |
| F6A.5 | Zillow's current files start earlier than the frozen panel (smoothed 2018-01-27, raw 2018-01-06, vs 2018-02-03). The build drops rows before 2018-02-03 so the holdout validation window matches the frozen backtest; recorded in the manifest. | `build_holdout_panel.py` | High | Minor | No (decision 1) |
| F6A.6 | OpenAI's `response.model` usually reports a dated snapshot. Under the strict check, the operator must pass that exact snapshot as `--model`, or the first call stops the run (by design). | `cot_holdout.py` | High | Procedural | No |
| F6A.7 | TimesFM holdout dry run (2 MSAs, pre-cap pseudo-holdout): lag-0 vs backtest forecasts 546 values, max diff 0; lag-2 at t vs lag-0 at t - 2 (2 steps further), 704 values, max diff 0; frozen-runner merge 232 cells, 0 missing. | dry-run outputs | High | n/a | No |

## Numbers

### UI single-model check (CLEANBLEED; pooled MAPE)

| Analysis | h | P-UI | B5-UI | SelBest-UI | Undamped FA-ETS | Seasonal | FA-ETS-d | Theta | Naive |
|---|---|---|---|---|---|---|---|---|---|
| Primary (COVID-excluded) | 4 | 17.863 | 17.769 | 34.033 | 17.709 | 17.836 | 17.868 | 17.760 | 18.816 |
| | 8 | 19.233 | 19.226 | 36.039 | 19.182 | 19.294 | 19.411 | 19.292 | 21.963 |
| | 13 | 20.578 | 20.562 | 57.891 | 20.552 | 20.613 | 20.791 | 20.688 | 24.716 |
| Secondary (COVID-included) | 4 | 18.583 | 18.559 | 18.580 | 18.494 | 18.584 | 18.628 | 18.559 | 19.490 |
| | 8 | 20.740 | 20.718 | 21.022 | 20.660 | 20.722 | 20.870 | 20.799 | 23.101 |
| | 13 | 22.731 | 22.712 | 65.730 | 22.701 | 22.700 | 22.924 | 22.911 | 26.379 |

P-UI vs SelBest-UI (primary): diff -16.17 / -16.81 / -37.31; DM p = 0.087 / 0.176 / 0.064; 95% block CI [-30.8, 0.4], [-31.7, 0.1], [-67.9, -1.5]; per-state BH 0 / 0 / 0. Per fold and selections: `phase6_ui_selbest_cells.csv`, `phase6_ui_selbest_selection.csv`.

## What surprised me and how I checked it

The rule chose seasonal naive in a third of the cells. I checked the mechanism by looking at the fold-1 validation window after COVID exclusion: it is essentially 2019, where year-ago values work, while the test targets' year-ago values fall in the 2020-21 surge. This is a property of the rule and the data, not a bug. I did not alter the rule after seeing it.

## Things I could not determine

Whether FRED has revised CPI, UNRATE or FEDFUNDS since July. That shows up only when the real build runs in Phase 7. Any difference will be reported by the revision check.

## Files changed

| Path | Owner | Why |
|---|---|---|
| `.git/config` (`core.autocrlf false`) | local clone setting | Prevent line-ending conversion before Phase 7 |
| `audit_reports/FINAL_CONFIG.md` (section 11 added) | audit | Phase 7 pre-run clarifications |
| `audit_reports/paper_mismatch_list.md`, `paper_claims_status.md` | audit | B15, resolutions, C17, C21 |
| `audit_reports/ui_protocol.md` (Addendum A) | audit | UI single-model pre-registration |
| `audit_reports/phase6_scripts/p6_ui_selbest.py`; `results_tables/phase6_ui_selbest_*.csv` (3) | new (audit) | UI check |
| `audit_reports/phase7/build_holdout_panel.py`, `timesfm_holdout.py` | new (audit) | Phase 7 data and TimesFM |
| `audit_reports/phase7/cot_holdout.py` (modified) | audit | Model checks, failure share |
| `tests_audit/test_phase7_prep.py` | new (audit) | Revision-check negative control; CoT model checks |
| `audit_reports/phase6_log.md`, `phase6_addendum_20261016.md` | audit | Log and this report |

The frozen runner and every hashed file are unchanged. All audit tests pass: 32 passed, 3 strict-xfail, 1 skipped (path-dependent parity check). Nothing pushed.

## Decisions needed from you

1. **Grid start for the holdout panel.** Accept dropping Zillow rows before 2018-02-03 so the holdout validation history matches the frozen backtest (my default)? The alternative is following `align_data` literally and including the earlier weeks (smoothed one extra week, raw four).
2. **FRED key in Phase 7.** `build_holdout_panel.py --phase7-go` reads `FRED_API_KEY` from `.env`, as `pull_macro.py` always has. May I run it at Phase 7, or will you?
3. **CoT model name.** Run it with the exact dated snapshot name the API returns for gpt-4o-mini as `--model`, so the strict `response.model` check passes. The script's own self-report query will show it on the first call.
4. **UI wording in the paper.** I suggest: "In the unemployment domain the frozen recipe beats naive and the damped FA-ETS reference, but not the undamped FA-ETS, which is the better reference there; there is no evidence that fusion beats the best single model in either domain." Your call.
