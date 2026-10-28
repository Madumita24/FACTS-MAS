# Changelog: every change made during the audit, with before and after numbers

Branch `audit-oct2026` from `main` at `0a1aec5`. **No production file was modified or deleted.** Every change lives under `audit_reports/` and `tests_audit/`. The final system P is defined in `FINAL_CONFIG.md` and reproduced by `audit_reports/phase7/run_holdout.py`, which is gated to reproduce P from agent code. Porting to `facts_mas/` is pending.

Numbers are test MAPE at 4 / 8 / 13 weeks, smoothed cut, clean cells (CLEANBLEED), unless stated otherwise.

## 1. Changes to the forecasting system

| # | Change | Before | After | Numbers (before to after) | Where / commit |
|---|---|---|---|---|---|
| 1 | Seasonal smoothing window re-selected on validation, nested by fold | 5 (chosen on fold-6 test incl. reserved window) | 7 (chosen in 6 of 6 folds) | 2.094 / 4.986 / 7.019 to 2.089 / 4.952 / 7.058 (R1) | artifact `seasonal_w7_r1.pkl`, `50df9b2` |
| 2 | Fix 2: a regime cell with fewer than 13 distinct validation weeks uses fold weights | Equal weights 0.2 per channel (incl. naive clones) | Fold weights | R1 + fix 2: 2.089 / 4.816 / 7.262 | `audit_core.min_support_weeks`, `e65bc06` |
| 3 | F0.10: validation targets bounded to train_end | Targets up to train_end + 13 weeks (inside embargo) | Targets on or before train_end | R1c (= 1 + 2 + 3): 2.122 / 4.468 / 6.987 | `audit_core.bound_targets`, `e65bc06` |
| 4 | Temporal channel at 8 and 13 weeks: FA-ETS-d (pre-registered promotion rule) | Holt | FA-ETS-d (damped chosen in every fold) | 8 / 13 weeks: 4.468 / 6.987 to 3.622 / 5.735 (C2) | `p3_build_baselines.py`, `bbfd9f5` |
| 5 | Event and intrinsic channels retired | 5 channels | 3 channels (temporal, seasonal, macro) | C2 to P: at most 0.005 (p >= 0.057); final P 2.118 / 3.622 / 5.731 | `FINAL_CONFIG.md`, `a27a23a` |
| 6 | Attribution | Level shares (paper eq. 11) | Change shares primary | Reporting only | `p2_analysis.py` |
| - | Unchanged | T = 0.05, median skill vs naive, 104-week validation, Router, regime threshold ±0.25, macro lags | | | |

Candidates tested and **not** adopted (pre-registered rule; numbers are test MAPE):
- Listing-flow channel (P5a): 1.982 / 3.648 / 5.755.
- Delta fusion (P5b): 2.009 / 3.702 / 5.887.
- Temperature and shrinkage tuning (P5c): 1.976 / 3.639 / 5.753.
- B5: 1.977 / 3.617 / 5.700.
- Window grid {7, 9, 11, 13}: parked, not run (the Phase 5 re-scope capped the phase at three candidates; `phase5_preregistration.md`).

## 2. Evaluation changes

| Change | Before | After |
|---|---|---|
| Reserved-window cells | 165 fold-6 cells with targets in Jan-Feb 2026 scored | Excluded (CLEANBLEED primary), and a guard raises on any such target (`tests_audit/audit_core.py`) |
| Baselines | Naive, Holt, TimesFM (fold 1), CoT (fold 1) | Plus seasonal naive, Naive2, damped Holt, FA-ETS, FA-ETS-d, Theta, equal-weight combinations, validation-selected single model, TimesFM on all folds and the holdout, CoT on the holdout |
| Statistics | Point MAPE, MSA win counts | Panel DM, per-MSA DM with BH, block bootstrap, equivalence, MCS, Holm / wild bootstrap / Driscoll-Kraay, regime placebo |
| Robustness | None | Raw Zillow cut; 2-week publication lag; unemployment-claims domain |
| Holdout | None | One fold, origins from 2026-02-28, pre-registered H7.1-H7.4 |

## 3. Audit code (new files only, plus audit-owned edits)

| Date | Commit | Change |
|---|---|---|
| 10-14 | `a550e63` | Phase 0 reproduction scripts |
| 10-15 | `ee0a41b` | `tests_audit/audit_core.py` (guarded loader, reserved-window guard, weighting mirror pinned bit-for-bit to production), leakage tests |
| 10-15 | `50df9b2`, `e65bc06` | R1 artifact; fix 2 and F0.10 options (defaults unchanged); component tests |
| 10-15 | `bbfd9f5`, `00906d2` | Baselines, candidates, leakage test for baselines; raw-cut panels |
| 10-15 | `dcbe2f1` | Phase 4 statistics (`stats_lib.py` from `345cf8f`) |
| 10-15 | `3f99bae`, `dec0c4d` | Flow channel and its leakage test; skill-reference option |
| 10-16 | `a27a23a` | **Freeze** (17 files sha256-locked) |
| 10-16 | `28eefcb`, `af9d505` | Holdout runner; gate v1 failed (4-decimal check), v2 passed (Holt optimizer noise ±0.001) |
| 10-16 | `91dc983`, `62a136a` | `build_holdout_panel.py` (revision check, key safety), CoT script, TimesFM holdout script |
| 10-16/17 | `9c63542`, `c4d205b`, `ab1ad1d` | FINAL_CONFIG 11.2 rules; CoT retry rules and tests |
| 10-21 | `f124368` | CoT model changed to gpt-5.6-luna; `--reasoning-effort`; progress lines; `run_cot.ps1` |
| 10-22 | `51926c0` | CoT transport fixes (brotli crash; hidden SDK retries off; timeout); preflight; per-cell checkpoint and resume; simulated-server tests |
| 10-22 | `d58e72b` | Post-hoc lag-2 statistics after a runner bug (FINAL_CONFIG 11.5) |

## 4. Bugs found

| Bug | Where | Status |
|---|---|---|
| Zero-support regimes got equal weights | production `fusion_baseline.py:268-270`, `factor_attribution.py:133-134` | Fixed in P (fix 2); production unchanged until porting |
| Intrinsic reversion schedule horizon-inconsistent | production `intrinsic_agent.py` | Channel retired; documented, not fixed |
| Calibration sweeps unbounded (event tables) | production `calibrate_event_*` | Bounded re-run reported; production scorer had been fixed in `0a1aec5` |
| `linearmodels` import broken in base env | environment | Separate `facts-audit` env |
| MCS failed on duplicate methods | audit `p4_stats.py` | Fixed before reporting (duplicates merged) |
| Gate v1 too strict for optimizer noise | audit runner gate | Criterion v2, accepted by Umang |
| CoT client: brotli decode crash; hidden SDK retries breaking the 3-attempt rule | audit `cot_holdout.py` | Fixed before any forecast call |
| Progress file: a cut-off line swallowed the next record | audit `cot_holdout.py` (new code) | Found by the resume test, fixed before use |
| `run_cot.ps1` exit-code handling | audit script (new code) | Found in the simulated run, fixed before use |
| Runner stats fail when an external forecast is missing for some MSAs in a week | audit `run_holdout.py` + `stats_lib.weekly` | Not patched (frozen); post-hoc statistics per 11.5 |

## 5. Environment

- Base: `C:\Anaconda3\python.exe`, pinned stack, never modified.
- New conda environments: `facts-audit` (linearmodels) and `facts-timesfm`.
- During Phase 7 the laptop's AC sleep timeout was set to 0; it was restored to 7200 s on 2026-10-22.
