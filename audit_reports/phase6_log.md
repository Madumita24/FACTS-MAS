# Phase 6 log

## Freeze (2026-10-16)

- Freeze commit `a27a23ab7b1a136c860b110e6871ad1bb4c7dfa8` (FINAL_CONFIG.md FROZEN; holdout option A; H7.1-H7.4; 2-week-lag secondary analysis; methods; runner gate; CoT holdout only). `d38191a` records the hash. 17 forecast-path files hashed; `audit_reports/phase7/check_freeze.py` verifies them (hashes are of the working copy, so a line-ending conversion by git on checkout would raise a false alarm).

## 6a. Runner and gate

- `audit_reports/phase7/run_holdout.py` computes every channel from frozen agent code (not the cache) and weights with `audit_core`.
- **Gate v1 FAILED** (2026-10-16 00:48). 8 and 13 weeks reproduced Phase 4 exactly (max cell diff ~1e-15). 4 weeks: pooled P 2.11828 vs 2.11759 (diff 0.00069, inside the 1e-3 tolerance), but my extra check "equal to 4 decimals" failed (2.1183 vs 2.1176); max per-cell diff 0.039 MAPE points. Naive identical (1.4e-14).
- **Diagnosis** (`diagnose_gate.py`, all 5,160 4-week cells): Holt refits at h = 13 sliced equal refits at h = 4 exactly (0 difference), so slicing is not the cause. Refits in today's pinned environment vs the cached AR values (fitted 2026-08-12 in another environment): median relative difference 2.6e-9, p99 3.3e-4, max 2.1e-3; 158 cells above 1e-4. That is optimizer-level variation across environments, which shifts 4-week validation skills slightly and moves pooled P by 0.0007. Only the 4-week system uses Holt, which is why 8 and 13 weeks match exactly.
- **Gate criterion v2** (encodes the stated tolerance "AR optimizer noise" instead of my stricter 4-decimal check; decided before re-running): (1) pooled P within 1e-3 of Phase 4 at each horizon; (2) naive identical; (3) attribution: with the cached AR values substituted for the refits, P must match Phase 4 on every cell to 1e-9. If (3) holds, the whole remaining gap is the AR optimizer. The v1 failure and this change are reported to Umang, who can reject v2.
- **Gate v2 PASSED** (2026-10-16). Pooled P vs Phase 4: 4wk +0.00069, 8wk 0.0, 13wk 9e-16. Naive identical (1.4e-14). Attribution: with cached AR substituted, max per-cell P difference 7.1e-15, so the whole 4-week gap is the Holt optimizer (refit vs cache relative difference over all cells: median 4.4e-9, p99 7.2e-4, max 1.5%; only the 4-week system uses Holt). Runner sha256 06dee8fd40831734adbdc706a75ed160494c07a2d236ee3694d612485c519dce recorded in FINAL_CONFIG section 9.
- Dry runs of the holdout code path (pseudo-holdout: train_end 2025-02-01, origins 2025-02-08 onward with targets before 2026-01-01; pre-cap data only; labelled DRY RUN, no conclusions): smoothed lag 0, smoothed lag 2 and raw lag 0 all ran end to end (cells, damped choice, Router modes, P / B5 / baselines, holdout statistics with the L = h block rule).
- CoT holdout script `audit_reports/phase7/cot_holdout.py` prepared (no API key; operator supplies the provider's stated cutoff; script also logs the model's self-reported cutoff, run date, panel hash, failures; marks results not out-of-sample if the stated cutoff is not before the earliest holdout target). Verified with `--dry-run-prompts` (no API call). Disclosure: the NEXUS CoT system prompt says "Your knowledge cutoff date is January 2025" regardless of model; kept unchanged for comparability.

## 6c. TimesFM

- Env `facts-timesfm`: Python 3.11.17, timesfm 2.0.2, torch 2.14.1+cpu, numpy 2.4.6, pandas 3.0.6. Checkpoint google/timesfm-2.5-200m-pytorch (same spec as `timesfm_baseline.py`: 52-week context, same ForecastConfig, clip at 0).
- Timing test (75 contexts, fold 1): per-horizon calls 4.16 s/context; one h = 13 call 0.81 s/context (measured while other jobs shared the CPU). Parity with the existing fold-1 results file: max |MAPE diff| 2.0e-5 for both call styles, so the single h = 13 call (sliced to 4 / 8 / 13) is used. Estimated six-fold run: 2,610 contexts x 0.81 s, about 35 minutes (< 3 hours, so run without asking, per instruction).
