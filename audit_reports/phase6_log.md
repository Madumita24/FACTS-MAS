# Phase 6 log

## Freeze (2026-10-16)

- Freeze commit `a27a23ab7b1a136c860b110e6871ad1bb4c7dfa8` (FINAL_CONFIG.md FROZEN; holdout option A; H7.1-H7.4; 2-week-lag secondary analysis; methods; runner gate; CoT holdout only). `d38191a` records the hash. 17 forecast-path files hashed; `audit_reports/phase7/check_freeze.py` verifies them (hashes are of the working copy, so a line-ending conversion by git on checkout would raise a false alarm).

## 6a. Runner and gate

- `audit_reports/phase7/run_holdout.py` computes every channel from frozen agent code (not the cache) and weights with `audit_core`.
- **Gate v1 FAILED** (2026-10-16 00:48). 8 and 13 weeks reproduced Phase 4 exactly (max cell diff ~1e-15). 4 weeks: pooled P 2.11828 vs 2.11759 (diff 0.00069, inside the 1e-3 tolerance), but my extra check "equal to 4 decimals" failed (2.1183 vs 2.1176); max per-cell diff 0.039 MAPE points. Naive identical (1.4e-14).
- **Diagnosis** (`diagnose_gate.py`, all 5,160 4-week cells): Holt refits at h = 13 sliced equal refits at h = 4 exactly (0 difference), so slicing is not the cause. Refits in today's pinned environment vs the cached AR values (fitted 2026-08-12 in another environment): median relative difference 2.6e-9, p99 3.3e-4, max 2.1e-3; 158 cells above 1e-4. That is optimizer-level variation across environments, which shifts 4-week validation skills slightly and moves pooled P by 0.0007. Only the 4-week system uses Holt, which is why 8 and 13 weeks match exactly.
- **Gate criterion v2** (encodes the stated tolerance "AR optimizer noise" instead of my stricter 4-decimal check; decided before re-running): (1) pooled P within 1e-3 of Phase 4 at each horizon; (2) naive identical; (3) attribution: with the cached AR values substituted for the refits, P must match Phase 4 on every cell to 1e-9. If (3) holds, the whole remaining gap is the AR optimizer. The v1 failure and this change are reported to Umang, who can reject v2.
- **Gate v2 PASSED** (2026-10-16). Pooled P vs Phase 4: 4wk +0.00069, 8wk 0.0, 13wk 9e-16. Naive identical (1.4e-14). Attribution: with cached AR substituted, max per-cell P difference 7.1e-15, so the whole 4-week gap is the Holt optimizer (refit vs cache relative difference over all cells: median 4.4e-9, p99 7.2e-4, max 1.5%; only the 4-week system uses Holt). Runner sha256 06dee8fd40831734adbdc706a75ed160494c07a2d236ee3694d612485c519dce recorded in FINAL_CONFIG section 9.
