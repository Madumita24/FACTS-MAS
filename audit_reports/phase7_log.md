# Phase 7 log

## 2026-10-20

1. Machine check. On AC (Win32_Battery BatteryStatus 2, charge 91%). Sleep was NOT disabled: AC standby timeout was 7200 s. Set to 0 with `powercfg /change standby-timeout-ac 0` (verified 0x0). AC hibernate was already 0. Restore to 7200 s (`powercfg /change standby-timeout-ac 120`) after Phase 7. DC standby is 7200 s: if AC is lost the machine still sleeps after 2 hours.
2. Authorisation recorded in FINAL_CONFIG 11.3 and committed (`829701b`) before any step.
3. `check_freeze.py`: "freeze check OK: 17 files match FINAL_CONFIG.md section 9", exit 0. Runner sha256 `06dee8fd...c519dce`, equal to `gate_record.json`.
4. `build_holdout_panel.py --phase7-go --archive-dir ../facts_mas_phase7_archive/archive_20261020` (outside the repo, per section 6).
   - Zillow downloads completed (22:52 local):
     - `Metro_invt_fs_uc_sfrcondo_sm_week.csv` sha256 `73fe35d7dbcfcb3b8951456e35629ccf4bc619aaf794eacc4c300990bd58f9bb`
     - `Metro_invt_fs_uc_sfrcondo_week.csv` sha256 `6ddf9594027d729608eac160750b35966b38bab9c11bcfae51828e0ea1ba13b5`
   - FRED step failed: `RuntimeError: FRED_API_KEY not found in environment or .env file` (no env var, no `.env` in repo or parent). Exit 1.
   - The panel build, revision check and secret check did not run. No post-cap value was parsed, summarised or scored.
   - Status: BLOCKED on the FRED key. This archive is partial and kept as is. The re-run must use a new archive directory (the script refuses an existing one). Both archives get reported.
5. Operator added `FACTS-MAS/.env` (45 bytes, git-ignored; not opened by the audit).
6. `build_holdout_panel.py --phase7-go --archive-dir ../facts_mas_phase7_archive/archive_20261020b`, built 2026-10-20 23:51, exit 0.
   - Zillow downloads are byte-identical to the partial archive in step 4 (same sha256 for both files): no vintage change between the two attempts.
   - Panels: smoothed and raw, each 6,810 rows, 15 MSAs, 2018-02-03 to 2026-10-10, 0 NaN. sha256 `panel_smoothed.csv` 282e9d32...5f1d, `panel_raw.csv` bb7b73b8...d454 (full hashes in `manifest.json`).
   - Grid start: rows before 2018-02-03 dropped (smoothed file starts 2018-01-27, raw 2018-01-06).
   - Revision check: 0 differences on or before 2026-02-21, macro and inventory. Inventory: 6,315 pre-cap cells per cut, 0 changed, MATERIAL = false on both cuts.
   - Filled rows: 9 smoothed, 20 raw, all dated 2018-2023. None after 2026-02-21, so no holdout target can contain a filled value.
   - Secret check: "FRED key absent from all 13 output files".
7. TimesFM holdout (`facts-timesfm` env, `timesfm_holdout.py`), 2026-10-20 23:51 to 2026-10-21 00:13, all exit 0. Output in `audit_reports/phase7/timesfm_20261020/`. Each file: 435 (msa, origin) contexts, 8,640 rows, 0 NaN, origins 2026-02-28 to 2026-09-12, h in {4, 8, 13}.
   - `timesfm_holdout_smoothed_lag0.csv` 23d82094ca6190c2d582d156497925203a898a0ac8105e330587944bb9d14c04
   - `timesfm_holdout_smoothed_lag2.csv` 68d9a043307750a7897838ced28815a7f1bc0ce881859e5e264ce4f10ed97d6a
   - `timesfm_holdout_raw_lag0.csv` 9b7499ee0cb3a7e2757f7609a04420c95927b70b9d7258c1b595dfec8f7b4515
   - `timesfm_holdout_raw_lag2.csv` d1de51e8aa4412c9f72a9049f07b19b0560d0ab7d3bedba04117c30e4894fb79
   - Holdout grid: 29 / 25 / 20 origins at 4 / 8 / 13 weeks (data to 2026-10-10, one week more than the 28 / 24 / 19 estimated in section 6), 1,110 cells per cut and lag.
8. STOP for CoT (operator run, smoothed cut, lag 0 and lag 2).
9. 2026-10-21/22: CoT model changed to gpt-5.6-luna (FINAL_CONFIG 11.4, committed `f124368` before any call).
10. 2026-10-22 00:29, first CoT attempt (operator): stopped on the first call (cutoff question) with `TypeError: process() takes no keyword arguments` in httpx2's brotli decoder (installed brotli 1.0.9 too old). No forecast call was made. Folder `audit_reports/phase7/cot_20261022_0029_lag0` is empty (kept); console log `cot_logs/cot_20261022_0029_lag0.txt` (checked: no key in it).
11. Fixes (FINAL_CONFIG 11.4 items 5-6), all offline, no API call by the audit:
    - gzip-only transport, SDK retries off (they had silently exceeded the 3-attempt rule), 180 s timeout;
    - preflight, per-cell checkpoint, resume;
    - `run_cot.ps1` rewritten (preflight first, `-Stamp` to continue).

    Tests: 18 pass in `test_phase7_prep.py`, including:
    - a negative control that reproduces the brotli crash through the real SDK;
    - a simulated-server test of headers, body and retries;
    - a crash-and-resume test.

    The resume test found a bug (a cut-off last line swallowed the next record), fixed before commit. A full `run_cot.ps1` run against a simulated server (kill at request 600, then `-Stamp` resume) gave 1,110 / 1,110 cells per lag, no duplicate calls, and output keys identical to the TimesFM files. That run also found a bug in the ps1 exit-code handling, fixed. Freeze check OK.
12. CoT run (operator, `run_cot.ps1`, stamp `20261022_0043`), gpt-5.6-luna, reasoning effort low. Preflight OK.
    - Lag 0: 2026-10-22 00:43-02:06. 1,110 / 1,110 cells, 0 failures, 0 retries.
    - Lag 2: started 02:06. Stopped by the operator at 331 cells (overnight), resumed 11:11:54 (logged), finished.
      - 27 failures (2.43%), all `APIConnectionError` within 11:28:12-11:30:21, a network drop. Each was tried 3 times (83 failed attempts in all; 56 retries).
      - Below the 5% threshold, so no re-run pass is allowed (11.2 I) and with/without-CoT statistics are not required (11 item 5).
      - The runner drops these 27 lag-2 smoothed cells for every method in that run's paired statistics.
    - response.model always `gpt-5.6-luna`. Stated cutoff 2026-02-16 (out_of_sample = true); self-reported cutoff "June 2024" (recorded, not used).
    - Tokens 2,070,830 in / 691,657 out, about $1.24 at list price. No key string in any output.
