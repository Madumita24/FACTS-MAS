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
