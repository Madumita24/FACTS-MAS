# Commit hash map

Commit messages on branch `audit-oct2026` were edited on 2026-10-23, before the first push, to remove a trailer line. File contents, dates and authors are unchanged (identical trees), but every commit hash changed. The audit documents cite the **old** hashes. Use this table to find the commit each one refers to.

| Old hash (cited in documents) | New hash (on GitHub) | Date | Commit |
|---|---|---|---|
| `a550e63` | `6e6c296` | 2026-10-14 | Phase 0 audit: orientation report, code map, headline reproduction |
| `ef28f47` | `c9c84cc` | 2026-10-15 | Phase 1: pre-register hypotheses and selection rules before running |
| `ee0a41b` | `f8106e6` | 2026-10-15 | Audit: guarded loader, reserved-window guard, fast weighting mirror, leakage tests |
| `b2bdaeb` | `effd36b` | 2026-10-15 | Phase 1: analysis scripts, results tables, run outcomes |
| `6abd6ea` | `3c79379` | 2026-10-15 | Phase 1 report: leakage and protocol audit |
| `50df9b2` | `17b3d30` | 2026-10-15 | Phase 2: pre-register fix 2 (13-week support fallback) and hypotheses; R1 artifact |
| `e65bc06` | `185acc2` | 2026-10-15 | Phase 2: component tests, analysis script, tables, outcomes |
| `5e9e783` | `71e2926` | 2026-10-15 | Phase 2: script for regime-cell and Router-by-fold tables |
| `78772b8` | `3d02d3f` | 2026-10-15 | Phase 2 report with Phase 0-2 re-evaluation; start paper_claims_status.md |
| `325dace` | `05114bf` | 2026-10-15 | Phase 3: pre-register baselines, candidates C1-C3, promotion rule, expectations |
| `bbfd9f5` | `c94df60` | 2026-10-15 | Phase 3: baselines, candidates, leakage test for new baselines, outcomes |
| `00906d2` | `f87b73f` | 2026-10-15 | Phase 3: raw vs smoothed Zillow robustness; capped panels (<= 2026-02-21) with hashes |
| `831d380` | `b643215` | 2026-10-15 | Phase 3 report: baselines and candidates; claims file updated |
| `fc0c7c8` | `d802494` | 2026-10-15 | Pre-register Phase 5 (re-scoped) and Phase 4 before any Phase 4 result; record Phase 3 decisions |
| `dcbe2f1` | `ba100c7` | 2026-10-15 | Phase 4: cell losses, statistics, macro re-check, regime placebo, outcomes |
| `4a095a4` | `4112f50` | 2026-10-15 | Phase 4 report with Phase 0-4 re-evaluation; claims file updated |
| `4025af4` | `fb4339c` | 2026-10-15 | Phase 5 pre-registration addendum A (P5a rules) before any flow data; record Phase 4 decisions (C6 withdrawn) |
| `345cf8f` | `2563996` | 2026-10-15 | Phase 5 step 0: raw-cut CIs and DM tests (reporting only); stats_lib reproduces Phase 4 |
| `3f99bae` | `29f1f1f` | 2026-10-15 | Phase 5 P5a: flow data capped with hashes, L = 2 weeks (timing not confirmable), flow channel builder, leakage test (3 pass) |
| `dec0c4d` | `af40ac6` | 2026-10-15 | Phase 5: B5, P5a, P5b, P5c results; nothing promoted; P stands; skill-reference option in audit_core |
| `b5e50ee` | `4162891` | 2026-10-15 | Phase 5 report, proposed FINAL_CONFIG.md and Phase 6 scope; claims file updated |
| `a27a23a` | `0900350` | 2026-10-16 | FREEZE: production configuration P (FINAL_CONFIG.md), holdout option A pre-registered |
| `d38191a` | `49d5781` | 2026-10-16 | Record frozen commit hash in FINAL_CONFIG.md (documentation only) |
| `28eefcb` | `f7412d7` | 2026-10-16 | Phase 6a: runner, gate v1 failure and diagnosis (AR optimizer noise), gate criterion v2; Phase 6b mismatch list |
| `cd2599b` | `d390be9` | 2026-10-16 | Phase 6d: pre-register UI-claims protocol, reserved window, COVID handling before any UI result |
| `af9d505` | `cc88745` | 2026-10-16 | Phase 6a: runner gate v2 passed (residual 4-week gap = Holt optimizer noise); runner hash recorded |
| `a6ef2ed` | `67af77e` | 2026-10-16 | Phase 6: dry runs of holdout path, CoT holdout script (no key), TimesFM scripts and timing/parity |
| `72800d6` | `91c2090` | 2026-10-16 | Phase 6c/6d: TimesFM all folds (P and FA-ETS-d beat it at every horizon); UI domain results; claims updated |
| `fef924c` | `5fe965c` | 2026-10-16 | Phase 6 report with Phase 0-6 re-evaluation; TimesFM forecasts artifact |
| `33e83d3` | `3502019` | 2026-10-16 | FINAL_CONFIG section 11: Phase 7 pre-run clarifications (13wk descriptive only, data build, CoT and TimesFM rules); mismatch resolutions; C17 withdrawn |
| `4a920d8` | `3d2d58b` | 2026-10-16 | UI protocol addendum A: pre-register validation-selected single-model check (added after Phase 6 results; disclosed) |
| `91dc983` | `a5c98b4` | 2026-10-16 | UI SelBest check (added after Phase 6, disclosed); build_holdout_panel.py with self-test (0 revision diffs); CoT model checks; prep tests |
| `62a136a` | `359b738` | 2026-10-16 | Pre-Phase-7: TimesFM holdout script (dry-run parity exact), UI claim C21 updated, addendum report |
| `9c63542` | `e6fadd9` | 2026-10-16 | FINAL_CONFIG 11.2: filled-target, revision, CoT retry, FRED key, scope and Phase 7 order rules; key safety, revision materiality, CoT retries, filled-target check, tests |
| `c4d205b` | `991665e` | 2026-10-16 | FINAL_CONFIG 11.2 (CoT re-run pass rule, retry and cut rules, raw-cut wording; section 6 pointer); CoT connection-error retries, Retry-After, single re-run pass; filled-target cut check; tests |
| `ab1ad1d` | `72f9e62` | 2026-10-17 | CoT Retry-After hardening: numeric only with backoff fallback, 60 s cap (longer = transport failure); FINAL_CONFIG 11.2 D; tests |
| `829701b` | `13c5366` | 2026-10-20 | FINAL_CONFIG 11.3: Phase 7 authorisation (Umang Bid, 2026-10-20; co-author sign-off waived by the operator) |
| `7b1fc7c` | `c8a6597` | 2026-10-20 | Phase 7 log: freeze check OK; panel build blocked at FRED step (no FRED_API_KEY); partial Zillow archive recorded |
| `a95fa92` | `ba6c865` | 2026-10-20 | Phase 7 log: holdout panels built; 0 revisions; no post-cap filled rows; secret check OK |
| `be038e6` | `d9f4eba` | 2026-10-21 | Phase 7: TimesFM holdout forecasts (smoothed/raw x lag 0/2); log |
| `0d127dd` | `ae0d89c` | 2026-10-21 | Before/after comparison of the system (pre-holdout, backtest folds only) |
| `5d77bf5` | `17575b4` | 2026-10-21 | Before/after comparison: correct audit commit count |
| `f124368` | `8be9c0e` | 2026-10-21 | FINAL_CONFIG 11.4: CoT model gpt-5.6-luna (cutoff 2026-02-16; GPT-6 family excluded, cutoffs inside holdout); --reasoning-effort (temperature omitted); progress lines; run_cot.ps1; test |
| `51926c0` | `ac2b744` | 2026-10-22 | CoT run safety: gzip-only transport (brotli 1.0.9 crash), SDK retries off, timeout; preflight, per-cell checkpoint, resume; run_cot.ps1 v2; tests (simulated server, crash/resume); first attempt logged |
| `a3e7e39` | `03c8f73` | 2026-10-22 | Phase 7: CoT holdout forecasts (gpt-5.6-luna; lag 0: 0 failures; lag 2: 27 connection failures, 2.43%); log |
| `ba1141a` | `f59038a` | 2026-10-22 | Phase 7: holdout results (smoothed lag 0, raw lag 0, raw lag 2); smoothed lag 2 stats failed (runner bug: incomplete weeks after CoT connection failures), cell losses kept; filled-target checks 0; log |
| `d58e72b` | `2e1b03b` | 2026-10-22 | FINAL_CONFIG 11.5: post-hoc smoothed lag-2 statistics after runner bug (decided before viewing lag-2 numbers); script |
| `a66f77e` | `f9439bf` | 2026-10-22 | Phase 7: post-hoc smoothed lag-2 statistics (11.5): without CoT (all cells) and with CoT (complete weeks); filled-target flags 0 |
| `15b6dff` | `09cabe4` | 2026-10-22 | Phase 7 report, SUMMARY.md, CHANGELOG.md; results tables; claims and mismatch files updated with holdout results |
| `19db735` | `3b863fb` | 2026-10-22 | Paper edits part 1 (abstract to event channel); mismatch H5/B17 (weights per fold, not per origin) |
| `cbda0d2` | `eed4a76` | 2026-10-22 | Paper edits: single file covering the whole paper in order (60 steps, search phrase + action + new LaTeX); replaces part 1 |
| `51e8b8e` | `4568d15` | 2026-10-22 | Post hoc P vs CoT holdout comparison (not significant); paper edits corrected: no 'more accurate than CoT' claim |
| `838f45f` | `184ec4e` | 2026-10-23 | run_cot.ps1: neutral wording in on-screen messages (display text only) |
