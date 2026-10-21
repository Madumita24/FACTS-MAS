# FACTS-MAS: before and after the October 2026 audit

Date: 2026-10-21. Written for Umang, ahead of briefing Prof. Pan. Phase 7 (the untouched holdout) has **not** been scored yet, so every number here is from the six backtest folds. Sources: phase reports 0-6, `results_tables/`, `paper_claims_status.md`, `paper_mismatch_list.md`, `FINAL_CONFIG.md`.

## 1. Bottom line

- **The numbers went up, a lot at 8 and 13 weeks.** On the same cells the paper used, MAPE falls from 2.243 / 5.038 / 7.020 to 2.118 / 3.618 / 5.624 (4 / 8 / 13 weeks).
- **But the story went down.** Almost all of the gain comes from two things:
  1. a properly chosen seasonal smoothing window;
  2. swapping in a strong single model, factor-adjusted damped ETS (FA-ETS-d), as the temporal channel at 8 and 13 weeks.
- The fusion then hands that one model about 92-99% of the weight. **FA-ETS-d on its own is as good as or better than the full system at every horizon.** In housing there is no evidence that multi-agent fusion beats the best single model.
- **The paper's central claims about regime-aware weighting, the Router and the macro effect do not survive.** 13 of 27 checked claims are withdrawn and 8 weakened. 5 hold, and 1 is reclassified as future work.
- **Production code in the repo is unchanged.** Every change so far is defined in the audit code and `FINAL_CONFIG.md`. Porting it into `facts_mas/` comes after Phase 7.

## 2. What "before" and "after" mean

There are three "before" versions. They differ, which is itself a finding.

| Label | What it is | 4wk | 8wk | 13wk |
|---|---|---|---|---|
| **Paper** (draft 2026-09-26) | The 2026-07-29 run: old seasonal agent (no smoothing) and old event agent | 2.243 | 5.038 | 7.020 |
| **Repo as of `main` 0a1aec5** ("PROVISIONAL") | What the code and cache actually produce today; smoothing window 5, chosen on test data | 2.094 | 4.973 | 6.855 |
| Naive (same cells) | Last value carried forward | 4.194 | 7.371 | 11.168 |

All three include the 165 fold-6 cells whose targets fall in Jan-Feb 2026, inside the reserved window ("DISCLOSURE" cells).

"After" is **P**, the frozen production system in `FINAL_CONFIG.md`:
- **4 weeks:** channels {Holt, seasonal, macro}.
- **8 and 13 weeks:** channels {FA-ETS-d, seasonal, macro}.
- **Protocol:** seasonal window 7, fix 2 and F0.10 (section 4), the Router kept, T = 0.05.
- **Event and intrinsic channels:** retired.

## 3. What changed in the repository

| Area | Before (`main` 0a1aec5) | After (branch `audit-oct2026`, 41 commits) |
|---|---|---|
| Production code (`facts_mas/`, root agents) | As written | **Unchanged.** 0 lines modified or deleted. |
| Audit code | None | `tests_audit/` (guarded loader, reserved-window guard, fast weighting mirror pinned bit-for-bit to production, leakage tests), `audit_reports/phase*_scripts/`, frozen runner `audit_reports/phase7/run_holdout.py` |
| Tests | 175 production tests, all pass | Still 175, all pass, plus the audit suite: 43 pass, 1 skipped, 3 strict-xfail. The xfails document known problems that remain in the unchanged production code. |
| Documentation of results | Paper draft; `factor_attribution.md` (July numbers) | 7 phase reports with config blocks, claims file, mismatch list (38 items), `FINAL_CONFIG.md` (frozen, sha256-locked), results CSVs with config headers |
| Data | Zillow smoothed, pulled July 2026 | Same data, verified unrevised (two re-downloads; 0 differences up to 2026-02-21). Raw cut added as a robustness check. Holdout panels built 2026-10-20 (data to 2026-10-10). |

Consequence: **anyone who runs the repo's own pipeline today gets the PROVISIONAL system (2.094 / 4.973 / 6.855), not P.** P is reproduced only by the audit runner.

## 4. Algorithm changes, one by one

| # | Component | Before | After | Why | Effect on test MAPE (CLEANBLEED, 4 / 8 / 13) |
|---|---|---|---|---|---|
| 1 | Seasonal smoothing window | 5, chosen on fold 6's **test** window, including Jan-Feb 2026 reserved data | **7**, re-selected on validation only, nested by fold (7 wins in all 6 folds) | Test-set and reserved-window tuning (F0.5, F1.4a) | 2.094 / 4.986 / 7.019 to 2.089 / 4.952 / 7.058 (small; 13 weeks slightly worse) |
| 2 | Regime cell with no validation data | Every channel got weight 0.2, including the two naive clones (an undocumented accident) | Fold-level weights if a regime has fewer than 13 distinct validation weeks (**fix 2**) | Bug (F1.11): 780 test cells were fused with accidental equal weights | Combined with item 3 below |
| 3 | Validation targets | Could run up to 13 weeks past `train_end`, inside the embargo | Bounded to `train_end` (**F0.10**) | Contradicted the embargo's purpose (not a test leak) | Items 2 + 3 together ("R1c"): 2.122 / 4.468 / 6.987. The 8-week gain rests mostly on one fold (fold 3 hiking). 4 weeks is slightly worse in 5 of 6 folds. |
| 4 | Temporal channel at 8 and 13 weeks | Holt (trend ETS, ignores seasonality) | **FA-ETS-d**: divide by the window-7 seasonal factors, fit damped ETS, multiply back | Chosen by a pre-registered promotion rule (validation win AND better on test in at least 5 of 6 folds): passed at 8 and 13 weeks, failed at 4 | 4.468 / 6.987 to 3.622 / 5.735 at 8 / 13 weeks. **This is the big gain.** |
| 5 | Event channel | In the forecast path | **Retired** | Its output equals naive in all 15,480 cells; impact scale calibrated to 0 | P vs 5-channel P: at most 0.005 MAPE (p >= 0.057) |
| 6 | Intrinsic channel | In the forecast path, rho_max = 0.30 | **Retired** | Calibration chose 0 in 6 of 6 folds but was never adopted; the schedule is horizon-inconsistent (t+4 pulls 0.30 if H = 4 but 0.028 if H = 13) | Included in item 5 |
| 7 | Attribution | Level shares `w_k * y_hat_k` (paper eq. 11) | Change shares `w_k * (y_hat_k - y_t)` primary; level in an appendix | Level shares credited a naive clone with 4.5-5.6% of the forecast while it contributed 0% of the change | Reporting only |
| - | Softmax temperature T | 0.05, hand-set | 0.05 (unchanged) | Validation-tuned T was tested in Phase 5 and **not** promoted | None |
| - | Skill rule, 104-week validation, Router, regime threshold +/-0.25, macro lags 6/10/10/5 | As before | Unchanged | Kept so P differs from the original only where a defect or a rule required it | None |

## 5. Headline numbers before and after

### Same cells as the paper (2,610 per horizon, the 165 bleed cells included)

| System | 4wk | 8wk | 13wk |
|---|---|---|---|
| Paper (July 2026 run) | 2.243 | 5.038 | 7.020 |
| Repo today (PROVISIONAL) | 2.094 | 4.973 | 6.855 |
| **P (after)** | **2.118** | **3.618** | **5.624** |
| FA-ETS-d alone | 1.974 | 3.605 | 5.560 |
| Naive | 4.194 | 7.371 | 11.168 |

### Clean cells (CLEANBLEED; the primary column from now on)

| System | 4wk | 8wk | 13wk | Reduction vs naive |
|---|---|---|---|---|
| Paper's system (July agents) | 2.243 | 5.059 | 7.180 | 46.5% / 30.0% / 33.6% |
| **P (after)** | **2.118** | **3.622** | **5.731** | **49.5% / 49.9% / 47.0%** |
| FA-ETS-d alone | 1.974 | 3.608 | 5.664 | 52.9% / 50.1% / 47.6% |
| Naive | 4.194 | 7.225 | 10.814 | |

P relative to the paper's system: **-0.125 (-5.6%) / -1.437 (-28.4%) / -1.449 (-20.2%)**.

Excluding the bleed cells makes everything look a little worse. Those cells were hard for naive and easy for the seasonal channel, so including them flattered the system.

## 6. Where the improvement comes from (CLEANBLEED, step by step)

| Step | 4wk | 8wk | 13wk | What changed |
|---|---|---|---|---|
| Paper's system (July agents) | 2.243 | 5.059 | 7.180 | |
| + smoothing window 5 (= repo today) | 2.094 | 4.986 | 7.019 | 98-100% of the paper-to-repo change; but window 5 was tuned on test |
| window 7 instead (R1, clean selection) | 2.089 | 4.952 | 7.058 | Clean replacement for the test-tuned window |
| + fix 2 + F0.10 (R1c) | 2.122 | 4.468 | 6.987 | Bug fix and protocol fix; the 8-week gain is mostly one fold |
| + FA-ETS-d at 8 / 13, event and intrinsic retired (= **P**) | 2.118 | 3.622 | 5.731 | The large 8 / 13-week gain |
| FA-ETS-d alone, no fusion | 1.974 | 3.608 | 5.664 | Better than P at every horizon |

The event-magnitude removal (August 2026) changed nothing measurable (at most 0.002).

## 7. Does fusion beat the best single model?

| Question | Before (paper) | After (audit) |
|---|---|---|
| Strongest baseline compared | "Strongest standalone temporal model" (Holt: 2.368 / 5.430 / 10.225) | FA-ETS-d (1.974 / 3.608 / 5.664), plus seasonal naive, Naive2, Theta, damped Holt, equal-weight combinations, validation-selected single model, TimesFM; CoT pending |
| FACTS-MAS vs the best single model | Claimed better | **No.** P vs FA-ETS-d: 4wk +0.144 (P worse; DM p = 0.12; 90% CI [0.022, 0.261]); 8wk +0.014 (**equivalent** within 2%); 13wk +0.067 (FA-ETS-d marginally better; 95% CI [0.001, 0.152]) |
| Weight on the strong single model | n/a | 92-99% at 8 and 13 weeks |
| Any variant that beats FA-ETS-d? | n/a | None. Listing-flow channel, delta fusion, temperature/shrinkage tuning: all fail the pre-registered rule. The best one is +0.002 (p = 0.42). |

## 8. Statistical evidence

| | Before | After |
|---|---|---|
| Tests in the paper | Point MAPE; per-MSA win counts; one macro panel regression | Panel Diebold-Mariano (Newey-West); per-MSA DM (HLN, Benjamini-Hochberg); moving block bootstrap over whole weeks; equivalence tests; Model Confidence Set; Holm, wild cluster bootstrap and Driscoll-Kraay for the macro panel; regime placebo |
| P vs naive | "Beats naive" | **Holds, strongly.** DM t = -6.9 / -6.3 / -5.7 (p < 1e-4); significant in 15 of 15 MSAs at every horizon; 95% CI of the difference [-2.66, -1.42] / [-4.72, -2.52] / [-7.11, -3.53] |
| Uncertainty | Not reported | Block-bootstrap CIs are 4-7x wider than naive iid CIs. P at 13 weeks: [4.54, 7.41] |
| Macro effect: "FFR p = 0.042, crosses 5%" | Claimed | Reproduces exactly but fails every re-check: Holm 0.170, wild bootstrap 0.073, Driscoll-Kraay 0.71. **Withdrawn.** |
| Regime information | Central contribution | Placebo: 46% (8wk) / 27% (13wk) of 200 random label sets do as well or better than the true regimes. **No evidence that regime labels carry information.** |

## 9. Channel weights (mean over clean test cells)

| | 4wk | 8wk | 13wk |
|---|---|---|---|
| Before (repo today): AR / seasonal / macro / event / intrinsic | 0.72 / 0.27 / 0.00 / 0.00 / 0.00 | 0.45 / 0.30 / 0.17 / 0.05 / 0.03 | 0.06 / 0.68 / 0.17 / 0.06 / 0.04 |
| After: temporal / seasonal / macro (event and intrinsic retired) | 0.67 / 0.32 / 0.00 (Holt) | 0.97 / 0.03 / 0.00 (FA-ETS-d) | 0.92 / 0.08 / 0.00 (FA-ETS-d) |

The 8 / 13-week figures are from the 5-channel C2, which differs from P by at most 0.005 MAPE.

- **Before:** about a quarter of the 8 / 13-week weight went to naive or near-naive channels. Macro's 13-week weight came from one thin regime cell (fold 3 hiking) plus the equal-weight bug.
- **After:** that weight is gone. The "temporal-seasonal handoff" across horizons disappears once the temporal channel is seasonally adjusted, and it does not appear in the unemployment data either (claim withdrawn).

## 10. Robustness checks that did not exist before

| Check | Result |
|---|---|
| Raw (unsmoothed) Zillow series | Zillow's series is a four-week trailing average. On the raw series every method is worse. P 3.642 / 5.159 / 7.189 vs naive 4.785 / 7.882 / 11.438, a 4-week reduction of **24% instead of 49%**. FA-ETS-d is still best (3.402 / 5.105 / 7.164). At 4 weeks P is no better than the seasonal channel alone (p = 0.83). |
| Publication lag | Zillow posts each week 7-13 days late; the backtest assumes data at t is known at t. Every method is optimistic by 1-2 weeks. The holdout includes a 2-week-lag analysis. |
| TimesFM 2.5 (foundation model), all 6 folds | 2.361 / 4.860 / 8.269. Beats naive; **P and FA-ETS-d beat TimesFM at every horizon** (p <= 0.006). Before: fold 1 only. |
| Unemployment claims (second domain) | The frozen recipe beats naive and damped FA-ETS at 8 / 13 weeks, but not undamped FA-ETS. Agreed wording: "In neither domain is there evidence that fusion beats the best single model." Before: the paper said this domain was "analyzed" but reported nothing. |
| Data revisions | 0 revisions in Zillow or FRED up to 2026-02-21 (checked 2026-10-15 and 2026-10-20) |

## 11. Evaluation protocol before and after

| | Before | After |
|---|---|---|
| Reserved window (Jan 2026 onward) | Claimed untouched; fold 6 actually scored 165 targets in Jan-Feb 2026 | Those cells excluded (CLEANBLEED primary); a code guard raises on any scored target in the window |
| Design choices | Several made with test data: smoothing window, adaptive vs granular, Router motivated by test failures, calibration constants chosen across all folds | Every choice made on validation, nested by fold, with hypotheses and rules committed to git **before** each run |
| Final configuration | Not frozen | Frozen at commit `a27a23a`; 17 files sha256-locked; runner gated to reproduce P from agent code |
| Truly out-of-sample test | None | Holdout: origins from 2026-02-28 (data to 2026-10-10; 29 / 25 / 20 origins at 4 / 8 / 13 weeks), hypotheses H7.1-H7.4 pre-registered. **Not scored yet.** |

## 12. Paper claims scorecard (27 claims checked)

| Status | Count | Claims |
|---|---|---|
| Withdrawn | 13 | C1 headline numbers; C3 regime weighting "corrected" failures; C4 mode selection stable; C6 Router as a central contribution; C7 regime weights "when sample sufficient"; C8 only pre-test information used; C9 reserved period untouched; C12 rho_max not yet calibrated; C15 lags calibrated; C17 temporal-seasonal handoff; C20 FFR p = 0.042; C23 beats the strongest temporal model; C25 no access to test outcomes |
| Weakened | 8 | C5 adaptive vs granular; C13 intrinsic schedule; C14 level attribution; C19 beats the NEXUS CoT baseline; C21 unemployment generalisation; C24 temporal channel vs persistence; C26 size of the gain over naive; C27 data available at each origin |
| Holds | 5 | C2 beats naive (strongly); C10 event magnitude calibrates to 0 (numbers replaced); C11 shorter event windows (grid A); C16 macro adds little; C18 TimesFM beats naive |
| Future work | 1 | C22 spatial spillover (never evaluated) |

## 13. Better, worse, unchanged

**Better**
- Accuracy: lower MAPE at every horizon; 28% lower at 8 weeks and 20% lower at 13 weeks than the paper's system.
- Every number is clean: no test-set tuning, no reserved-window data, and every choice is validation-based and pre-registered.
- One real bug fixed (accidental equal weights for regimes with no validation data).
- Simpler: three channels instead of five, with two dead channels removed.
- Proper statistics, strong baselines, a robustness cut, a second domain, and a holdout ready to run.

**Worse (for the paper's story)**
- The best single model matches or beats the fusion, so "multi-agent fusion improves accuracy" cannot be claimed in housing.
- The regime-aware Router and macro significance results are withdrawn.
- The gain over naive is about half as large on the raw series.
- 13 claims must come out and 8 must be reworded.

**Unchanged**
- MAPE as the metric (definition identical), the 15 MSAs, the 6 folds, the data vintage, naive's numbers, T = 0.05, the macro lags and the Router mechanics.

## 14. What is still open

1. **Phase 7 holdout.** Panels and TimesFM are done. Waiting on the CoT decision (pay roughly $1-3 for gpt-4o-mini, or drop CoT before any scoring). Until it runs, every number above is from backtest folds that the team has looked at many times.
2. **Porting P into `facts_mas/`** after Phase 7, so the repo's own pipeline produces P.
3. **Paper rewrite** per `paper_mismatch_list.md` (38 items).
4. Restore the laptop's AC sleep timeout to 2 hours after Phase 7.

## 15. Suggested one-paragraph summary for Prof. Pan

"We audited FACTS-MAS end to end. The paper's headline came from an older run, and some design choices had used test data, including the reserved January-February 2026 window. We replaced those choices with validation-selected ones, fixed a weighting bug, retired two channels that calibrated to null, and froze the configuration under pre-registered rules. The frozen system's errors are 6% / 28% / 20% lower than the paper's at 4 / 8 / 13 weeks, and it beats naive decisively. But a single factor-adjusted damped ETS model is as good or better at every horizon, and the regime and macro claims do not survive proper tests. So the contribution is best framed as a rigorous protocol, not as accuracy from fusion. A pre-registered holdout on post-February-2026 data is built and ready to score."
