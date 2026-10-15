# Phase 1 log: pre-registration and running notes

This file is written BEFORE each run. Hypotheses and selection rules below were committed before any Phase 1 number was computed (see git history of this file). Results go in `phase1_YYYYMMDD.md`, not here, except short "outcome" lines appended after each run.

## Standing rules for Phase 1 (from Umang, 2026-10-15)

- Baseline = current cache, labelled **PROVISIONAL** in every table (seasonal window was tuned on fold 6 test).
- Report every headline both with and without the 165 fold-6 bleed cells (Option A). Option B (exclude) becomes primary in final tables.
- Hard guard: scoring any target dated on or after 2026-01-01 raises, unless the run is flagged `disclosure=True`. Disclosure runs are labelled DISCLOSURE in all outputs.
- Nothing dated after 2026-02-21 is loaded or summarized. The audit loader truncates `aligned_weekly.csv` at 2026-02-21 on read and asserts it. Phase 7 holdout = origins after 2026-02-21.
- New files only under `tests_audit/` and `audit_reports/`.
- T = 0.05 is flagged for nested re-selection in Phase 5. Not touched in Phase 1.

## Pre-registered hypotheses

**H1.1 Seasonal truncation.** `compute_seasonal_factors` filters to dates on or before the origin before taking the centred MA, so forecasts from a series truncated at t and from the full (capped) series will be bit-identical. Expect PASS. The end-of-sample ratios use a one-sided partial window (`min_periods=26`); that is a bias question, not a leak, and I will describe it, not "fix" it.

**H1.2 Calibration targets.** The current scorer (`calibration.py:229`) drops any origin whose last target is after the window end. A spy test will confirm every scored target is on or before `train_end`. The paper's event impact-scale table (5.813 ... 9.357) was produced by a script that did NOT bound targets and was never re-run. Prediction: an exact re-run without the bound reproduces it; with the bound, every value moves by under 5% relative, the ordering stays monotonic, and 0 still wins 6 of 6 folds.

**H1.3 Event window sweep.** Same as H1.2 for the declaration-window table in `calibration.md` (5.442 ... 8.905 and 5.321 ... 6.974). Prediction: ordering unchanged, 0 still best.

**H1.4 F0.10 (bound validation targets to train_end in fusion weights, regime grid and mode selection).** Removes the last h weeks of origins from each validation window per horizon. Prediction: every fold weight moves by under 0.05 absolute; at most 2 of 18 (fold, horizon) mode choices flip; pooled test MAPE moves by under 0.10 at every horizon. Direction not predicted.

**H1.5 Imputed values.** No NaN or inf enters the skill. Validation cells touching an imputed date are under 1% of the window; excluding them moves weights by under 0.01. Test headline is unchanged by construction (0 test cells touch an imputed date, Phase 0).

**H1.6 Bleed exclusion.** Excluding the 165 bleed cells changes the pooled 8wk and 13wk numbers by under 0.10 (45 and 120 of 2,610 cells). Fold 6 alone may move more. Naive moves too; I will report both.

**H1.7 Version decomposition (disclosure only).** Steps: V0 = July agents (seasonal unsmoothed, event magnitude 0.10, window 8) under today's harness; V1 = V0 + smoothing window 5; V2 = V1 + event magnitude removed (= current). Also V1b = V0 + event removal only, to check order dependence.
- V0 must reproduce 2.243 / 5.038 / 7.020. If it does not, some other change exists between July and now and I stop that line and report.
- Prediction: smoothing explains at least 80% of the 4-week gain (2.243 to 2.094). Event removal explains under 0.05 at each horizon, because the event channel differed from naive in only a few dozen test cells and held 0.4 to 5% weight.

**H1.8 Nested seasonal window re-selection.** Selection rule, fixed now:
- Candidates: window in {1 (no smoothing), 3, 5, 7}. Same grid as the original sweep. No other values.
- For each fold f separately: choose the window with the lowest mean per-cell MAPE of the **standalone seasonal channel**, over all validation cells of fold f (origins in [train_end - 104 weeks, train_end], all 15 MSAs, horizons 4, 8, 13 pooled), **with targets bounded to on or before train_end**. Tie: smaller window.
- Standalone channel MAPE is the criterion because that is what the original sweep used.
- Then rebuild seasonal outputs with fold f's window for fold f's validation cells (weights) and test cells (scoring), and rerun the full adaptive pipeline. Two reruns: R1 = nested window, production harness otherwise; R2 = nested window + F0.10 bound. Both with and without bleed cells.
- Prediction: interior windows (3 or 5) chosen in at least 4 of 6 folds; R1 test MAPE within 0.05 of the PROVISIONAL baseline at every horizon if 5 is mostly chosen.

**H1.9 Regime episodes.** The regime label is national (fed funds), so it is identical across MSAs. Prediction: each validation window has at most 2 hiking episodes and at most 1 cutting episode; folds 2 and 5 have zero validation origins in the regime that dominates their test hiking / cutting cells. From reading the code, `skill_score_median([])` returns -1.0 for every agent, so softmax gives **uniform 0.2 weights** to every channel (including the two naive clones) for such a regime. If fold 2 (hiking) and fold 5 (cutting) use regime weights at 8 and 13 weeks, those test cells are fused with equal weights. I will confirm from the weight grid.

**H1.10 COVID declarations.** No effect on current forecasts (event output = naive everywhere). In the July version, COVID rows affect the forecast only while age <= 8 weeks from the 2020-01-20 incident start, i.e. at most a few weeks in early 2020, which lie only in folds 1-2 validation windows.

**H1.11 Static vintage.** Replacing the intrinsic channel's output with naive (same weights) moves fused test MAPE by under 0.07 at every horizon. A vintage effect can only act through this channel, so this bounds its practical size.

**Seasonal sweep rebuild (reproducibility of the original selection).** The original table (window none / 3 / 5 / 7 on fold 6 test, 15 metros) will be rebuilt as a DISCLOSURE run. I will try at most these three definitions, in this order, and stop at the first that matches to 3 decimals: (a) standalone seasonal channel, mean per-cell MAPE on fold 6 test cells; (b) fused adaptive FACTS-MAS; (c) fused fold-level FACTS-MAS. If none matches I report the mismatch and do not search further.

## Running notes

(appended below as runs complete)
