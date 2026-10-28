# FACTS-MAS audit: summary

Audit of October 2026 (Phases 0-7), branch `audit-oct2026`. Paper draft: `FACTS_MAS.pdf` (2026-09-26). Frozen configuration: `FINAL_CONFIG.md`. Details: phase reports, `paper_claims_status.md`, `paper_mismatch_list.md`, `CHANGELOG.md`, `results_tables/`.

## Top 10 findings (most important first)

1. **A single model matches the fused system.** Factor-adjusted damped ETS (FA-ETS-d) is as accurate as the final system P at every horizon, both in the backtest and on the untouched holdout:
   - 8 weeks: statistically equivalent.
   - 4 weeks: FA-ETS-d better (holdout +0.105, p = 0.044).
   - 13 weeks: within 2% (descriptive).

   No fused variant beat it, so there is no accuracy claim for fusion over the best single model.
2. **The paper's headline is stale and mixes system versions.** 2.243 / 5.038 / 7.020 comes from a July 2026 run with older agents, while the ablation and weights come from the August code.
3. **Some design choices used test data, including the reserved window.**
   - The seasonal smoothing window (5) was chosen on fold 6's test window, which included January-February 2026 targets. Fold 6 also scored 165 targets in the reserved window.
   - The window was re-selected on validation only (7), and those cells were excluded.
4. **The pre-registered holdout supports all four hypotheses.** On post-February-2026 data:
   - P beats naive by about 60% (4wk −1.88 [−3.03, −0.90]; 8wk −3.52 [−5.68, −1.70]).
   - FA-ETS-d also beats naive.
   - P ≈ FA-ETS-d at 8 weeks.
   - FA-ETS-d ≤ P at 4 weeks.
5. **The regime-aware weighting claims do not survive.**
   - About 20% of the Router's 13-week gain was an equal-weight fallback artifact, and 60-75% of it came from one fold.
   - 27-46% of randomly permuted regime labels did as well as the true ones.
   - Horizon-only weights match or beat the Router.
6. **There was a weighting bug.** Regimes with no validation data gave every channel weight 0.2, including the naive clones; this affected 780 test cells. Fixed by falling back to fold weights when a regime has fewer than 13 validation weeks.
7. **The macro "federal funds p = 0.042" result does not survive re-checking:** Holm 0.170, wild cluster bootstrap 0.073, Driscoll-Kraay 0.71. The macro channel adds little.
8. **The target is Zillow's smoothed series, a four-week trailing average, which flatters every model.**
   - On the raw series the gain over naive roughly halves (holdout 4wk: 60% to 39%; backtest: 49% to 24%).
   - On raw data P is essentially the seasonal channel.
   - With a realistic 2-week publication lag, every error rises and no conclusion changes.
9. **The foundation-model and LLM baselines.**
   - TimesFM 2.5 loses to P and FA-ETS-d everywhere (backtest p ≤ 0.006; holdout 4 / 8 weeks p = 0.016 / < 0.0001).
   - CoT with gpt-5.6-luna, out-of-sample by its documented cutoff, is competitive (holdout 1.259 / 2.225 / 3.268) but not better than FA-ETS-d.
   - The paper's NEXUS-CoT comparison used fold 1, the July system and an in-sample LLM, so it is superseded.
10. **Two channels calibrated to null and were retired.**
    - The event channel's output equals naive in every cell.
    - The intrinsic channel's calibrated reversion is 0, and its schedule is horizon-inconsistent.
    - The temporal-seasonal "handoff" appears in neither domain.
    - In unemployment claims the recipe beats naive and damped FA-ETS, but not undamped FA-ETS.

## What changed in the numbers (MAPE, 4 / 8 / 13 weeks)

| | Paper | Backtest: repo before audit (Sept/Oct) | Backtest: final P | Backtest: FA-ETS-d | Holdout: final P | Holdout: FA-ETS-d |
|---|---|---|---|---|---|---|
| Smoothed | 2.243 / 5.038 / 7.020 | 2.094 / 4.986 / 7.019 | **2.118 / 3.622 / 5.731** | 1.974 / 3.608 / 5.664 | **1.243 / 1.929 / 3.045** | 1.138 / 1.936 / 3.080 |
| Naive (same cells) | 4.194 / 7.371 / 11.168 | 4.194 / 7.225 / 10.814 | 4.194 / 7.225 / 10.814 | | 3.122 / 5.445 / 8.438 | |
| Raw cut, P (naive) | not reported | | 3.642 / 5.159 / 7.189 (4.785 / 7.882 / 11.438) | 3.402 / 5.105 / 7.164 | 2.053 / 2.671 / 3.644 (3.355 / 5.306 / 7.925) | 2.028 / 2.709 / 3.736 |

- Backtest columns use clean cells (CLEANBLEED), except the paper column, which includes the 165 reserved-window cells.
- Holdout 13-week numbers are descriptive.
- Backtest and holdout absolute levels are not comparable, because the holdout period was easier for every method.
- Against the paper's system on the same clean cells, P is 6% / 28% / 20% lower. Almost all of the 8- and 13-week gain comes from adding FA-ETS-d as the temporal channel.

## What the paper must now say

1. Report P, FA-ETS-d and naive on both cuts with block-bootstrap CIs, for the backtest and the holdout. Name the smoothed cut in the abstract and give the gain over naive on both cuts.
2. Make no claim that fusion beats the best single model. P is equivalent to FA-ETS-d at 8 weeks and slightly worse at 4 weeks. Frame the contribution as the protocol:
   - every channel tested against the strongest endogenous reference;
   - nulls allowed and reported (event, intrinsic, listing flows, macro);
   - pre-registered rules;
   - an untouched holdout.
3. Withdraw:
   - the regime-correction table and text;
   - adaptive weighting as a central contribution;
   - "mode selection stable across folds";
   - FFR "crosses 5%";
   - "lags calibrated";
   - "no access to test outcomes";
   - "reserved period untouched";
   - the adaptive-vs-granular table.
4. Disclose:
   - the test-tuned smoothing window and its validation replacement;
   - the reserved-window cells;
   - the cross-fold calibration;
   - the 1-2 week publication lag;
   - the ±0.001 numerical tolerance at 4 weeks;
   - the NEXUS prompt's hard-coded "January 2025" cutoff.
5. Replace Table 6 with TimesFM on all six folds and on the holdout, and CoT on the holdout. For CoT, name gpt-5.6-luna (reasoning effort low), its documented cutoff (2026-02-16) and its self-reported cutoff ("June 2024").
6. Add the unemployment section with the agreed wording: "In unemployment claims, the frozen recipe beats naive and the damped FA-ETS reference, but not undamped FA-ETS, which is the better reference for that series. In neither domain is there evidence that fusion beats the best single model."
7. Describe the frozen configuration: three channels, with FA-ETS-d as the temporal channel at 8 and 13 weeks, and event and intrinsic retired as tested nulls. Use change-based attribution, and mark spatial spillover as future work.
