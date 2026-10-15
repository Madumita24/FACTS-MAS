# Paper claims status (running file)

Paper: `../FACTS_MAS.pdf` (draft of 2026-09-26). Line numbers refer to its extracted text. Updated at the end of every phase. Status values:

- **holds**: the claim survives the audit so far.
- **weakened**: the claim is directionally supported but overstated, fragile, or based on contaminated evidence.
- **withdrawn**: the claim is false for the current system or rests on an invalid procedure; it must be removed or rewritten.
- **not yet checked**: scheduled for a later phase.

"R1" = current code with the seasonal window re-selected on validation (window 7). "R1c" = R1 + zero-support fallback + validation targets bounded to train_end (proposed baseline, Phase 2). CLEANBLEED = the 165 fold-6 cells with targets in Jan-Feb 2026 excluded. All are PROVISIONAL for T = 0.05, the Router design and the adaptive choice.

Last updated: Phase 2, 2026-10-15.

| # | Claim (paper location) | Status | Evidence | Updated |
|---|---|---|---|---|
| C1 | Headline MAPE 2.243 / 5.038 / 7.020 vs naive 4.194 / 7.371 / 11.168 (abstract l.66-68, Table 5, l.1031, l.1139) | **withdrawn** | From the 2026-07-29 run with older agents. Today's code gives 2.094 / 4.973 / 6.855, and that number used a test-tuned seasonal window. R1 CLEANBLEED: 2.089 / 4.952 / 7.058 (naive 4.194 / 7.225 / 10.814). R1c CLEANBLEED: 2.122 / 4.468 / 6.987. Final number awaits the baseline decision. | P0, P1, P2 |
| C2 | FACTS-MAS beats naive at all three horizons (l.721) | **holds** | Holds in every configuration tested (PROVISIONAL, R1, R1c, DISCLOSURE and CLEANBLEED). Statistical tests in Phase 4. | P2 |
| C3 | Regime-aware weighting "corrected" the 13-week failures in hiking and cutting; pooled 8.773 / 14.309 vs adaptive 7.270 / 10.001 vs naive 8.059 / 13.043 (l.763-779, l.981-985) | **weakened** | Single substantive hiking episode in the whole sample (2022-06 to 2023-09); all cutting test cells are one 13-week episode in fold 5; about 20% of the Router's 13-week gain came from the equal-weight fallback artifact (P1). Phase 2 adds: under R1 the pooled fusion already beats naive in both regimes (hiking 6.387 vs 8.059; cutting 12.034 vs 13.043), so the motivating failure no longer exists. The cutting "correction" (10.181) is entirely the equal-weight artifact. In hiking the Router makes 13 weeks worse under R1 (7.236 vs 6.387). Under R1c the Router changes cutting by 0 and hiking by 0.10. **Auditor's view: the "corrected failures" sentence and table should be withdrawn; the author decides.** | P1, P2 |
| C4 | Mode selection "stable across all six folds" (l.~1031) | **withdrawn** | Current system: folds 4 and 6 choose fold weights at 8 weeks. | P0 |
| C5 | Adaptive vs adaptive-granular comparison (13wk 7.020 vs 7.645, l.710-714, l.1040) | **weakened** | Choice made on test data (including reserved-window cells); numbers from the July system. | P1 |
| C6 | Adaptive (Router) weighting as a central contribution; its 13-week value (contribution 2, l.78-80; Sec. 3.5) | **weakened** | Router effect at 13wk is 1.031 (R1), 0.827 (R1 + fix 2), 1.142 (R1c): sensitive to configuration. 60-75% of it comes from fold 4, and under R1c it sits in stable-regime cells, which suggests horizon conditioning (more seasonal weight at 13 weeks) rather than regime conditioning. At 8 weeks the Router hurts under R1 (-0.519) and is neutral under R1c (+0.009). The "- Router" ablation also removes horizon conditioning, so it cannot isolate regime value. Phase 4 placebo pending. | P2 |
| C7 | Weights depend on regime "when the validation sample is sufficient" (contribution 2, l.80; l.189) | **withdrawn (as implemented)** | Production gave equal weights (0.2 each, incl. naive clones) to regimes with zero validation weeks (fold 2 hiking, fold 5 cutting). No sufficiency rule existed. Fix 2 implements one (13 distinct weeks). | P1, P2 |
| C8 | Fitting, calibration, lags, fusion weights and the weighting rule use only pre-test information (l.476-477) | **withdrawn (as stated)** | Seasonal smoothing window chosen on fold 6 test incl. reserved window; adaptive vs granular chosen on test; calibration constants chosen once across all folds (later folds' validation windows postdate earlier test windows). | P1 |
| C9 | Reserved period untouched / 13-week embargo protects the test boundary (Sec. 4.1) | **withdrawn (as stated)** | Fold 6 scores 165 targets in Jan-Feb 2026. Fusion validation reads targets inside the embargo (not a test leak). | P0, P1 |
| C10 | Event magnitude calibrates to zero; Table 10 values 5.813 ... 9.357 (l.662-675, l.939-954) | **holds (numbers replaced)** | Original sweep did not bound targets. Bounded re-run: 5.704 / 5.869 / 6.112 / 6.413 / 7.110 / 8.121 / 9.234; 0 wins 6/6. Table 10 to be replaced with footnote (Umang, P1 decision). | P1 |
| C11 | Shorter event windows are better; extending the grid to 0 (event duration study, l.~660) | **holds (grid B numbers unverified)** | Grid A reproduces exactly; bounded values +6%, same conclusion. Grid B table cannot be reproduced; if no script is found it is dropped from the paper (P1 decision). | P1 |
| C12 | rho_max = 0.30 "should eventually" be calibrated (p.16) | **withdrawn (outdated)** | It was calibrated: 0 wins 6/6 (bounded re-run 7.923 vs 10.259 at 0.30). Production still uses 0.30. | P0 |
| C13 | Quadratic intrinsic schedule gives weak short-horizon pull (l.641-642) | **weakened** | The schedule is horizon-inconsistent: the forecast for t+4 pulls 0.30 of the gap if H = 4 but 0.028 if H = 13 (a 10.6x difference). | P2 |
| C14 | Factor attribution C_k = w_k * y_hat_k summarizes which channels were useful (eq. 11, l.400-422) | **weakened** | Level attribution credits naive clones: event gets 4.5% / 5.6% level share at 8 / 13 weeks but contributes exactly 0 to the forecasted change. Change-based attribution w_k * (y_hat_k - y_t) separates them. | P2 |
| C15 | Lags are calibrated (intro, contribution 3, l.477) | **withdrawn** | Macro lags 6/10/10/5 weeks are literature priors, not fitted (`macro_agent.py:24-29`). | P1 |
| C16 | The exogenous (macro) channel adds little robust incremental skill (l.882) | **holds** | Standalone test MAPE at 13 weeks is worse than naive in 5 of 6 folds; its 13-week weight came from one thin regime cell (fold 3 hiking) and the equal-weight fallback. | P2 |
| C17 | Temporal and seasonal channels exchange roles across horizons (not in the paper; in commit messages) | **holds** | R1 mean weights: temporal 0.673 / 0.403 / 0.063, seasonal 0.320 / 0.342 / 0.677. R1c: temporal 0.674 / 0.590 / 0.260, seasonal 0.317 / 0.405 / 0.733. | P2 |
| C18 | Beats naive on 15/15 MSAs at 4 and 8 weeks, 14/15 at 13 weeks (l.751) | not yet checked | Phase 3 table. | |
| C19 | Noticeably improves on the NEXUS chain-of-thought baseline, fold 1 (l.68, l.745-757) | not yet checked | Phase 6; wording ("NEXUS-paper CoT baseline", not the NEXUS system) to fix; LLM look-ahead risk in 2021. | |
| C20 | Only federal funds crosses 5% in the macro panel (p = 0.042) (l.612) | not yet checked | Phase 4 re-specification and multiple-testing correction. | |
| C21 | Unemployment domain generalization (l.69-70, Sec. 4) | not yet checked | Phase 6. | |
| C22 | Spatial spillover graph (reference doc says "Built") | not yet checked | Phase 6; ships switched off. | |
