# Phase 2 log: pre-registration and running notes

Written and committed BEFORE any Phase 2 number is computed. Baseline = **R1** (Phase 1; seasonal window 7 in every fold, artifact `audit_reports/artifacts/seasonal_w7_r1.pkl`, sha256 `3b530099...`). Primary column CLEANBLEED (165 fold-6 bleed cells excluded); DISCLOSURE shown alongside. T = 0.05, Router design and adaptive choice remain PROVISIONAL.

Known facts used to write these (all from Phase 1, no new numbers): regime label is national; validation support in weeks per fold is in `results_tables/phase1_regime_episodes.csv`.

## Fix 2: zero-support fallback (pre-registered rule)

- For each fold, regime r and horizon h, **support** = number of DISTINCT origin weeks with label r among that fold's validation cells at horizon h (after the F0.10 bound when F0.10 is on).
- If support < **13 weeks**, the (r, h) cell uses the **fold weights** instead of its own softmax weights. Support >= 13 keeps regime weights. Exactly 13 passes.
- Applied in BOTH places regime weights are used: the fold-vs-regime mode-selection comparison on validation, and test fusion.
- Threshold 13 is fixed. Sensitivity at 4 and 26 weeks is DISCLOSURE only and will not be used to choose.
- Reported: R1, R1 + fix 2, R1 + F0.10, R1 + fix 2 + F0.10 (the combined change). CLEANBLEED and DISCLOSURE. Per fold. Count of test cells whose weight set changes, and every (fold, h) mode-selection flip.

Expected cells that fall back (from Phase 1 episode counts): fold 1 hiking (2 weeks; no hiking test cells), fold 2 hiking (0), fold 5 cutting (0). With F0.10 also: fold 3 hiking at all horizons (13 weeks unbounded, fewer than 13 once the last h weeks are removed), fold 6 cutting.

**H2.0 predictions.**
- R1 + fix 2 vs R1 (CLEANBLEED): 13wk worse by 0.20 to 0.35 (the equal-weight benefit in folds 2 and 5 goes away; Phase 1 measured about 0.285 under PROVISIONAL); 8wk better by 0.05 to 0.15; 4wk unchanged to 3 decimals (every fold uses fold mode at 4wk), unless a mode flips.
- R1 + F0.10 alone vs R1: about -0.39 at 13wk (Phase 1 artifact), as in PROVISIONAL.
- Combined vs R1: 13wk worse (the fold-3 artifact is removed too, since fold 3 hiking falls back under the bound); 8wk better. I do not predict exact sizes.
- At most 3 of 18 (fold, h) mode choices flip under the combined change.

## Other Phase 2 hypotheses

**H2.1 Intrinsic horizon inconsistency.** `rho_h = rho_max * (h/H)^2` means the step-4 forecast pulls 0.30 of the gap when H = 4 but 0.30 * (4/13)^2 = 0.028 when H = 13. The other four channels are horizon-consistent: AR's ETS fit does not depend on H, seasonal and event are deterministic per step, macro applies one constant modifier. A strict-xfail test will record the intrinsic inconsistency; passing tests will confirm the other four.

**H2.2 Weight ordering anomaly** (13wk under PROVISIONAL: exogenous 0.160 > temporal 0.060 > event 0.053, although exogenous is worse than naive on test). Predicted mechanism: (a) at 13wk every fold uses regime weights; (b) in some (fold, regime) cells the median of pointwise skill ratios ranks macro above AR because macro's skill is near 0 with low spread while AR's 13wk skill is widely spread with a low median; (c) the zero-support cells give macro and event 0.2 each. Prediction: window-level skill (1 - MAE_k / MAE_naive, and 1 - MAPE_k / MAPE_naive) ranks AR above macro in at least 5 of 6 folds at 13wk, while the median skill ranks macro above AR in at least one (fold, regime) cell that carries test cells. I will report all three skill definitions per fold and per (fold, regime, h), and a decomposition of mean weight by (fold, regime).

**H2.3 Persistence clones.** Event output equals naive in 15,480 of 15,480 cached cells (validation and test). Total weight on event + macro + intrinsic under R1: under 0.03 at 4wk, 0.15 to 0.30 at 13wk.

**H2.4 Attribution.** Level attribution (paper eq. 11, w_k * y_k) gives each channel a share close to its weight, so naive clones look like large contributors. Change attribution (w_k * (y_k - y_t)) gives event exactly 0 and macro under 5% of the absolute forecasted change at every horizon.

**H2.5 Regime threshold sensitivity.** Under 10% of labelled weeks have a 13-week fed funds change within 0.05 of +/-0.25. I will also count how many week labels flip at thresholds 0.20 and 0.30 (descriptive only).

**H2.6 Ablation definitions.** Read from code: "- Router" (`- Factor agent`) = fold-level weights pooled over horizons AND regimes, so it removes horizon conditioning as well as regime conditioning. Channel removal = renormalise the existing weights over the remaining channels, without re-estimating mode selection. No numerical prediction.

## Rules

- Max variants for fix 2: the four configurations above plus two DISCLOSURE threshold sensitivities (4, 26). Nothing else.
- Anything else that looks worth trying goes into "Proposed for later phases", not into a run.

## Running notes

(appended as runs complete)
- 2026-10-15 `tests_audit/test_components.py`: 7 passed, 1 strict-xfail. H2.1 CONFIRMED (AR, seasonal, macro, event horizon-consistent; intrinsic not). Event == naive in 15,480 of 15,480 cached cells. R1 pinned.
- 2026-10-15 `p2_analysis.py` outcomes (CLEANBLEED unless stated):
  - H2.0 fix 2 alone CONFIRMED: 4wk unchanged (2.089); 8wk 4.952 -> 4.816 (-0.136, in predicted -0.05..-0.15); 13wk 7.058 -> 7.262 (+0.204, in predicted +0.20..+0.35). Exactly the 780 zero-support cells change (390 at 8wk, 390 at 13wk). No mode flips.
  - H2.0 F0.10 alone CONFIRMED (artifact as in Phase 1): 13wk 6.651.
  - H2.0 combined: 8wk better as predicted (4.468, -0.484). 13wk prediction "worse" WRONG: 6.987 (-0.071). 4wk +0.033 (fold weights bounded; worse in 5 of 6 folds). Mechanism checked: fold 3 hiking (13 weeks of support unbounded, < 13 bounded) falls back from regime weights macro 0.78 / AR 0.00 to fold weights AR 0.75 / seasonal 0.25: -0.370 (8wk), -0.273 (13wk) pooled. Fold 5 cutting loses the lucky equal weights: +0.171 at 13wk. Net change rests on one fold. No mode flips (<= 3 predicted).
  - Threshold sensitivity (DISCLOSURE only, not used): threshold 4 -> 8wk 4.648 / 13wk 6.987; threshold 26 -> 8wk 4.468 / 13wk 7.337 (fold 5 stable at 13wk also falls back).
  - H2.2 PARTLY CONFIRMED: macro's 0.169 mean weight at 13wk = 0.137 from fold 3 hiking (13-week cell, AR median skill -0.599) + 0.032 from the equal-weight cells. Window-level skill ranks AR above macro at 13wk in 5/6 folds (MAE ratio) but only 4/6 (MAPE ratio) (predicted >= 5 for both). Median skill pooled over regimes ranks AR above macro in 6/6 folds; the disagreement is regime-cell level, not median vs aggregate.
  - H2.3 CONFIRMED: naive-like weight (event + macro + intrinsic) under R1 0.007 / 0.255 / 0.260; under combined 0.009 / 0.005 / 0.007.
  - H2.4 PARTLY WRONG: event change share 0 (confirmed); macro change share 0.113 (8wk) and 0.092 (13wk), not < 0.05 (fold 3 hiking weight).
  - H2.5 WRONG: 43 of 344 labelled weeks (12.5%) lie within 0.05 of +/-0.25 (predicted < 10%). Threshold 0.20 flips 24 labels, 0.30 flips 19.
  - Regime cells (13wk, CLEANBLEED): under R1, fold-weight (Router-off) fusion already beats naive in hiking (6.387 vs 8.059) and cutting (12.034 vs 13.043). The Router's cutting gain (10.181) is entirely the equal-weight artifact; in hiking the Router hurts (7.236). Under combined the Router's 13wk gain sits in stable cells (6.633 vs 8.317), and 60-75% of it comes from fold 4.
- Decision needed recorded for Umang: adopt R1 + fix2 + F0.10 ("R1c") as baseline.
- Proposed for Phase 4 (not run): a horizon-conditioned, regime-blind weight set as the correct reference for the Router ablation, since "- Router" currently removes horizon conditioning too.
