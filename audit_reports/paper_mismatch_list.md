# Paper mismatch list

Every place where the paper draft (`../FACTS_MAS.pdf`, 2026-09-26; line numbers from the extracted text) disagrees with the code or the audited results. Built 2026-10-16 (Phase 6). It complements `paper_claims_status.md`, which tracks the status of each claim; this file lists **what has to change in the text**.

Abbreviations:
- **P**: frozen production system (`FINAL_CONFIG.md`).
- **FA-ETS-d**: factor-adjusted damped ETS, the strongest baseline.
- **CLEANBLEED**: the 165 fold-6 cells with targets in Jan-Feb 2026 excluded.

Numbers are TEST, CLEANBLEED, smoothed cut, unless stated.

## A. Headline numbers and tables

| # | Location | Paper says | Code / results say | Evidence | Required change |
|---|---|---|---|---|---|
| A1 | Abstract l.66-68; Table 5 (l.725-730); l.1031; l.1139 | FACTS-MAS 2.243 / 5.038 / 7.020 vs naive 4.194 / 7.371 / 11.168 | These are the 2026-07-29 run (older seasonal and event agents). Frozen P: **2.118 / 3.622 / 5.731** vs naive 4.194 / 7.225 / 10.814; FA-ETS-d 1.974 / 3.608 / 5.664. Raw cut: P 3.642 / 5.159 / 7.189 vs naive 4.785 / 7.882 / 11.438. | P0 F0.1-F0.3; P1 version decomposition; P4 F4.1-F4.2 | Replace with P, FA-ETS-d, naive, both cuts, with block-bootstrap CIs; name the smoothed cut in the abstract and give both gains |
| A2 | Table 6 (l.732-737), l.745-757 | Fold-1 FACTS-MAS 2.156 / 5.126 / 9.155 vs TimesFM and NEXUS-CoT | July system exactly (V0 fold 1). Current P fold 1: 2.088 / 4.139 / 7.684; FA-ETS-d 1.937 / 4.112 / 7.395. vs TimesFM: significant at 4 weeks only (29 weeks of data). | P3 F3.9; P4 F4.9 | Recompute; add TimesFM on all folds (Phase 6c); CoT on holdout only |
| A3 | Table 11 (l.996-1015) | Per-regime MAPE for fold / regime / adaptive (July numbers); 13-week hiking and cutting "corrected" | July system. Under the current code pooled fusion already beats naive in hiking and cutting; the cutting "correction" was an equal-weight fallback artifact; regime labels carry no detectable information (placebo) | P1 F1.11; P2 F2.6; P4 F4.8; claims C3, C6 (both withdrawn) | Remove the table and the "corrected failures" text (l.763-779, l.981-985) |
| A4 | Table 12 (l.1017-1022), l.710-714, l.1040 | Adaptive vs adaptive-granular comparison | Chosen on test data (incl. reserved-window cells); July numbers | P1 F1.4(c) | Remove, or report as a test-informed exploration with current numbers |
| A5 | Tables 3 (l.671) and 10 (l.947) | Event impact-scale table printed twice; 5.813 ... 9.357 | Duplicate tables. Values come from a sweep that did not bound targets to train_end. Bounded: 5.704 / 5.869 / 6.112 / 6.413 / 7.110 / 8.121 / 9.234 (0 still wins 6/6) | P1 F1.2 | Keep one table, with bounded values and a footnote on why they changed (authors' decision) |
| A6 | l.659-661 (event duration) | "progressively shorter windows ... zero influence best in all six folds" | Grid A reproduces (bounded values +6%, same conclusion). The extended grid (0-8 weeks, calibration.md) cannot be reproduced | P1 F1.2 | If no script is found, drop the grid-B numbers (authors' decision) |
| A7 | Table 9 (l.893-899), l.612 | Federal funds p = 0.042, "crosses the conventional 5% threshold" | Reproduces exactly, but does not survive Holm (0.170), a wild cluster bootstrap (0.073) or Driscoll-Kraay (0.71) | P4 F4.7; claim C20 (withdrawn) | Report the corrected p-values; remove the "crosses 5%" sentence |
| A8 | l.566-570 (Sec. 4.2.1) | ETS vs naive 2.73 / 5.51 / 7.79 vs 4.69 / 9.34 / 13.67; "14 of 15 MSAs" | Single-origin numbers from a superseded script (one forecast per fold at train_end, scoring horizons in the embargo). Full origin set: Holt 2.368 / 5.449 / 10.309 vs naive 4.194 / 7.371 / 11.168 (DISCLOSURE); AR beats naive in 15 / 14 / 11 MSAs | commit `c8ae265` note; P3 table | Replace with full-origin numbers |
| A9 | l.1029-1031 | Mode selection "stable across all six folds" | Folds 4 and 6 choose fold weights at 8 weeks | P0 F0.4 | Remove |

## B. Method descriptions

| # | Location | Paper says | Code / results say | Evidence | Required change |
|---|---|---|---|---|---|
| B1 | Sec. 3.4 eq. 8 (l.323-330) | "temperature-controlled softmax" with tau, value never given | T = 0.05, hand-set to give the temporal channel about 90-98% of the weight; not swept. Phase 5's validation-chosen temperature was not promoted | `fusion_baseline.py:204`, `:226-232`; P5 F5.4 | State T = 0.05 and how it was set |
| B2 | Intro l.52, contribution 3 l.83, l.441, l.477, l.498 | Lags are calibrated | Macro lags 6 / 10 / 10 / 5 weeks are literature priors, not fitted | `macro_agent.py:24-29` | Say the lags are pre-specified |
| B3 | l.816 (p.16) | rho_max = 0.30 "is still a modeling choice and should eventually" be calibrated | It was calibrated: 0 wins 6/6 folds. Production kept 0.30; the intrinsic channel is now retired | `rho_max_calibration.csv`; claims C12 | Report the calibration result and the retirement |
| B4 | l.641-642 | Quadratic schedule rho_h = rho_max (h/H)^2, "weak short-horizon pull" | Horizon-inconsistent: the t+4 forecast pulls 0.30 of the gap if H = 4 but 0.028 if H = 13. Known issue, not fixed (channel retired) | P2 F2.1 | Disclose as a known issue of the retired channel |
| B5 | Sec. 3.7 eq. 11 (l.400-422) | Attribution C_k = w_k * y_hat_k | Level attribution credits naive clones (event 4.5-5.6% level share, 0% of the forecasted change) | P2 F2.4 | Change-based attribution w_k (y_hat_k - y_t) primary; level-based in an appendix (authors' decision) |
| B6 | Eq. 9 w_k(H,t); Sec. 3.5 | Weights depend on horizon | At 4 weeks every fold uses fold weights, which pool skill over all three horizons, so the 4-week forecast is not horizon-conditioned | P0 F0.11 | Describe the actual pooling |
| B7 | Contribution 2 (l.78-80), l.189 | Regime weights are used "when the validation sample is sufficient" | No sufficiency rule existed: regimes with zero validation weeks got equal weights (0.2 each, incl. naive clones). Frozen P adds a 13-distinct-week rule | P1 F1.11; P2 fix 2 | Describe the rule actually used |
| B8 | Figure 1 caption (l.205-207); l.228-229 | A calibration step in the per-origin loop; "a calibration step sets the channel weights" | Calibration is a manual, offline sweep whose result a human applies; weights come from softmax over validation skill, not from calibration | `calibration.py` docstring ("a human applies") | Redraw Figure 1 and fix the sentence |
| B9 | Sec. 2.4 (l.182-190) | "Efficiency extensions ... early-exit gating, spillover analysis, statistical output constraints, and batched reasoning" | The repo implements these four NEXUS modifications (A-D, commit `cc41229`), but they are not in the forecast path and are never evaluated in the paper. Ambiguous whether they are claimed for FACTS-MAS | `facts_mas/nodes/*`; `evaluate_nexus_mods.py` | Say explicitly whether they are part of FACTS-MAS and, if so, report them; otherwise state they are NEXUS's |
| B10 | Sec. 4 (event channel), throughout | LLM used for event interpretation and the CoT baseline is never named | `gpt-4o-mini` (event agent and CoT baseline, temperature 0.1 for CoT) | `event_agent.py:26`, `cot_baseline.py:7`, `:28` | Name the model, version and stated knowledge cutoff; note the look-ahead risk for 2021-2023 folds |
| B11 | Sec. 4 (seasonal channel) | Ratio-to-moving-average seasonal agent | The seasonal channel is exactly M4's Naive2 with leakage-safe window-7 factors; window 7 was re-selected on validation (window 5 had been chosen on fold-6 test incl. reserved-window cells) | P1 F1.4(a), F1.13; P3 F3.6 | Name it Naive2; report the window and how it was selected |
| B12 | Data section | Target described as housing inventory | Zillow's smoothed series (a four-week trailing average per Zillow's 2020 notes) | `pull_zillow.py:9`; P5 F5.8 | State the cut; raw results in the main table |
| B13 | l.63-64 | "models are trained only on observations available through that week" | Zillow posts each week 7-13 days later; every method treats inventory at t as known at t | P5 F5.7; claim C27 | Disclose |
| B14 | Sec. 4 (channels) | Five channels in the housing forecast | Event and intrinsic retired from the forecast path (both calibrated to null); production uses temporal, seasonal and exogenous; at 8 and 13 weeks the temporal channel is FA-ETS-d | `FINAL_CONFIG.md` | Describe the frozen configuration; keep the retired channels as tested nulls |

## C. Protocol claims

| # | Location | Paper says | Code / results say | Evidence | Required change |
|---|---|---|---|---|---|
| C1 | l.476-477, l.519 | Fitting, calibration, lags, weights and the weighting rule use only pre-test information; "all weights and constants are chosen without access to test outcomes" | Seasonal window chosen on fold-6 test (incl. reserved window); adaptive vs granular chosen on test; calibration constants chosen once across all six folds | P1 F1.3, F1.4 | Rewrite; disclose each test-informed choice and its replacement |
| C2 | l.690-691 | Data from January 2026 onward reserved | Fold 6 scores 165 targets in Jan-Feb 2026 (45 at 8 weeks, 120 at 13 weeks); the clean holdout now starts after 2026-02-21 | P0 F0.6; P1 F1.7 | Disclose; report CLEANBLEED as primary |
| C3 | Embargo description (l.479-481, l.692-694) | 13-week embargo protects the train/test boundary | True for test; but fusion validation read targets inside the embargo (fixed in P by bounding validation targets to train_end) | P0 F0.10 | Describe the bound |

## D. Comparative claims and wording

| # | Location | Paper says | Code / results say | Evidence | Required change |
|---|---|---|---|---|---|
| D1 | Baselines l.514; Sec. 4.2 | Compared with "the strongest standalone temporal model" | FA-ETS-d is stronger than every FACTS-MAS variant; P is not better than FA-ETS-d at any horizon (equivalent at 8 weeks, marginally worse at 13, worse at 4) | P3 F3.1-F3.3; P4 F4.2 | Add FA-ETS-d as the strongest baseline; no accuracy claim over it |
| D2 | l.68, l.89, l.192, l.745-757 | "improves on the NEXUS chain-of-thought baseline" | It is a NEXUS-*paper-style* CoT baseline run by the authors with gpt-4o-mini, not the NEXUS system; fold 1 only; inside the LLM's training window | `cot_baseline.py` | Make every mention precise; CoT moves to the holdout |
| D3 | l.880 | "The temporal channel improves strongly on persistence" | True at 4 weeks; at 13 weeks Holt is worse than naive in folds 3-6; on the raw cut Holt does not beat naive at any horizon | P3 F3.7; P5 raw stats | Qualify by horizon and cut |
| D4 | Table 11 / text | Router 13-week gain presented as regime-driven | 60-75% of the Router's 13-week effect comes from one fold (fold 4, stable regime); random labels do as well | P2 F2.7; P4 F4.8 | Remove the regime interpretation (C6 withdrawn) |
| D5 | Table 11, 8-week rows | Adaptive picks regime weights at 8 weeks (hiking cell shows fold 4.733 vs adaptive 5.401) without comment | In the then-current system the Router cost 0.445 MAPE overall at 8 weeks (ablation); not disclosed. Under P it is neutral (+0.009) | P2 F2.7 | Disclose |

## E. Content claimed but absent, or present in the repo but absent from the paper

| # | Location | Issue | Required change |
|---|---|---|---|
| E1 | Sec. 4 opening (l.462-467) | Says three cases (housing, energy, unemployment) "are analyzed"; energy has only a planned channel map (Table 7) and no results; unemployment results exist in the repo (`domains/ui_claims/`) but are not in the paper | Say energy is a planned configuration; add the unemployment results from Phase 6d or remove the claim |
| E2 | Whole paper | Ablation table, weights by horizon (the temporal/seasonal handoff), and standalone channel MAPEs are absent | Add the ablation (with the corrected definitions, P2 F2.8), the channel weights, and the standalone table (P3) |
| E3 | Sec. 2.4, Figure 1 | Spatial spillover is presented as a capability; the reference doc marks it "Built"; it ships switched off and was never evaluated | Mark as **future work, not evaluated** |
| E4 | Discussion | The listing-flow channel (Phase 5) is a tested null and supports the protocol framing | Add as a tested channel (authors' choice) |

## F. Reference document (not the paper)

| # | Item | Finding |
|---|---|---|
| F1 | Event confidence gate | The reference doc says "< 0.60"; the code (`event_agent.py:147-168`) and the paper (l.650) both use **<= 0.60**. The reference doc is wrong, not the paper. |
| F2 | Spillover | Marked "Built"; it ships switched off. |
