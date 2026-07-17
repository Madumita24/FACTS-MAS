# Macro Agent — Build & Validation Notes

Status: Step 4 (agent) and Step 4b (tests + marginal-contribution check) built and passing. **Macro signal investigation complete and closed** — three independent, increasingly rigorous validation rounds all converge on a well-earned null. `macro_agent.py` was never modified based on any of these findings; that's a decision for later, not made here.

---

## Scope

The Macro Agent produces a **modifier**, not a full forecast: a ridge regression of weekly inventory pct-change on 4 lagged FRED features (mortgage_rate, fed_funds, cpi, unemployment), matching Umang's `MacroModifier` schema (`model_name`, `modifier`, `lag_weeks_used`). It's meant to adjust the AR Agent's output, not replace it.

## Files

| File | Role |
|---|---|
| [macro_agent.py](macro_agent.py) | `fit_and_forecast` / `run_macro_agent` (Step 4) — unmodified throughout the investigation below |
| [test_macro_agent.py](test_macro_agent.py) | Unit tests + compounding/additive combiner comparison (Step 4b) |
| [validate_macro_signal.py](validate_macro_signal.py) | Validation round 1: fold-based correlation, single-week modifier |
| [validate_macro_lags.py](validate_macro_lags.py) | Validation round 2: lag grid search + horizon-level regression target |
| [validate_macro_panel.py](validate_macro_panel.py) | Validation round 3: two-way FE panel regression, MSA-clustered SEs — the rigorous final test |

---

## Step 4 — Agent design

`fit_and_forecast(df, msa, fold_train_end, horizon_weeks)` predicts a **single constant pct-change "pressure" value** from one macro snapshot at `fold_train_end - lag_weeks` per feature, then repeats it across the whole horizon. This is a deliberate v1 simplification: predicting a *different* modifier per horizon week would require future macro values that don't exist yet for the shorter-lag features (unemployment's 5wk lag < the 13wk horizon).

**Fixed midpoint lags**, taken from the transmission-lag ranges documented in `align_data.py`: mortgage_rate 6wk, fed_funds 10wk, cpi 10wk, unemployment 5wk — starting points from domain literature, not fitted.

**No-lookahead guard**: `assert_macro_no_lookahead` raises if any macro source date used is later than `fold_train_end - lag_weeks`, verified with a dedicated injection test (deliberately-violating input, confirmed to raise) plus an exact-boundary test (confirmed *not* to raise on the legal edge case).

## Step 4b — Combiner comparison (AR alone vs. AR + macro modifier)

Two ad-hoc diagnostic combination rules were tested (neither is Umang's real fusion logic):

| Horizon | AR alone | + macro (compounding) | + macro (additive, 1x) |
|---|---|---|---|
| 4wk | 2.73% | 3.71% (-0.98pp HURTS) | 2.98% (-0.25pp, mild HURT) |
| 8wk | 5.51% | 6.95% (-1.44pp HURTS) | 5.55% (-0.05pp, NEUTRAL) |
| 13wk | 7.79% | 10.71% (-2.92pp HURTS) | 7.80% (-0.01pp, NEUTRAL) |

**Finding**: compounding a constant weekly modifier `(1+modifier)^week` across up to 13 weeks mechanically amplifies a small, noisy per-week estimate into a large swing — most of the "macro hurts" story under compounding was a combiner artifact, not a property of the macro signal itself. This motivated checking the signal directly, independent of any combiner.

---

## The macro-signal investigation

Three rounds, each addressing a specific weakness found in the previous one. Read together, not in isolation — the later rounds are more trustworthy, but the full arc is the actual evidence.

### Round 1 — `validate_macro_signal.py`: fold-based correlation, single-week modifier

Directly correlated the agent's actual modifier output (90 msa/fold pairs, midpoint lags) against realized inventory pct-change, per horizon:

| Horizon | n | r | p |
|---|---|---|---|
| 4wk | 90 | +0.011 | 0.920 |
| 8wk | 90 | -0.025 | 0.814 |
| 13wk | 90 | -0.177 | 0.095 |

**Clean null.** No horizon distinguishable from noise.

### Round 2 — `validate_macro_lags.py`: lag grid search + horizon-level regression target

**Step A** (raw lagged feature level vs. realized 13wk pct-change, swept across each feature's full documented lag range):

| Feature | best lag | r | p |
|---|---|---|---|
| mortgage_rate | 7wk | +0.628 | <0.0001 |
| fed_funds | 12wk | +0.678 | <0.0001 |
| cpi | 12wk | +0.607 | <0.0001 |
| unemployment | 5wk | -0.240 | 0.0225 |

These numbers looked strong but **were flagged as unreliable at the time, not treated as a finding**: `align_data.py` broadcasts macro data identically across all 15 MSAs per date, so within each of the 6 folds the "90 observations" are really 6 distinct macro values repeated 15 times each — true n≈6 for the macro side, not 90. At n=6, none of these correlations would clear significance. Both macro and inventory levels also trend substantially over 2018-2025 (rate hikes, CPI's monotonic rise, the post-COVID inventory crash/recovery), which independently risks spurious level-correlation between two trending series. This round's headline numbers were the direct motivation for Round 3, not evidence of signal.

**Step B** (horizon-level regression target — predict the realized horizon-total pct-change directly, per-MSA fitted models, using Step A's lags):

| Horizon | r | p |
|---|---|---|
| 4wk | -0.131 | 0.219 |
| 8wk | -0.248 | **0.018** |
| 13wk | -0.391 | **0.0001** |

Structurally more sound than Step A (per-MSA fitted coefficients, not a shared raw value), but the significant results are **negative** — predictions trending opposite to actual outcomes. A statistically-real wrong-signed effect is a flag that something's backwards, not a usable signal.

### Round 3 — `validate_macro_panel.py`: two-way fixed-effects panel, MSA-clustered SEs (the rigorous final test)

Directly fixes both Round 2 weaknesses: **MSA fixed effects** absorb level differences, **year-quarter fixed effects** absorb the common trend (using quarter buckets rather than per-date FE specifically so the nationally-broadcast macro regressor isn't perfectly collinear with time dummies), and **standard errors clustered by MSA** (15 clusters) correctly account for the non-independence that inflated Round 2's apparent significance. Lags used were the documented midpoints, deliberately *not* Round 2's data-mined "best" lags, to avoid baking that round's flawed-selection bias into the final test.

n = 5,850 (msa, date) rows, 2018-04-14 to 2025-09-27 (respects the Jan 2026 held-out cutoff on both the row date and the 13-week-forward target date):

| Feature | lag | coef | SE (clustered) | p |
|---|---|---|---|---|
| mortgage_rate | 6wk | +0.0075 | 0.0133 | 0.575 |
| fed_funds | 10wk | **-0.0129** | 0.0063 | **0.042** |
| cpi | 10wk | -0.0047 | 0.0028 | 0.085 |
| unemployment | 5wk | -0.0015 | 0.0025 | 0.548 |

**Honest reading of this result, not just the p-value table:**
- Only 1 of 4 features crosses p<0.05 — roughly what you'd expect from chance alone across 4 tests at that threshold (~0.2 expected false positives), not strong evidence on its own.
- It's the *wrong* feature to be the "hit": mortgage_rate — the variable most directly tied to buyer affordability and seller lock-in, the textbook housing-inventory channel — is the least significant of the four (p=0.575, coefficient ≈0). Fed funds being marginally significant while mortgage rate is dead null runs against what housing economics would predict, which argues for skepticism rather than confidence in the fed_funds result.
- cpi is borderline in the same direction (p=0.085) — plausibly the same 2022-23 tightening episode bleeding across two correlated regressors, not two independent confirmations of a real effect.
- R²(within)=0.548 is the whole model (mostly the 29 quarter dummies absorbing common trend), not macro's contribution specifically — don't read it as "macro explains half of inventory change."

---

## Summary table — all three rounds side by side

| Round | Method | Key fix over prior round | Result |
|---|---|---|---|
| 1 | Fold-based correlation, single-week modifier (90 pts) | — | Clean null (all p≥0.095) |
| 2 (Step A) | Lag grid search, raw feature level (90 pts) | — | Looked strong (p<0.0001) but flagged as pseudo-replicated + spurious-trend-prone at the time |
| 2 (Step B) | Horizon-level regression target, per-MSA models | Per-MSA fitting (less pseudo-replication) | Significant at 8/13wk but **wrong-signed** |
| 3 | Two-way FE panel + MSA-clustered SEs (5,850 pts) | Fixes both pseudo-replication and spurious trend directly | 1/4 features marginally significant (fed_funds, p=0.042), theoretically the less-expected one; others null |

## Final recommendation

**A well-earned null, not an artifact of any single test's weakness.** The investigation deliberately got more rigorous at each round specifically to try to *find* a real signal if one existed, and the strongest, most defensible test (Round 3) still returns essentially nothing usable — one borderline, theoretically-backwards result out of four. Recommend deprioritizing the Macro Agent in the fusion pipeline rather than continuing to tune lags, targets, or combiners. If revisited later, the more promising angle is probably a different feature set or a genuinely different specification (e.g. macro *changes* rather than levels, or MSA-specific rather than pooled coefficients) — not further tuning of this same lag/target/combiner space, which has now been reasonably well covered.

`macro_agent.py` itself was never changed based on any of this — building a "why doesn't this help" case first, before deciding what (if anything) to do about it, per explicit instruction throughout.

## Open items

1. **Decision needed**: does the Macro Agent stay in the fusion pipeline as a documented-null / inactive component, or get removed/deprioritized entirely for now? Not decided here.
2. If revisited: consider macro *changes* (deltas) rather than *levels* as regressors, or MSA-specific coefficients instead of a pooled panel — both untested angles.
3. `ar_agent_stub.py` (Step 5, AR Agent handoff) is done; no equivalent stub exists for the Macro Agent — flag if Umang's fusion harness needs one regardless of the null result.
