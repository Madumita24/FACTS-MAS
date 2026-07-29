# Factor Discovery & Attribution Agent (Agent 6) — Final Writeup

Status: built, validated, wired into `run_backtest.py`. **Production default is `"adaptive"`, not `"adaptive_granular"`** — see Section 9 for the final decision. Both modes documented in full below; nothing hidden or removed.

---

## 1. The discovery that started this

The full 15×6×3 backtest (fold-level weighting, `skill_score_median` + softmax, all 5 agents) beat naive overall (3/3 horizons) but **lost to naive at 13 weeks specifically during hiking and cutting regimes**:

| Regime | Fused MAPE | Naive MAPE |
|---|---|---|
| 13wk, hiking | 8.77 | 8.06 (**loss**) |
| 13wk, cutting | 14.31 | 13.04 (**loss**) |

Masked in the pooled 13wk number because "stable" origins (5,310) outnumber hiking+cutting combined (1,935+585). AR held 95%+ of fold-level weight in most folds, so its trend-smoothing behavior (a known weakness on fast-moving series — see the Miami/Phoenix limitation in `ar_agent.md`) was the prime suspect.

## 2. Diagnostic build (Steps 1-5)

Built `compute_regime_weights` and `compute_all_regime_weights` in `factor_attribution.py`, reusing `skill_score_median` and `compute_weights_softmax` from `fusion_baseline.py` (not reimplemented), filtering the validation window by `classify_regime` before scoring. Validated on fold 6:

- **AR's regime skill at 13wk**: overall +0.066 → hiking +0.006 (weakens, but never goes negative) → cutting +0.116 (*improves*). The original hypothesis ("AR fails in hiking/cutting") was only half right — AR never actually goes negative.
- **The real driver at 13wk**: Seasonality overtakes AR in cutting (0.373 vs 0.116) and stable (0.461 vs 0.087), and softmax hands it 99%+ of the weight there — this explained a previously-unexplained puzzle (fold 6's overall weight table showing Seasonality spiking to 32%).
- **8 gating rules emerged automatically**, all in hiking, targeting macro/seasonality/intrinsic — never AR. Sample sizes: hiking n=525, cutting n=195, stable n=855 per horizon — all well above the 30-point reliability threshold.

## 3. Wiring into the backtest — four modes, tested head to head

`run_backtest.py` gained `weighting_mode`: `"fold"` (original), `"regime"` (one weight set per (regime, horizon) cell, looked up per origin), `"adaptive"` (per (fold, horizon), pick whichever mode scores lower on that fold's own **validation window**, never the test window), and `"adaptive_granular"` (same idea, per (fold, horizon, regime) instead of per (fold, horizon)).

### Full 9-cell results, all four modes plus naive

| h | regime | naive | fold | regime | adaptive | adaptive_granular |
|---|---|---|---|---|---|---|
| 4 | hiking | 4.126 | 2.191 | 2.618 | **2.191** | 2.191 |
| 4 | cutting | 6.164 | 3.291 | 4.941 | **3.291** | 3.291 |
| 4 | stable | 4.001 | 2.146 | 2.269 | **2.146** | 2.146 |
| 8 | hiking | 6.213 | **4.733** | 5.401 | 5.401 | **4.733** |
| 8 | cutting | 10.116 | 7.954 | **7.564** | **7.564** | 7.954 |
| 8 | stable | 7.491 | 4.950 | **4.627** | **4.627** | 4.627 |
| 13 | hiking | 8.059 | 8.773 | **7.270** | **7.270** | 8.499 |
| 13 | cutting | 13.043 | 14.309 | **10.001** | **10.001** | 14.309 |
| 13 | stable | 12.094 | 9.444 | **6.600** | **6.600** | 6.600 |

**Overall per-horizon (n-weighted across all MSAs, folds, regimes):**

| Horizon | fold | regime | adaptive | adaptive_granular |
|---|---|---|---|---|
| 4wk | 2.191→2.243* | — | 2.243 | 2.243 |
| 8wk | — | — | 5.038 | **4.902** |
| 13wk | — | — | **7.020** | 7.645 |

*(per-cell numbers above are n-weighted within horizon; the "overall" row pools regimes, so exact figures differ slightly from a simple per-cell average — see the acceptance-bar tables below for the precise overall numbers actually used for pass/fail.)*

### `adaptive`: strong, one known gap

Matches fold-level exactly at 4wk (no cost) and regime-aware exactly at 13wk (fixing both original failing cells). One gap: **8wk/hiking**, where the pooled per-horizon comparison favored "regime" (since cutting+stable pulled the pooled validation number toward "regime" being better) even though "fold" was actually better specifically within hiking (4.733 vs 5.401 on test data). Root cause: `adaptive` selects one mode per (fold, horizon), evaluated on all three regimes pooled together — it cannot see a regime-specific reversal hiding inside a horizon-level average.

**Mode selection was identical across all 6 folds** (fold @ 4wk, regime @ 8wk and 13wk, no exceptions) — a genuinely stable pattern across 2021-2025, not a fold-6 artifact.

### `adaptive_granular`: fixes the target gap, but at a real cost elsewhere

Selecting per (fold, horizon, regime) instead of per (fold, horizon) **does fix 8wk/hiking** (5.401 → 4.733, matching fold-level's best result exactly) — confirming the intended mechanism works. But it is **not a strict improvement**: three cells got *worse* relative to `adaptive`:

| Cell | adaptive | adaptive_granular | Δ |
|---|---|---|---|
| 8wk, hiking | 5.401 | **4.733** | fixed (as intended) |
| 8wk, cutting | 7.564 | 7.954 | **regressed** |
| 13wk, hiking | 7.270 | 8.499 | **regressed** |
| 13wk, cutting | 10.001 | 14.309 | **regressed back to fold-level's original failing value** |

**Root cause, found by inspecting the actual mode-selection log**: cutting and (in some folds) hiking are rare regimes. When the granular selection criterion is restricted to a single regime within a single fold's 104-week validation window, several (fold, regime) combinations have **zero validation points at all** for that regime (e.g. Fold 2's hiking regime, Folds 3-5's cutting regime never occurred in their respective validation windows) — both candidate MAPEs come back as `inf`, and the tie-break silently defaults to `"fold"`. In several of those folds, `"fold"` turns out to be the *wrong* choice once applied to that regime's actual test-window data, which does contain the regime that was absent from the training-window validation slice.

This is the same tension the night's sample-size checks were built to catch (`MIN_RELIABLE_N`, `reliability_notes`), now manifesting as a genuine performance cost rather than a theoretical risk: **granularity and statistical reliability trade off directly** — finer selection criteria see less data per cell, and the cells that most need regime-awareness (rare regimes) are exactly the ones most likely to run out of data to make that call reliably.

**Net effect**: 4wk unchanged, 8wk improves slightly overall (5.038→4.902), **13wk gets worse overall** (7.020→7.645) — worse at the exact horizon this whole investigation started from, even though it's still comfortably better than fold-level's original 9.641 and still clears naive.

## 4. Acceptance bar — met by both `adaptive` and `adaptive_granular`

| Mode | 4wk | 8wk | 13wk | Bar |
|---|---|---|---|---|
| adaptive | 2.243 vs naive 4.194 ✅ | 5.038 vs 7.371 ✅ | 7.020 vs 11.168 ✅ | MET 3/3 |
| adaptive_granular | 2.243 vs naive 4.194 ✅ | 4.902 vs 7.371 ✅ | 7.645 vs 11.168 ✅ | MET 3/3 |

Both pass. The bar doesn't distinguish them — the honest comparison is `adaptive` vs `adaptive_granular` directly, not either vs naive.

## 5. Production wiring

`run_backtest()`'s `weighting_mode` default is `"adaptive"` (see Section 9 for the full decision history — it was briefly set to `"adaptive_granular"` before being reverted once Section 3's regressions were found). `"adaptive_granular"` remains fully implemented and available as an explicit option, not deleted.

## 6. Auto-generated gating rules (fold 6 example, from the diagnostic build)

```
downweight_macro_hiking_4wk       downweight_seasonality_hiking_4wk    downweight_intrinsic_hiking_4wk
downweight_macro_hiking_8wk       downweight_seasonality_hiking_8wk    downweight_intrinsic_hiking_8wk
downweight_macro_hiking_13wk      downweight_seasonality_hiking_13wk
```

All 8 rules are hiking-specific and never target AR — consistent with Section 2's finding that hiking is where macro/seasonality/intrinsic break down, not where AR does. `generate_production_report()` (in `factor_attribution.py`) annotates each rule with which weighting mode (`fold` or `regime`) that specific cell actually uses in production, and produces a plain-language per-fold summary, e.g.:

> Fold 6: 4wk uses pooled fold-level weighting across all regimes; 8wk uses regime-conditional weighting during cutting and stable, pooled fold-level weighting during hiking; 13wk uses regime-conditional weighting across all regimes.

## 7. Files

| File | Role |
|---|---|
| `facts_mas/factor_attribution.py` | `compute_regime_weights`, `compute_all_regime_weights`, `generate_factor_attribution_output`, `evaluate_validation_mape` (with `regime_filter` for granular selection), `generate_production_report` |
| `facts_mas/run_backtest.py` | `weighting_mode` parameter: `"fold"`, `"regime"`, `"adaptive"` (default), `"adaptive_granular"` |
| `facts_mas/factor_attribution.md` | this document |

## 8. Summary for the record

Regime-aware weighting is real and it works — the original 13wk hiking/cutting losses are fixed, decisively, by both `adaptive` and `adaptive_granular`. The finer-grained version fixes one more specific gap (8wk/hiking) but is not a strict improvement once checked properly — it trades that fix for three regressions elsewhere, driven by data sparsity in rare-regime validation slices.

## 9. Final decision: production default is `"adaptive"`, not `"adaptive_granular"`

`"adaptive_granular"` was briefly set as the default (Section 5's earlier draft), then reverted once the regressions below were confirmed. **Production default is `"adaptive"`, not `"adaptive_granular"`, despite the latter being built as an intended improvement.** `adaptive_granular` fixes 8wk/hiking but regresses 13wk/hiking, 13wk/cutting (back to fold-level's original failing value), and 8wk/cutting — all traced to the same root cause: empty validation windows in rare-regime cells causing a silent, incorrect fallback to fold-level weighting. Documented as a known limitation with a scoped fix (data-sufficiency fallback to the pooled `adaptive` decision, using the same `MIN_RELIABLE_N = 30` threshold already used for gating rules) for a future session, not implemented tonight.

`adaptive_granular`'s code is fully intact in `factor_attribution.py` and `run_backtest.py` and remains available via `weighting_mode="adaptive_granular"` — kept as a real, documented, partially-working experiment, not deleted.

**Confirmed production numbers (`weighting_mode="adaptive"`, the default)**:

| Horizon | Fused MAPE | Naive MAPE | Beats naive |
|---|---|---|---|
| 4wk | 2.243 | 4.194 | ✅ |
| 8wk | 5.038 | 7.371 | ✅ |
| 13wk | 7.020 | 11.168 | ✅ |

Acceptance bar: **MET, 3/3**. The only known imperfection is the 8wk/hiking cell (5.401 vs. fold-level's 4.733) — a small, well-understood, documented gap, not a silent one.
