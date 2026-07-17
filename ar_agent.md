# AR Agent — Build Notes

Status: Steps 1-4 done and validated. Step 5 (schema-matching stub for the fusion-harness handoff) not started.

---

## Scope

The AR Agent is a per-MSA ETS (Holt-Winters, additive trend, **no seasonality**) forecaster for weekly housing inventory. Seasonality is deliberately excluded — it's a separate agent owned by a teammate, so approximating it here would create two competing seasonality signals in the eventual fusion step.

## Files

| File | Role |
|---|---|
| [fold_boundaries.py](fold_boundaries.py) | 6-fold expanding-window backtest schedule (Step 1) |
| [naive_baseline.py](naive_baseline.py) | Last-value baseline, the minimum bar (Step 2) |
| [ar_agent.py](ar_agent.py) | ETS forecaster (Step 3) |
| [test_naive_baseline.py](test_naive_baseline.py), [test_ar_agent.py](test_ar_agent.py) | Unit tests (Steps 2b, 3b) |
| [compare_ar_vs_naive.py](compare_ar_vs_naive.py) | Head-to-head MAPE/RMSE comparison (Step 3c) |
| [diagnose_ar_outliers.py](diagnose_ar_outliers.py) | Read-only diagnostic on the two underperforming MSAs |

---

## Step 1 — Fold boundaries

Original design (in `evaluation_protocol.md`) assumed a January 2025 live-eval cutoff. **That cutoff was confirmed wrong mid-build — the real cutoff is January 2026.** `fold_boundaries.py` recomputes the schedule against the corrected date; `evaluation_protocol.md` is deliberately left untouched for now (explicit instruction: don't update it until all agents are validated against the new dates).

Design, unchanged in spirit from the original:
- Training data usable from **2019-02-02** (52-week feature warm-up past the 2018-02-03 data start).
- Each fold's training window ends a **13-week embargo** before its own test window starts — not a formality: a training row's 13-week-ahead label would otherwise reference a value that hadn't occurred yet as of that fold's simulated "today." Re-applied fresh at every fold boundary.
- Expanding window: once a fold's test period is chronologically past, it becomes training data for the next fold with no leakage.
- Hard ceiling: no fold's test window may extend past 2025-12-31 (enforced with an in-code assertion, not just a comment).

Moving the cutoff a year later grew the test-fold size from 20 weeks (old design) to **29 weeks**. Final schedule (6 folds, expanding):

| fold | train_end | test_start | test_end |
|---|---|---|---|
| 1 | 2021-01-23 | 2021-05-01 | 2021-11-13 |
| 2 | 2021-11-13 | 2022-02-19 | 2022-09-03 |
| 3 | 2022-09-03 | 2022-12-10 | 2023-06-24 |
| 4 | 2023-06-24 | 2023-09-30 | 2024-04-13 |
| 5 | 2024-04-13 | 2024-07-20 | 2025-02-01 |
| 6 | 2025-02-01 | 2025-05-10 | 2025-11-22 |

Fold 6 ends 5 weeks before the Jan 2026 boundary.

## Step 2 — Naive baseline

`forecast_naive(series, horizon_weeks)` repeats the series' last value. `run_naive_baseline` produces one row per (msa, fold_index, horizon) — 270 rows total (15 MSAs × 6 folds × 3 horizons) — using only data up to and including each fold's `train_end`.

**Output schema decision**: `forecast` is a list column (length = horizon), not exploded into `forecast_week_1..N` columns. Reasoning: horizons differ (4/8/13), so exploded columns would leave NaN-padding on shorter-horizon rows; a list column also matches the schema the AR Agent needs to produce (`point_forecast: list[float]`), so the two are directly comparable. `train_end` and `last_training_value` are included as audit columns.

**No-lookahead guard**: `assert_no_lookahead(series, as_of)` raises `ValueError` if a series contains any date after `as_of`. Called as defense-in-depth right after each fold's slicing — under normal operation it should never trigger, since the slicing already does it correctly, but it exists so a future bug that accidentally passes untruncated data fails loudly instead of silently leaking test-window data into a forecast. **This guard is reused unchanged by the AR Agent** (imported from `naive_baseline.py`), so both agents enforce the identical no-lookahead rule rather than maintaining two copies of the same safety check.

Verified: `last_training_value` for a sample MSA (Atlanta) cross-checked directly against `aligned_weekly.csv` at each fold's `train_end` — exact match, all 6 folds.

## Step 3 — AR Agent

`fit_and_forecast(series, horizon_weeks)` fits `statsmodels.tsa.holtwinters.ExponentialSmoothing(series, trend="add", seasonal=None)` and returns:
```python
{"model_name": "ets", "point_forecast": [...], "prediction_intervals": None}
```

**Non-negativity**: output is clipped at 0 (`np.clip(forecast, 0, None)`). Documented in-code as a *temporary local safeguard* against ETS's additive-trend extrapolation going negative on a sharply declining series — explicitly not a substitute for the boundary-constraint layer owned downstream by a teammate; that layer still needs to own this concern for the fused output.

**Refit-per-horizon**: `run_ar_agent` calls `fit_and_forecast` separately for each of the 3 horizons per (msa, fold), re-fitting ETS each time rather than fitting once at the max horizon and slicing. This is less efficient (3x the fits) but matches the exact `fit_and_forecast(series, horizon_weeks)` signature as specified — a known, accepted tradeoff, not an oversight.

**Output schema matches Step 2 exactly**: `msa, fold_index, horizon, forecast, train_end, last_training_value` — verified with an explicit test (`test_output_schema_matches_naive_baseline`) rather than assumed by inspection.

## Step 3b — Tests

7 pytest cases in `test_ar_agent.py`, all passing:
- Output length equals horizon for 4/8/13 weeks.
- **Non-negativity actually exercised**: a synthetic sharply-declining series (5000 → 50 over 60 weeks) is used specifically to push ETS's linear extrapolation toward/below zero, confirming the clip engages — not just tested on data where it never triggers.
- Schema match against spec (`model_name`, `point_forecast` type, `prediction_intervals`).
- No-lookahead guard raises `ValueError` when given out-of-bounds data.
- Full-column schema equality against `naive_baseline`'s output.

## Step 3c — AR vs. naive comparison

`compare_ar_vs_naive.py` scores both agents' forecasts against actual `aligned_weekly.csv` values at each fold's `train_end + 1..horizon` weeks, per (msa, fold, horizon), then aggregates to per-MSA-per-horizon MAPE/RMSE.

**Note on scope**: this evaluates every fold starting right at `train_end`, including horizons that land inside that fold's 13-week embargo gap. That's fine here — the embargo exists to keep *training-label construction* from leaking future data, not to forbid checking a forecast against dates that have already occurred by now. This script is a smoke comparison, not the official backtest in `evaluation_protocol.md`.

### Results

| Horizon | AR beats naive | Overall MAPE (naive → AR) | Overall RMSE (naive → AR) |
|---|---|---|---|
| 4 weeks | 14/15 MSAs | 4.69% → 2.73% | 613 → 388 |
| 8 weeks | 14/15 MSAs | 9.34% → 5.51% | 1,165 → 790 |
| 13 weeks | 13/15 MSAs | 13.67% → 7.79% | 1,707 → 1,192 |

AR's advantage over naive grows with horizon — naive's error compounds linearly since it just holds the last value flat, while ETS captures trend. Biggest wins: Seattle and San Francisco (both >10pp MAPE improvement at 13wk).

**Consistent underperformer: Miami** (naive beats AR at all 3 horizons, gap widening from -0.6pp at 4wk to -3.3pp at 13wk). **Phoenix** also flips at 13wk (-1.5pp). Everywhere else, AR wins clearly.

## Diagnostic — why Miami and Phoenix underperform

Read-only investigation (`diagnose_ar_outliers.py`), no changes made to `ar_agent.py`. Tested two hypotheses: (1) genuinely volatile/choppy series where any trend model would struggle, vs. (2) ETS specifically extrapolating a stale trend into a reversal it should have anticipated.

| MSA | mean | std | CV | 13wk rolling-trend reversals (full series) | ETS sign-disagreement with actual 13wk move |
|---|---|---|---|---|---|
| Miami | 36,870 | 10,531 | 0.29 | 12 | 2/6 folds |
| Phoenix | 15,562 | 3,849 | 0.25 | 23 | 1/6 folds |

**Finding: hypothesis (1), not (2).** Sign-disagreement is low for both (2/6 and 1/6) — ETS is not confidently extrapolating the wrong direction. The real problem is **magnitude**: e.g. Miami fold 1, ETS predicted a mild uptrend while the actual 13-week move was -12,726 (≈34% of the mean) — even a directionally-correct forecast gets crushed by MAPE when swings this large occur. ETS's smoothing structurally can't react fast enough to abrupt moves of that size, regardless of which direction it currently leans. Phoenix reverses direction roughly every other rolling window (23 reversals across ~413 weeks) — inherently choppy, not a specific reversal ETS missed.

**Conclusion, logged as a known limitation, not fixed here**: Miami and Phoenix are volatile-series cases where the shared ETS model structurally underperforms even a trivial baseline. Candidate for a **per-MSA model override in Phase 3** (e.g. a less-smoothed / faster-reacting model, or explicitly wider uncertainty bounds for these two) rather than a change to the shared AR Agent.

## Open items

1. **Step 5 not built yet**: `ar_agent_stub.py`, a fake `run_ar_agent` matching the exact output schema/shape, needed to unblock the fusion-harness teammate before the real model is fully validated.
2. **Miami/Phoenix**: decide in Phase 3 whether these need a per-MSA override or stay a documented limitation of the shared-model approach.
3. **`evaluation_protocol.md` still has stale Jan-2025 dates** — intentionally not updated yet; needs a pass once all agents (not just AR) are validated against the corrected Jan-2026 schedule.
