# Evaluation Protocol

## ⚠️ Needs your decision: backtest start date

Data starts 2018-02-03 (Zillow inventory) / 2018-01-01 (macro). Before any fold can be evaluated, two buffers eat into that history:

1. **Feature warm-up (52 weeks)** — if the Macro/feature layer builds seasonal or rolling features (e.g. year-over-year comparisons), the first ~52 weeks can't produce a fully-formed feature row. This pushes the earliest usable row to **2019-02-02**.
2. **Horizon embargo (13 weeks)** — a training row's label for the 13-week-horizon model is `inventory[t+13wk]`. Any training row within 13 weeks of a fold's test start would use a label that hadn't actually occurred yet as of that fold's simulated "today" — real lookahead bias, not just a formality. Each fold's train window must end 13 weeks before its own test window starts.

**Proposed cutoff: training data usable from 2019-02-02, with the last 6-fold test window ending 2024-11-09 and 2025-01-04 onward reserved entirely as a held-out live-evaluation period** (matching the "Jan 2025 evaluation cutoff" mentioned in the original data-pull request). **Please confirm or adjust — in particular, confirm whether Jan 2025 should be a hard reserved boundary (untouched by any backtest fold) or just a soft target.**

## 6-fold rolling backtest

Expanding-window design: training start is fixed at 2019-02-02; training end grows each fold to include the previous fold's now-realized test period. Test windows are 20 weeks each, separated from their own fold's training end by a 13-week embargo (not between consecutive folds — once a fold's test period is chronologically in the past, it legitimately becomes training data for the next fold with no leakage).

| Fold | Train start | Train end  | Train length | Test start | Test end   |
|------|-------------|------------|--------------|------------|------------|
| 1    | 2019-02-02  | 2021-01-23 | 104 weeks    | 2021-05-01 | 2021-09-11 |
| 2    | 2019-02-02  | 2021-09-11 | 137 weeks    | 2021-12-18 | 2022-04-30 |
| 3    | 2019-02-02  | 2022-04-30 | 170 weeks    | 2022-08-06 | 2022-12-17 |
| 4    | 2019-02-02  | 2022-12-17 | 203 weeks    | 2023-03-25 | 2023-08-05 |
| 5    | 2019-02-02  | 2023-08-05 | 236 weeks    | 2023-11-11 | 2024-03-23 |
| 6    | 2019-02-02  | 2024-03-23 | 269 weeks    | 2024-06-29 | 2024-11-09 |

Fold 6 ends 8 weeks before 2025-01-04, leaving a buffer before the reserved live-evaluation period. Within each fold's test window, generate a forecast at every weekly origin (not just one per fold) so a 20-week test window yields ~20 overlapping forecast origins per horizon.

## Metrics

Computed **per MSA per horizon**, then averaged across MSAs for an overall number. Let `y_t` = actual inventory, `ŷ_t` = forecast, over the `n` forecast origins in a fold's test window:

- **MAPE** (Mean Absolute Percentage Error):
  `MAPE = (100/n) * Σ |y_t - ŷ_t| / |y_t|`
- **RMSE** (Root Mean Squared Error):
  `RMSE = sqrt( (1/n) * Σ (y_t - ŷ_t)^2 )`

Report both **per-MSA** (15 rows) and an **overall average across the 15 MSAs** (unweighted mean of per-MSA scores, so no single large metro like New York dominates the headline number).

## Horizons — reported separately, never averaged

Report MAPE and RMSE independently for the 4-week, 8-week, and 13-week horizons. Averaging across horizons would hide a model that's accurate short-term but degrades badly at 13 weeks (or vice versa) — exactly the failure mode a horizon-specific model choice needs visibility into.

## Regime-stratified error

Regimes are defined from the Fed funds rate trajectory already present in `aligned_weekly.csv` (`fed_funds` column), using its 3-month (13-week) change as of each forecast origin `t`:

- **Rate-hiking**: `fed_funds[t] - fed_funds[t-13wk] > +0.25` (25bps)
- **Rate-cutting**: `fed_funds[t] - fed_funds[t-13wk] < -0.25` (25bps)
- **Stable**: everything in between

Each forecast origin is tagged with its regime at the time the forecast was made (not at the target date), and MAPE/RMSE are reported separately for each regime, in addition to the pooled/overall numbers — per horizon, per regime, per MSA-average. This surfaces whether a model quietly relies on the "stable" regime (which likely dominates row counts) to hit its headline accuracy while doing poorly exactly when the macro backdrop is moving fastest — arguably the most decision-relevant moments for a housing forecast.
