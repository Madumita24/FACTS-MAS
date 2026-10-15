# FACTS-MAS code map (Phase 0, commit 0a1aec5)

Where each mechanism lives. Paths are relative to the `FACTS-MAS/` repo root. Owners per `KNOWLEDGE_TRANSFER.md`: Umang owns `facts_mas/` framework, seasonality and intrinsic agents; Madumita owns root-level AR, macro, event cores, data pulls, and `domains/ui_claims/`.

## Data

| Item | Location | Notes |
|---|---|---|
| Zillow series | `pull_zillow.py:9` downloads `Metro_invt_fs_uc_sfrcondo_sm_week.csv` | **Smoothed** cut (`_sm`). Saturday anchor. |
| Panel | `align_data.py:40-53` writes `aligned_weekly.csv` | Macro joined by backward `merge_asof`. Missing inventory forward-filled per MSA (`:49`), logged in `filled_rows_log.csv` (9 rows). Data runs 2018-02-03 to 2026-06-20, so the file contains the reserved window. |
| Loaded by | `run_phase34.py:158`, `facts_mas/run_backtest.py:527`, `facts_mas/agent_cache.py` callers | |
| Static features | `intrinsic_static.csv` (ACS 2024, NOAA 2016-2025) | Vintage later than early folds. |

## Protocol

| Item | Location | Notes |
|---|---|---|
| Fold definitions | `facts_mas/run_backtest.py:56-69` (`FOLDS`); generator `fold_boundaries.py:41-70` | 6 expanding folds, train start 2019-02-02, 29-week test windows, last test origin 2025-11-22. |
| Embargo | `fold_boundaries.py:24` (13 weeks); check `run_backtest.py:113-124` (`gap >= 91 days`) | `test_start = train_end + 14 weeks`. |
| Reserved window | `fold_boundaries.py:26` (`HELD_OUT_START = 2026-01-01`) | Bounds test **origins** only. Fold 6 targets reach 2026-02-21. |
| Test cells | `agent_cache.py:96-124` (`cells(test_only=True)`) | 2,610 per horizon. |
| Validation window | `fusion_baseline.py:386-387`; `factor_attribution.py:166-167`, `:280-281`; `calibration.py:160-169` | `[train_end - 104 weeks, train_end]` on origins. Fusion and mode selection do not bound targets (they reach `train_end + 13`); the calibration scorer does (`calibration.py:229`). |
| Metric | `run_backtest.py:131-141` (`compute_mape`, `compute_rmse`) | MAPE per cell over the length-H path (steps 1..H). RMSE per cell, then averaged over cells. |

## Fusion

| Item | Location | Notes |
|---|---|---|
| Skill | `fusion_baseline.py:413-446` | Per cell: `1 - MAPE_agent / MAPE_naive`, each MAPE over the 1..H path. Cells with naive MAPE 0 are skipped (`:427`). Agent failure scores -1 (`:446`). Pooled over all 3 horizons and all regimes. |
| Aggregation | `fusion_baseline.py:454-456`, `:261-270` | Median of per-cell skill. |
| Softmax | `fusion_baseline.py:202-247` | **Temperature 0.05** (`:204`, `:342`; also `factor_attribution.py:58`, `:146`). Hand-set. |
| Regime grid | `factor_attribution.py:51-187` | Same skill and softmax, per (regime, horizon). |
| Mode selection ("Router") | `factor_attribution.py:244-330`; used at `run_backtest.py:329-338` and `run_phase34.py:95-99` | Per (fold, horizon): fold vs regime weights, lower validation MAPE wins, ties go to fold. |
| Fuse | `fusion_baseline.py:290-332` (live path, rounds to 2 dp); `ablation.py:52-78` (cache path, renormalises over live agents) | |
| Production entry | `run_phase34.py:71-119` (`build_weight_providers`) | Source of every committed Phase 4 number. |

## Regimes

| Item | Location | Notes |
|---|---|---|
| Labelling | `run_backtest.py:78-106` | `fed_funds[t] - fed_funds[13 rows back]`; hiking if > 0.25, cutting if < -0.25. `fed_funds` is monthly FEDFUNDS shifted +32 days and forward-filled (`README.md`). Fewer than 14 rows gives "stable". |

## Agents

| Channel | Adapter | Core | Notes |
|---|---|---|---|
| Temporal (AR) | `facts_mas/agents/ar_agent.py` | `ar_agent.py:18-39` | statsmodels ETS, additive trend, no seasonality, refit per origin and horizon. |
| Seasonal | `facts_mas/agents/seasonality_agent.py:67-218` | same | 52-week centred MA (`min_periods=26`) on data up to the origin, median ratio per ISO week, circular smoothing window **5** (`:44`, chosen on fold 6 test), forecast `last * f(week+h) / f(week)`. |
| Exogenous (macro) | `facts_mas/agents/macro_agent.py` | `macro_agent.py:73-122` | Ridge on lagged macro levels, fixed lags 6/10/10/5 weeks (`:24-29`, not fitted). One-week pct-change applied once. |
| Event | `facts_mas/agents/event_agent.py:59-116` | cached LLM output `event_agent_weekly.csv` | Point forecast is always the last value (magnitude removed 08-12). Window constant 8 weeks only gates `confidence`, which fusion ignores. |
| Intrinsic | `facts_mas/agents/intrinsic_agent.py:135-220` | same | Ridge on 5 static features predicts a level; pull `0.30 * (step/H)^2` toward it. **Production rho_max = 0.30** (`:55`, used at `:208`). |

## Calibration

| Item | Location | Notes |
|---|---|---|
| Framework | `facts_mas/calibration.py:259-383` | Each candidate scored on each fold's validation window; one global value; approval needs improvement in 5 of 6 folds (`:71`), dead band 0.01 (`:85`), min 30 affected origins (`:79`). |
| Studies | `calibrate_event_window.py`, `calibrate_event_impact_scale.py`, `calibrate_rho_max.py` (`rho_max_calibration.csv`) | Event impact scale 0 was adopted (08-12). rho_max 0 was not adopted. |
| Feedback loop | none automated | Recommendations are applied by hand. |

## Results artefacts

| Artefact | Built | Used for |
|---|---|---|
| `full_backtest_*_fused_results.pkl` / `*_naive_results.csv` | 2026-07-29 (`da77e8e`), old agents | Paper headline, regime tables, adaptive vs granular |
| `agent_output_cache.pkl` | 2026-08-12 (committed 08-18), current agents | `phase34_results/*` (ablation, weights, standalone channels) |
