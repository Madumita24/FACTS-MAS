# FACTS-MAS Data Pipeline

Pulls FRED macro data and Zillow MSA-level inventory, then aligns them into a single weekly panel for the housing-inventory forecasting pipeline.

## Scripts (run in order)
1. `pull_macro.py` -> `macro_weekly.csv`
2. `pull_zillow.py` -> `zillow_inventory_weekly.csv`
3. `align_data.py` -> `aligned_weekly.csv`

Requires `FRED_API_KEY` in `.env` (already present) and the `fredapi` package (installed).

## Date range
- Macro data: 2018-01-01 to 2026-06-29 (444 weekly rows, Monday-anchored). Fetched from FRED starting 2017-09-01 as a buffer so monthly series (FEDFUNDS/CPI/UNRATE) are already available at the 2018-01-01 grid start with no NaNs.
- Zillow inventory: 2018-02-03 to 2026-06-20 (Zillow's own history starts here, not Jan 2018 — the dataset simply doesn't go back further).
- Aligned output: bounded by Zillow's range, 2018-02-03 to 2026-06-20, 6,570 rows (15 MSAs x 438 weeks).

## Weekly-anchor convention
- **macro_weekly.csv** is anchored on **Monday** (`W-MON`).
- **zillow_inventory_weekly.csv** keeps Zillow's native **Saturday** anchor.
- **aligned_weekly.csv** uses Zillow's Saturday dates as the canonical grid. Macro values are attached via `merge_asof(direction="backward")`, i.e. each Zillow week gets the most recent macro row dated on or before it — never a future one. This avoids forcing a shared calendar day while staying lookahead-safe.

## MSA coverage
All 15 requested MSAs were found in Zillow's data under these exact `RegionName` values — **no substitutions needed**:
New York, Los Angeles, Chicago, Dallas, Houston, Washington DC ("Washington, DC"), Miami, Philadelphia, Atlanta, Phoenix, Boston, San Francisco, Riverside, Detroit, Seattle.

## Missing data gaps
- `inventory_count`: 9 NaN rows total across the full panel (Chicago: 4, Los Angeles: 2, Miami: 2, Atlanta: 1) — these are gaps in Zillow's own smoothed series (early history / smoothing-window edge effects for a few metros), not introduced by the pipeline.
- `mortgage_rate`, `fed_funds`, `cpi`, `unemployment`: 0 missing values in the final aligned output.

## No-lookahead handling for monthly macro series
FRED indexes monthly series (FEDFUNDS, CPIAUCSL, UNRATE) by the first day of the reference month, not by the date the value was actually published. To avoid leaking future information into past weeks, each monthly observation is shifted forward by an approximate real-world publication lag before being forward-filled:
- `fed_funds` (FEDFUNDS): +32 days
- `unemployment` (UNRATE): +35 days (BLS employment situation report, first Friday of the following month)
- `cpi` (CPIAUCSL): +42 days (BLS CPI release, ~10th-13th of the following month)

**Caveat for methodology discussion:** these are calendar approximations based on typical BLS/Fed release schedules, not true ALFRED real-time vintage data. Actual release dates vary by a few days month to month. If point-in-time precision matters for the backtest, consider switching to `fredapi`'s ALFRED vintage endpoints (`get_series_as_of_date` / `get_series_all_releases`).

`MORTGAGE30US` is already a native weekly release, so it's as-of joined directly with no added lag.

## Documented (not yet applied) transmission lags
Recorded as comments in `align_data.py` for the downstream Macro Agent to use as a starting point for lag search — these are directional research priors from housing-finance literature, not fitted values:
- `mortgage_rate`: ~4-8 weeks
- `fed_funds`: ~8-12 weeks (indirect, via mortgage rates and credit conditions)
- `cpi`: ~8-12 weeks (affordability effects build gradually)
- `unemployment`: ~4-6 weeks (labor-market shifts affect household formation/forced sales fastest of the four)
