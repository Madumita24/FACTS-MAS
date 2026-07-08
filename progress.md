# FACTS-MAS Data Pipeline — Progress Log

Tracks what's been built, why, and what came out of it, so the state of Phase 1 is legible at a glance. Extends the NEXUS multi-agent forecasting paper toward weekly housing-inventory forecasting for 15 US metros at 4/8/13-week horizons.

## Status: Phase 1 (data pipeline) complete, pending your review on the evaluation protocol's backtest start date.

---

## 1. Weekly time-series layer

**Scripts:** [pull_macro.py](pull_macro.py), [pull_zillow.py](pull_zillow.py), [align_data.py](align_data.py)

**What it does:**
- Pulls 4 FRED macro series (mortgage rate, fed funds, CPI, unemployment) via `fredapi`, 2018–present, resampled to a Monday-anchored weekly grid.
- Pulls Zillow's weekly for-sale inventory (smooth, all homes) for 15 MSAs directly from `files.zillowstatic.com`, no auth needed.
- Aligns them via `merge_asof` (backward), using Zillow's native Saturday dates as the canonical grid.

**Key decisions:**
- **Weekly anchor**: macro data anchored Monday; Zillow kept its native Saturday anchor; joined via backward `merge_asof` rather than forcing a shared calendar day — keeps the join lookahead-safe by construction.
- **No-lookahead handling for monthly macro series**: FRED indexes FEDFUNDS/CPIAUCSL/UNRATE by the *first of the reference month*, not the actual publish date. Shifted each by an approximate real-world publication lag (FEDFUNDS +32d, UNRATE +35d, CPIAUCSL +42d) before forward-filling, so no week ever sees a macro value before it was actually knowable. **Caveat**: these are calendar approximations from typical BLS/Fed release schedules, not true ALFRED point-in-time vintages — flagged as a simplification worth mentioning if precise real-time data matters later.
- **Transmission lags** (mortgage ~4-8wk, fed funds ~8-12wk, CPI ~8-12wk, unemployment ~4-6wk) are documented as comments in `align_data.py` only, not applied — that's the downstream Macro Agent's job, not Phase 1's.

**Results:**
- `macro_weekly.csv`: 444 rows, 2018-01-01 → 2026-06-29, zero missing values.
- `zillow_inventory_weekly.csv`: 6,570 rows (15 MSAs × 438 weeks), 2018-02-03 → 2026-06-20. All 15 requested MSAs matched Zillow's data exactly, **no substitutions needed**.
- `aligned_weekly.csv`: 6,570 rows, same columns plus the 4 macro fields broadcast nationally across all MSAs.
- Zillow's own history starts 2018-02-03, not Jan 2018 as originally hoped — the dataset simply doesn't go back further.

**Follow-up fix — missing inventory values:**
`aligned_weekly.csv` had 9 missing `inventory_count` rows (Chicago: 4, Los Angeles: 2, Miami: 2, Atlanta: 1), all gaps in Zillow's own smoothed series. Added a per-MSA forward-fill to `align_data.py` (never fills across MSAs, never uses a future value) and an audit trail: [filled_rows_log.csv](filled_rows_log.csv) records exactly which (date, msa) rows were filled and with what value. All 9 gaps landed on MSAs whose panel starts 2018-02-03, so none were an unfillable "first observation" case. **One thing worth a footnote**: Chicago's gap was 4 *consecutive* weeks (2021-01-16 to 2021-02-06), all filled with the same carried-forward value — a real 4-week flat patch, not independent noise, which could suppress week-over-week deltas in that window if a downstream model is sensitive to that.

**Tests:** [test_align_data.py](test_align_data.py) — 5 pytest cases against synthetic fixtures (no network calls), covering: fillable-gap correctness, no cross-MSA leakage, unfillable-leading-gap handling, log accuracy, and no-lookahead join behavior. All passing.

---

## 2. Intrinsic (static) layer

**Script:** [pull_intrinsic.py](pull_intrinsic.py) → [intrinsic_static.csv](intrinsic_static.csv)

**What it does:** one row per MSA — population, median household income, total housing units, density, and a hazard index.

**Key decisions:**
- Population/income/housing units come directly from **Census ACS 2024 5-year estimates** (2020–2024 vintage, most recent available) queried at CBSA geography — no county aggregation needed, ACS publishes these totals natively per CBSA.
- **Density** required land area, which ACS doesn't provide — pulled `ALAND_SQMI` from the Census 2023 CBSA Gazetteer file and computed `population / land_area_sqmi`.
- **Hazard index — source swap from the original plan**: FEMA's National Risk Index was the first choice (a pre-aggregated composite score would've been cleaner), but it turned out to have no stable programmatic download — automated fetches get a 403, the data-resources page redirects to an interactive JS tool (RAPT) rather than a static file, and it isn't in the OpenFEMA API catalog. Verified this live rather than assuming. Switched to **NOAA Storm Events Database** (2016–2025, 10 years), which does have reliable bulk CSVs. `hazard_index` = count of storm events with recorded property/crop damage or injury/death, summed across each MSA's constituent counties via the Census county↔CBSA delineation file.
  - **Caveat**: NOAA logs each event against either a county or an NWS forecast zone; this pipeline only counts county-reported events (no zone→county crosswalk built yet, ~57% of records nationally in a 2024 spot check). `hazard_index` is therefore a real but partial signal, likely undercounted for zone-heavy hazard types like flooding.
- All 15 MSAs resolved to unambiguous CBSA codes with **no substitutions**. Some now carry updated official titles post the Census's 2023 delineation (e.g. Houston is "Houston-Pasadena-The Woodlands" not "...Sugar Land"; SF is "...Fremont" not "...Berkeley") — same metro, renamed, not a different area.

**Results:** 15 rows, zero missing values. Notable spread: hazard_index ranges from 17 (Seattle — Pacific Northwest's milder severe-storm climate, plausible not a bug) to 3,765 (Washington DC, whose CBSA spans many counties across 4 states). Population ranges from New York (19.8M) to Seattle (4.1M).

**Bug caught and fixed mid-run:** first run crashed on the final merge (`Series` object had no named column to join on — `msa` was the Series index, not a column). Fixed by setting `.index.name = "msa"` and merging via `right_index=True`. All the actual data fetching (ACS, Gazetteer, delineation, NOAA) succeeded on both attempts — only the final join step was broken.

---

## 3. Events layer — scoping only, not built

**Doc:** [events_layer_scoping.md](events_layer_scoping.md)

Explicitly a documented Phase 2 dependency, nothing implemented yet. Covers why local event data (employer moves, zoning changes, housing initiatives) has no clean API — it's locally generated, heterogeneous in structure, and inconsistently digitized, unlike the stable federal/commercial endpoints used everywhere else in this pipeline. Ranked candidates:
1. **FEMA disaster declarations API** (`fema.gov/api/open`) — free, no key, clean, confirmed present in the OpenFEMA catalog. Covers natural disasters only.
2. **WARN Act mass layoff notices** — no unified API, would need per-state scraping (15+ states/DC involved across the 15 MSAs).
3. **Google News RSS per MSA** — trivial to query but noisy; would need an LLM extraction step with no ground truth to validate against.

**Recommendation:** start Phase 2 with FEMA disaster declarations only as MVP event coverage; defer WARN/news scraping until there's a specific question natural-disaster events alone can't answer.

---

## 4. Evaluation protocol

**Doc:** [evaluation_protocol.md](evaluation_protocol.md)

**⚠️ Needs your decision, not yet confirmed:** proposed that training data becomes usable from **2019-02-02** (after a 52-week feature warm-up past the 2018-02-03 data start), with a 6-fold expanding-window backtest ending 2024-11-09, and **everything from 2025-01-04 onward reserved as a fully held-out live-evaluation period** — matching the "Jan 2025 evaluation cutoff" mentioned in the original data-pull request. Open question: should Jan 2025 be a *hard* reserved boundary untouched by any backtest fold, or just a soft target? This choice drives the entire fold table below it.

**Fold design:** each fold's training window ends 13 weeks (the longest horizon) before its own test window starts — this is a real lookahead-prevention mechanism, not a formality: a training row's 13-week-ahead label would otherwise reference a value that hadn't occurred yet as of that fold's simulated "today." Once a fold's test period is chronologically past, it legitimately rolls into the next fold's training data with no leakage (that's the expanding-window design).

**Metrics:** MAPE and RMSE, computed per MSA per horizon, then averaged across MSAs (unweighted, so New York doesn't dominate the headline number).

**Horizons:** 4/8/13-week reported separately, never averaged — averaging would hide a model that's accurate short-term but degrades badly further out.

**Regime stratification:** 3 regimes from the Fed funds rate's 3-month (13-week) change already in `aligned_weekly.csv` — hiking (>+25bps), cutting (<-25bps), stable (between). Error reported separately per regime, per horizon, per MSA-average, to surface whether a model's headline accuracy is quietly propped up by the (likely dominant) stable-regime rows while it does worse exactly when the macro backdrop is moving fastest.

---

## Open items for you

1. **Confirm the backtest start date / fold design** in `evaluation_protocol.md` — specifically whether the Jan 2025 boundary should be hard or soft.
2. **Sanity-check `hazard_index`** against local knowledge, especially Seattle's low count (17) and DC's high one (3,765) — both have plausible real-world explanations (climate, multi-state county span) but haven't been validated against a second source.
3. Events layer (Task 3) is scoping only — no code exists yet, by design.

## All files produced so far

| File | What |
|---|---|
| `pull_macro.py`, `macro_weekly.csv` | FRED macro pull |
| `pull_zillow.py`, `zillow_inventory_weekly.csv` | Zillow inventory pull |
| `align_data.py`, `aligned_weekly.csv`, `filled_rows_log.csv` | Alignment + forward-fill audit |
| `test_align_data.py` | Tests for the alignment/fill logic |
| `pull_intrinsic.py`, `intrinsic_static.csv` | Census + NOAA static features |
| `events_layer_scoping.md` | Phase 2 events-layer research note |
| `evaluation_protocol.md` | Backtest design, metrics, regime stratification |
| `README.md` | Task 1 data-pull documentation (date ranges, MSA coverage, gaps, anchor convention) |
