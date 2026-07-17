# FACTS-MAS Data Pipeline — Progress Log

Tracks what's been built, why, and what came out of it, so the state of Phase 1 is legible at a glance. Extends the NEXUS multi-agent forecasting paper toward weekly housing-inventory forecasting for 15 US metros at 4/8/13-week horizons.

## Status: Phase 1 (data pipeline) complete. AR Agent and Macro Agent built and validated (Macro Agent's validation resulted in a well-earned null — see below). Event Agent (first LLM-based component) build is **complete** — finalized prompt design, full batch run, and mocked test suite all done — see [event_agent.md](event_agent.md). Ready for fusion-harness handoff alongside the AR Agent's real output.

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

## 5. AR Agent (ETS) + naive baseline

**Full writeup:** [ar_agent.md](ar_agent.md) — this section is a summary pointer, not a duplicate.

Built a naive last-value baseline (the minimum bar) and an ETS-based AR Agent, backtested both against the corrected fold schedule below, and confirmed AR beats naive at all 3 horizons on the large majority of MSAs. Two MSAs (Miami, Phoenix) underperform and were diagnosed rather than patched — see `ar_agent.md` for the finding and `diagnose_ar_outliers.py` for the diagnostic script.

**Headline result:** AR (ETS) beats naive on MAPE for 14/15 MSAs at 4wk and 8wk horizons, 13/15 at 13wk. Overall MAPE roughly halves at every horizon (e.g. 13wk: 13.67% naive → 7.79% AR). Full per-MSA table in `ar_agent.md`. Step 5 (`ar_agent_stub.py`, schema-matching fake output) is also done — unblocked the fusion-harness teammate.

---

## 6. Macro Agent — built, then investigated to a well-earned null

**Full writeup:** [macro_agent.md](macro_agent.md) — this section is a summary pointer, not a duplicate.

Built a ridge-regression Macro Agent (lagged FRED features → a pct-change modifier on top of AR's forecast). Initial combiner testing looked bad, but the harm turned out to be mostly a compounding artifact, not the macro signal itself — so the signal was tested directly, independent of any combiner, across three progressively more rigorous rounds:

1. Fold-based correlation (single-week modifier vs. actual change): clean null, all p≥0.095.
2. Lag grid search + horizon-level regression target: apparent strong correlations (p<0.0001) turned out to be pseudo-replicated (national macro data is identical across all 15 MSAs per fold, so true n≈6, not 90) and vulnerable to spurious trend-correlation (both macro and inventory trend heavily over 2018-2025). The one structurally sounder sub-result (horizon-level, per-MSA models) was statistically significant at 8/13wk but **wrong-signed**.
3. **Two-way fixed-effects panel regression with MSA-clustered SEs** (5,850 rows, MSA + year-quarter FE) — the rigorous final test, fixing both flaws from round 2 directly. Result: only fed_funds crosses p<0.05 (p=0.042), and it's the *theoretically less-expected* feature (mortgage_rate, the textbook housing-inventory channel, is dead null at p=0.575). Read as noise-level, not a real finding.

**Headline result: well-earned null.** `macro_agent.py` was never modified based on any of this — the investigation was entirely diagnostic, building the case before any decision gets made about what to do with it.

---

## 7. Event Agent — first LLM-based component, prompt design finalized after 5 rounds

**Full writeup:** [event_agent.md](event_agent.md) — this section is a summary pointer, not a duplicate.

Built the FEMA data fetch (Step 1, reuses Phase 1's county↔CBSA crosswalk) and the LLM interpretation function (Step 2), and validated the prompt on a fixed 9-declaration sample (Miami/Houston/Chicago) before touching the full batch — that discipline caught two real bugs, not just tuning opportunities:

1. **Round 1** (thin metadata): no severity differentiation possible (Ian = Nicole = -1.00), confidence gate never fired even on an obscure minor brush fire.
2. **Round 2** (enriched metadata): **every one of 9 declarations flipped sign** — root cause was that impact_magnitude's direction was never causally anchored to actual inventory effect, so the model's sign convention wasn't stable across prompt edits. Worse bug than the one being fixed.
3. **Round 3** (explicit causal mechanism + reasoning field + gate boundary fix): sign consistency fixed; also fixed a real off-by-one (confidence exactly at 0.60 passed through ungated under `<`, changed to `<=`, decided explicitly). Ian/Nicole tied at -0.70 — investigated and confirmed as a **genuine data ceiling** (byte-for-byte identical FEMA metadata), not a bug.
4. **Round 4** (open-ended "use program count" instruction): failed — Beryl (3 programs) stayed tied with Ian/Nicole (2 programs) despite genuinely different input.
5. **Round 5** (explicit numeric rubric by program_count): worked, verified across all 9 declarations — clean separation by program-count tier (2 programs → -0.50, 3 programs → -0.75).

**Known limitation, not chased further**: the model resolves to one canonical value per program-count tier rather than continuous variation within a tier — effectively a 4-tier classifier, not a true continuous scale. Likely the real resolution the input data supports. Downstream fusion should weight cross-tier differences as signal, within-tier differences as noise.

**Step 3 (full batch) — complete and sanity-checked.** `run_event_agent.py` ran `interpret_event` once per unique (msa, declaration_id) — 153 calls (not 130, since disasters spanning multiple MSAs get one interpretation per MSA) — for **$0.031 total cost**. Results: 152 disaster / 1 zoning_change, gate fired on 57/153 (37%), non-gated magnitudes split -0.75×35 / -0.50×61 exactly matching the validated tiers.

Two spot-checks before marking it done, both passed:
- **The zoning_change classification** is a genuine edge case, not a bug: FEMA's own `incidentType="Other"` for `EM-3553-DC` ("59th Presidential Inauguration") — a security/logistics declaration that doesn't fit any of the 4 ontology categories well. Landed at confidence=0.60 → gated → impact=0, zero downstream effect. Logged as a known ontology gap (no "special event" bucket), not fixed.
- **Gate behavior confirmed non-arbitrary at scale**: 56/57 gated rows have program_count=1, 55/57 are EM-type; the 96 non-gated rows cleanly split into program_count 2/3 matching the -0.50/-0.75 tiers. Gate rate jumped from ~0% on the 9-declaration sample to 37% at full scale, and the jump tracks real severity signals, not noise.

Outputs: `event_agent_declarations.csv` (153 rows, one per unique declaration interpretation), `event_agent_weekly.csv` (8,601 rows, weekly grid joined with interpretations).

**Step 4 (mocked-LLM tests) — done.** `test_event_agent.py`: 13 tests, fully mocked (no real API calls, no API key needed, ~1.3s runtime), all passing — happy path, confidence gate incl. exact 0.60/0.61 boundary, ontology reject+retry, schema validation (5 malformed-input cases), and rubric pass-through for the two empirically-validated tiers. **Event Agent build is now complete**, ready for fusion-harness handoff alongside the AR Agent's real output.

---

## Open items for you

1. ~~Confirm the backtest start date~~ **Resolved**: live-eval cutoff is confirmed as **January 2026**, not the originally-assumed January 2025. `fold_boundaries.py` has the corrected 6-fold schedule (test folds now 29 weeks each, ending 2025-11-22). **`evaluation_protocol.md` itself is intentionally NOT yet updated** to reflect this — deliberately deferred until all agents are validated against the new dates, per explicit instruction.
2. **Sanity-check `hazard_index`** against local knowledge, especially Seattle's low count (17) and DC's high one (3,765) — both have plausible real-world explanations (climate, multi-state county span) but haven't been validated against a second source.
3. Events layer (Task 3) is scoping only — no code exists yet, by design.
4. **Decide on Miami/Phoenix handling for Phase 3** — both underperform naive with the shared ETS model. Diagnosed as genuinely volatile series (large, fast swings), not a fixable ETS bug — see `ar_agent.md` for the evidence. Open question: per-MSA model override, or accept as a known limitation of the shared-model approach.
5. **Decide on Macro Agent's fate** — keep it in the fusion pipeline as a documented-null/inactive component, or deprioritize/remove it? See `macro_agent.md` for the full three-round investigation. Not decided here.
6. No Macro Agent equivalent of `ar_agent_stub.py` exists — flag if Umang's fusion harness needs a macro-modifier stub regardless of the null result.
7. **Event Agent ontology gap** — no category fits "planned special event" FEMA declarations (found one: a presidential inauguration). Self-neutralized by the confidence gate this time; worth a 5th ontology category if more turn up in later data pulls.
8. If finer Event Agent severity resolution is ever needed, pulling OpenFEMA's Individual/Public Assistance funded-project datasets (dollar-figure severity data) is the identified next step — not started, deliberately out of scope for now.

## All files produced so far

| File | What |
|---|---|
| `pull_macro.py`, `macro_weekly.csv` | FRED macro pull |
| `pull_zillow.py`, `zillow_inventory_weekly.csv` | Zillow inventory pull |
| `align_data.py`, `aligned_weekly.csv`, `filled_rows_log.csv` | Alignment + forward-fill audit |
| `test_align_data.py` | Tests for the alignment/fill logic |
| `pull_intrinsic.py`, `intrinsic_static.csv` | Census + NOAA static features |
| `events_layer_scoping.md` | Phase 2 events-layer research note |
| `evaluation_protocol.md` | Backtest design, metrics, regime stratification (dates now stale, update deferred) |
| `macro_agent.py`, `test_macro_agent.py` | Macro Agent (ridge regression modifier) + tests + combiner comparison |
| `validate_macro_signal.py`, `validate_macro_lags.py`, `validate_macro_panel.py` | Three-round macro-signal investigation (see `macro_agent.md`) |
| `macro_agent.md` | Full Macro Agent writeup: design, combiner test, all three validation rounds, final null verdict |
| `README.md` | Task 1 data-pull documentation (date ranges, MSA coverage, gaps, anchor convention) |
| `fold_boundaries.py` | Corrected 6-fold expanding-window schedule (Jan 2026 cutoff) |
| `naive_baseline.py`, `test_naive_baseline.py` | Last-value baseline + tests |
| `ar_agent.py`, `test_ar_agent.py` | ETS-based AR Agent + tests |
| `compare_ar_vs_naive.py` | AR vs naive MAPE/RMSE comparison, per MSA per horizon |
| `diagnose_ar_outliers.py` | Miami/Phoenix underperformance diagnostic (read-only, no agent changes) |
| `ar_agent.md` | Full AR Agent writeup: schema, decisions, results, known limitations |
| `ar_agent_stub.py`, `test_ar_agent_stub.py` | Schema-valid fake AR output (Step 5) to unblock the fusion-harness teammate |
| `fetch_fema_events.py`, `fema_events_weekly.csv` | Event Agent Step 1: FEMA disaster declarations, weekly activity grid |
| `event_agent.py` | Event Agent Step 2: `interpret_event`, final prompt design (5 rounds), 9-declaration validation harness |
| `run_event_agent.py`, `event_agent_declarations.csv`, `event_agent_weekly.csv` | Event Agent Step 3: full batch runner + outputs (153 calls, $0.031) |
| `test_event_agent.py` | Event Agent Step 4: 13 mocked tests, no real API calls, ~1.3s runtime |
| `event_agent.md` | Full Event Agent writeup: all 5 prompt-design rounds, Step 3 batch results + sanity checks, Step 4 tests, known limitations, open items |
