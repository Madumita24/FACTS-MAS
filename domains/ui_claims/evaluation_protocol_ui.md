# Evaluation Protocol — UI Claims Pilot

Parallels `evaluation_protocol.md` (the housing pipeline's own protocol) in
structure and rigor, adapted for this domain's minimal-scope pilot: a
**single train/test split**, not a 6-fold backtest (see Scope, below).

## Entity selection methodology — a genuine improvement over the housing precedent

Worth recording plainly, including for the eventual paper's methodology
section: the original 15-MSA housing list has **no documented selection
criteria** anywhere in that codebase — `README.md` lists the 15 MSAs as
"requested," with no stated rationale (confirmed by inspection, not
assumed). This pilot's 15-state list was instead built to **explicit,
stated criteria**: full Census-region coverage, economic-base diversity
(tech/finance/energy/manufacturing/agriculture/tourism/logistics, not one
dominant industry), a data-quality floor excluding the thinnest-volume
states, and a deliberate split between states overlapping the housing
MSAs' states (10 of 15, leaving a cross-domain comparison open) and
genuinely new states covering regions the housing list has zero
representation in (Mountain West, Gulf South, Upper Midwest, Mid-South,
non-tech Pacific NW — 5 of 15). This document is the record of that
methodology; the housing project has no equivalent to point to.

## History window: 2013-01-01 → present

Not a data-availability limit — FRED's ETA 539 mirror goes back to 1984-86
for every target state, far deeper than this pilot uses. The window is a
deliberate scope decision:

- **Avoids the 2008-09 crisis and its multi-year elevated-claims recovery
  tail.** National claims did not return to a stable pre-crisis baseline
  until roughly 2013 — starting there means the window begins in a
  comparable regime to where it ends, rather than opening inside an
  extreme outlier era whose dynamics (extended federal benefit programs,
  a labor market still absorbing the crisis) don't resemble the rest of
  the window.
- **Overlaps the housing pipeline's own macro regime for comparability.**
  `pull_macro.py`'s window starts 2018; 2013-2026 fully contains that and
  extends it earlier, so `fed_funds`/`cpi` here span the same
  post-crisis-ZIRP → 2015-2018 hikes → 2019 cuts → COVID-ZIRP → 2022-23
  hiking cycle → current arc already used elsewhere in this project,
  rather than a disjoint or non-overlapping macro history.

## COVID-19 (2020-2021): included deliberately, not excised — a known limitation, not an oversight

National initial claims went from roughly 200K/week to ~6M/week within a
few weeks in April 2020 — confirmed in this pilot's own data too (Step 1
sanity check: California peaked at 23.6x its pre-COVID baseline, Colorado
at 46.1x). Two reasons this stays in the window rather than being cut out:

1. **Series continuity.** AR/ETS depends on lag structure; removing 2020-21
   entirely would leave a discontinuous series, breaking exactly the
   lag-based features the AR agent's logic depends on. This is the same
   reason the housing pipeline never drops weeks outright, only fills or
   flags gaps.
2. **It is this domain's single most important stress-test.** A pilot that
   never had to face its hardest period would be a weaker piece of
   evidence than one that did, gated appropriately.

**Explicitly flagged as a known limitation, not a solved problem**: this
AR+Macro-only pilot has **no Event-agent equivalent** to gate or flag an
anomaly like COVID the way the housing pipeline's Event Agent gates FEMA
declarations. Left ungated, a shock of this size would dominate any metric
it falls inside. The train/test split below is designed specifically so
that risk lands in training (where the model gets to learn from it, but
isn't scored against an un-gated shock it had no way to see coming) rather
than in the held-out test window.

## Single train/test split

**Not a 6-fold backtest** — per the team's agreed minimal scope for this
pilot (AR + Macro agents only). A full multi-fold expanding-window backtest,
matching `evaluation_protocol.md`'s design, is **Phase UI-4 (stretch)**,
contingent on this minimal pilot's results justifying the added scope.

**Horizons: 4/8/13 weeks, same as the housing pipeline — confirmed, not
assumed.** This is a deliberate comparability decision, not a claim that
unemployment claims data calls for these specific horizons on its own
merits: this pilot exists to answer "does the FACTS-MAS architecture
generalize to a second domain," and that question is only answerable if
the horizons match housing's exactly, so a horizon-by-horizon result here
is directly comparable to the corresponding housing-domain number rather
than requiring a horizon-adjustment caveat on every comparison.

Embargo: **13 weeks (91 days)**, identical to the housing pipeline's
convention (`facts_mas/run_backtest.py`'s `validate_embargo`), sized to
this pilot's longest horizon (13wk, per the above).

| | Date | Note |
|---|---|---|
| Train start | 2013-01-05 | First date in `aligned_weekly_ui_claims.csv` |
| **Train end** | **2023-06-24** | See below |
| Embargo | 91 days (13wk) | Matches housing's `validate_embargo` exactly |
| **Test start** | **2023-09-23** | `train_end` + exactly 91 days — a real grid Saturday, no rounding |
| **Test end** | **2026-08-22** | Last available date |

- **Train**: 547 weeks/state (7,105 cells across 15 states)
- **Test**: 153 weeks/state (2,295 cells across 15 states)

**Why 2023-06-24, evidence-based rather than a round-number guess**:
checked when each sampled state's claims actually normalized post-COVID
(8-week trailing mean back within 20% of its 2019 baseline) rather than
assuming a date. Result: Colorado 2021-10-23, Texas 2021-11-27, California
2022-03-05 (the slowest of the three). `train_end` sits **~15 months after
the slowest-normalizing sampled state's recovery**, so training ends well
inside a settled, non-pandemic-distorted regime, not partway through the
recovery — and the entire acute shock plus its resolution both sit inside
training, none of it inside test.

## Metrics

Same convention as `evaluation_protocol.md`:

- **MAPE** and **RMSE**, computed **per state**, then averaged across the
  15 states (unweighted, so California's ~60K/week mean doesn't dominate
  the headline number the way New York would in the housing set).
- Reported per horizon, **never averaged across horizons**.

## Scope note

This is a **single train/test split**, not a 6-fold rolling backtest. That
is the agreed minimal scope for this pilot (AR + Macro agents only, one
split, no Event/Seasonality/Intrinsic agents, no fusion weighting). A full
multi-fold backtest matching the housing pipeline's design is the natural
next step (**Phase UI-4, stretch**) if this pilot's single-split results
are promising enough to justify the added build and compute cost.
