# UI Claims Pilot — Second-Domain Generalization Test for FACTS-MAS

## 1. Purpose

A second-domain generalization pilot for the FACTS-MAS architecture, per
Prof. Pan's request: does the multi-agent forecasting pattern validated on
weekly housing inventory (15 MSAs) transfer to a structurally different
domain — weekly Unemployment Insurance initial claims by state (15
states)? This directory is the complete, self-contained answer.

## 2. Scope

**Minimal pilot, per team-agreed scope**: AR + Macro agents only (no
Event/WARN-Act, Seasonality, or Intrinsic agents), a **single train/test
split** (not a 6-fold backtest), and a simple naive equal-weight blend
rather than the housing pipeline's full skill-weighted fusion mechanism.
Full methodology and reasoning in [evaluation_protocol_ui.md](evaluation_protocol_ui.md).

## 3. Headline result

**The architecture pattern generalizes, with a real caveat.** AR beats
naive on a majority of states at every horizon (10/15, 12/15, 8/15 at
4/8/13wk); Macro does not (0/15 states at every horizon) — the same shape
housing shows. But AR's edge here is meaningfully weaker than housing's
near-unanimous win rate (14/15, 14/15, 13/15) — flagged as a real,
undiagnosed difference (unconfirmed hypothesis: state-level claims may be
structurally more volatile on average than MSA-level housing inventory),
not explained away. Full results and both caveats in
[pilot_results_ui.md](pilot_results_ui.md).

## 4. Key validating finding

A naive equal-weight AR+Macro blend is **strictly worse than AR alone on
every one of 15 states, at every horizon, no exceptions**. This is
independent, second-domain evidence for the housing pipeline's
skill-weighted fusion design (`facts_mas/fusion_baseline.py`) — blind
averaging actively hurts here too, not just in housing, which is a
stronger argument for the general principle than either domain's result
alone.

## 5. Two caught near-misses — evidence of methodological rigor, not footnotes

1. **COVID-driven spurious significance**, caught before being reported.
   A first-pass lag-grid sweep for the Macro agent's fed_funds/cpi
   features found 55/72 (horizon, feature, lag) cells "significant" —
   but with coefficients 10-100x larger than anything in housing's own
   investigation and signs flipping between adjacent lags, both red
   flags. Refitting with the 2020-2021 COVID window excluded collapsed
   this to 39/72 (26/72 sign-stable) — claims spiked 20-46x baseline in
   April 2020 (housing inventory never moved anywhere near that), and a
   handful of extreme-leverage weeks were dominating the regression, not
   a real relationship. See [lag_grid_search_ui.py](lag_grid_search_ui.py).

2. **Serial-correlation-inflated significance**, caught via a
   Driscoll-Kraay follow-up. Of the 26 COVID-excluded cells, 24 were cpi
   cells that survived Benjamini-Hochberg correction (reusing
   `facts_mas.factor_scoring._apply_fdr` directly) at 24/24 — but with
   p-values as small as 2.2e-16, an anomaly worth distrusting rather than
   celebrating. Diagnosed directly (AR(1)-on-residuals test, cross-state
   ACF check: both significant at all 24 cells, coefficient climbing
   +0.38 to +0.67 with horizon, exactly matching the overlapping-forecast-
   window mechanism) and refit with Driscoll-Kraay SEs: only 15/24 (62%)
   survive, and the collapse tracks the same horizon gradient — 4wk mostly
   survives (10/12), 13wk fully collapses (0/4). Resolved to a coherent,
   mechanistically-explained **partial result**, not a clean win or a
   clean null. See [serial_correlation_check_ui.py](serial_correlation_check_ui.py).

Neither near-miss was left as an unresolved caveat — both were diagnosed,
tested, and the actual finding (not the more convenient-looking first
answer) is what's recorded in `pilot_results_ui.md`.

## 6. Files in this directory

| File | Role |
|---|---|
| `pull_ui_claims.py` | Pulls weekly initial claims (FRED's ETA 539 mirror) for all 15 states |
| `pull_macro_ui.py` | Pulls fed_funds + cpi from FRED (adapted from `pull_macro.py`, unemployment excluded — circular with the claims target) |
| `align_data_ui.py` | Lag-safe join of claims + macro data (adapted from `align_data.py`) |
| `ui_claims_weekly.csv` | Raw claims pull output |
| `macro_weekly_ui.csv` | Raw macro pull output |
| `aligned_weekly_ui_claims.csv` | Final aligned dataset (date, state, claims_count, fed_funds, cpi) |
| `filled_rows_log_ui.csv` | Forward-fill audit log (0 rows filled — raw data had no gaps) |
| `evaluation_protocol_ui.md` | Entity selection, history window, COVID-inclusion reasoning, train/test split |
| `naive_baseline_ui.py` | Last-value baseline + shared `assert_no_lookahead` guard |
| `agents/ar_agent_ui.py` | ETS forecaster (adapted from `ar_agent.py`) |
| `agents/macro_agent_ui.py` | Lag-aware ridge regression modifier (adapted from `macro_agent.py`) |
| `evaluate_ui_claims_pilot.py` | Step 3's full single-split evaluation runner |
| `ui_claims_eval_per_origin.csv` / `_per_state.csv` / `_summary.csv` | Evaluation outputs at increasing levels of aggregation |
| `pilot_results_ui.md` | The citable pilot result — full tables, caveats, both near-misses resolved |
| `lag_grid_search_ui.py` | Panel-FE, clustered-SE, COVID-robust, BH-corrected lag sweep for Macro's fed_funds/cpi |
| `lag_grid_search_results_ui.csv` | Full 72-cell lag-sweep output |
| `serial_correlation_check_ui.py` | Steps 6-9: serial-correlation diagnosis + Driscoll-Kraay refit of the 24 surviving cpi cells |
| `serial_correlation_check_results_ui.csv` | Per-cell clustered vs. Driscoll-Kraay comparison |
| `README.md` | This file |

## 7. Status

**Pilot complete and documented.** Phase UI-4 (Event/WARN-Act agent,
Seasonality agent, full multi-fold expanding-window backtest matching
`evaluation_protocol.md`'s design, and porting the real skill-weighted
fusion mechanism) is the justified next step — **not yet started**,
contingent on this pilot's result being judged sufficient to build on.

## 8. Isolation confirmed — housing pipeline untouched throughout

No file under `facts_mas/`, and none of `pull_macro.py`, `pull_zillow.py`,
`align_data.py`, `ar_agent.py`, `macro_agent.py`, `naive_baseline.py`, or
`aligned_weekly.csv`, was modified at any point in this pilot. Verification
checkpoints performed along the way, not just claimed at the end:

- After Step 1.4 (alignment): `git status` checked, only `domains/` new.
- After Step 2.4 (agent sanity check): `git status` checked again post-port.
- Throughout: every housing file referenced was opened with Read only, for
  logic reference — never Edited. Every domains/ui_claims/ agent file's own
  docstring states explicitly what housing logic it was adapted from and
  that it was copied, not imported.
- The one deliberate, explicit exception: `serial_correlation_check_ui.py`
  and `lag_grid_search_ui.py` import `facts_mas.factor_scoring._apply_fdr`
  directly (read-only import of a generic statistical utility, not
  domain-specific agent logic) — done on explicit instruction, to reuse a
  tool already built and fixed for a real bug rather than reimplementing
  it. This is an import of housing code into the pilot's diagnostics, never
  the reverse, and touches no housing file.
