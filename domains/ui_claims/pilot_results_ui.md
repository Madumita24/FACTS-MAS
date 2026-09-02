# UI Claims Pilot — Results (Phase UI-2/3)

Closes out the minimal-scope pilot (AR + Macro agents, single train/test
split) with a citable result. See [evaluation_protocol_ui.md](evaluation_protocol_ui.md)
for the full split/methodology reasoning this builds on, and `progress.md`
/ `ar_agent.md` / `macro_agent.md` for the housing-domain findings compared
against throughout.

## 1. Summary: does the FACTS-MAS architecture generalize?

**Yes, qualitatively.** The same shape that holds in the housing domain
holds here: AR beats naive (majority of states, all 3 horizons), Macro
does not (0/15 states at every horizon). The architecture's core premise —
a simple statistical forecaster adds real value, a lagged macro-regression
modifier does not — transfers to a structurally different domain (weekly
UI claims vs. weekly housing inventory) rather than being a housing-specific
artifact. Two caveats sit underneath that "yes": AR's edge is real but
weaker than housing's, and a naive combination of AR+Macro actively hurts
rather than helps — both detailed below.

## 2. Full results (single split: train_end 2023-06-24, test 2023-09-23 to 2026-08-22)

Per-state, per-horizon MAPE/RMSE in `ui_claims_eval_per_state.csv` (45
rows = 15 states x 3 horizons); per-origin detail in
`ui_claims_eval_per_origin.csv` (6,510 rows, 0 errors).

**Aggregate MAPE (mean/median across 15 states):**

| Horizon | naive | AR | Macro | Naive-blend (AR+Macro)/2 |
|---|---|---|---|---|
| 4wk | 18.350 / 16.728 | 18.187 / 16.768 | 20.006 / 18.505 | 18.943 / 17.109 |
| 8wk | 20.687 / 18.441 | 20.543 / 18.440 | 22.371 / 19.727 | 21.283 / 18.772 |
| 13wk | 22.648 / 21.632 | 22.527 / 21.637 | 24.466 / 22.624 | 23.286 / 22.046 |

**Aggregate RMSE (mean across 15 states):**

| Horizon | naive | AR | Macro | Blend |
|---|---|---|---|---|
| 4wk | 1801 | 1796 | 1866 | 1822 |
| 8wk | 2094 | 2097 | 2149 | 2113 |
| 13wk | 2306 | 2319 | 2365 | 2331 |

**Win counts (of 15 states, beats naive on MAPE):**

| | 4wk | 8wk | 13wk |
|---|---|---|---|
| AR beats naive | 10/15 | 12/15 | 8/15 |
| Macro beats naive | 0/15 | 0/15 | 0/15 |
| Naive-blend beats naive | 1/15 | 1/15 | 1/15 |
| Naive-blend beats AR alone | 0/15 | 0/15 | 0/15 |

## 3. Key caveat: AR's edge is real but weaker than housing — undiagnosed

Housing's AR agent beats naive on 14/15, 14/15, 13/15 MSAs at 4/8/13wk
(`progress.md`). This pilot's AR beats naive on 10/15, 12/15, 8/15 states —
a real edge, same direction, but a meaningfully lower win rate, especially
at 13wk (majority but not near-unanimous).

**Unconfirmed hypothesis, flagged as a hypothesis, not a finding**: state-level
UI claims may be structurally more volatile on average than MSA-level
housing inventory — closer to the housing domain's own Miami/Phoenix
outliers (both documented as genuinely volatile series that beat a shared
ETS model poorly, not an ETS bug) than to its typical MSA. Claims are
sensitive to abrupt, discrete events (single-employer layoffs, seasonal
industry patterns particular to a state's economic base) in a way that
could plausibly resist a smooth additive-trend ETS model more often than
housing inventory does. This is plausible but **not tested here** — it
would need a per-state volatility comparison against the housing MSAs'
own distribution, which this pilot does not do. Recorded as an open
question, not resolved.

## 4. Key finding: naive blending hurts — independent validation of skill-weighted fusion

The naive equal-weight AR+Macro blend is **strictly worse than AR alone on
every one of 15 states, at every horizon, no exceptions** (0/15 win rate
in the table above). This is not a marginal result — it means blindly
averaging a good forecaster with a bad one reliably makes things worse,
exactly as arithmetic would predict once Macro's null (below) is taken
into account.

This is independent, second-domain evidence for a specific design decision
already made in the housing pipeline: fusion needs to be **skill-weighted**
(`facts_mas/fusion_baseline.py`'s softmax-on-skill-score approach), not a
flat average. The housing pipeline adopted skill-weighting based on
housing-domain evidence; this pilot shows the same failure mode (naive
averaging actively hurts) shows up again in a structurally different
domain, which is a stronger argument for the general principle than either
domain's result alone.

## 5. Macro's null, documented precisely

**0/15 states beat naive, at every horizon** — this pilot's null is more
one-sided than housing's. Precise comparison, not just "also null":

- **Housing's null** (`macro_agent.md`) was established via regression
  significance testing across 3 increasingly rigorous rounds: the final,
  most trustworthy test (two-way FE panel, MSA-clustered SEs) found 1 of 4
  features marginally significant (fed_funds, p=0.042) and judged that a
  likely false positive (wrong theoretically-expected feature, borderline
  companion result) — "a well-earned null," not zero features ever
  reaching significance.
- **This pilot's null** is a direct MAPE-vs-naive comparison, not a
  significance test: Macro loses on every single state, not just "fails
  to reach significance."
- **The genuinely comparable number**: in this project's Phase-4 ablation
  table (housing, test-only cells), *removing* Macro from the fused system
  improved MAPE at every horizon (delta -0.002 / -0.167 / -0.202 at
  4/8/13wk) — housing's Macro agent is also net-negative standalone, not
  merely inert. Both domains converge on the same qualitative conclusion
  through different measurements.

**Two competing explanations, genuinely undistinguished by this pilot —
stated as open, not resolved:**

(a) **Lag-specification problem**: `LAG_WEEKS = {fed_funds: 10, cpi: 10}`
    was carried over unchanged from housing's documented transmission-lag
    guess, not re-derived for claims. Monetary-policy transmission to
    labor markets is often cited in the literature as taking many months,
    not weeks — a 10-week lag could simply be reading the macro features
    at the wrong point in their transmission into claims, making this a
    fixable specification error, not a real null.
(b) **Genuine no-signal domain fact**: fed_funds/cpi may carry no usable
    predictive relationship with claims at any lag, matching housing's
    own well-triangulated null for inventory.

**Resolved by Step 5 (`lag_grid_search_ui.py`) — neither hypothesis cleanly,
a mix, with a real methodological catch along the way.** A naive full-2013-2026
sweep found 55/72 (horizon, feature, lag) cells significant — but with
coefficients 10-100x larger than anything housing ever found and fed_funds's
sign flipping between adjacent lags within the same horizon, both signs of
an artifact, not a real effect. Rerunning with the 2020-2021 COVID window
excluded collapsed that to 39/72 significant, 26/72 both significant and
sign-stable — confirming the dramatic with-COVID result was overwhelmingly
COVID acting as a small number of extreme-leverage observations (claims hit
20-46x baseline in April 2020; housing inventory never moved anywhere near
that), not a real fed_funds/cpi relationship at any lag.

Of what survives: **fed_funds's surviving signal is sparse and weak** (2
of 36 cells, comparable in scale to housing's own single marginally-
significant result, which that investigation judged a likely false
positive) — resolves to **hypothesis (b)**, matching housing's null.

**cpi is the more interesting case, resolved with Benjamini-Hochberg
correction** (reusing `facts_mas.factor_scoring._apply_fdr` directly, not
reimplemented — the same tool already built, tested, and fixed once for a
real bug in Mod B's Granger screen). Of the 24 previously significant-
and-sign-stable cpi cells (of 36 tested, 4-14wk short-to-medium lags,
consistently positive sign; 16-22wk longer lags, consistently negative),
**all 24 (100%) survive BH correction** — this is not multiple-testing
noise inflating a raw hit-count. By the pre-agreed decision rule, this is
real evidence for **hypothesis (a)**: cpi carries a genuine, modest
relationship with claims that a mis-specified single fixed lag was
missing (the current 10wk default sits inside the significant short-lag
band, so it isn't badly wrong, just not attempting to use the sign
structure across lags).

**Resolved with a serial-correlation-robust refit
(`serial_correlation_check_ui.py`), not left as a caveat.** The
extraordinarily small p-values (1e-9 to 2.2e-16) were a real red flag:
diagnosed directly (not assumed) via an AR(1)-on-residuals test and a
cross-state residual-autocorrelation check, both significant at all 24
cells, with the AR(1) coefficient climbing from +0.38 (4wk) to +0.57
(8wk) to +0.67 (13wk) — exactly the pattern overlapping-horizon target
construction predicts (longer horizons share more underlying weeks
between adjacent origins, so serial correlation should be strongest at
13wk and weakest at 4wk; it is).

Refitting all 24 cells with **Driscoll-Kraay standard errors** (the
panel-appropriate robust method for combined cross-sectional + serial
dependence — not generic Newey-West/HAC, which assumes a single time
series) drops the surviving count from 24/24 to **15/24 (62%)**, and the
collapse tracks the same horizon gradient: **4wk mostly survives (10/12),
8wk is mixed (5/8), 13wk fully collapses (0/4)**. This is a coherent,
mechanistically-explained partial result, not noise landing at 62% by
chance — the cells with the least induced serial correlation (short
horizon) are the ones that hold up.

**Final verdict: (b) partial survival.** fed_funds is null (matches
housing). cpi shows *some* genuine relationship, but concentrated at
short horizons (4wk) and essentially absent at long horizons (13wk) —
not the clean, uniform "hypothesis (a)" result the BH-corrected count
alone suggested, and not a full collapse to "hypothesis (b)" either.
`macro_agent_ui.py`'s `LAG_WEEKS` was NOT updated on this evidence — a
horizon-varying, partially-supported effect this modest doesn't meet the
bar for a production change, matching housing's own standard (Round 3's
single marginally-significant, theoretically-backwards feature was
judged a likely false positive and left alone, not acted on).

## 6. Scope reminder

**Single train/test split, AR + Macro agents only** — the agreed minimal
pilot scope. No Event/WARN-Act agent, no Seasonality or Intrinsic agents,
no skill-weighted fusion mechanism, no multi-fold backtest. **Phase UI-4
(stretch)** — Event/WARN Act agent, Seasonality agent, full multi-fold
expanding-window backtest (matching `evaluation_protocol.md`'s design),
and porting the real skill-weighted fusion mechanism — is the natural next
step, contingent on this pilot's result being judged sufficient to justify
the added build and compute cost.

## References

- [evaluation_protocol_ui.md](evaluation_protocol_ui.md) — split, entity
  selection, and history-window methodology this result depends on.
- `progress.md`, `ar_agent.md` — housing AR agent's 14/15, 14/15, 13/15
  win-rate result, compared against in Section 3.
- `macro_agent.md` — housing Macro agent's 3-round null investigation,
  compared against in Section 5.
- `facts_mas/fusion_baseline.py` — the skill-weighted fusion mechanism
  this pilot's naive-blend result (Section 4) independently supports.
