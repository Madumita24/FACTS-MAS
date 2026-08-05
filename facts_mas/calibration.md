# Calibration Agent — Writeup

Status: built, tested (14/14), and run against the Event Agent's two
self-flagged arbitrary constants. **Headline finding: the Event Agent's
magnitude signal is net-harmful at every value tested, and the fitted
optimum is zero.** Details in §3–§4.

---

## 1. What it is

The Calibration Agent learns forecasting constants from backtest error
instead of having them hand-set. It sweeps candidate values for a constant,
scores each on every fold's training window, and promotes a new value only
if it improves MAPE *consistently across folds* — not merely on the pooled
average.

It is a producer for calibration models that already existed but had never
been wired to real data. `facts_mas/state/micro_reasoning.py` (from the
Phase-1 NEXUS state work) defines `FoldResult`, `CalibrationGuideline`, and
`TightenedCalibrationResult`, including the Mod-D fold-consistency gate.
Those were tested in isolation and unused. This module is what feeds them.

| File | Role |
|---|---|
| `facts_mas/calibration.py` | The agent: sweep, fold-consistency gate, reporting |
| `facts_mas/test_calibration.py` | 14 tests (no LLM calls, no full backtest) |
| `calibrate_event_window.py` | Study 1: `DECLARATION_ACTIVE_WINDOW_WEEKS` |
| `calibrate_event_impact_scale.py` | Study 2: `IMPACT_SCALE` |

## 2. Two design constraints treated as non-negotiable

**No lookahead.** Candidates are scored only on a 104-week window ending at
`fold.train_end` — the same lookback `fusion_baseline.compute_fold_weights`
uses, so calibration and weighting share one slice of history rather than
two silently different ones. A fold's test window is never read while
choosing. `evaluate_on_test_window()` reports out-of-sample performance but
is a separate call that `calibrate_target()` never invokes, so no execution
path can let a test number influence a choice. Four tests assert this,
including one that spies on every origin the scorer proposes and fails if
any falls outside the window.

**No edits to agents owned by others.** Both constants live in
`facts_mas/agents/event_agent.py`, which is Madumita's under "one file, one
owner." Sweeping a constant requires varying it, so `override_constant()` is
a context manager that sets the module attribute and restores it on exit —
including on exception, which is tested, because one failed sweep leaving a
teammate's constant mutated would be a nasty and invisible bug. The source
file is never modified. **This module recommends; a human applies.**

The gate is the NEXUS Mod-D rule: "the same direction of correction in at
least 4 of the 5 training folds individually." This backtest has 6 folds, so
the equivalent is 5 of 6 — the nearest integer at least as strict as the
proposal. One test asserts 5/6 passes and 4/6 fails.

One subtlety worth stating: `CalibrationGuideline`'s own consistency check is
direction-*agnostic*. It confirms folds agree, not that they agree on an
improvement. A consistently-*worse* candidate passes that check. Approval
therefore also requires `dominant_direction == "decrease"`. There is a test
named for exactly this.

## 3. Study 1 — `DECLARATION_ACTIVE_WINDOW_WEEKS` (currently 8)

`event_agent.py` says the value is "a reasonable-but-arbitrary starting
assumption pending real calibration data." Sweeping `(2, 4, 6, 8, 12, 16,
26)` over origins where the candidates actually differ:

| window | 2 | 4 | 6 | **8** | 12 | 16 | 26 |
|---|---|---|---|---|---|---|---|
| mean val. MAPE | **5.442** | 6.067 | 6.530 | *6.671* | 6.797 | 7.943 | 8.905 |

Window 2 was approved (better in 6/6 folds) and held up out-of-sample:
−2.27 MAPE mean on test windows, better in 2 of the 3 folds that had
qualifying origins and unchanged in the third.

**But the ordering is perfectly monotonic**, which is equally consistent
with "the optimum is near 2" and with "any event impact hurts, so less is
always better." Extending the grid down to 0 settles it:

| window | **0 (impact off)** | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| mean val. MAPE | **5.321** | 5.844 | 6.094 | 6.318 | 6.532 | 6.855 | 6.974 |

Zero wins, in 6 folds out of 6. Shortening the window was treating a
symptom: it helped only by applying the wrong magnitude less often.

## 4. Study 2 — `IMPACT_SCALE` (currently 0.10)

This is the constant that converts the LLM's qualitative −1..+1 severity
score into a weekly percentage change. `event_agent.py` calls it an
"ARBITRARY, UNVALIDATED CHOICE … a documented guess, not a fitted value,"
and a TODO directly below predicts its failure mode: a WILDFIRES declaration
whose "-7.5% adjustment overshot [the actual −1.5%] roughly 5x."

A 5x overshoot at 0.10 predicts a fitted scale near 0.02. Sweeping:

| IMPACT_SCALE | **0.0** | 0.01 | 0.02 | 0.03 | 0.05 | 0.075 | **0.10** |
|---|---|---|---|---|---|---|---|
| mean val. MAPE | **5.813** | 5.982 | 6.227 | 6.529 | 7.229 | 8.242 | *9.357* |

The fitted value is **0.0**, approved at 6/6 folds. There is no interior
optimum. The overshoot diagnosis was directionally right but understated:
the problem is not that 0.10 is ~5x too large, it is that no positive scale
beats applying no adjustment at all. At the production 0.10 the constant is
costing **3.5 MAPE points** on the origins where it fires.

*Method note:* `IMPACT_SCALE` is a function-local variable, not a module
attribute, so `override_constant()` cannot reach it. Study 2 reproduces the
single line that consumes it (`factor = 1 + impact_magnitude * IMPACT_SCALE`)
against the same cached CSV, rather than editing Madumita's file. A parity
check against the real `run_event_agent` runs on every invocation and raises
if the two ever diverge. **Recommendation: promote `IMPACT_SCALE` to a
module constant** so future sweeps need no reimplementation.

## 5. What this does and does not mean

It does **not** mean events are irrelevant to housing inventory. It means
this agent should not be emitting a *point forecast* derived from that
severity score. That is consistent with `event_agent.md`'s own finding that
the LLM resolves to a 4-tier classifier rather than a continuous scale —
the score carries ordinal information, and the pipeline is reading it as
cardinal.

It also puts the Event Agent in the same position the Macro Agent already
occupies: a well-earned null on magnitude. Two of five agents now have no
usable magnitude signal, which is a result worth stating plainly in the
paper rather than burying.

## 6. Open — not measured here

**Fusion-level impact is not yet quantified.** Everything above is measured
at the agent level, on origins where a declaration is active. The Event
Agent does carry non-trivial fusion weight (see §7), so the effect should
propagate, but the size of it at the fused-forecast level is unmeasured. A
full 5-agent backtest re-run with the calibrated value is the natural next
step; it was not run here because a single fold's weight computation alone
takes ~15 minutes at production settings.

Also open: whether the right fix is scale=0, or restructuring the Event
Agent to contribute a *gate* or a confidence signal rather than a point
forecast. Calibration can answer "what value of this constant is best"; it
cannot answer "should this constant exist." That is a design decision.

## 7. Fusion weight context

The miscalibrated magnitude is not filtered out by the fusion layer before
it reaches the forecast. A spot check of `compute_fold_weights` at fold 6
(2 MSAs, 26-week lookback — **indicative, not the production configuration**)
gives:

| agent | ar | macro | event | seasonality | intrinsic |
|---|---|---|---|---|---|
| weight | 0.852 | 0.073 | **0.070** | 0.000 | 0.005 |

Event carries roughly 7% of the fused forecast under this configuration —
small, but far from the zero that would make the calibration finding
cosmetic. Softmax is not silently neutralising it.

The full-configuration weight run (15 MSAs, 104-week lookback) takes ~15
minutes per fold and is not included here; it should be attached before the
paper cites any weight figure. Treat the table above as a sanity check that
the finding matters, not as a number to quote.
