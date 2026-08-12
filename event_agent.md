# Event Agent — Build & Validation Notes

Status: **Build complete, Steps 1-4 done.** Step 1 (FEMA fetch) and Step 2 (LLM interpretation) built and validated on a 9-declaration sample across 3 MSAs; prompt design finalized after 5 iterative rounds surfaced and fixed two real bugs (not just tuning). Step 3 (full batch across all unique declarations) run and passed a post-hoc sanity check on both the gate behavior and the one ontology edge case it produced. Step 4 (13 mocked-LLM tests, no real API calls) written and passing. Ready to hand off to the fusion harness.

---

## Scope and boundary

The Event Agent is the first LLM-based component in this pipeline — everything upstream (AR, Macro) is pure statistics. Strict boundary: **the LLM never outputs a forecast number.** It classifies `event_type` and a bounded, qualitative `impact_magnitude` (-1.0..+1.0) from FEMA disaster declaration text — a directional pressure signal for Umang's Synthesizer to weight, not an inventory estimate in itself.

**Provider note**: spec called for Claude at temperature 0.1. Only an OpenAI key was available — confirmed intentional (budget constraint, not a design preference) — so this uses **gpt-4o-mini** at the same temperature=0.1, for reproducibility across backtest folds.

## Files

| File | Role |
|---|---|
| [fetch_fema_events.py](fetch_fema_events.py) | Step 1: OpenFEMA pull, county→MSA crosswalk reuse, weekly activity grid |
| [event_agent.py](event_agent.py) | Step 2: `interpret_event`, prompt, validation, confidence gate, `build_declaration_input` (shared with Step 3) |
| [run_event_agent.py](run_event_agent.py) | Step 3: full batch runner, `(msa, declaration_id)` caching, cost-confirmation gate |
| `fema_events_weekly.csv` | Step 1 output: 8,601 rows, 130 unique declarations, all 15 MSAs |
| `event_agent_declarations.csv` | Step 3 output: 153 rows, one per unique (msa, declaration_id) interpretation |
| `event_agent_weekly.csv` | Step 3 output: 8,601 rows, Step 1's weekly grid joined with each declaration's interpretation |
| [test_event_agent.py](test_event_agent.py) | Step 4: 13 tests, fully mocked (no real API calls, no API key needed, runs in ~1.3s) |

---

## Step 1 — FEMA data fetch

Pulls `DisasterDeclarationsSummaries` from the OpenFEMA API (no key needed) for every county belonging to the 15 target MSAs, reusing `pull_intrinsic.py`'s exact county↔CBSA crosswalk rather than rebuilding it. Output is long-format: one row per (msa, week, active declaration) — sparse, since most weeks have zero active declarations for a given MSA.

**Design choice, deliberately deviating from the literal spec wording**: the spec said "cache by (msa, week)" in Step 3, but some declarations span years (COVID-19: ~172 active weeks). Caching by `(msa, week)` wouldn't actually deduplicate those — each week is a distinct key even though the declaration text is identical. So Step 1's output preserves `declaration_id` per row specifically so Step 3 can cache by `(msa, declaration_id)` instead, which is what actually achieves the stated goal ("the same declaration shouldn't be re-interpreted multiple times"). Flagged explicitly rather than silently deviating.

**Result**: 8,601 rows, 130 unique declarations, all 15 MSAs covered. Washington DC and Philadelphia have disproportionately many rows (1,393 and 1,379) because their CBSAs span multiple states, each of which filed separate COVID-19 declarations.

## Step 2 — LLM interpretation: 5 rounds to a working prompt

The sanity-check requirement ("show me real results before the full batch") did its job — it caught two genuine bugs that a shortcut straight to Step 3 would have baked into 130 declarations × however many MSA/fold combinations before anyone looked closely.

### Round 1 (original prompt, thin metadata)
Declaration text = title + incident type + county only. Result: Hurricane Ian (catastrophic) and Hurricane Nicole (minor, glancing) both scored impact=-1.00 — no severity differentiation possible from what the model was given. **Confidence gating never fired** (9/9 ungated), including a small, obscure brush fire at confidence=0.80 despite the prompt explicitly asking for low confidence on ambiguous/minor input.

### Round 2 (Step 1b/2b: enriched metadata, first calibration pass)
Added `declarationType` (DR/EM) and the four FEMA assistance-program flags (real severity proxies, not fabricated). **Result: every single one of the 9 declarations flipped sign**, negative → positive. Root cause: neither prompt version had ever defined what positive vs. negative *means* for inventory — "-1.0 = negative pressure" was never anchored to an actual causal direction, so the model's implicit sign convention wasn't stable across prompt edits. This was a more serious bug than the one the round was meant to fix — an unstable sign convention would silently corrupt anything downstream that assumes a consistent meaning.

### Round 3 (Step 2d/2e: explicit causal grounding + reasoning field + gate boundary fix)
Rewrote the prompt to require the model to choose between two explicit causal mechanisms before assigning a sign (disrupted transactions → decrease, vs. distressed sales/relocation → increase), added a required `reasoning` field (auditable, not a black box), and fixed a real off-by-one: confidence exactly at 0.60 was passing through ungated because the check was `< 0.60`, not `<= 0.60` — decided and documented `<=` explicitly. **Result**: sign consistency fixed (all 8 non-gated declarations negative, no flips), gate correctly caught the Pauline Road Fire case at confidence=0.40. But: Ian and Nicole were now tied at exactly -0.70 — and so was *every other* non-gated declaration. Investigated and confirmed **not a bug**: Ian and Nicole have byte-for-byte identical FEMA severity metadata (same declaration type, same program flags) — the model had no basis in the given data to differentiate them. A genuine data ceiling, not a prompt failure.

### Round 4 (Step 2g: open-ended program-count instruction)
Beryl has one more active program than Ian/Nicole (3 vs. 2) — a real, non-ceiling distinguishing signal the prompt wasn't using. Added an open-ended instruction: "more active programs = more severe, use this as a fine-grained signal." **Result: failed.** Beryl still landed on the same -0.70 as Ian/Nicole despite genuinely different input. The model appeared to anchor on a single "generic severe disaster" value once past the gating threshold, regardless of a stated qualitative distinction.

### Round 5 (Step 2h: explicit numeric rubric) — final design
Replaced the open-ended instruction with an exact rubric: `program_count=0 → [0.0,0.2]`, `1 → [0.2,0.4]`, `2 → [0.4,0.6]`, `3-4 → [0.6,1.0]` (magnitude size; sign still comes from the causal-mechanism choice). **Result: worked, verified across all 9 declarations, not just the pair explicitly checked:**

| program_count | Declarations | impact_magnitude |
|---|---|---|
| 2 | Nicole, Ian, both Chicago storm declarations, Pauline Fire (pre-gate) | -0.50 |
| 3 | Beryl, COVID-19 (all 3 MSAs) | -0.75 |

Every declaration landed exactly where its program count predicted. Two qualitative rounds failed where one quantitative rubric succeeded — worth remembering as a general lesson for this model, not just this prompt.

---

## Final design

- **Model**: gpt-4o-mini, temperature=0.1. Confirmed intentional (existing API access / budget), not a capability judgment — no comparison against Claude was run.
- **Sign convention**: impact_magnitude measures effect on **for-sale inventory count specifically**, via one of two explicit causal mechanisms the model must choose between (disrupted transactions → decrease; distressed sales/relocation → increase) before assigning a sign. Required `reasoning` field makes the choice auditable.
- **Magnitude size**: explicit numeric rubric keyed to `program_count` (FEMA assistance programs activated, 0-4), not open-ended severity language.
- **Confidence gate**: `confidence <= 0.60` (not `< 0.60`) forces `impact_magnitude` to 0.0; `confidence` and `raw_impact_magnitude` are both retained in the output for audit even when gated.
- **Schema**: `{"event_type", "impact_magnitude", "confidence", "reasoning", "raw_impact_magnitude", "gated"}`, validated with one retry on malformed/out-of-ontology output.

## Known limitation, deliberately not chased further

The model resolves to **one canonical value per program-count bucket** (-0.50 for every 2-program case, -0.75 for every 3-program case) rather than genuine continuous variation within a bucket, despite the prompt asking for "a specific value within the range... not reused from a different declaration." Net effect: this is really a **4-tier severity classifier wearing a continuous-scale interface**, not a true continuous scale. Likely reflects the real resolution the input data (declaration type + 4 boolean program flags) can support, rather than a fixable prompt issue — two qualitative attempts to get finer differentiation already failed on a *coarser* problem (Beryl vs. Ian/Nicole) before the rubric fixed that specific case; there's no evidence a within-bucket refinement would fare better.

**Guidance for downstream fusion**: treat cross-tier differences (e.g. -0.50 vs. -0.75) as real signal; treat within-tier differences as unlikely to exist at all given current behavior — don't weight small magnitude deltas between same-tier events as meaningful.

---

## Step 3 — full batch run

`run_event_agent.py` runs `interpret_event` once per unique `(msa, declaration_id)` pair (not per `(msa, week)` — see the Step 1 caching note above) and broadcasts each result back to every week that declaration was active. Prints a cost estimate (gpt-4o-mini published pricing: $0.15/1M input, $0.60/1M output tokens) before calling, and gates on explicit confirmation above 50 calls (bypassable with `--yes` for pre-authorized runs — used here after explicit chat authorization).

**Result**: 153 unique (msa, declaration_id) calls (higher than 130 unique declarations, since disasters spanning multiple MSAs — e.g. `DR-4482-CA` affecting both Riverside and San Francisco — get their own interpretation per MSA). Actual cost: $0.031.

| Metric | Value |
|---|---|
| `event_type` | 152 disaster, 1 zoning_change |
| Gated (confidence ≤ 0.60) | 57/153 (37%) |
| impact_magnitude, non-gated | -0.75 × 35, -0.50 × 61 |

### Post-hoc sanity check (before marking Step 3 done)

**The one zoning_change classification**: `EM-3553-DC`, "59TH PRESIDENTIAL INAUGURATION," FEMA's own `incidentType` field is literally `"Other"` — a genuine edge case, not a misclassification. This isn't a disaster at all, it's a security/logistics declaration for a planned event, and none of the 4 ontology categories (disaster / zoning_change / major_employer_shift / housing_initiative) fit it well. The model's reasoning ("temporary changes in local zoning or regulations... unlikely to significantly disrupt housing inventory") is coherent given the ontology gap, not a hallucination. It landed at confidence=0.60 → gated → impact_magnitude=0, so it has zero downstream effect regardless of the type label. **Logged as a known ontology gap** (no "special event/other" bucket) — not fixed, since it's one record out of 153 and self-neutralizes via the gate.

**Gate behavior at scale — confirmed non-arbitrary, not just "inert vs. now firing"**:

| | Gated (57) | Non-gated (96) |
|---|---|---|
| declaration_type | 55 EM, 1 FM, 1 DR | 70 DR, 25 FM, 1 EM |
| program_count | 56 have 1, 1 has 2 | 61 have 2, 35 have 3 |

Near-perfect separation: 56/57 gated rows have `program_count=1` and 55/57 are EM-type, while the 96 non-gated rows sit cleanly at `program_count` 2 or 3 — exactly matching the validated -0.50/-0.75 tiers. Gate rate jumped from ~0% on the 9-declaration test sample to 37% at full scale, and the jump tracks real severity signals (single-program EM filings — winter storms, tropical storms, single-state COVID filings) rather than looking arbitrary.

**Step 3 marked complete** on the strength of this sanity check.

---

## Step 4 — mocked-LLM test suite

`test_event_agent.py`: 13 tests, all mocked via a `MagicMock` standing in for the OpenAI client (passed through `interpret_event`'s explicit `client` parameter) — no real API calls, no `OPENAI_API_KEY` needed, runs in ~1.3s. Covers:

- Happy path (full schema, correct keys).
- Confidence gate, including the exact boundary decided in Step 2e: confidence=0.55 gates (and preserves the original confidence + raw magnitude for audit); confidence=0.60 gates (`<=`, not `<`); confidence=0.61 does not.
- Ontology enforcement: an out-of-ontology `event_type` is rejected and retried (verified via `call_count == 2`, not just an eventual exception), with a companion test confirming recovery works if the retry returns a valid type.
- Schema validation: missing field, wrong type, both bounds checks, missing `reasoning` — each raises a specific, matchable error rather than silently corrupting output.
- Rubric pass-through for `program_count` 2 and 3, using the exact empirically-validated values from Step 3 (-0.50, -0.75).

**Honest scope note, stated in the test file's docstring too**: the rubric tests verify `interpret_event`'s code doesn't corrupt an already-correct magnitude on the way through (no accidental rounding/clipping/mutation) — they cannot and do not verify that the LLM actually *produces* rubric-compliant values from a given `program_count`. That verification already happened for real, against real declarations, in Step 2h and Step 3.

**Event Agent build is now complete** — ready to hand off to the fusion harness alongside the AR Agent's real (non-stub) output.

## Update — magnitude adjustment removed (facts_mas/agents/event_agent.py)

IMPACT_SCALE removed per Umang's calibration sweep (`facts_mas/calibration.py`, `calibrate_event_impact_scale.py`) — 0.0 beat every positive value tested, 6/6 folds, no interior optimum. Event Agent no longer adjusts forecast magnitude; retains event_type classification, confidence gating, and the 8-week decay window. Magnitude signal treated as an unreliable ordinal indicator, not a usable point-forecast modifier (see `calibration.md` and `factor_attribution.md` for full history).

Concretely: `run_event_agent`'s `values` is now `last_inventory` held flat across the horizon, unconditionally — the same value regardless of whether a declaration is active, what its `impact_magnitude`/`event_type`/`program_count` are, or how old it is. Those signals still gate `confidence` (0.0 when no declaration is active, or when one is active but past the 8-week decay window, or — upstream, in `interpret_event`'s own gate, unaffected by this change — below the 0.60 confidence bar); they just no longer move the forecast's magnitude. Confirmed via 3 real active-declaration spot-checks (Phoenix 2021-06-12, Miami 2022-11-12, Chicago 2020-01-25) that `values` is flat at `last_inventory` and `confidence` still reflects the correct gated value in each case.

This is the minimal, literal application of the calibration finding — not the confidence-weight-boost redesign considered and abandoned earlier, and not a change to what "AR's own forecast" would look like (the wrapper still anchors on `last_inventory`, same as before; it just no longer perturbs it).

## Open items

1. **The Ian/Nicole-style data ceiling recurs at scale, as expected**: any two declarations sharing declaration_type + program_count tie on magnitude, by design (correct behavior, not a bug) — this is why the full batch collapses to essentially 2 non-gated values (-0.50, -0.75) plus 0 (gated). Not a red flag; a direct consequence of the documented 4-tier-classifier behavior.
2. **Ontology gap**: no category fits "planned special event" declarations (e.g. presidential inaugurations, large public gatherings) that FEMA files EM declarations for. One instance found in this batch, self-neutralized by gating; worth a 5th ontology category if more turn up.
3. If finer severity resolution is ever needed, the concrete next step (not started, out of scope for this round) is pulling OpenFEMA's Individual/Public Assistance funded-project datasets for dollar-figure severity signal — richer than declaration-summary metadata can support.
