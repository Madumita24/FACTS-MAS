"""Event Agent -- the first LLM-based component in this pipeline. Everything
upstream (AR, Macro) is pure statistics; this interprets non-numerical
evidence (FEMA declaration text) into a structured signal.

Boundary, strictly enforced: the LLM NEVER outputs a forecast number. It
classifies event_type and a bounded, qualitative impact_magnitude
(-1.0..+1.0) -- a directional pressure signal, not an inventory estimate.
Turning that signal into an actual forecast adjustment is Umang's
Synthesizer's job (weighted fusion across all agents), not this agent's.

Provider note: the original spec called for Claude at temperature 0.1.
Only an OpenAI key was available, so this uses gpt-4o-mini instead (cheap,
well-suited to a bounded structured-classification task like this) at the
same temperature=0.1 for reproducibility across backtest folds. Swap
MODEL/the client if an Anthropic key becomes available later.
"""
import json
import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

EVENT_TYPES = ["disaster", "zoning_change", "major_employer_shift", "housing_initiative"]
CONFIDENCE_GATE = 0.60
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.1

SYSTEM_PROMPT = """You are an event classifier for a housing-inventory forecasting pipeline.
You will be given the text and metadata of a single real-world declaration/event record,
including FEMA severity indicators (declaration type and which assistance programs were
actually activated).

Rules, followed strictly:
- Classify ONLY the event described in the given text. Never invent details, never infer
  additional events beyond what's stated, never speculate about effects not evidenced by the text.
- event_type must be exactly one of: disaster, zoning_change, major_employer_shift, housing_initiative.
- confidence is 0.0-1.0: how confident you are in event_type and impact_magnitude given the text alone.

impact_magnitude -- WHAT IT MEASURES (read carefully, this is specific):
impact_magnitude represents the effect on FOR-SALE INVENTORY COUNT specifically -- the number of
homes listed for sale in the local market -- NOT general economic damage, distress, or severity.
A severe event is not automatically "negative"; the direction depends on WHICH causal mechanism
actually applies to for-sale inventory:
  (1) DECREASE inventory (negative magnitude) if the event disrupts the local market's ability to
      list or close on homes: damaged infrastructure blocking access/inspections, evacuation
      orders, halted transactions, sellers pulling listings during the crisis, agents/title
      offices unable to operate.
  (2) INCREASE inventory (positive magnitude) if the event drives MORE homes onto the market:
      distressed/forced sales, owners relocating away permanently, damaged homes being listed
      as-is, insurance payouts enabling sales, post-event population outflow.
Before assigning a magnitude, decide which mechanism (1) or (2) you believe actually applies here,
based on the event type and what's known about it -- then set the sign to match. Do not default to
one direction out of habit; a hurricane could plausibly go either way depending on whether it mainly
halted transactions (mechanism 1) or forced distressed sales/relocations (mechanism 2).
Scale: -1.0 = strong DECREASE in inventory, +1.0 = strong INCREASE in inventory, 0.0 = negligible/
unclear effect on inventory either way. This is NOT a forecast and NOT a precise unit/percent
estimate -- it's a coarse directional signal for a downstream statistical model to weigh.

Severity calibration (use the FEMA fields provided, don't guess beyond them) -- severity affects
MAGNITUDE SIZE (how strongly mechanism 1 or 2 applies), not which mechanism you pick:
- You will be given a "program count": the number of FEMA assistance programs actually activated,
  out of 4 possible (Individual Housing, Individual Assistance, Public Assistance, Hazard
  Mitigation). Use this EXACT numeric rubric for the SIZE of impact_magnitude (|impact_magnitude|),
  regardless of which mechanism/sign applies:
    program_count = 0  -> |impact_magnitude| in [0.0, 0.2]
    program_count = 1  -> |impact_magnitude| in [0.2, 0.4]
    program_count = 2  -> |impact_magnitude| in [0.4, 0.6]
    program_count = 3 or 4 -> |impact_magnitude| in [0.6, 1.0]
  Pick a SPECIFIC value within the applicable range, not just the boundary. Then apply the sign
  from whichever mechanism (1: decrease, negative / 2: increase, positive) you determined applies.
  Example: program_count=3, mechanism 1 (decrease) -> a value like -0.75, NOT -0.60 or -0.70 by
  default and NOT reused from a different declaration's value.
  Declaration type (DR vs EM) is a secondary signal: within the range implied by program_count,
  lean toward the higher end of that range for DR and the lower end for EM.
- Two declarations with the SAME program_count (and same declaration type) have no basis in the
  given data to be assigned different magnitudes -- giving them the same magnitude in that case is
  CORRECT, not a failure to differentiate. Two declarations with DIFFERENT program_count MUST land
  in different ranges per the rubric above -- do not default to a single familiar-feeling number
  regardless of program_count.
- Confidence anchor: reserve confidence above 0.85 ONLY for declarations with clear severity
  indicators (DR type AND program count >= 2). Use confidence below 0.5 for declarations with
  sparse metadata, program count of 0, or ambiguous/minor-sounding incident types/titles.
- Default to LOW confidence (below 0.5) when the text is ambiguous, minor/routine, or doesn't give
  enough detail (beyond the FEMA fields provided) to judge real-world housing impact.

reasoning: a short 1-2 sentence explanation of WHICH mechanism (1 or 2) you chose and why, given
the specific event. This must be consistent with your sign: if reasoning describes disrupted
transactions (mechanism 1), impact_magnitude must be negative; if it describes distressed sales or
relocations (mechanism 2), impact_magnitude must be positive.

Respond with ONLY a JSON object:
{"event_type": "...", "impact_magnitude": <float>, "confidence": <float>, "reasoning": "..."}
No other text, no markdown formatting.
"""


def _client() -> OpenAI:
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env file")
    return OpenAI(api_key=key)


def _call_llm(client: OpenAI, declaration_text: str, declaration_metadata: dict) -> dict:
    user_prompt = (
        f"Declaration text: {declaration_text}\n\n"
        f"Metadata: {json.dumps(declaration_metadata, default=str)}"
    )
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def _validate(parsed: dict) -> dict:
    if not isinstance(parsed, dict) or "event_type" not in parsed or parsed["event_type"] not in EVENT_TYPES:
        raise ValueError(f"event_type {parsed.get('event_type') if isinstance(parsed, dict) else parsed!r} "
                          f"not in approved ontology {EVENT_TYPES}")
    if "impact_magnitude" not in parsed or not isinstance(parsed["impact_magnitude"], (int, float)):
        raise ValueError(f"impact_magnitude missing or not numeric: {parsed.get('impact_magnitude')}")
    if not (-1.0 <= parsed["impact_magnitude"] <= 1.0):
        raise ValueError(f"impact_magnitude {parsed['impact_magnitude']} out of bounds [-1, 1]")
    if "confidence" not in parsed or not isinstance(parsed["confidence"], (int, float)):
        raise ValueError(f"confidence missing or not numeric: {parsed.get('confidence')}")
    if not (0.0 <= parsed["confidence"] <= 1.0):
        raise ValueError(f"confidence {parsed['confidence']} out of bounds [0, 1]")
    if "reasoning" not in parsed or not isinstance(parsed["reasoning"], str) or not parsed["reasoning"].strip():
        raise ValueError(f"reasoning missing or not a non-empty string: {parsed.get('reasoning')!r}")
    return parsed


def interpret_event(declaration_text: str, declaration_metadata: dict, client: OpenAI = None) -> dict:
    """Returns {"event_type", "impact_magnitude", "confidence", "reasoning", "raw_impact_magnitude", "gated"}.

    impact_magnitude is confidence-gated: if confidence <= 0.60, impact_magnitude is forced to 0
    (per EventRecord design) while confidence itself is kept unchanged for audit. raw_impact_magnitude
    preserves the pre-gating LLM output for the same audit purpose.

    Gate boundary decision: <= 0.60, not < 0.60. A confidence score of exactly the threshold reads
    as "at or below the minimum bar", not "just above it" -- documented explicitly here because the
    off-by-one direction is easy to get backwards silently. (Confirmed bug in an earlier version:
    a confidence=0.60 declaration passed through ungated under a strict `<` check.)
    """
    client = client or _client()

    last_error = None
    validated = None
    for _ in range(2):  # one retry if the LLM returns something outside the ontology/schema
        try:
            raw = _call_llm(client, declaration_text, declaration_metadata)
            validated = _validate(raw)
            break
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
    if validated is None:
        raise ValueError(f"interpret_event failed schema validation after 2 attempts: {last_error}")

    raw_impact = float(validated["impact_magnitude"])
    confidence = float(validated["confidence"])
    gated = confidence <= CONFIDENCE_GATE

    return {
        "event_type": validated["event_type"],
        "impact_magnitude": 0.0 if gated else raw_impact,
        "confidence": confidence,
        "reasoning": validated["reasoning"],
        "raw_impact_magnitude": raw_impact,
        "gated": gated,
    }


# Results from Run 4 (Step 2g: open-ended program-count instruction, no numeric rubric) --
# sign-consistent, Ian/Nicole correctly tied, but Beryl (3 programs) still stuck at the same
# -0.70 as Ian/Nicole (2 programs) -- the open-ended instruction wasn't enough to move the number.
# Hardcoded as the "old" baseline for the Step 2h comparison below (explicit numeric rubric).
RUN4_RESULTS = {
    ("Miami", "Hurricane Nicole"): {"impact_magnitude": -0.70, "confidence": 0.80},
    ("Miami", "Hurricane Ian"): {"impact_magnitude": -0.70, "confidence": 0.80},
    ("Miami", "COVID-19"): {"impact_magnitude": -0.70, "confidence": 0.85},
    ("Houston", "Pauline Road Fire"): {"impact_magnitude": 0.00, "confidence": 0.40},
    ("Houston", "Hurricane Beryl"): {"impact_magnitude": -0.70, "confidence": 0.90},
    ("Houston", "COVID-19"): {"impact_magnitude": -0.70, "confidence": 0.85},
    ("Chicago", "Severe Storms/Tornadoes/Flooding"): {"impact_magnitude": -0.70, "confidence": 0.85},
    ("Chicago", "Severe Storms/Flooding"): {"impact_magnitude": -0.70, "confidence": 0.80},
    ("Chicago", "COVID-19"): {"impact_magnitude": -0.70, "confidence": 0.85},
}

# Keyword heuristic ONLY, for a quick automated flag -- not a substitute for actually reading
# the reasoning field, which is printed in full for manual review below.
_DECREASE_KEYWORDS = ["decrease", "disrupt", "halt", "reduc", "pause", "block", "delay"]
_INCREASE_KEYWORDS = ["increase", "distress", "relocat", "forced sale", "displac", "sell off"]


def _reasoning_sign_consistent(reasoning: str, impact_magnitude: float) -> bool:
    text = reasoning.lower()
    decrease_hit = any(kw in text for kw in _DECREASE_KEYWORDS)
    increase_hit = any(kw in text for kw in _INCREASE_KEYWORDS)
    if impact_magnitude < 0 and decrease_hit and not increase_hit:
        return True
    if impact_magnitude > 0 and increase_hit and not decrease_hit:
        return True
    if impact_magnitude == 0:
        return True  # negligible effect, no mechanism claim to check
    return not (decrease_hit or increase_hit)  # no clear keyword either way -- can't call it inconsistent


def _short_label(row) -> str:
    title = row.declaration_title.title()
    if "Covid" in title or "Coronavirus" in title:
        return "COVID-19"
    if "Nicole" in title:
        return "Hurricane Nicole"
    if "Ian" in title:
        return "Hurricane Ian"
    if "Beryl" in title:
        return "Hurricane Beryl"
    if "Pauline" in title:
        return "Pauline Road Fire"
    if "Tornado" in title:
        return "Severe Storms/Tornadoes/Flooding"
    return "Severe Storms/Flooding"


def _severity_label(row):
    dtype = row.declaration_type
    dtype_note = "Major Disaster -- generally more severe" if dtype == "DR" else \
                 "Emergency -- generally less severe than DR" if dtype == "EM" else "unknown"
    programs_active = [name for name, flag in [
        ("Individual Housing", row.ih_program_declared),
        ("Individual Assistance", row.ia_program_declared),
        ("Public Assistance", row.pa_program_declared),
        ("Hazard Mitigation", row.hm_program_declared),
    ] if flag]
    program_count = len(programs_active)
    programs_note = ", ".join(programs_active) if programs_active else "NONE activated"
    return dtype_note, programs_note, program_count


def build_declaration_input(row) -> tuple:
    """Builds (declaration_text, metadata) for a single declaration row from
    fema_events_weekly.csv, in the exact format validated through Steps 2c-2h.
    Shared by the sample harness below and run_event_agent.py's full batch,
    so the batch never drifts from what was actually validated.
    """
    dtype_note, programs_note, program_count = _severity_label(row)
    declaration_text = (
        f"{row.declaration_title} ({row.incident_type}) -- {row.designated_area}. "
        f"Declaration type: {row.declaration_type} ({dtype_note}). "
        f"Assistance programs activated ({program_count} of 4): {programs_note}."
    )
    metadata = {
        "declaration_id": row.declaration_id,
        "declaration_type": row.declaration_type,
        "ih_program_declared": bool(row.ih_program_declared),
        "ia_program_declared": bool(row.ia_program_declared),
        "pa_program_declared": bool(row.pa_program_declared),
        "hm_program_declared": bool(row.hm_program_declared),
        "program_count": program_count,
        "incident_begin_date": row.incident_begin_date,
        "incident_end_date": row.incident_end_date,
        "designated_area": row.designated_area,
    }
    return declaration_text, metadata


if __name__ == "__main__":
    df = pd.read_csv("fema_events_weekly.csv")
    dedup = df.drop_duplicates(subset=["msa", "declaration_id"])

    sample_msas = ["Miami", "Houston", "Chicago"]
    client = _client()

    print("STEP 2h -- re-test with explicit numeric rubric replacing open-ended program-count instruction.\n")

    after_results = {}
    for msa in sample_msas:
        msa_declarations = dedup[dedup["msa"] == msa]
        non_covid = msa_declarations[msa_declarations["incident_type"] != "Biological"].head(2)
        covid = msa_declarations[msa_declarations["incident_type"] == "Biological"].head(1)
        sample = pd.concat([non_covid, covid])

        for row in sample.itertuples():
            declaration_text, metadata = build_declaration_input(row)
            result = interpret_event(declaration_text, metadata, client=client)
            after_results[(msa, _short_label(row))] = result

    for key, before in RUN4_RESULTS.items():
        msa, label = key
        after = after_results.get(key)
        if after is None:
            print(f"{msa} / {label}: NOT FOUND IN NEW SAMPLE")
            continue
        consistent = _reasoning_sign_consistent(after["reasoning"], after["impact_magnitude"])
        print(f"\n{msa} / {label}")
        print(f"  impact_magnitude: {before['impact_magnitude']:+.2f} (Run 4) -> {after['impact_magnitude']:+.2f} (Run 5)")
        print(f"  confidence:       {before['confidence']:.2f} (Run 4) -> {after['confidence']:.2f} (Run 5)")
        print(f"  gated: {after['gated']}")
        print(f"  reasoning: {after['reasoning']}")
        print(f"  reasoning/sign consistent (keyword heuristic)? {'YES' if consistent else 'NO -- CHECK MANUALLY'}")

    print("\n" + "=" * 70)
    print("Specific checks:")

    all_signs = [r["impact_magnitude"] for r in after_results.values() if r["impact_magnitude"] != 0]
    all_negative = all(s < 0 for s in all_signs)
    all_positive = all(s > 0 for s in all_signs)
    sign_consistent = all_negative or all_positive
    print(f"  Sign CONSISTENT across all 9, no flips reappeared? {'YES' if sign_consistent else 'NO -- MIXED SIGNS REMAIN'}  "
          f"(signs: {[f'{s:+.2f}' for s in all_signs]})")

    ian = after_results.get(("Miami", "Hurricane Ian"))
    nicole = after_results.get(("Miami", "Hurricane Nicole"))
    if ian and nicole:
        tied = ian["impact_magnitude"] == nicole["impact_magnitude"]
        print(f"  Ian vs Nicole still tied? {'YES' if tied else 'NO'}  "
              f"(Ian={ian['impact_magnitude']:+.2f}, Nicole={nicole['impact_magnitude']:+.2f}) -- "
              f"{'EXPECTED: identical program count (2), genuine data ceiling, not a bug' if tied else 'unexpected -- investigate'}")

    fire = after_results.get(("Houston", "Pauline Road Fire"))
    beryl = after_results.get(("Houston", "Hurricane Beryl"))
    if ian and beryl:
        beryl_in_range = 0.6 <= abs(beryl["impact_magnitude"]) <= 1.0  # 3 programs
        ian_in_range = 0.4 <= abs(ian["impact_magnitude"]) <= 0.6      # 2 programs
        rubric_followed = beryl_in_range and ian_in_range and abs(beryl["impact_magnitude"]) > abs(ian["impact_magnitude"])
        print(f"  Beryl (3 programs) in [-1.0,-0.6] AND Ian/Nicole (2 programs) in [-0.6,-0.4]? "
              f"{'YES -- explicit rubric succeeded where open-ended instruction failed' if rubric_followed else 'NO -- rubric not followed either'}  "
              f"(Beryl={beryl['impact_magnitude']:+.2f} [in range: {beryl_in_range}], "
              f"Ian={ian['impact_magnitude']:+.2f} [in range: {ian_in_range}])")
    if fire and beryl:
        lower_conf = fire["confidence"] < beryl["confidence"]
        lower_impact = abs(fire["impact_magnitude"]) < abs(beryl["impact_magnitude"])
        print(f"  Pauline Road Fire lower confidence than Hurricane Beryl? "
              f"{'YES' if lower_conf else 'NO'}  (fire={fire['confidence']:.2f}, beryl={beryl['confidence']:.2f})")
        print(f"  Pauline Road Fire lower |impact| than Hurricane Beryl? "
              f"{'YES' if lower_impact else 'NO'}  (fire={fire['impact_magnitude']:+.2f}, beryl={beryl['impact_magnitude']:+.2f})")
        print(f"  Pauline Road Fire gates out (confidence<=0.60)? "
              f"{'YES' if fire['gated'] else 'NO'}  (confidence={fire['confidence']:.2f})")

    gated_count = sum(1 for r in after_results.values() if r["gated"])
    print(f"\n  Gate (confidence<=0.60) fires on: {gated_count}/9 declarations")
