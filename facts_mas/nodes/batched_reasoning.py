"""
NEXUS Modification D (first half) — Batched Micro-Reasoning.

Solves the remaining part of Limitation 2. The published micro-reasoning
agent forecasts one timestep at a time: a 26-week horizon costs 26
sequential language-model calls from that one agent alone. The paper's own
authors give this as the reason they could never run the full system more
than once, which in turn is why they could never report variance or
confidence intervals on any of their numbers.

The fix is unglamorous: ask for the whole horizon in a single call, get back
a structured array, and validate it. If the array comes back malformed, fall
back to the original one-call-per-step method -- as a backup, never as the
default.

    13-week horizon, sequential:  13 calls
    13-week horizon, batched:      1 call

The second half of Mod D, the tightened fold-consistency rule for
calibration, is already implemented in `facts_mas/calibration.py`.

Design notes:

    The prompt follows the mandatory 5-part structure from the Master Spec
    (role and goal, valid output vocabulary, constraints, process, worked
    example), the same shape `event_agent.py` uses.

    `step_reasoning` is returned but is quarantined by contract: it belongs
    in the graph state's internal traces key and must never reach the
    Synthesizer. Only `forecast_values` crosses the layer boundary. That is
    the prompt-leakage rule, and it is why this function returns the model
    rather than a bare list -- the type carries the quarantine with it.

    The client is injected, so the whole module is testable without an API
    key or a live call, exactly like `test_event_agent.py` does for the
    Event Agent.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

import numpy as np

from facts_mas.state.micro_reasoning import BatchedReasoningOutput

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.1

SYSTEM_PROMPT = """You are a housing-inventory micro-reasoning agent in a forecasting pipeline.

ROLE AND GOAL
Given a city's recent weekly for-sale inventory history and its context, produce a
forecast for EVERY week of the requested horizon in a single response, together with
one short reason per week.

VALID OUTPUT
A single JSON object, nothing else:
{"forecasts": [<float>, ...], "reasoning": ["<short string>", ...]}
Both arrays must have exactly HORIZON entries, in chronological order.

CONSTRAINTS
- Inventory counts are non-negative. Never return a negative number.
- Do not return percentages, changes, or deltas. Return absolute inventory counts.
- Do not invent events, policies or news that were not given to you.
- Week-over-week moves in this data are typically under 10 percent. A forecast that
  doubles or halves within one week is almost certainly wrong.
- No markdown, no commentary, no text outside the JSON object.

PROCESS
1. Read the recent trend and its direction.
2. Read any context provided (season, regime, active events).
3. Project week by week, carrying the level forward and applying the trend.
4. Write one short reason per week explaining what drove that week's value.

EXAMPLE
Input: recent weekly inventory [10000, 10200, 10400], horizon 3, context "spring, rising"
Output: {"forecasts": [10600.0, 10800.0, 11000.0], "reasoning": ["spring build continues", "steady listing pace", "trend holds into late spring"]}
"""


# ═══════════════════════════════════════════════════════════════════════════
# Client
# ═══════════════════════════════════════════════════════════════════════════

def _client():
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)


def _build_user_prompt(history: list[float], horizon: int, context: str) -> str:
    recent = [round(float(v), 1) for v in history[-12:]]
    return (
        f"Recent weekly inventory (oldest to newest): {recent}\n"
        f"HORIZON: {horizon} weeks\n"
        f"Context: {context or 'none provided'}\n"
        f"Return exactly {horizon} forecasts and {horizon} reasons."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Parsing and validation
# ═══════════════════════════════════════════════════════════════════════════

def parse_batch_response(raw: str, horizon: int) -> BatchedReasoningOutput:
    """
    Turn a raw model response into a validated output.

    Never raises on bad input. A malformed batch is a normal, expected
    outcome that the architecture already has an answer for, so it is
    reported as `is_valid_batch=False` and the caller falls back to
    sequential mode. Raising here would turn a handled case into an outage.

    `forecast_values` is set to an empty list on failure rather than to
    zeros or the horizon length, because a zero-filled forecast is a
    plausible-looking wrong answer and an empty one cannot be mistaken for
    a real result.
    """
    def invalid() -> BatchedReasoningOutput:
        return BatchedReasoningOutput(
            forecast_values=[], step_reasoning=[], horizon_weeks=horizon,
            is_valid_batch=False, model_name=MODEL,
        )

    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text)
    except Exception:
        return invalid()

    if not isinstance(parsed, dict):
        return invalid()

    values = parsed.get("forecasts")
    reasons = parsed.get("reasoning", [])
    if not isinstance(values, list) or len(values) != horizon:
        return invalid()

    try:
        nums = [float(v) for v in values]
    except (TypeError, ValueError):
        return invalid()
    if any(np.isnan(v) or np.isinf(v) or v < 0 for v in nums):
        return invalid()

    if not isinstance(reasons, list):
        reasons = []
    reasons = [str(r) for r in reasons][:horizon]

    return BatchedReasoningOutput(
        forecast_values=nums, step_reasoning=reasons, horizon_weeks=horizon,
        is_valid_batch=True, model_name=MODEL,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Batched and sequential paths
# ═══════════════════════════════════════════════════════════════════════════

def run_batched_reasoning(
    history: list[float],
    horizon_weeks: int,
    context: str = "",
    client=None,
    sequential_fallback: bool = True,
) -> tuple[BatchedReasoningOutput, int]:
    """
    One call for the whole horizon, with sequential fallback.

    Returns (output, n_llm_calls). The call count is returned rather than
    logged because it is the measurement the whole modification exists to
    improve -- Phase 4's cost analysis needs the real number, not an
    assumed one.
    """
    client = client or _client()
    calls = 0

    try:
        resp = client.chat.completions.create(
            model=MODEL, temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(history, horizon_weeks, context)},
            ],
        )
        calls += 1
        out = parse_batch_response(resp.choices[0].message.content, horizon_weeks)
    except Exception:
        calls += 1
        out = BatchedReasoningOutput(
            forecast_values=[], step_reasoning=[], horizon_weeks=horizon_weeks,
            is_valid_batch=False, model_name=MODEL,
        )

    if out.is_valid_batch or not sequential_fallback:
        return out, calls

    seq, seq_calls = run_sequential_reasoning(history, horizon_weeks, context, client)
    return seq, calls + seq_calls


def run_sequential_reasoning(
    history: list[float],
    horizon_weeks: int,
    context: str = "",
    client=None,
) -> tuple[BatchedReasoningOutput, int]:
    """
    The original one-call-per-timestep method, kept only as the fallback.

    Each step appends its own forecast to the running history, which is what
    makes this sequential in the first place and why it cannot be
    parallelised away. Included so the comparison in the paper is against a
    real implementation of the published approach rather than an estimate of
    one.
    """
    client = client or _client()
    running = list(history)
    values, reasons = [], []
    calls = 0

    for step in range(horizon_weeks):
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(
                        running, 1, f"{context} (step {step + 1} of {horizon_weeks})")},
                ],
            )
            calls += 1
            one = parse_batch_response(resp.choices[0].message.content, 1)
            if one.is_valid_batch:
                v = one.forecast_values[0]
                reasons.append(one.step_reasoning[0] if one.step_reasoning else "")
            else:
                v = running[-1]
                reasons.append("fallback: malformed step response")
        except Exception:
            calls += 1
            v = running[-1]
            reasons.append("fallback: call failed")

        values.append(float(v))
        running.append(float(v))

    return BatchedReasoningOutput(
        forecast_values=values, step_reasoning=reasons,
        horizon_weeks=horizon_weeks, is_valid_batch=True, model_name=MODEL,
    ), calls


# ═══════════════════════════════════════════════════════════════════════════
# Measuring the saving
# ═══════════════════════════════════════════════════════════════════════════

def batching_savings(horizon_weeks: int, n_forecasts: int,
                     batch_failure_rate: float = 0.0) -> dict:
    """
    Calls required with and without batching.

    `batch_failure_rate` is charged honestly: a failed batch costs its own
    call *and* the full sequential run afterwards, so it is more expensive
    than never batching at all. Quoting the saving without that term would
    overstate the benefit.
    """
    sequential = n_forecasts * horizon_weeks
    clean = n_forecasts * (1.0 - batch_failure_rate)
    failed = n_forecasts * batch_failure_rate
    batched = clean + failed * (1 + horizon_weeks)
    return {
        "horizon_weeks": horizon_weeks,
        "n_forecasts": n_forecasts,
        "sequential_calls": int(sequential),
        "batched_calls": int(round(batched)),
        "calls_saved": int(round(sequential - batched)),
        "reduction": 1.0 - batched / sequential if sequential else 0.0,
        "assumed_batch_failure_rate": batch_failure_rate,
    }
