"""Exact NEXUS-paper Chain-of-Thought baseline (Appendix C prompt template,
verbatim, not simplified). This is a comparison baseline, not one of our
production agents -- it exists to answer "how does our fused system compare
to the paper's own zero-shot LLM-forecasting approach," using the paper's
literal prompt, not our own house style.

Provider/model: gpt-4o-mini, temperature=0.1 -- same choice and same reason
as event_agent.py (existing OpenAI API access, not a capability judgment;
no comparison against Claude was run here either).

Point-in-time discipline: forecast_cot_nexus takes an already-truncated
historical_series (the caller is responsible for cutting it off at the
correct forecast origin -- this function does not know about folds or
train_end, it just uses the series' own last index entry as "now"). This
matches the no-lookahead discipline used by every other agent tonight,
just enforced by the caller instead of internally, since the exact NEXUS
prompt template has no notion of a fold/origin -- only a plain historical
window ending at some date.
"""
import json
import os
import re

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.1
HISTORY_WEEKS = 52

SYSTEM_PROMPT = (
    "You are a contextual numerical forecasting agent. Your task is to predict "
    "future values based on the provided historical context. You have to solely "
    "rely on your own deductive reasoning based on the provided context. Your "
    "knowledge cutoff date is January 2025."
)

USER_TEMPLATE = """**Target Metric:** {target_name}
**Target Location:** {domain}
**Forecast Time (Cut-off):** {last_date}
**Prediction Horizon:** Next {horizon} steps ({frequency} Forecast)

**A. Historical Records ({start_date} to {last_date})**
{history_values_str}

**B. Event Intelligence**
{history_text_str}
(Note: Consider how the events listed above interact with the typical seasonality of the {target_name}.)

**TASK:**
Predict the future {frequency} values for {target_name} for the next {horizon} steps based on the historical data and event intelligence above.
Since you are generating an {frequency} forecast, you MUST output EXACTLY {horizon} data points.

**STRICT CONSTRAINTS:**
1. **BRIEF ANALYSIS:** Provide a reasoning to explain your prediction.
2. **FORMAT:** Comma-separated values ONLY for the numerical array.
3. **WRAPPER:** Wrap the final sequence of forecasting numbers strictly inside `<prediction>` tags."""

CORRECTION_TEMPLATE = (
    "Your previous response did not satisfy the format requirements: {problem} "
    "You MUST wrap EXACTLY {horizon} comma-separated numeric values inside "
    "<prediction></prediction> tags -- no more, no fewer, no extra text inside the tags. "
    "Please provide a corrected response now."
)


def _client() -> OpenAI:
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env file")
    return OpenAI(api_key=key)


def _build_user_prompt(historical_series: pd.Series, msa: str, horizon_weeks: int, event_text: str) -> tuple[str, str, str]:
    window = historical_series.tail(HISTORY_WEEKS)
    history_lines = "\n".join(f"{idx.strftime('%Y-%m-%d')}: {val:.1f}" for idx, val in window.items())

    last_date = window.index[-1].strftime("%Y-%m-%d")
    start_date = window.index[0].strftime("%Y-%m-%d")

    prompt = USER_TEMPLATE.format(
        target_name="weekly for-sale housing inventory count",
        domain=msa,
        last_date=last_date,
        horizon=horizon_weeks,
        frequency="weekly",
        start_date=start_date,
        history_values_str=history_lines,
        history_text_str=event_text or "No significant events recorded.",
    )
    return prompt, start_date, last_date


def _parse_prediction(raw_text: str, horizon_weeks: int) -> list:
    match = re.search(r"<prediction>(.*?)</prediction>", raw_text, re.DOTALL)
    if not match:
        raise ValueError("no <prediction>...</prediction> tags found in response")

    inner = match.group(1).strip()
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    try:
        values = [float(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"non-numeric value inside <prediction> tags: {e}")

    if len(values) != horizon_weeks:
        raise ValueError(f"expected exactly {horizon_weeks} values, got {len(values)}")

    return values


def forecast_cot_nexus(
    historical_series: pd.Series,
    msa: str,
    horizon_weeks: int,
    event_text: str = None,
    client: OpenAI = None,
) -> dict:
    """
    Runs the exact NEXUS-paper Appendix C CoT prompt for one (msa, origin,
    horizon). historical_series must already be truncated by the caller to
    the correct point-in-time forecast origin (its last index entry becomes
    the prompt's "Forecast Time (Cut-off)").

    Retry discipline (same as event_agent.py's malformed-output handling):
    one retry if parsing fails or the value count doesn't match, but as an
    explicit CONVERSATIONAL CORRECTION (the model sees its own bad response
    and is told exactly what was wrong), not a blind fresh retry. If it
    fails twice, this raises -- it does NOT silently default to naive or
    any other fallback value.

    Returns {"model_name": "cot_nexus", "point_forecast": list[float],
    "reasoning": str (the model's response text outside the <prediction>
    tags), "raw_response": str, "attempts": int}.
    """
    client = client or _client()
    user_prompt, start_date, last_date = _build_user_prompt(historical_series, msa, horizon_weeks, event_text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_error = None
    for attempt in range(1, 3):
        resp = client.chat.completions.create(model=MODEL, temperature=TEMPERATURE, messages=messages)
        raw_text = resp.choices[0].message.content

        try:
            values = _parse_prediction(raw_text, horizon_weeks)
            reasoning = re.sub(r"<prediction>.*?</prediction>", "", raw_text, flags=re.DOTALL).strip()
            return {
                "model_name": "cot_nexus",
                "point_forecast": values,
                "reasoning": reasoning,
                "raw_response": raw_text,
                "attempts": attempt,
            }
        except ValueError as e:
            last_error = e
            if attempt == 1:
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "user", "content": CORRECTION_TEMPLATE.format(
                    problem=str(e) + ".", horizon=horizon_weeks,
                )})

    raise ValueError(
        f"forecast_cot_nexus failed to get a valid {horizon_weeks}-value prediction for "
        f"msa={msa} after 2 attempts: {last_error}"
    )
