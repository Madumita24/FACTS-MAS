"""Tests for event_agent.interpret_event -- entirely mocked, no real LLM calls.

Mocking strategy: interpret_event(declaration_text, declaration_metadata, client)
takes an explicit client parameter, so a MagicMock standing in for the OpenAI
client bypasses _client() (which reads OPENAI_API_KEY) entirely -- these tests
need no API key, run fast, and cost nothing.

Note on the rubric-consistency tests specifically: they verify interpret_event's
own code doesn't corrupt an already-rubric-compliant magnitude on the way
through (rounding, clipping, accidental mutation). They do NOT and cannot verify
that the LLM actually follows the numeric rubric -- that requires real API calls
and was already validated against real declarations in Step 2h/Step 3.
"""
import json
from unittest.mock import MagicMock

import pytest

from event_agent import CONFIDENCE_GATE, EVENT_TYPES, interpret_event


def _fake_response(payload: dict):
    message = MagicMock()
    message.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def make_mock_client(*payloads):
    client = MagicMock()
    client.chat.completions.create.side_effect = [_fake_response(p) for p in payloads]
    return client


VALID_PAYLOAD = {"event_type": "disaster", "impact_magnitude": -0.5, "confidence": 0.80, "reasoning": "test"}


# --- Happy path -------------------------------------------------------------

def test_valid_response_passes_through_with_full_schema():
    client = make_mock_client(VALID_PAYLOAD)
    result = interpret_event("some declaration", {}, client=client)
    assert result.keys() == {"event_type", "impact_magnitude", "confidence", "reasoning",
                              "raw_impact_magnitude", "gated"}
    assert result["event_type"] == "disaster"
    assert result["gated"] is False
    assert client.chat.completions.create.call_count == 1


# --- Confidence gate ----------------------------------------------------------

def test_confidence_below_gate_forces_impact_to_zero_but_preserves_confidence():
    payload = {"event_type": "disaster", "impact_magnitude": -0.6, "confidence": 0.55, "reasoning": "minor event"}
    client = make_mock_client(payload)
    result = interpret_event("x", {}, client=client)
    assert result["gated"] is True
    assert result["impact_magnitude"] == 0.0
    assert result["confidence"] == 0.55  # preserved for audit, not overwritten
    assert result["raw_impact_magnitude"] == -0.6  # pre-gating value also preserved


def test_confidence_exactly_at_gate_boundary_is_gated():
    # Per the Step 2e decision: <= 0.60, not < 0.60.
    payload = {"event_type": "disaster", "impact_magnitude": -0.5, "confidence": 0.60, "reasoning": "borderline"}
    client = make_mock_client(payload)
    result = interpret_event("x", {}, client=client)
    assert result["confidence"] == CONFIDENCE_GATE
    assert result["gated"] is True
    assert result["impact_magnitude"] == 0.0


def test_confidence_just_above_gate_boundary_is_not_gated():
    payload = {"event_type": "disaster", "impact_magnitude": -0.5, "confidence": 0.61, "reasoning": "above bar"}
    client = make_mock_client(payload)
    result = interpret_event("x", {}, client=client)
    assert result["gated"] is False
    assert result["impact_magnitude"] == -0.5


# --- Ontology enforcement -----------------------------------------------------

def test_invalid_event_type_is_rejected_after_retry_not_passed_through():
    bad_payload = {"event_type": "flood_warning", "impact_magnitude": -0.5, "confidence": 0.9, "reasoning": "bad"}
    client = make_mock_client(bad_payload, bad_payload)  # both retry attempts return the invalid type
    with pytest.raises(ValueError, match="not in approved ontology"):
        interpret_event("x", {}, client=client)
    assert client.chat.completions.create.call_count == 2  # confirms it retried, didn't accept on the first try
    assert "flood_warning" not in EVENT_TYPES  # sanity: this really is outside the ontology


def test_invalid_event_type_recovers_if_retry_returns_valid_ontology():
    bad_payload = {"event_type": "flood_warning", "impact_magnitude": -0.5, "confidence": 0.9, "reasoning": "bad"}
    client = make_mock_client(bad_payload, VALID_PAYLOAD)
    result = interpret_event("x", {}, client=client)
    assert result["event_type"] == "disaster"
    assert client.chat.completions.create.call_count == 2


# --- Schema validation ---------------------------------------------------------

def test_missing_required_field_raises_clear_error():
    bad_payload = {"event_type": "disaster", "confidence": 0.9, "reasoning": "test"}  # missing impact_magnitude
    client = make_mock_client(bad_payload, bad_payload)
    with pytest.raises(ValueError, match="impact_magnitude"):
        interpret_event("x", {}, client=client)


def test_wrong_type_for_impact_magnitude_raises_clear_error():
    bad_payload = {"event_type": "disaster", "impact_magnitude": "very bad", "confidence": 0.9, "reasoning": "test"}
    client = make_mock_client(bad_payload, bad_payload)
    with pytest.raises(ValueError, match="impact_magnitude"):
        interpret_event("x", {}, client=client)


def test_impact_magnitude_out_of_bounds_raises():
    bad_payload = {"event_type": "disaster", "impact_magnitude": 1.5, "confidence": 0.9, "reasoning": "test"}
    client = make_mock_client(bad_payload, bad_payload)
    with pytest.raises(ValueError, match="out of bounds"):
        interpret_event("x", {}, client=client)


def test_confidence_out_of_bounds_raises():
    bad_payload = {"event_type": "disaster", "impact_magnitude": -0.5, "confidence": 1.5, "reasoning": "test"}
    client = make_mock_client(bad_payload, bad_payload)
    with pytest.raises(ValueError, match="out of bounds"):
        interpret_event("x", {}, client=client)


def test_missing_reasoning_raises():
    bad_payload = {"event_type": "disaster", "impact_magnitude": -0.5, "confidence": 0.9}
    client = make_mock_client(bad_payload, bad_payload)
    with pytest.raises(ValueError, match="reasoning"):
        interpret_event("x", {}, client=client)


# --- Rubric consistency (pass-through only -- see module docstring) -----------

def test_program_count_2_magnitude_passes_through_correctly():
    # -0.50 is the empirically validated value for program_count=2 from Step 3's
    # full batch run (61/96 non-gated declarations landed exactly here).
    payload = {"event_type": "disaster", "impact_magnitude": -0.50, "confidence": 0.80, "reasoning": "2 programs"}
    client = make_mock_client(payload)
    result = interpret_event("x", {"program_count": 2}, client=client)
    assert result["impact_magnitude"] == -0.50
    assert result["gated"] is False


def test_program_count_3_magnitude_passes_through_correctly():
    # -0.75 is the empirically validated value for program_count=3 from Step 3's
    # full batch run (35/96 non-gated declarations landed exactly here).
    payload = {"event_type": "disaster", "impact_magnitude": -0.75, "confidence": 0.90, "reasoning": "3 programs"}
    client = make_mock_client(payload)
    result = interpret_event("x", {"program_count": 3}, client=client)
    assert result["impact_magnitude"] == -0.75
    assert result["gated"] is False
