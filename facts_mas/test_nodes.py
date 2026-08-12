"""
Test Suite — NEXUS Modifications A/B/C/D (facts_mas/nodes/).

Run with:  python facts_mas/test_nodes.py

No live LLM calls and no API key required. Mod D is tested against a mock
client, the same approach test_event_agent.py uses, so the batch/fallback
logic is exercised deterministically and for free.
"""

import sys
import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from facts_mas.nodes.batched_reasoning import (
    batching_savings,
    parse_batch_response,
    run_batched_reasoning,
)
from facts_mas.nodes.boundary_node import (
    boundary_report,
    clamp_corrector,
    enforce_boundaries,
)
from facts_mas.nodes.gate_node import (
    default_baseline,
    evaluate_gate,
    gate_savings_report,
)
from facts_mas.nodes.spatial_node import (
    apply_spillover,
    build_spatial_graph,
    compute_spillover_signal,
)
from facts_mas.state.boundary import BoundaryConfig
from facts_mas.state.gate import GateConfig
from facts_mas.state.ontology import GateTriggerType
from facts_mas.state.spatial import SpilloverSignal

WEEKLY_DF = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
AS_OF = datetime.date(2025, 2, 1)

passed = 0
failed = 0


def check(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        failed += 1


# ═══════════════════════════════════════════════════════════════════════════
# Mod B — Spatial Spillover
# ═══════════════════════════════════════════════════════════════════════════

_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_spatial_graph(
            WEEKLY_DF, ["Los Angeles", "Riverside", "Phoenix", "Seattle"], as_of=AS_OF
        )
    return _GRAPH


def test_graph_has_all_ordered_pairs():
    g = _graph()
    # 4 cities -> 4*3 = 12 ordered pairs, no self-loops
    assert len(g.edges) == 12, f"expected 12 ordered pairs, got {len(g.edges)}"
    assert all(e.source_msa_id != e.target_msa_id for e in g.edges)


def test_graph_edges_are_directional():
    """A->B and B->A are separate tests; spillover is not symmetric."""
    g = _graph()
    fwd = [e for e in g.edges if e.source_msa_id == "Los Angeles"
           and e.target_msa_id == "Riverside"]
    rev = [e for e in g.edges if e.source_msa_id == "Riverside"
           and e.target_msa_id == "Los Angeles"]
    assert len(fwd) == 1 and len(rev) == 1, "both directions must be present"


def test_neighbors_sorted_by_strength():
    g = _graph()
    for msa in g.msa_ids:
        ns = g.neighbors_of(msa)
        ps = [e.p_value for e in ns]
        assert ps == sorted(ps), f"{msa} neighbours not sorted by p-value"
        assert all(e.is_significant for e in ns), "neighbors_of returned a non-significant edge"


def test_top_k_caps_neighbours():
    g = _graph()
    for msa in g.msa_ids:
        assert len(g.neighbors_of(msa, top_k=2)) <= 2


def test_spillover_signal_shape():
    g = _graph()
    s = compute_spillover_signal(WEEKLY_DF, g, "Riverside", AS_OF)
    assert s.target_msa_id == "Riverside"
    assert s.neighbor_count == len(s.contributing_msa_ids)
    assert s.max_neighbor_shock >= 0.0
    assert np.isfinite(s.aggregate_neighbor_trend)


def test_spillover_never_reads_the_future():
    """The signal at an early date must not change when later data exists."""
    g = _graph()
    early = compute_spillover_signal(WEEKLY_DF, g, "Riverside", datetime.date(2022, 6, 4))
    truncated = WEEKLY_DF[WEEKLY_DF["date"] <= pd.Timestamp("2022-06-04")]
    same = compute_spillover_signal(truncated, g, "Riverside", datetime.date(2022, 6, 4))
    assert abs(early.aggregate_neighbor_trend - same.aggregate_neighbor_trend) < 1e-12, \
        "signal changed when future rows were removed -- lookahead"


def test_spillover_neutral_when_no_neighbours():
    g = _graph()
    s = SpilloverSignal(target_msa_id="X", neighbor_count=0,
                        aggregate_neighbor_trend=0.0, max_neighbor_shock=0.0)
    fc = np.array([100.0, 100.0, 100.0, 100.0])
    assert np.array_equal(apply_spillover(fc, s, strength=0.5), fc)


def test_apply_spillover_is_off_by_default():
    """strength=0 is the default and must be a no-op -- an unvalidated
    magnitude constant is exactly what went wrong with the Event Agent."""
    s = SpilloverSignal(target_msa_id="Riverside", neighbor_count=2,
                        aggregate_neighbor_trend=0.25, max_neighbor_shock=0.4,
                        contributing_msa_ids=["Los Angeles", "Phoenix"])
    fc = np.array([100.0, 100.0, 100.0, 100.0])
    assert np.array_equal(apply_spillover(fc, s), fc), "default strength must not adjust"


def test_apply_spillover_ramps_and_stays_non_negative():
    s = SpilloverSignal(target_msa_id="Riverside", neighbor_count=2,
                        aggregate_neighbor_trend=-0.5, max_neighbor_shock=0.5,
                        contributing_msa_ids=["Los Angeles", "Phoenix"])
    fc = np.array([100.0, 100.0, 100.0, 100.0])
    adj = apply_spillover(fc, s, strength=1.0)
    assert (adj >= 0).all(), "spillover produced a negative forecast"
    assert abs(adj[0] - 100.0) < abs(adj[-1] - 100.0), "adjustment should ramp in"


# ═══════════════════════════════════════════════════════════════════════════
# Mod A — Early-Exit Gate
# ═══════════════════════════════════════════════════════════════════════════

def _flat_baseline(df, msa, as_of, horizon):
    md = df[(df["msa"] == msa) & (df["date"] <= pd.Timestamp(as_of))]
    last = float(md.sort_values("date")["inventory_count"].iloc[-1])
    return np.full(horizon, last, dtype=np.float64)


def test_gate_always_returns_a_baseline():
    """Even when dormant, the baseline must exist: it is the output on the
    skip path and Mod C's fallback on the correction path."""
    d = evaluate_gate(WEEKLY_DF, "Phoenix", AS_OF, 4, baseline_fn=_flat_baseline)
    assert len(d.baseline_forecast) == 4
    assert all(v >= 0 for v in d.baseline_forecast)


def test_gate_fires_on_text_event():
    d = evaluate_gate(WEEKLY_DF, "Phoenix", AS_OF, 4,
                      event_trigger=GateTriggerType.NATURAL_DISASTER,
                      baseline_fn=_flat_baseline)
    assert d.should_activate_heavy_loop
    assert d.trigger_type == GateTriggerType.NATURAL_DISASTER


def test_gate_ignores_event_not_on_trigger_list():
    cfg = GateConfig(text_event_triggers=[GateTriggerType.FED_RATE_ANNOUNCEMENT])
    d = evaluate_gate(WEEKLY_DF, "Phoenix", AS_OF, 4, config=cfg,
                      event_trigger=GateTriggerType.NATURAL_DISASTER,
                      baseline_fn=_flat_baseline)
    assert d.trigger_type != GateTriggerType.NATURAL_DISASTER, \
        "gate fired on an event type not in its config"


def test_gate_stays_shut_when_nothing_happens():
    """Impossible thresholds mean nothing can fire; the gate must be dormant
    and must report trigger NONE, which GateDecision itself enforces."""
    cfg = GateConfig(anomaly_zscore_threshold=1e9,
                     neighbor_anomaly_zscore_threshold=1e9)
    d = evaluate_gate(WEEKLY_DF, "Phoenix", AS_OF, 4, config=cfg,
                      baseline_fn=_flat_baseline)
    assert not d.should_activate_heavy_loop
    assert d.trigger_type == GateTriggerType.NONE


def test_gate_fires_on_self_anomaly():
    cfg = GateConfig(anomaly_zscore_threshold=1e-9)  # anything exceeds this
    d = evaluate_gate(WEEKLY_DF, "Phoenix", AS_OF, 4, config=cfg,
                      baseline_fn=_flat_baseline)
    assert d.should_activate_heavy_loop
    assert d.trigger_type == GateTriggerType.BASELINE_ANOMALY
    assert d.baseline_error_zscore is not None


def test_gate_fires_on_neighbour_shock():
    """Mod A <-> Mod B: a neighbour's move must be able to wake this city."""
    cfg = GateConfig(anomaly_zscore_threshold=1e9,
                     neighbor_anomaly_zscore_threshold=0.01)
    s = SpilloverSignal(target_msa_id="Riverside", neighbor_count=1,
                        aggregate_neighbor_trend=0.30, max_neighbor_shock=0.30,
                        contributing_msa_ids=["Los Angeles"])
    d = evaluate_gate(WEEKLY_DF, "Riverside", AS_OF, 4, config=cfg,
                      spillover=s, baseline_fn=_flat_baseline)
    assert d.should_activate_heavy_loop
    assert d.trigger_type == GateTriggerType.NEIGHBOR_SPILLOVER
    assert d.neighbor_anomaly_msa_id == "Los Angeles"


def test_text_event_outranks_anomaly():
    cfg = GateConfig(anomaly_zscore_threshold=1e-9)
    d = evaluate_gate(WEEKLY_DF, "Phoenix", AS_OF, 4, config=cfg,
                      event_trigger=GateTriggerType.FED_RATE_ANNOUNCEMENT,
                      baseline_fn=_flat_baseline)
    assert d.trigger_type == GateTriggerType.FED_RATE_ANNOUNCEMENT, \
        "a declared event should outrank a statistical hint"


def test_gate_savings_report():
    ds = [
        evaluate_gate(WEEKLY_DF, "Phoenix", AS_OF, 4,
                      config=GateConfig(anomaly_zscore_threshold=1e9,
                                        neighbor_anomaly_zscore_threshold=1e9),
                      baseline_fn=_flat_baseline),
        evaluate_gate(WEEKLY_DF, "Phoenix", AS_OF, 4,
                      event_trigger=GateTriggerType.NATURAL_DISASTER,
                      baseline_fn=_flat_baseline),
    ]
    r = gate_savings_report(ds)
    assert r["n_forecasts"] == 2 and r["n_activated"] == 1
    assert abs(r["skip_rate"] - 0.5) < 1e-9
    assert r["heavy_loop_calls_saved"] == 1


def test_default_baseline_never_raises():
    v = default_baseline(WEEKLY_DF, "Phoenix", AS_OF, 8)
    assert len(v) == 8 and np.isfinite(v).all()


# ═══════════════════════════════════════════════════════════════════════════
# Mod C — Boundary Constraints
# ═══════════════════════════════════════════════════════════════════════════

def _cfg(max_pct=0.10, retries=2):
    return BoundaryConfig(msa_id="TEST", max_historical_pct_change=max_pct,
                          max_correction_retries=retries)


def test_clean_forecast_passes_untouched():
    fc = [100.0, 102.0, 104.0, 106.0]
    o = enforce_boundaries(fc, 100.0, _cfg())
    assert o.result.is_within_bounds
    assert not o.corrected and not o.fell_back
    assert o.final_forecast == fc


def test_negative_value_is_caught_and_corrected():
    o = enforce_boundaries([100.0, -50.0, 100.0, 100.0], 100.0, _cfg())
    assert all(v >= 0 for v in o.final_forecast), "negative survived correction"
    assert o.corrected


def test_excessive_jump_is_pulled_to_the_boundary():
    """A 100% jump against a 10% bound should be clamped to +10%."""
    o = enforce_boundaries([200.0, 200.0, 200.0, 200.0], 100.0, _cfg(0.10))
    assert abs(o.final_forecast[0] - 110.0) < 1e-6, \
        f"expected clamp to 110.0, got {o.final_forecast[0]}"
    assert o.result.is_within_bounds or o.corrected


def test_correction_is_sequential_not_independent():
    """Bounds are relative to the previous value, so fixing step i changes
    what step i+1 may be. A one-pass independent fix would leave violations."""
    o = enforce_boundaries([300.0, 400.0, 500.0, 600.0], 100.0, _cfg(0.10))
    prev = 100.0
    for v in o.final_forecast:
        assert abs((v - prev) / prev) <= 0.10 + 1e-6, \
            f"step {v} still violates the bound relative to {prev}"
        prev = v


def test_retry_limit_is_respected():
    o = enforce_boundaries([1e9, 1e9, 1e9, 1e9], 100.0, _cfg(0.01, retries=2))
    assert o.attempts <= 2, f"exceeded retry budget: {o.attempts}"


def test_falls_back_to_baseline_after_retries():
    baseline = [100.0, 100.0, 100.0, 100.0]
    cfg = _cfg(0.01, retries=0)
    o = enforce_boundaries([1e9, 1e9, 1e9, 1e9], 100.0, cfg,
                           baseline_forecast=baseline)
    assert o.fell_back, "should have fallen back to baseline"
    # The returned series must satisfy the bound it was checked against.
    prev = 100.0
    for v in o.final_forecast:
        assert abs((v - prev) / prev) <= cfg.max_historical_pct_change + 1e-9, \
            f"fallback returned {v}, which still violates against {prev}"
        prev = v


def test_always_terminates_with_a_usable_forecast():
    """The guarantee that matters: no input leaves us without an answer."""
    for bad in ([-1e6] * 4, [1e12] * 4, [0.0] * 4, [100.0, -1.0, 1e9, 50.0]):
        o = enforce_boundaries(bad, 100.0, _cfg(0.05))
        assert len(o.final_forecast) == 4
        assert all(np.isfinite(v) and v >= 0 for v in o.final_forecast), \
            f"unusable output for input {bad}"


def test_violations_are_logged_with_their_range():
    o = enforce_boundaries([500.0, 500.0], 100.0, _cfg(0.10))
    assert o.violation_log, "violations should be logged"
    assert any("pct change" in m.lower() or "excessive" in m for m in o.violation_log)


def test_boundary_report_counts():
    outs = [
        enforce_boundaries([100.0, 101.0], 100.0, _cfg()),
        enforce_boundaries([500.0, 500.0], 100.0, _cfg()),
    ]
    r = boundary_report(outs)
    assert r["n_forecasts"] == 2 and r["clean"] == 1
    assert r["violation_rate"] == 0.5


def test_clamp_corrector_respects_physical_floor():
    cfg = _cfg()
    out = clamp_corrector([-10.0, -20.0], [], 100.0, cfg)
    assert all(v >= cfg.physical_floor for v in out)


# ═══════════════════════════════════════════════════════════════════════════
# Mod D — Batched Micro-Reasoning (mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════

class _MockClient:
    """Minimal stand-in for the OpenAI client, returning canned text."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        text = self._replies.pop(0) if self._replies else self._replies_default()
        return type("R", (), {"choices": [
            type("C", (), {"message": type("M", (), {"content": text})()})()
        ]})()

    @staticmethod
    def _replies_default():
        return '{"forecasts": [1.0], "reasoning": ["x"]}'


def test_parse_valid_batch():
    o = parse_batch_response('{"forecasts": [1.0, 2.0, 3.0], "reasoning": ["a","b","c"]}', 3)
    assert o.is_valid_batch and o.forecast_values == [1.0, 2.0, 3.0]


def test_parse_strips_markdown_fence():
    o = parse_batch_response('```json\n{"forecasts": [1.0, 2.0], "reasoning": []}\n```', 2)
    assert o.is_valid_batch and len(o.forecast_values) == 2


def test_parse_rejects_wrong_length():
    o = parse_batch_response('{"forecasts": [1.0, 2.0], "reasoning": []}', 5)
    assert not o.is_valid_batch


def test_parse_rejects_negative_values():
    o = parse_batch_response('{"forecasts": [1.0, -2.0], "reasoning": []}', 2)
    assert not o.is_valid_batch, "negative inventory must invalidate the batch"


def test_parse_rejects_garbage():
    for bad in ("not json", "", "[1,2,3]", '{"forecasts": "nope"}',
                '{"forecasts": [1.0, null]}'):
        assert not parse_batch_response(bad, 2).is_valid_batch, f"accepted {bad!r}"


def test_invalid_batch_returns_empty_not_zeros():
    """A zero-filled forecast is a plausible-looking wrong answer; empty
    cannot be mistaken for a real result."""
    o = parse_batch_response("garbage", 4)
    assert o.forecast_values == []


def test_batched_path_uses_one_call():
    c = _MockClient(['{"forecasts": [10.0, 11.0, 12.0, 13.0], "reasoning": ["a","b","c","d"]}'])
    out, calls = run_batched_reasoning([100.0] * 12, 4, client=c)
    assert out.is_valid_batch
    assert calls == 1, f"batched path should cost 1 call, cost {calls}"


def test_malformed_batch_falls_back_to_sequential():
    replies = ["garbage"] + ['{"forecasts": [10.0], "reasoning": ["r"]}'] * 4
    c = _MockClient(replies)
    out, calls = run_batched_reasoning([100.0] * 12, 4, client=c)
    assert out.is_valid_batch, "fallback should still yield a usable forecast"
    assert len(out.forecast_values) == 4
    assert calls == 5, f"expected 1 failed batch + 4 sequential, got {calls}"


def test_fallback_can_be_disabled():
    c = _MockClient(["garbage"])
    out, calls = run_batched_reasoning([100.0] * 12, 4, client=c,
                                        sequential_fallback=False)
    assert not out.is_valid_batch and calls == 1


def test_reasoning_is_carried_but_separate_from_values():
    """Prompt-leakage contract: reasoning exists on the model, values are
    the only field meant to cross into Layer 2."""
    c = _MockClient(['{"forecasts": [1.0, 2.0], "reasoning": ["r1","r2"]}'])
    out, _ = run_batched_reasoning([100.0] * 12, 2, client=c)
    assert out.step_reasoning == ["r1", "r2"]
    assert out.forecast_values == [1.0, 2.0]


def test_batching_savings_math():
    s = batching_savings(13, 100)
    assert s["sequential_calls"] == 1300 and s["batched_calls"] == 100
    assert abs(s["reduction"] - (1 - 100 / 1300)) < 1e-9


def test_batching_savings_charges_for_failures():
    """A failed batch costs its own call plus the full sequential run, so it
    is worse than never batching. The saving must reflect that."""
    clean = batching_savings(13, 100, batch_failure_rate=0.0)["batched_calls"]
    lossy = batching_savings(13, 100, batch_failure_rate=0.2)["batched_calls"]
    assert lossy > clean, "failures must increase the call count"


# ═══════════════════════════════════════════════════════════════════════════

print("\n-- Mod B: Spatial Spillover --")
check("Graph covers every ordered pair, no self-loops", test_graph_has_all_ordered_pairs)
check("Edges are directional (A->B separate from B->A)", test_graph_edges_are_directional)
check("neighbors_of sorted by strength, significant only", test_neighbors_sorted_by_strength)
check("top_k caps the neighbour count", test_top_k_caps_neighbours)
check("Spillover signal has a valid shape", test_spillover_signal_shape)
check("Spillover signal never reads the future", test_spillover_never_reads_the_future)
check("Neutral signal when no neighbours", test_spillover_neutral_when_no_neighbours)
check("Spillover adjustment is OFF by default", test_apply_spillover_is_off_by_default)
check("Adjustment ramps in and stays non-negative", test_apply_spillover_ramps_and_stays_non_negative)

print("\n-- Mod A: Early-Exit Gate --")
check("Baseline is always returned, even when dormant", test_gate_always_returns_a_baseline)
check("Fires on a configured text event", test_gate_fires_on_text_event)
check("Ignores an event not on the trigger list", test_gate_ignores_event_not_on_trigger_list)
check("Stays shut when nothing happens", test_gate_stays_shut_when_nothing_happens)
check("Fires on self anomaly", test_gate_fires_on_self_anomaly)
check("Fires on neighbour shock (Mod A<->B link)", test_gate_fires_on_neighbour_shock)
check("Text event outranks statistical anomaly", test_text_event_outranks_anomaly)
check("Savings report counts skips correctly", test_gate_savings_report)
check("Default baseline never raises", test_default_baseline_never_raises)

print("\n-- Mod C: Boundary Constraints --")
check("Clean forecast passes untouched", test_clean_forecast_passes_untouched)
check("Negative value caught and corrected", test_negative_value_is_caught_and_corrected)
check("Excessive jump pulled to the boundary", test_excessive_jump_is_pulled_to_the_boundary)
check("Correction is sequential, not independent", test_correction_is_sequential_not_independent)
check("Retry limit respected", test_retry_limit_is_respected)
check("Falls back to baseline after retries", test_falls_back_to_baseline_after_retries)
check("Always terminates with a usable forecast", test_always_terminates_with_a_usable_forecast)
check("Violations logged with their range", test_violations_are_logged_with_their_range)
check("Boundary report counts correctly", test_boundary_report_counts)
check("Clamp corrector respects the physical floor", test_clamp_corrector_respects_physical_floor)

print("\n-- Mod D: Batched Micro-Reasoning --")
check("Parses a valid batch", test_parse_valid_batch)
check("Strips markdown fences", test_parse_strips_markdown_fence)
check("Rejects wrong-length arrays", test_parse_rejects_wrong_length)
check("Rejects negative inventory", test_parse_rejects_negative_values)
check("Rejects garbage input", test_parse_rejects_garbage)
check("Invalid batch returns empty, not zeros", test_invalid_batch_returns_empty_not_zeros)
check("Batched path costs exactly one call", test_batched_path_uses_one_call)
check("Malformed batch falls back to sequential", test_malformed_batch_falls_back_to_sequential)
check("Fallback can be disabled", test_fallback_can_be_disabled)
check("Reasoning carried separately from values", test_reasoning_is_carried_but_separate_from_values)
check("Batching savings math", test_batching_savings_math)
check("Savings charges for batch failures", test_batching_savings_charges_for_failures)

print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("All tests passed!")
else:
    sys.exit(1)
