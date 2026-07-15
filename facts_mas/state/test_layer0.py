"""
Comprehensive test suite for the NEXUS state models.

Tests cover all 4 architectural modifications plus Layer 0 and baseline:
  - Layer 0: SharedContext with NEXUS extensions (I_t, N_t)
  - Mod A: Smart Early-Exit Gate (trigger consistency, neighbor spillover)
  - Mod B: Spatial Spillover (Granger edges, graph queries, adjacency matrix)
  - Mod C: Boundary Constraints (check_forecast, retry limit, baseline fallback)
  - Mod D: Batched Reasoning (horizon validation, calibration fold consistency)
  - Baseline: Honest Control (metric computation, comparison summaries)
  - Graph State: NEXUSGraphState key completeness
"""

import sys
import os
sys.path.insert(0, ".")

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from facts_mas.state import *

passed = 0
failed = 0


def check(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:
        failed += 1
        print(f"  FAIL  {name}: {e}")



# ══════════════════════════════════════════════════════════════════════════
# LAYER 0 — Original + NEXUS Extensions
# ══════════════════════════════════════════════════════════════════════════

print("\n── Layer 0: SharedContext ──")


def test_inventory_non_negative():
    try:
        InventoryHistory(
            dates=["2026-01-01"], values=[-10.0], series_id="TEST"
        )
        raise AssertionError("Should reject negative inventory")
    except ValueError:
        pass

check("InventoryHistory rejects negative values", test_inventory_non_negative)


def test_inventory_pct_changes():
    inv = InventoryHistory(
        dates=["2026-01-01", "2026-01-08", "2026-01-15"],
        values=[100.0, 110.0, 99.0],
        series_id="TEST",
    )
    pct = inv.pct_changes()
    assert len(pct) == 2
    assert abs(pct[0] - 0.1) < 1e-9  # +10%
    assert abs(pct[1] - (-0.1)) < 1e-9  # -10%

check("InventoryHistory.pct_changes() computes correctly", test_inventory_pct_changes)


def test_neighbor_context_aligned():
    try:
        NeighborContext(
            neighbor_msa_id="12060",
            granger_p_value=0.03,
            spillover_lag_weeks=4,
            recent_inventory_values=[100, 105],
            recent_inventory_dates=["2026-01-01"],  # misaligned
        )
        raise AssertionError("Should reject misaligned arrays")
    except ValueError:
        pass

check("NeighborContext rejects misaligned values/dates", test_neighbor_context_aligned)


def test_shared_context_with_nexus_extensions():
    ctx = SharedContext(
        msa_id="38060",
        forecast_date="2026-07-07",
        horizon_weeks=4,
        X_t=ZillowHistory(
            dates=["2026-06-02", "2026-06-09", "2026-06-16", "2026-06-23"],
            values=[350000.0, 351000.0, 351500.0, 352000.0],
            series_id="ZHVI-SFR-38060",
        ),
        M_t=MacroFeatures(
            columns=[
                MacroFeatureColumn(
                    factor="mortgage_rate_30y",
                    fred_series_id="MORTGAGE30US",
                    values=[6.5, 6.48, 6.45, 6.42],
                    lag_weeks=1,
                ),
            ]
        ),
        E_t=[],
        C_t=CalendarContext(
            week_of_year=[28, 29, 30, 31],
            month=[7, 7, 7, 8],
            quarter=[3, 3, 3, 3],
            is_spring_selling_season=[False, False, False, False],
            is_holiday_period=[False, False, False, False],
        ),
        I_t=InventoryHistory(
            dates=["2026-06-02", "2026-06-09", "2026-06-16", "2026-06-23"],
            values=[1200.0, 1180.0, 1195.0, 1210.0],
            series_id="INVEN-SFR-38060",
        ),
        N_t=[
            NeighborContext(
                neighbor_msa_id="40140",
                granger_p_value=0.02,
                spillover_lag_weeks=4,
                recent_inventory_values=[800.0, 790.0, 785.0, 780.0],
                recent_inventory_dates=[
                    "2026-05-05", "2026-05-12", "2026-05-19", "2026-05-26",
                ],
                recent_pct_changes=[-0.0125, -0.0063, -0.0064],
            ),
        ],
    )
    assert ctx.I_t is not None
    assert len(ctx.N_t) == 1
    assert ctx.N_t[0].granger_p_value == 0.02
    # JSON round-trip
    json_str = ctx.model_dump_json()
    ctx2 = SharedContext.model_validate_json(json_str)
    assert ctx2.I_t.series_id == "INVEN-SFR-38060"
    assert len(ctx2.N_t) == 1

check("SharedContext with I_t and N_t + JSON round-trip", test_shared_context_with_nexus_extensions)


# ══════════════════════════════════════════════════════════════════════════
# MOD A — Smart Early-Exit Gate
# ══════════════════════════════════════════════════════════════════════════

print("\n── Mod A: Smart Early-Exit Gate ──")


def test_gate_dormant():
    g = GateDecision(
        should_activate_heavy_loop=False,
        trigger_type="none",
        baseline_forecast=[100.0, 102.0, 103.0, 104.0],
    )
    assert not g.should_activate_heavy_loop

check("Gate dormant: no trigger, accepts baseline", test_gate_dormant)


def test_gate_activates_on_anomaly():
    g = GateDecision(
        should_activate_heavy_loop=True,
        trigger_type="baseline_anomaly",
        trigger_details="Baseline error z-score 2.7 exceeded threshold 2.0",
        baseline_forecast=[100.0, 102.0, 103.0, 104.0],
        baseline_error_zscore=2.7,
    )
    assert g.should_activate_heavy_loop
    assert g.trigger_type == GateTriggerType.BASELINE_ANOMALY

check("Gate activates on baseline anomaly z-score", test_gate_activates_on_anomaly)


def test_gate_activates_on_neighbor_spillover():
    g = GateDecision(
        should_activate_heavy_loop=True,
        trigger_type="neighbor_spillover",
        trigger_details="LA inventory shock detected, Granger-linked neighbor",
        baseline_forecast=[100.0, 102.0, 103.0, 104.0],
        neighbor_anomaly_msa_id="31080",
        neighbor_error_zscore=3.1,
    )
    assert g.trigger_type == GateTriggerType.NEIGHBOR_SPILLOVER
    assert g.neighbor_anomaly_msa_id == "31080"

check("Gate activates on neighbor spillover (Mod A↔B)", test_gate_activates_on_neighbor_spillover)


def test_gate_inconsistent_rejected():
    try:
        GateDecision(
            should_activate_heavy_loop=True,
            trigger_type="none",  # inconsistent!
            baseline_forecast=[100.0],
        )
        raise AssertionError("Should reject activate=True with trigger=none")
    except ValueError:
        pass

check("Gate rejects inconsistent activate=True / trigger=none", test_gate_inconsistent_rejected)


def test_gate_neighbor_missing_msa():
    try:
        GateDecision(
            should_activate_heavy_loop=True,
            trigger_type="neighbor_spillover",
            baseline_forecast=[100.0],
            # missing neighbor_anomaly_msa_id!
        )
        raise AssertionError("Should reject neighbor trigger without MSA ID")
    except ValueError:
        pass

check("Gate rejects neighbor trigger without MSA ID", test_gate_neighbor_missing_msa)


# ══════════════════════════════════════════════════════════════════════════
# MOD B — Spatial Spillover Agent
# ══════════════════════════════════════════════════════════════════════════

print("\n── Mod B: Spatial Spillover ──")


def test_granger_self_loop_rejected():
    try:
        GrangerEdge(
            source_msa_id="38060",
            target_msa_id="38060",
            optimal_lag_weeks=4,
            p_value=0.01,
            f_statistic=12.5,
            is_significant=True,
        )
        raise AssertionError("Should reject self-loops")
    except ValueError:
        pass

check("GrangerEdge rejects self-loops", test_granger_self_loop_rejected)


def test_spatial_graph_neighbors():
    graph = SpatialGraph(
        msa_ids=["38060", "40140", "12060"],
        edges=[
            GrangerEdge(
                source_msa_id="40140", target_msa_id="38060",
                optimal_lag_weeks=4, p_value=0.02, f_statistic=8.5,
                is_significant=True,
            ),
            GrangerEdge(
                source_msa_id="12060", target_msa_id="38060",
                optimal_lag_weeks=6, p_value=0.15, f_statistic=2.1,
                is_significant=False,
            ),
            GrangerEdge(
                source_msa_id="38060", target_msa_id="40140",
                optimal_lag_weeks=5, p_value=0.04, f_statistic=5.2,
                is_significant=True,
            ),
        ],
    )
    # Only significant neighbors
    neighbors = graph.neighbors_of("38060")
    assert len(neighbors) == 1
    assert neighbors[0].source_msa_id == "40140"

check("SpatialGraph.neighbors_of() returns only significant edges", test_spatial_graph_neighbors)


def test_spatial_adjacency_matrix():
    graph = SpatialGraph(
        msa_ids=["A", "B", "C"],
        edges=[
            GrangerEdge(
                source_msa_id="B", target_msa_id="A",
                optimal_lag_weeks=4, p_value=0.05, f_statistic=4.0,
                is_significant=True,
            ),
        ],
    )
    ids, mat = graph.adjacency_matrix()
    assert ids == ["A", "B", "C"]
    assert mat[0, 1] > 0  # A←B edge exists
    assert mat[1, 0] == 0  # no B←A edge
    assert mat[2, 2] == 0  # no self-loop

check("SpatialGraph.adjacency_matrix() correct", test_spatial_adjacency_matrix)


# ══════════════════════════════════════════════════════════════════════════
# MOD C — Statistical Boundary Constraints
# ══════════════════════════════════════════════════════════════════════════

print("\n── Mod C: Boundary Constraints ──")


def test_boundary_from_historical():
    config = BoundaryConfig.from_historical_series(
        msa_id="38060",
        historical_values=[1000, 1050, 1020, 1100, 980],
    )
    assert config.msa_id == "38060"
    assert config.max_historical_pct_change > 0
    assert config.max_correction_retries == 2

check("BoundaryConfig.from_historical_series() computes max pct change", test_boundary_from_historical)


def test_boundary_check_passes():
    config = BoundaryConfig(
        msa_id="38060",
        max_historical_pct_change=0.15,
    )
    result = BoundaryCheckResult.check_forecast(
        forecast_values=[1010.0, 1020.0, 1030.0, 1040.0],
        last_historical_value=1000.0,
        config=config,
    )
    assert result.is_within_bounds
    assert len(result.violations) == 0

check("BoundaryCheckResult passes clean forecast", test_boundary_check_passes)


def test_boundary_check_detects_violation():
    config = BoundaryConfig(
        msa_id="38060",
        max_historical_pct_change=0.05,  # tight bound: 5%
    )
    result = BoundaryCheckResult.check_forecast(
        forecast_values=[1000.0, 1200.0],  # +20% spike!
        last_historical_value=1000.0,
        config=config,
    )
    assert not result.is_within_bounds
    assert len(result.violations) == 1
    assert result.violations[0].violation_type == "excessive_pct_change"

check("BoundaryCheckResult detects excessive pct change", test_boundary_check_detects_violation)


def test_boundary_fallback_after_retries():
    config = BoundaryConfig(
        msa_id="38060",
        max_historical_pct_change=0.05,
        max_correction_retries=2,
    )
    baseline = [1010.0, 1020.0]
    result = BoundaryCheckResult.check_forecast(
        forecast_values=[1000.0, 1500.0],  # way out of bounds
        last_historical_value=1000.0,
        config=config,
        retry_count=2,  # already at limit
        baseline_forecast=baseline,
    )
    assert not result.is_within_bounds
    assert result.fell_back_to_baseline
    assert result.corrected_forecast is not None
    assert result.corrected_forecast[1] == 1020.0  # substituted from baseline

check("Boundary falls back to baseline after 2 retries", test_boundary_fallback_after_retries)


def test_boundary_negative_inventory():
    config = BoundaryConfig(
        msa_id="38060",
        max_historical_pct_change=0.50,
        physical_floor=0.0,
    )
    result = BoundaryCheckResult.check_forecast(
        forecast_values=[500.0, -10.0],
        last_historical_value=1000.0,
        config=config,
    )
    assert not result.is_within_bounds
    assert any(v.violation_type == "below_physical_floor" for v in result.violations)

check("Boundary detects negative inventory (physical floor)", test_boundary_negative_inventory)


# ══════════════════════════════════════════════════════════════════════════
# MOD D — Batched Micro-Reasoning & Calibration
# ══════════════════════════════════════════════════════════════════════════

print("\n── Mod D: Batched Reasoning & Calibration ──")


def test_batched_valid():
    out = BatchedReasoningOutput(
        forecast_values=[100.0, 102.0, 103.0, 104.0],
        horizon_weeks=4,
        model_name="gemini-3.1-pro",
    )
    assert out.is_valid_batch
    assert len(out.forecast_values) == 4

check("BatchedReasoningOutput valid batch accepted", test_batched_valid)


def test_batched_wrong_length_rejected():
    try:
        BatchedReasoningOutput(
            forecast_values=[100.0, 102.0],
            horizon_weeks=4,  # mismatch!
            model_name="gemini-3.1-pro",
            is_valid_batch=True,
        )
        raise AssertionError("Should reject mismatched lengths")
    except ValueError:
        pass

check("BatchedReasoningOutput rejects length mismatch", test_batched_wrong_length_rejected)


def test_batched_invalid_allows_any_length():
    out = BatchedReasoningOutput(
        forecast_values=[100.0],
        horizon_weeks=4,
        model_name="gemini-3.1-pro",
        is_valid_batch=False,  # marked invalid → fallback
    )
    assert not out.is_valid_batch

check("BatchedReasoningOutput invalid batch allows mismatch (fallback)", test_batched_invalid_allows_any_length)


def test_calibration_guideline_consistency():
    gl = CalibrationGuideline(
        guideline_text="Decrease forecasts by 2% during holiday weeks",
        fold_results=[
            FoldResult(fold_index=0, correction_direction="decrease", mape_before=0.12, mape_after=0.09),
            FoldResult(fold_index=1, correction_direction="decrease", mape_before=0.11, mape_after=0.08),
            FoldResult(fold_index=2, correction_direction="decrease", mape_before=0.13, mape_after=0.10),
            FoldResult(fold_index=3, correction_direction="decrease", mape_before=0.10, mape_after=0.09),
            FoldResult(fold_index=4, correction_direction="increase", mape_before=0.09, mape_after=0.10),
        ],
        min_consistent_folds=4,
    )
    assert gl.passes_consistency_check  # 4/5 folds say "decrease"
    assert gl.dominant_direction == "decrease"
    assert gl.fold_consistency_count == 4

check("CalibrationGuideline passes 4/5 fold consistency", test_calibration_guideline_consistency)


def test_calibration_guideline_fails_consistency():
    gl = CalibrationGuideline(
        guideline_text="Increase forecasts in spring",
        fold_results=[
            FoldResult(fold_index=0, correction_direction="increase", mape_before=0.12, mape_after=0.10),
            FoldResult(fold_index=1, correction_direction="decrease", mape_before=0.11, mape_after=0.09),
            FoldResult(fold_index=2, correction_direction="increase", mape_before=0.13, mape_after=0.11),
            FoldResult(fold_index=3, correction_direction="neutral", mape_before=0.10, mape_after=0.10),
            FoldResult(fold_index=4, correction_direction="decrease", mape_before=0.09, mape_after=0.08),
        ],
        min_consistent_folds=4,
    )
    assert not gl.passes_consistency_check  # only 2/5 say "increase"

check("CalibrationGuideline fails 4/5 fold consistency", test_calibration_guideline_fails_consistency)


# ══════════════════════════════════════════════════════════════════════════
# BASELINE — Honest Control
# ══════════════════════════════════════════════════════════════════════════

print("\n── Baseline: Honest Control ──")


def test_single_model_compute_metrics():
    r = SingleModelResult(
        model_name="gemini-3.1-pro",
        system_type="cot_baseline",
        forecast_values=[100.0, 200.0, 300.0],
    )
    r2 = r.compute_metrics([110.0, 190.0, 310.0])
    assert r2.mape is not None
    assert r2.rmse is not None
    assert r2.rmse > 0

check("SingleModelResult.compute_metrics() pure NumPy", test_single_model_compute_metrics)


def test_baseline_comparison_summaries():
    cot = [
        SingleModelResult(
            model_name="gemini-3.1-pro", system_type="cot_baseline",
            forecast_values=[100.0], actual_values=[110.0],
            mape=0.0909, rmse=10.0, repetition_index=0,
        ),
        SingleModelResult(
            model_name="gemini-3.1-pro", system_type="cot_baseline",
            forecast_values=[105.0], actual_values=[110.0],
            mape=0.0455, rmse=5.0, repetition_index=1,
        ),
    ]
    ma = [
        SingleModelResult(
            model_name="nexus-multi-agent", system_type="multi_agent",
            forecast_values=[108.0], actual_values=[110.0],
            mape=0.0182, rmse=2.0, repetition_index=0,
        ),
    ]
    comp = BaselineComparison(
        msa_id="38060", horizon_weeks=4,
        cot_results=cot, multi_agent_results=ma,
    )
    assert comp.cot_mean_mape is not None
    assert comp.multi_agent_mean_mape is not None
    assert comp.cot_mean_mape > comp.multi_agent_mean_mape

check("BaselineComparison auto-computes summary statistics", test_baseline_comparison_summaries)


# ══════════════════════════════════════════════════════════════════════════
# GRAPH STATE
# ══════════════════════════════════════════════════════════════════════════

print("\n── Graph State ──")


def test_nexus_graph_state_keys():
    keys = list(NEXUSGraphState.__annotations__.keys())
    expected = {
        "shared_ctx",
        "gate_config", "gate_decision",
        "spatial_graph", "spillover_signal",
        "ar_forecast", "macro_modifiers", "event_impacts", "batched_reasoning",
        "_l1_internal_traces",
        "boundary_config", "boundary_check",
        "weights", "fused_forecast", "explanation", "calibration",
        "baseline_comparison",
    }
    assert set(keys) == expected, f"Missing: {expected - set(keys)}, Extra: {set(keys) - expected}"

check("NEXUSGraphState has all 17 expected keys", test_nexus_graph_state_keys)


def test_deprecated_alias():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from facts_mas.state.graph_state import FACTSGraphState
        assert len(w) == 1
        assert "deprecated" in str(w[0].message).lower()
        assert FACTSGraphState is NEXUSGraphState

check("FACTSGraphState deprecated alias works", test_deprecated_alias)


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
else:
    print("All tests passed! ✓")
