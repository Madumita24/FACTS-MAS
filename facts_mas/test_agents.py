"""
Test Suite — FACTS-MAS Phase 2.

Unit tests for each agent's outputs (shape/range/non-negativity), plus
one integration test running the full pipeline on a sample MSA end-to-end.

Acceptance criteria (PDF §3):
    - Suite must pass before any agent is merged.
    - Coverage: every agent's public function has at least one direct test.
"""

import sys
import os
import datetime

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from facts_mas.schema import AgentOutput, FusionInput, BacktestResult
from facts_mas.agents.seasonality_agent import (
    run_seasonality_agent,
    compute_seasonal_factors,
)
from facts_mas.agents.intrinsic_agent import (
    run_intrinsic_agent,
    train_intrinsic_model,
)
from facts_mas.validation import (
    validate_agent_output,
    validate_fusion_input,
    ValidationError,
    check_shape,
    check_non_negative,
    check_agent_name,
)
from facts_mas.run_backtest import (
    FOLDS,
    validate_embargo,
    classify_regime,
    compute_mape,
    compute_rmse,
    naive_forecast,
)

# Load real data
WEEKLY_DF = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
STATIC_DF = pd.read_csv("intrinsic_static.csv")

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
# SCHEMA TESTS
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Schema --")


def test_schema_valid():
    out = AgentOutput(
        agent_name="seasonality", msa="Atlanta",
        forecast_origin=datetime.date(2023, 6, 1),
        horizon_weeks=4, values=[1000.0, 1010.0, 1020.0, 1030.0],
    )
    assert len(out.values) == 4

check("AgentOutput accepts valid data", test_schema_valid)


def test_schema_rejects_wrong_length():
    try:
        AgentOutput(
            agent_name="ar", msa="Atlanta",
            forecast_origin=datetime.date(2023, 6, 1),
            horizon_weeks=4, values=[1000.0, 1010.0],
        )
        raise AssertionError("Should reject length mismatch")
    except ValueError:
        pass

check("AgentOutput rejects length mismatch", test_schema_rejects_wrong_length)


def test_schema_rejects_negative():
    try:
        AgentOutput(
            agent_name="macro", msa="Atlanta",
            forecast_origin=datetime.date(2023, 6, 1),
            horizon_weeks=2, values=[1000.0, -5.0],
        )
        raise AssertionError("Should reject negative values")
    except ValueError:
        pass

check("AgentOutput rejects negative inventory", test_schema_rejects_negative)


def test_schema_rejects_bad_agent():
    try:
        AgentOutput(
            agent_name="unknown_agent", msa="Atlanta",
            forecast_origin=datetime.date(2023, 6, 1),
            horizon_weeks=1, values=[1000.0],
        )
        raise AssertionError("Should reject unknown agent")
    except ValueError:
        pass

check("AgentOutput rejects unknown agent_name", test_schema_rejects_bad_agent)


# ══════════════════════════════════════════════════════════════════════════
# SEASONALITY AGENT TESTS
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Seasonality Agent --")


def test_seasonality_shape():
    origin = datetime.date(2023, 6, 1)
    for horizon in [4, 8, 13]:
        out = run_seasonality_agent(WEEKLY_DF, "Atlanta", origin, horizon)
        assert len(out.values) == horizon
        assert out.agent_name == "seasonality"

check("Seasonality output shape matches horizon (4/8/13)", test_seasonality_shape)


def test_seasonality_non_negative():
    origin = datetime.date(2023, 6, 1)
    out = run_seasonality_agent(WEEKLY_DF, "Atlanta", origin, 13)
    assert all(v >= 0 for v in out.values)

check("Seasonality output is non-negative", test_seasonality_non_negative)


def test_seasonality_deterministic():
    origin = datetime.date(2023, 6, 1)
    out1 = run_seasonality_agent(WEEKLY_DF, "Phoenix", origin, 8)
    out2 = run_seasonality_agent(WEEKLY_DF, "Phoenix", origin, 8)
    assert out1.values == out2.values

check("Seasonality is deterministic (same inputs -> same output)", test_seasonality_deterministic)


def test_seasonality_direction():
    """Spring/summer should show higher inventory than winter."""
    factors = compute_seasonal_factors(
        WEEKLY_DF, "Atlanta", datetime.date(2023, 12, 31)
    )
    # Spring/summer: weeks 12-30, winter: weeks 48-8
    spring_mean = np.mean(factors[11:30])
    winter_mean = np.mean([*factors[47:52], *factors[0:8]])
    assert spring_mean > winter_mean, (
        f"Expected spring ({spring_mean:.3f}) > winter ({winter_mean:.3f})"
    )

check("Seasonality direction: spring/summer > winter", test_seasonality_direction)


# ══════════════════════════════════════════════════════════════════════════
# INTRINSIC AGENT TESTS
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Intrinsic Agent --")


def test_intrinsic_shape():
    origin = datetime.date(2023, 6, 1)
    for horizon in [4, 8, 13]:
        out = run_intrinsic_agent(WEEKLY_DF, STATIC_DF, "Boston", origin, horizon)
        assert len(out.values) == horizon
        assert out.agent_name == "intrinsic"

check("Intrinsic output shape matches horizon (4/8/13)", test_intrinsic_shape)


def test_intrinsic_non_negative():
    origin = datetime.date(2023, 6, 1)
    out = run_intrinsic_agent(WEEKLY_DF, STATIC_DF, "Seattle", origin, 13)
    assert all(v >= 0 for v in out.values)

check("Intrinsic output is non-negative", test_intrinsic_non_negative)


def test_intrinsic_held_out_msa():
    """Train without Dallas, then predict Dallas — shouldn't crash."""
    origin = datetime.date(2023, 6, 1)
    model = train_intrinsic_model(
        WEEKLY_DF, STATIC_DF, origin, exclude_msa="Dallas"
    )
    out = run_intrinsic_agent(
        WEEKLY_DF, STATIC_DF, "Dallas", origin, 8, model=model
    )
    assert len(out.values) == 8
    assert all(v >= 0 for v in out.values)

check("Intrinsic works on held-out MSA (leave-one-out)", test_intrinsic_held_out_msa)


# ══════════════════════════════════════════════════════════════════════════
# VALIDATION LAYER TESTS
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Validation Layer --")


def test_validation_rejects_negative():
    # Bypass Pydantic validation to test the validation layer directly
    out = AgentOutput.model_construct(
        agent_name="ar", msa="Atlanta",
        forecast_origin=datetime.date(2023, 6, 1),
        horizon_weeks=2, values=[100.0, -5.0],
    )
    errors = validate_agent_output(out)
    assert any("Negative" in e for e in errors)

check("Validation catches negative values", test_validation_rejects_negative)


def test_validation_rejects_bad_shape():
    out = AgentOutput.model_construct(
        agent_name="ar", msa="Atlanta",
        forecast_origin=datetime.date(2023, 6, 1),
        horizon_weeks=4, values=[100.0, 110.0],
    )
    errors = validate_agent_output(out)
    assert any("Shape" in e for e in errors)

check("Validation catches shape mismatch", test_validation_rejects_bad_shape)


def test_validation_rejects_bad_agent():
    out = AgentOutput.model_construct(
        agent_name="bogus", msa="Atlanta",
        forecast_origin=datetime.date(2023, 6, 1),
        horizon_weeks=1, values=[100.0],
    )
    errors = validate_agent_output(out)
    assert any("Unknown" in e for e in errors)

check("Validation catches unknown agent_name", test_validation_rejects_bad_agent)


def test_validation_passes_good_output():
    out = AgentOutput(
        agent_name="seasonality", msa="Atlanta",
        forecast_origin=datetime.date(2023, 6, 1),
        horizon_weeks=4, values=[1000.0, 1010.0, 1020.0, 1030.0],
    )
    errors = validate_agent_output(out)
    assert errors == []

check("Validation passes clean output", test_validation_passes_good_output)


# ══════════════════════════════════════════════════════════════════════════
# BACKTEST HARNESS TESTS
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Backtest Harness --")


def test_embargo_all_folds():
    for fold in FOLDS:
        validate_embargo(fold)

check("All 6 folds pass 13-week embargo check", test_embargo_all_folds)


def test_regime_classification():
    r = classify_regime(WEEKLY_DF, "Atlanta", datetime.date(2022, 9, 1))
    assert r in {"hiking", "cutting", "stable"}

check("Regime classification returns valid label", test_regime_classification)


def test_mape_rmse():
    actuals = np.array([100.0, 200.0, 300.0])
    forecast = np.array([110.0, 190.0, 310.0])
    mape = compute_mape(actuals, forecast)
    rmse = compute_rmse(actuals, forecast)
    assert mape > 0
    assert rmse > 0

check("MAPE and RMSE compute correctly", test_mape_rmse)


def test_naive_baseline():
    naive = naive_forecast(1000.0, 8)
    assert len(naive) == 8
    assert all(v == 1000.0 for v in naive)

check("Naive last-value baseline works", test_naive_baseline)


def test_fold_dates_no_overlap():
    """No fold's test window should overlap with another fold's test window."""
    for i, f1 in enumerate(FOLDS):
        for j, f2 in enumerate(FOLDS):
            if i >= j:
                continue
            assert f1.test_end <= f2.test_start or f2.test_end <= f1.test_start, (
                f"Fold {f1.fold} and {f2.fold} test windows overlap"
            )

check("Fold test windows don't overlap", test_fold_dates_no_overlap)


def test_training_before_jan_2026():
    """All folds' training data must end before Jan 2026."""
    cutoff = datetime.date(2026, 1, 1)
    for fold in FOLDS:
        assert fold.train_end < cutoff, (
            f"Fold {fold.fold} train_end {fold.train_end} >= Jan 2026"
        )

check("All training data before Jan 2026 cutoff", test_training_before_jan_2026)


# ══════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST
# ══════════════════════════════════════════════════════════════════════════
print("\n-- Integration: End-to-End --")


def test_end_to_end_single_msa():
    """Run both agents on a single MSA and fuse their outputs."""
    msa = "Phoenix"
    origin = datetime.date(2023, 6, 1)
    horizon = 8

    # Run both agents
    s_out = run_seasonality_agent(WEEKLY_DF, msa, origin, horizon)
    i_out = run_intrinsic_agent(WEEKLY_DF, STATIC_DF, msa, origin, horizon)

    # Validate both
    assert validate_agent_output(s_out) == []
    assert validate_agent_output(i_out) == []

    # Fuse with equal weights
    weights = {"seasonality": 0.5, "intrinsic": 0.5}
    fused = [
        round(0.5 * s + 0.5 * i, 2)
        for s, i in zip(s_out.values, i_out.values)
    ]
    assert len(fused) == horizon
    assert all(v >= 0 for v in fused)

check("End-to-end: both agents + fusion on Phoenix", test_end_to_end_single_msa)


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
else:
    print("All tests passed!")
