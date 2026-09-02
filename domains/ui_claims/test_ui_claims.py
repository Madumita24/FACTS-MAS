"""Minimal test coverage for the UI-claims pilot -- highest-value checks
only, not exhaustive. Same pytest style as the housing test suite
(test_ar_agent.py, etc.), not the custom check() harness style used in
facts_mas/test_agents.py.

Uses the real aligned_weekly_ui_claims.csv, same convention as
facts_mas/test_agents.py reading the real aligned_weekly.csv -- this
pilot's whole point is testing against the actual pipeline output, not a
synthetic fixture, for the split-boundary and agent-output checks. The
lookahead-guard test uses a small synthetic series, same as the housing
suite's own no-lookahead tests.
"""
import datetime

import numpy as np
import pandas as pd
import pytest

from domains.ui_claims.agents.ar_agent_ui import run_ar_agent_ui
from domains.ui_claims.agents.macro_agent_ui import run_macro_agent_ui
from domains.ui_claims.evaluate_ui_claims_pilot import TEST_END, TEST_START
from domains.ui_claims.naive_baseline_ui import (
    assert_no_lookahead,
    forecast_naive_ui,
    run_naive_baseline_ui,
)

DATA_PATH = "domains/ui_claims/aligned_weekly_ui_claims.csv"
HORIZONS = [4, 8, 13]

# Hardcoded from evaluation_protocol_ui.md's confirmed split -- see test
# below for why these are asserted as literals, not derived.
DOC_TRAIN_END = datetime.date(2023, 6, 24)
DOC_TEST_START = datetime.date(2023, 9, 23)
DOC_TEST_END = datetime.date(2026, 8, 22)
DOC_EMBARGO_DAYS = 91


@pytest.fixture(scope="module")
def weekly_df():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


# ═══════════════════════════════════════════════════════════════════════════
# 1. Lookahead guard actually raises
# ═══════════════════════════════════════════════════════════════════════════

def test_assert_no_lookahead_raises_on_violation():
    s = pd.Series([1, 2, 3], index=pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]))
    with pytest.raises(ValueError, match="lookahead violation"):
        assert_no_lookahead(s, pd.Timestamp("2020-01-08"))


def test_assert_no_lookahead_does_not_raise_on_legal_boundary():
    s = pd.Series([1, 2, 3], index=pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]))
    assert_no_lookahead(s, pd.Timestamp("2020-01-15"))  # exact boundary, should not raise


# ═══════════════════════════════════════════════════════════════════════════
# 2. AR / Macro agent output shape and non-negativity
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("horizon", HORIZONS)
@pytest.mark.parametrize("state", ["California", "Colorado"])
def test_ar_agent_output_shape_and_sign(weekly_df, state, horizon):
    origin = DOC_TRAIN_END
    out = run_ar_agent_ui(weekly_df, state, origin, horizon)
    assert len(out["values"]) == horizon
    assert all(v >= 0 for v in out["values"])


@pytest.mark.parametrize("horizon", HORIZONS)
@pytest.mark.parametrize("state", ["California", "Colorado"])
def test_macro_agent_output_shape_and_sign(weekly_df, state, horizon):
    origin = DOC_TRAIN_END
    out = run_macro_agent_ui(weekly_df, state, origin, horizon)
    assert len(out["values"]) == horizon
    assert all(v >= 0 for v in out["values"])


# ═══════════════════════════════════════════════════════════════════════════
# 3. Train/test split boundary regression test
# ═══════════════════════════════════════════════════════════════════════════

def test_split_dates_match_evaluation_protocol():
    """If evaluate_ui_claims_pilot.py's TEST_START/TEST_END ever drift from
    what evaluation_protocol_ui.md documents, every downstream result
    (pilot_results_ui.md, the lag grid search, the serial-correlation
    check) is silently evaluated on the wrong window. Hardcoded literals,
    not derived from the doc or the code -- that's the point of a
    regression test.
    """
    assert TEST_START == DOC_TEST_START
    assert TEST_END == DOC_TEST_END


def test_embargo_is_91_days():
    gap = (DOC_TEST_START - DOC_TRAIN_END).days
    assert gap == DOC_EMBARGO_DAYS


# ═══════════════════════════════════════════════════════════════════════════
# 4. Naive baseline: flat repeat, correct length
# ═══════════════════════════════════════════════════════════════════════════

def test_forecast_naive_ui_is_flat_repeat_of_last_value():
    s = pd.Series([100.0, 110.0, 95.0], index=pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]))
    forecast = forecast_naive_ui(s, 4)
    assert len(forecast) == 4
    assert np.all(forecast == 95.0)


def test_run_naive_baseline_ui_shape(weekly_df):
    out = run_naive_baseline_ui(weekly_df, "Texas", DOC_TRAIN_END, 13)
    assert len(out["values"]) == 13
    assert len(set(out["values"])) == 1  # flat -- every value identical
