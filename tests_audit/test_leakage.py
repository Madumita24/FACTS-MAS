"""
Phase 1 leakage and protocol tests.

Each test is written so that it FAILS if the leak it targets exists.
Where the audit confirmed a leak in production code (which this audit may
not edit), the test is marked xfail(strict=True): the suite records the
known failure, and turns red if the behaviour silently changes.

Run:  python -m pytest -q tests_audit
"""
import datetime
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit_core as ac  # noqa: E402

os.chdir(ac.REPO)

from facts_mas.agents.seasonality_agent import compute_seasonal_factors, run_seasonality_agent  # noqa: E402
from facts_mas.calibration import _validation_window, score_agent_over_window  # noqa: E402
from facts_mas.run_backtest import FOLDS  # noqa: E402
from facts_mas.schema import AgentOutput  # noqa: E402

PANEL = ac.load_panel()


@pytest.fixture(scope="module")
def cache():
    from facts_mas.agent_cache import AgentCache
    return AgentCache.load(os.path.join(ac.REPO, "agent_output_cache.pkl"))


# ── Guards ────────────────────────────────────────────────────────────────

def test_panel_is_capped():
    assert PANEL["date"].max() <= ac.DATA_CAP


def test_guard_blocks_reserved_targets():
    with pytest.raises(ac.ReservedWindowError):
        ac.assert_scoring_allowed([pd.Timestamp("2025-12-27"), pd.Timestamp("2026-01-03")])
    ac.assert_scoring_allowed([pd.Timestamp("2026-01-03")], disclosure=True)  # allowed when labelled
    ac.assert_scoring_allowed([pd.Timestamp("2025-12-27")])


def test_guard_blocks_beyond_cap_even_with_disclosure():
    with pytest.raises(ac.DataCapError):
        ac.assert_scoring_allowed([pd.Timestamp("2026-02-28")], disclosure=True)


@pytest.mark.xfail(strict=True, reason="CONFIRMED F1.7: fold 6 test cells score targets in Jan-Feb 2026 "
                                       "(45 at 8wk, 120 at 13wk). Production has no target bound.")
def test_production_test_cells_have_no_reserved_targets(cache):
    cells = ac.cells_from_cache(cache, PANEL)
    tds = [d for c in ac.test_cells(cells) for d in c.target_dates]
    ac.assert_scoring_allowed(tds)  # raises -> xfail


# ── Seasonal channel: centred MA must not see past the origin ─────────────

ORIGINS = [datetime.date(2020, 6, 6), datetime.date(2021, 1, 23), datetime.date(2021, 7, 3),
           datetime.date(2022, 9, 3), datetime.date(2023, 12, 2), datetime.date(2025, 2, 1),
           datetime.date(2025, 11, 22)]


@pytest.mark.parametrize("msa", ["Phoenix", "Seattle", "Miami", "New York"])
def test_seasonal_identical_on_truncated_series(msa):
    """Forecast at origin t from the full (capped) panel == from the panel cut at t."""
    for o in ORIGINS:
        trunc = PANEL[PANEL["date"] <= pd.Timestamp(o)]
        f_full = compute_seasonal_factors(PANEL, msa, o)
        f_trunc = compute_seasonal_factors(trunc, msa, o)
        assert np.array_equal(f_full, f_trunc), (msa, o)
        for h in (4, 8, 13):
            a = run_seasonality_agent(PANEL, msa, o, h).values
            b = run_seasonality_agent(trunc, msa, o, h).values
            assert a == b, (msa, o, h)


def test_seasonal_factors_change_when_future_is_altered():
    """Negative control: corrupting data AFTER the origin must not move the factors,
    corrupting data BEFORE it must (so the test above can detect a leak)."""
    o = datetime.date(2023, 12, 2)
    base = compute_seasonal_factors(PANEL, "Phoenix", o)
    fut = PANEL.copy()
    m = (fut["msa"] == "Phoenix") & (fut["date"] > pd.Timestamp(o))
    fut.loc[m, "inventory_count"] *= 3.0
    assert np.array_equal(base, compute_seasonal_factors(fut, "Phoenix", o))
    past = PANEL.copy()
    m = (past["msa"] == "Phoenix") & (past["date"].between(pd.Timestamp(o) - pd.Timedelta(weeks=20), pd.Timestamp(o)))
    past.loc[m, "inventory_count"] *= 3.0
    assert not np.array_equal(base, compute_seasonal_factors(past, "Phoenix", o))


# ── Calibration scorer: targets must stay inside the validation window ────

def test_calibration_scorer_targets_end_before_embargo():
    """Spy on every (origin, horizon) the scorer grades; every target <= train_end < test_start."""
    seen = []

    def spy(df, msa, origin, h):
        md = df[df["msa"] == msa].sort_values("date")
        fut = md[md["date"] > pd.Timestamp(origin)].head(h)
        seen.append((fut["date"].max(), origin))
        return AgentOutput(agent_name="ar", msa=msa, forecast_origin=origin, horizon_weeks=h,
                           values=[1.0] * h)

    for f in (FOLDS[0], FOLDS[5]):
        seen.clear()
        win = _validation_window(f)
        score_agent_over_window(PANEL, spy, ["Phoenix", "Miami"], win, [4, 8, 13])
        assert seen, "scorer graded nothing"
        assert max(t for t, _ in seen) <= pd.Timestamp(f.train_end)
        assert pd.Timestamp(f.train_end) < pd.Timestamp(f.test_start)


# ── Fusion weights: validation targets inside the embargo (F0.10) ─────────

@pytest.mark.xfail(strict=True, reason="CONFIRMED F0.10: compute_fold_weights grades validation origins "
                                       "up to train_end with targets up to train_end + 13 weeks.")
def test_fusion_validation_targets_end_by_train_end():
    from facts_mas.fusion_baseline import compute_fold_weights
    seen = []

    def spy(df, msa, origin, h):
        md = df[df["msa"] == msa].sort_values("date")
        seen.append(md[md["date"] > pd.Timestamp(origin)].head(h)["date"].max())
        return AgentOutput(agent_name="ar", msa=msa, forecast_origin=origin, horizon_weeks=h,
                           values=[1.0] * h)

    f = FOLDS[0]
    compute_fold_weights(PANEL, {"ar": spy}, ["Phoenix"], train_end=f.train_end)
    assert max(seen) <= pd.Timestamp(f.train_end)


def test_fusion_validation_targets_still_precede_test_start():
    """F0.10 is not a test leak: even unbounded, validation targets end before test_start."""
    for f in FOLDS:
        last_target = pd.Timestamp(f.train_end) + pd.Timedelta(weeks=13)
        assert last_target < pd.Timestamp(f.test_start)


# ── Imputed values: no NaN / inf may enter the skill ──────────────────────

def test_skill_inputs_finite_on_imputed_cells(cache):
    fl = pd.read_csv(os.path.join(ac.REPO, "filled_rows_log.csv"), parse_dates=["date"])
    filled = set(zip(fl["msa"], fl["date"]))
    cells = ac.cells_from_cache(cache, PANEL)
    touched = [c for c in cells if any((c.msa, d) in filled for d in c.target_dates)]
    assert touched, "expected some validation cells to touch imputed targets"
    for c in touched:
        nm = ac._mape_frac(c.actuals, c.naive)
        assert nm is not None and np.isfinite(nm)
        for a in ac.AGENTS:
            am = ac._mape_frac(c.actuals, np.asarray(c.values[a]))
            assert am is not None and np.isfinite(am)
