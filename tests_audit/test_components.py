"""
Phase 2 component-correctness tests.

Horizon consistency: the forecast for week t+k must not depend on how many
weeks were requested. A channel that fails this gives two different answers
to the same question depending on the caller's H.
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
from facts_mas.run_backtest import build_agent_runners  # noqa: E402

PANEL = ac.load_panel()
RUNNERS = build_agent_runners()
CASES = [("Phoenix", datetime.date(2022, 12, 10)), ("Seattle", datetime.date(2024, 7, 20)),
         ("New York", datetime.date(2021, 5, 1))]


def _first4(agent, msa, o):
    a = np.array(RUNNERS[agent](PANEL, msa, o, 4).values)
    b = np.array(RUNNERS[agent](PANEL, msa, o, 13).values)[:4]
    return a, b


@pytest.mark.parametrize("agent", ["ar", "seasonality", "macro", "event"])
def test_channel_is_horizon_consistent(agent):
    for msa, o in CASES:
        a, b = _first4(agent, msa, o)
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=0.011, err_msg=f"{agent} {msa} {o}")


@pytest.mark.xfail(strict=True, reason="CONFIRMED F2.1: intrinsic pull rho_max*(h/H)^2 depends on H; "
                                       "step 4 pulls 0.30 of the gap at H=4 but 0.028 at H=13.")
def test_intrinsic_is_horizon_consistent():
    for msa, o in CASES:
        a, b = _first4("intrinsic", msa, o)
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=0.011)


def test_intrinsic_inconsistency_size():
    """Pins the size: at step 4 the H=4 forecast closes 169/16 = 10.56x more of the gap than H=13."""
    from facts_mas.agents import intrinsic_agent as ia
    assert ia.TOTAL_REVERSION == 0.30
    assert 0.30 * (4 / 4) ** 2 / (0.30 * (4 / 13) ** 2) == pytest.approx(169 / 16)


def test_event_equals_naive_in_every_cached_cell():
    from facts_mas.agent_cache import AgentCache
    cache = AgentCache.load(os.path.join(ac.REPO, "agent_output_cache.pkl"))
    assert all(np.array_equal(e.agent_values["event"], e.naive) for e in cache.entries.values())
    assert len(cache.entries) == 15480


def test_r1_baseline_pinned():
    from facts_mas.agent_cache import AgentCache
    cache = AgentCache.load(os.path.join(ac.REPO, "agent_output_cache.pkl"))
    cells = ac.cells_from_cache(cache, PANEL)
    fn = ac.r1_values_fn(ac.load_r1_seasonal())
    d = ac.score(cells, ac.build_models(cells, fn), fn, exclude_bleed=True)
    s = d.groupby("h")["fused"].mean()
    assert s[4] == pytest.approx(2.0894, abs=5e-5)
    assert s[8] == pytest.approx(4.9522, abs=5e-5)
    assert s[13] == pytest.approx(7.0578, abs=5e-5)


def test_r1c_combined_fix_pinned():
    """R1 + fix 2 (13 distinct validation weeks) + F0.10 bound, CLEANBLEED."""
    from facts_mas.agent_cache import AgentCache
    cache = AgentCache.load(os.path.join(ac.REPO, "agent_output_cache.pkl"))
    cells = ac.cells_from_cache(cache, PANEL)
    fn = ac.r1_values_fn(ac.load_r1_seasonal())
    m = ac.build_models(cells, fn, min_support_weeks=13, bound_targets=True)
    s = ac.score(cells, m, fn, exclude_bleed=True).groupby("h")["fused"].mean()
    assert s[4] == pytest.approx(2.1224, abs=5e-5)
    assert s[8] == pytest.approx(4.4684, abs=5e-5)
    assert s[13] == pytest.approx(6.9874, abs=5e-5)
