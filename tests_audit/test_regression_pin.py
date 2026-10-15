"""
Regression pin: the fast audit mirror must reproduce PRODUCTION outputs
exactly, and naive must stay at the published values.

The fixture (fixtures/production_pin.json) is produced by running the
unchanged production code (make_production_fixture.py). If a later audit
change moves any of these numbers, this file says so.
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit_core as ac  # noqa: E402

FIX = os.path.join(HERE, "fixtures", "production_pin.json")
NAIVE = {4: 4.193649233250065, 8: 7.371381852702863, 13: 11.167666933582275}
PROVISIONAL = {4: 2.093957034215429, 8: 4.972910484734933, 13: 6.854970940991459}


@pytest.fixture(scope="module")
def setup():
    from facts_mas.agent_cache import AgentCache
    cache = AgentCache.load(os.path.join(ac.REPO, "agent_output_cache.pkl"))
    cells = ac.cells_from_cache(cache, ac.load_panel())
    return cells, ac.build_models(cells)


def test_naive_and_provisional_headline_pinned(setup):
    cells, models = setup
    d = ac.score(cells, models, disclosure=True)  # DISCLOSURE: includes the 165 bleed cells
    s = d.groupby("h")[["naive", "fused"]].mean()
    for h in (4, 8, 13):
        assert (d.h == h).sum() == 2610
        assert s.loc[h, "naive"] == pytest.approx(NAIVE[h], abs=1e-12)
        assert s.loc[h, "fused"] == pytest.approx(PROVISIONAL[h], abs=1e-9)


@pytest.mark.skipif(not os.path.exists(FIX), reason="run make_production_fixture.py first")
def test_mirror_matches_production_fixture(setup):
    cells, models = setup
    fx = json.load(open(FIX))
    for f, rec in fx["folds"].items():
        m = models[int(f)]
        for a, w in rec["fold_weights"].items():
            assert m.fold_weights[a] == pytest.approx(w, abs=1e-12), (f, a)
        for r, per_h in rec["grid"].items():
            for h, cell in per_h.items():
                hh = int(str(h).replace("wk", ""))  # production keys the grid "4wk", "8wk", "13wk"
                assert m.regime_grid[r][hh]["n"] == cell["n"], (f, r, h)
                for a, w in cell["weights"].items():
                    assert m.regime_grid[r][hh]["weights"][a] == pytest.approx(w, abs=1e-12)
        for h, (fm, rm) in rec["val_mape"].items():
            assert m.val_mape[int(h)][0] == pytest.approx(fm, abs=1e-9)
            assert m.val_mape[int(h)][1] == pytest.approx(rm, abs=1e-9)
            assert m.mode[int(h)] == rec["mode"][h]
    d = ac.score(cells, models, disclosure=True)
    for h, t in fx["test_mape"].items():
        assert d[d.h == int(h)].fused.mean() == pytest.approx(t["fused"], abs=1e-9)
        assert d[d.h == int(h)].naive.mean() == pytest.approx(t["naive"], abs=1e-12)
    by_key = {(c.msa, c.origin.isoformat(), c.h): c for c in ac.test_cells(cells)}
    for k, vals in fx["slice"].items():
        msa, o, h = k.split("|")
        c = by_key[(msa, o, int(h))]
        np.testing.assert_allclose(ac.fuse(c.values, ac.weights_for(c, models), c.h), vals, atol=1e-9)
