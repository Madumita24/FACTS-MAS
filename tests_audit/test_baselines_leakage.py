"""
Phase 3: the new standalone baselines (especially FA-ETS-d, the best method
found) must not see data after the forecast origin.

1. Truncated vs full panel: identical forecasts.
2. Negative control: corrupting data after the origin changes nothing;
   corrupting data before it changes the forecast.
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
sys.path.insert(0, os.path.join(ac.REPO, "audit_reports", "phase3_scripts"))
from p3_build_baselines import forecasts_at  # noqa: E402

PANEL = ac.load_panel()
CASES = [("Phoenix", datetime.date(2022, 12, 10)), ("Seattle", datetime.date(2024, 7, 20)),
         ("Miami", datetime.date(2021, 5, 1)), ("New York", datetime.date(2025, 2, 1))]


@pytest.mark.parametrize("msa,o", CASES)
def test_baselines_identical_on_truncated_panel(msa, o):
    full = forecasts_at(PANEL, msa, o)
    trunc = forecasts_at(PANEL[PANEL["date"] <= pd.Timestamp(o)], msa, o)
    for k in full:
        np.testing.assert_array_equal(full[k], trunc[k], err_msg=k)


def test_fa_ets_d_negative_control():
    msa, o = "Phoenix", datetime.date(2022, 12, 10)
    base = forecasts_at(PANEL, msa, o)["fa_ets_d"]
    fut = PANEL.copy()
    fut.loc[(fut.msa == msa) & (fut.date > pd.Timestamp(o)), "inventory_count"] *= 3.0
    np.testing.assert_array_equal(base, forecasts_at(fut, msa, o)["fa_ets_d"])
    past = PANEL.copy()
    past.loc[(past.msa == msa) & past.date.between(pd.Timestamp(o) - pd.Timedelta(weeks=8), pd.Timestamp(o)),
             "inventory_count"] *= 1.5
    assert not np.allclose(base, forecasts_at(past, msa, o)["fa_ets_d"])


def test_forecasts_match_stored_builder_output():
    """The refactored function reproduces the pickle the analysis used (if available)."""
    p = os.environ.get("P3_BASELINES_PKL")
    if not p or not os.path.exists(p):
        pytest.skip("set P3_BASELINES_PKL to the builder output to run this check")
    import pickle
    B = pickle.load(open(p, "rb"))
    for msa, o in CASES:
        fa = forecasts_at(PANEL, msa, o)
        for h in (4, 8, 13):
            for k in fa:
                np.testing.assert_allclose(B[(msa, o, h)][k], fa[k][:h], rtol=1e-10, err_msg=f"{msa} {o} {h} {k}")
