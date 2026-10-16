"""
Phase 5, P5a (Addendum A item 5): the listing-flow channel must use only inventory on or before the
origin and flow values dated on or before (origin - 2 weeks).
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
sys.path.insert(0, os.path.join(ac.REPO, "audit_reports", "phase5_scripts"))
from p5_flow_channel import LAG_WEEKS, flow_forecasts, load_flows  # noqa: E402

PANEL = ac.load_panel()
FLOWS = load_flows()
ORIGINS = [datetime.date(2021, 5, 1), datetime.date(2023, 12, 2)]


@pytest.mark.parametrize("o", ORIGINS)
def test_flow_channel_identical_on_truncated_data(o):
    full, _ = flow_forecasts(PANEL, FLOWS, o)
    ots = pd.Timestamp(o)
    trunc, _ = flow_forecasts(PANEL[PANEL.date <= ots], FLOWS[FLOWS.date <= ots - pd.Timedelta(weeks=LAG_WEEKS)], o)
    for m in full:
        np.testing.assert_array_equal(full[m], trunc[m], err_msg=m)


def test_flow_channel_negative_control():
    o = datetime.date(2023, 12, 2)
    ots = pd.Timestamp(o)
    base, _ = flow_forecasts(PANEL, FLOWS, o)
    after = FLOWS.copy()
    m = after.date > ots - pd.Timedelta(weeks=LAG_WEEKS)      # inside the lag window and later
    after.loc[m, ["new_listings", "new_pending", "price_cut_share"]] *= 5.0
    a, _ = flow_forecasts(PANEL, after, o)
    for k in base:
        np.testing.assert_array_equal(base[k], a[k])
    before = FLOWS.copy()
    m = (before.msa == "Phoenix") & (before.date == ots - pd.Timedelta(weeks=LAG_WEEKS))
    assert m.sum() == 1
    before.loc[m, "new_listings"] *= 3.0
    b, _ = flow_forecasts(PANEL, before, o)
    assert not np.allclose(base["Phoenix"], b["Phoenix"])
