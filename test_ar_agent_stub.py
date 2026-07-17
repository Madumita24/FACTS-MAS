import numpy as np
import pandas as pd

from ar_agent import run_ar_agent
from ar_agent_stub import run_ar_agent_stub


def test_stub_schema_exactly_matches_real_agent():
    dates = pd.date_range("2019-01-05", periods=150, freq="W-SAT")
    df = pd.DataFrame({
        "date": list(dates) * 2,
        "msa": ["A"] * 150 + ["B"] * 150,
        "inventory_count": list(10000 + np.arange(150) * 10) * 2,
    })
    fold_boundaries = pd.DataFrame([{
        "fold": 1,
        "train_start": dates[0].date(),
        "train_end": dates[100].date(),
        "train_weeks": 101,
        "embargo_end": dates[113].date(),
        "test_start": dates[114].date(),
        "test_end": dates[130].date(),
    }])

    real_out = run_ar_agent(df, fold_boundaries, horizons=[4])
    stub_out = run_ar_agent_stub(df, fold_boundaries, horizons=[4])

    assert list(stub_out.columns) == list(real_out.columns)
    assert list(stub_out.dtypes) == list(real_out.dtypes)
