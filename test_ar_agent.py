import numpy as np
import pandas as pd
import pytest

from ar_agent import fit_and_forecast
from naive_baseline import run_naive_baseline
from ar_agent import run_ar_agent


@pytest.fixture
def sample_series():
    # A trending series long enough for ETS to fit stably.
    dates = pd.date_range("2019-01-05", periods=150, freq="W-SAT")
    values = 10000 + np.arange(150) * 15 + np.sin(np.arange(150) / 5) * 50
    return pd.Series(values, index=dates, dtype=float)


@pytest.fixture
def declining_series():
    # Sharply declining series, deliberately chosen to push an additive-trend
    # ETS forecast toward/below zero, to exercise the non-negativity clip.
    dates = pd.date_range("2019-01-05", periods=60, freq="W-SAT")
    values = np.linspace(5000, 50, 60)
    return pd.Series(values, index=dates, dtype=float)


@pytest.mark.parametrize("horizon", [4, 8, 13])
def test_output_length_matches_horizon(sample_series, horizon):
    result = fit_and_forecast(sample_series, horizon)
    assert len(result["point_forecast"]) == horizon


def test_no_negative_values(declining_series):
    result = fit_and_forecast(declining_series, 13)
    assert all(v >= 0 for v in result["point_forecast"])


def test_schema_matches_spec(sample_series):
    result = fit_and_forecast(sample_series, 4)
    assert result["model_name"] == "ets"
    assert isinstance(result["point_forecast"], list)
    assert result["prediction_intervals"] is None


def test_no_lookahead_guard_raises():
    dates = pd.date_range("2020-01-04", periods=20, freq="W-SAT")
    series = pd.Series(range(100, 120), index=dates, dtype=float)
    train_end = dates[10]  # strictly before the series' last date

    from naive_baseline import assert_no_lookahead
    with pytest.raises(ValueError, match="lookahead violation"):
        assert_no_lookahead(series, train_end)


def test_output_schema_matches_naive_baseline():
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

    naive_out = run_naive_baseline(df, fold_boundaries, horizons=[4])
    ar_out = run_ar_agent(df, fold_boundaries, horizons=[4])

    assert list(ar_out.columns) == list(naive_out.columns)
