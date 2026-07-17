import numpy as np
import pandas as pd
import pytest

from naive_baseline import forecast_naive, assert_no_lookahead


@pytest.fixture
def sample_series():
    dates = pd.date_range("2020-01-04", periods=10, freq="W-SAT")
    return pd.Series(range(100, 110), index=dates, dtype=float)


@pytest.mark.parametrize("horizon", [4, 8, 13])
def test_output_length_matches_horizon(sample_series, horizon):
    forecast = forecast_naive(sample_series, horizon)
    assert len(forecast) == horizon


@pytest.mark.parametrize("horizon", [4, 8, 13])
def test_output_is_constant_repeat_of_last_value(sample_series, horizon):
    forecast = forecast_naive(sample_series, horizon)
    assert np.all(forecast == sample_series.iloc[-1])
    assert len(set(forecast.tolist())) == 1  # truly constant, not just equal on average


def test_assert_no_lookahead_raises_on_future_data(sample_series):
    as_of = sample_series.index[5]  # a date strictly before the series' last date
    with pytest.raises(ValueError, match="lookahead violation"):
        assert_no_lookahead(sample_series, as_of)


def test_assert_no_lookahead_passes_when_series_is_properly_truncated(sample_series):
    as_of = sample_series.index[-1]  # series' own last date -- should not raise
    assert_no_lookahead(sample_series, as_of)  # no exception
