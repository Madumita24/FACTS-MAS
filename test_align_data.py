"""Tests for align_data.py's merge + per-MSA forward-fill logic.

Uses small synthetic macro/zillow CSVs (no network calls) written into a
temp directory, since align_data.main() reads/writes fixed filenames in the
current working directory.
"""
import pandas as pd
import pytest

import align_data


@pytest.fixture
def synthetic_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Monday-anchored macro grid, 5 weeks.
    macro = pd.DataFrame({
        "date": pd.to_datetime([
            "2021-01-04", "2021-01-11", "2021-01-18", "2021-01-25", "2021-02-01",
        ]),
        "mortgage_rate": [2.7, 2.7, 2.8, 2.8, 2.9],
        "fed_funds": [0.1, 0.1, 0.1, 0.1, 0.1],
        "cpi": [261.5, 261.5, 261.9, 261.9, 262.2],
        "unemployment": [6.3, 6.3, 6.3, 6.2, 6.2],
    })
    macro.to_csv(tmp_path / "macro_weekly.csv", index=False)

    # Two MSAs, Saturday-anchored, 4 weeks each.
    # MSA A: gap in week 3 (fillable from week 2).
    # MSA B: gap in week 1 -- the very first observation, so it is
    # unfillable (no prior week exists) and must stay NaN.
    zillow = pd.DataFrame({
        "date": pd.to_datetime([
            "2021-01-09", "2021-01-16", "2021-01-23", "2021-01-30",  # MSA A
            "2021-01-09", "2021-01-16", "2021-01-23", "2021-01-30",  # MSA B
        ]),
        "msa": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "inventory_count": [100.0, 105.0, None, 110.0, None, 200.0, 205.0, 210.0],
    })
    zillow.to_csv(tmp_path / "zillow_inventory_weekly.csv", index=False)

    return tmp_path


def test_fillable_gap_uses_same_msa_prior_week(synthetic_inputs):
    align_data.main()
    out = pd.read_csv(synthetic_inputs / "aligned_weekly.csv", parse_dates=["date"])

    row = out[(out["msa"] == "A") & (out["date"] == "2021-01-23")].iloc[0]
    assert row["inventory_count"] == 105.0  # carried forward from A's 2021-01-16, not B's value


def test_no_cross_msa_leakage(synthetic_inputs):
    align_data.main()
    out = pd.read_csv(synthetic_inputs / "aligned_weekly.csv", parse_dates=["date"])

    # MSA B's real values must be untouched by MSA A's fill and vice versa.
    b_values = out[out["msa"] == "B"].sort_values("date")["inventory_count"].tolist()
    assert b_values[1:] == [200.0, 205.0, 210.0]


def test_unfillable_leading_gap_stays_nan_and_is_not_logged(synthetic_inputs):
    align_data.main()
    out = pd.read_csv(synthetic_inputs / "aligned_weekly.csv", parse_dates=["date"])
    log = pd.read_csv(synthetic_inputs / "filled_rows_log.csv", parse_dates=["date"])

    b_first = out[(out["msa"] == "B") & (out["date"] == "2021-01-09")].iloc[0]
    assert pd.isna(b_first["inventory_count"])
    assert not ((log["msa"] == "B") & (log["date"] == pd.Timestamp("2021-01-09"))).any()


def test_log_records_exactly_the_filled_row(synthetic_inputs):
    align_data.main()
    log = pd.read_csv(synthetic_inputs / "filled_rows_log.csv", parse_dates=["date"])

    assert len(log) == 1
    assert log.iloc[0]["msa"] == "A"
    assert log.iloc[0]["date"] == pd.Timestamp("2021-01-23")
    assert log.iloc[0]["filled_value"] == 105.0


def test_macro_join_never_uses_a_future_row(synthetic_inputs):
    align_data.main()
    out = pd.read_csv(synthetic_inputs / "aligned_weekly.csv", parse_dates=["date"])

    # Zillow 2021-01-09 (Sat) must attach the macro row dated 2021-01-04
    # (Mon, <= date), never 2021-01-11 or later.
    row = out[(out["msa"] == "A") & (out["date"] == "2021-01-09")].iloc[0]
    assert row["mortgage_rate"] == 2.7
    assert row["cpi"] == 261.5
