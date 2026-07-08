"""Pull macro series from FRED and resample to a Monday-anchored weekly grid.

Publication-lag handling: FRED indexes monthly series (FEDFUNDS, CPIAUCSL,
UNRATE) by the first day of the reference month, not the date the value was
actually released. To avoid lookahead leakage, each monthly observation is
shifted forward by an approximate publication lag before it is treated as
"known", then as-of joined onto the weekly grid (backward direction only):
  - FEDFUNDS  : +32 days (published in the first few business days of the
                following month)
  - UNRATE    : +35 days (BLS employment situation report, first Friday of
                the following month)
  - CPIAUCSL  : +42 days (BLS CPI release, ~10th-13th of the following month)
These are calendar approximations based on typical release schedules, not
true ALFRED real-time vintages -- flag as a simplifying assumption if exact
point-in-time data is needed later.

MORTGAGE30US is already a weekly release, so it is as-of joined directly
with no added lag.
"""
import os

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

GRID_START = "2018-01-01"
FETCH_START = "2017-09-01"  # buffer so monthly series are already available by GRID_START
WEEKLY_ANCHOR = "W-MON"  # each row date is a Monday

PUBLICATION_LAG_DAYS = {
    "FEDFUNDS": 32,
    "UNRATE": 35,
    "CPIAUCSL": 42,
}

SERIES = {
    "MORTGAGE30US": "mortgage_rate",
    "FEDFUNDS": "fed_funds",
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment",
}


def load_api_key() -> str:
    load_dotenv()
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY not found in environment or .env file")
    return key


def fetch_series(fred: Fred, series_id: str) -> pd.DataFrame:
    s = fred.get_series(series_id, observation_start=FETCH_START)
    df = s.rename("value").to_frame()
    df.index.name = "obs_date"
    df = df.reset_index().dropna(subset=["value"])
    lag = PUBLICATION_LAG_DAYS.get(series_id, 0)
    df["available_date"] = df["obs_date"] + pd.Timedelta(days=lag)
    return df.sort_values("available_date")


def asof_to_grid(grid: pd.DataFrame, series_df: pd.DataFrame, col_name: str) -> pd.Series:
    merged = pd.merge_asof(
        grid, series_df[["available_date", "value"]],
        left_on="date", right_on="available_date", direction="backward",
    )
    return merged["value"].rename(col_name)


def main():
    key = load_api_key()
    fred = Fred(api_key=key)

    end_date = pd.Timestamp.today().normalize()
    grid = pd.DataFrame({"date": pd.date_range(GRID_START, end_date, freq=WEEKLY_ANCHOR)})

    out = grid.copy()
    for series_id, col_name in SERIES.items():
        series_df = fetch_series(fred, series_id)
        out[col_name] = asof_to_grid(grid, series_df, col_name)

    out.to_csv("macro_weekly.csv", index=False)

    print(f"macro_weekly.csv written: {len(out)} rows, "
          f"{out['date'].min().date()} to {out['date'].max().date()}")
    print("Missing values per column:")
    print(out.isna().sum().to_string())
    print(out.head())


if __name__ == "__main__":
    main()
