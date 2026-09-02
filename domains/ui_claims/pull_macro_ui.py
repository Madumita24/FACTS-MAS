"""Pull macro series from FRED for the UI-claims pilot, resampled to a
Monday-anchored weekly grid.

Adapted from (not imported from) the housing pipeline's pull_macro.py --
copied so this domain's macro pull can never be silently affected by a
change made for housing, or vice versa. The publication-lag VALUES below
are the same real-world facts pull_macro.py uses (they describe when the
Fed/BLS actually release these series, which doesn't change by domain),
re-declared here rather than imported.

Deliberately only FEDFUNDS and CPIAUCSL. UNRATE is excluded on purpose:
initial claims and the unemployment rate are two views of the same labor-
market phenomenon, so including UNRATE as a "macro input" while claims is
the forecast target would make the Macro Agent partly regress the target
on a close proxy of itself -- the circularity flagged when this pilot was
scoped. Mortgage rate is also excluded: it has no first-order relevance to
UI claims the way it does to housing inventory.

Publication-lag handling, unchanged from pull_macro.py: FRED indexes
monthly series by the first day of the reference month, not the date the
value was actually released. Each observation is shifted forward by an
approximate publication lag before being treated as "known", then as-of
joined onto the weekly grid (backward direction only):
  - FEDFUNDS  : +32 days (published in the first few business days of the
                following month)
  - CPIAUCSL  : +42 days (BLS CPI release, ~10th-13th of the following month)
These are calendar approximations, not true ALFRED real-time vintages --
same simplifying-assumption flag pull_macro.py carries.
"""
import os

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

GRID_START = "2012-12-31"  # a Monday -- see note below
FETCH_START = "2012-09-01"  # buffer so monthly series are already available by GRID_START
WEEKLY_ANCHOR = "W-MON"  # each row date is a Monday

# GRID_START is 2012-12-31, not the pilot's nominal 2013-01-01, and that is
# deliberate: 2013-01-01 is a Tuesday, so a W-MON grid anchored there would
# not produce its first Monday until 2013-01-07 -- two days AFTER
# ui_claims_weekly.csv's first Saturday (2013-01-05). align_data_ui.py's
# backward merge_asof would then find no macro row at or before that first
# claims date and leave fed_funds/cpi NaN for it, in every state. Starting
# the grid on the Monday on/before the pilot's true start absorbs that
# instead of leaving a silent gap in row 1.

PUBLICATION_LAG_DAYS = {
    "FEDFUNDS": 32,
    "CPIAUCSL": 42,
}

SERIES = {
    "FEDFUNDS": "fed_funds",
    "CPIAUCSL": "cpi",
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
    fred = Fred(api_key=load_api_key())

    end_date = pd.Timestamp.today().normalize()
    grid = pd.DataFrame({"date": pd.date_range(GRID_START, end_date, freq=WEEKLY_ANCHOR)})

    out = grid.copy()
    for series_id, col_name in SERIES.items():
        series_df = fetch_series(fred, series_id)
        out[col_name] = asof_to_grid(grid, series_df, col_name)

    out_path = os.path.join(os.path.dirname(__file__), "macro_weekly_ui.csv")
    out.to_csv(out_path, index=False)

    print(f"macro_weekly_ui.csv written: {len(out)} rows, "
          f"{out['date'].min().date()} to {out['date'].max().date()}")
    print("Missing values per column:")
    print(out.isna().sum().to_string())
    print(out.head())


if __name__ == "__main__":
    main()
