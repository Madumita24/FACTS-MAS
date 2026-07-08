"""Align Zillow weekly inventory with national macro data.

Zillow's weekly inventory series is anchored on Saturdays; macro_weekly.csv
is anchored on Mondays (see pull_macro.py). Rather than forcing both onto the
same calendar day, Zillow's date is treated as the canonical weekly grid
(it's the forecast target), and each Zillow row is matched via merge_asof
(direction="backward") to the most recent macro row on or before that date.
This keeps the join lookahead-safe: a Saturday 2018-02-03 inventory row is
matched with the Monday 2018-02-01 (or earlier) macro row, never a later one.

Macro data is national-level and is broadcast identically across all 15
MSAs for a given date.

Missing inventory_count values (gaps in Zillow's own smoothed series) are
forward-filled per MSA using that MSA's most recent prior week only -- never
across MSAs, never using a future value. Every filled row is logged to
filled_rows_log.csv (date, msa, filled_value) for audit.

--- Documented transmission lags (informational only -- NOT applied here;
    lag application happens downstream in the Macro Agent) ---
  mortgage_rate : ~4-8 weeks   -- rate moves affect buyer affordability and
                                  listing decisions with a multi-week lag
  fed_funds     : ~8-12 weeks  -- transmits indirectly, through mortgage
                                  rates and broader credit conditions
  cpi           : ~8-12 weeks  -- inflation affects real income and
                                  affordability gradually
  unemployment  : ~4-6 weeks   -- labor market shifts affect household
                                  formation / forced-sale behavior faster
                                  than the above
These are directional research priors (rough consensus from housing-finance
literature), not fitted values -- treat as a starting point for lag search,
not fixed truth.
"""
import pandas as pd

OUTPUT_COLUMNS = ["date", "msa", "inventory_count", "mortgage_rate", "fed_funds", "cpi", "unemployment"]


def main():
    macro = pd.read_csv("macro_weekly.csv", parse_dates=["date"]).sort_values("date")
    zillow = pd.read_csv("zillow_inventory_weekly.csv", parse_dates=["date"]).sort_values("date")

    merged = pd.merge_asof(zillow, macro, on="date", direction="backward")
    merged = merged[OUTPUT_COLUMNS].sort_values(["msa", "date"]).reset_index(drop=True)

    # Forward-fill missing inventory_count within each MSA only, using strictly
    # prior weeks for that same MSA (no cross-MSA fill, no future values).
    was_missing = merged["inventory_count"].isna()
    merged["inventory_count"] = merged.groupby("msa")["inventory_count"].ffill()

    filled_log = merged.loc[was_missing & merged["inventory_count"].notna(), ["date", "msa", "inventory_count"]]
    filled_log = filled_log.rename(columns={"inventory_count": "filled_value"})
    filled_log.to_csv("filled_rows_log.csv", index=False)

    still_missing = merged["inventory_count"].isna().sum()

    merged.to_csv("aligned_weekly.csv", index=False)

    print(f"aligned_weekly.csv written: {len(merged)} rows, "
          f"{merged['date'].min().date()} to {merged['date'].max().date()}")
    print(f"filled_rows_log.csv written: {len(filled_log)} rows forward-filled")
    if still_missing:
        print(f"WARNING: {still_missing} rows still missing inventory_count "
              f"(no prior-week value available within their MSA)")
    print("Missing values per column (after fill):")
    print(merged.isna().sum().to_string())
    print(merged.head(10))


if __name__ == "__main__":
    main()
