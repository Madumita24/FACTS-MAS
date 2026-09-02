"""Align weekly UI claims with national macro data for the ui_claims pilot.

Adapted from (not imported from) the housing pipeline's align_data.py --
copied so a change to one domain's join logic can never silently affect the
other. The lookahead-safety discipline is identical:

ui_claims_weekly.csv is Saturday-anchored (FRED's native ETA 539 convention,
confirmed in the Step 1.3 sanity check); macro_weekly_ui.csv is Monday-
anchored (pull_macro_ui.py's W-MON grid) -- same day-of-week pairing the
housing pipeline has between Zillow (Saturday) and macro_weekly.csv
(Monday). Claims is treated as the canonical weekly grid (it's the forecast
target) and each claims row is matched via merge_asof (direction=
"backward") to the most recent macro row on or before that date, so a
2013-01-05 claims row can only ever see a macro value dated 2013-01-05 or
earlier -- never a later one. pull_macro_ui.py's GRID_START was moved back
to 2012-12-31 specifically so this join has no leading-row gap (see that
file's own note); confirmed empirically here (see "still missing" check
below) rather than assumed.

Macro data is national-level and is broadcast identically across all 15
states for a given date, same as the housing pipeline broadcasts across
MSAs.

Missing claims_count values (gaps in FRED's series, none found in the raw
pull as of this writing but handled the same way regardless) are forward-
filled per state using that state's most recent prior week only -- never
across states, never using a future value. Every filled row is logged to
filled_rows_log_ui.csv for audit, same discipline as align_data.py.
"""
import os

import pandas as pd

HERE = os.path.dirname(__file__)
OUTPUT_COLUMNS = ["date", "state", "claims_count", "fed_funds", "cpi"]


def main():
    macro = pd.read_csv(os.path.join(HERE, "macro_weekly_ui.csv"), parse_dates=["date"]).sort_values("date")
    claims = pd.read_csv(os.path.join(HERE, "ui_claims_weekly.csv"), parse_dates=["date"]).sort_values("date")

    merged = pd.merge_asof(claims, macro, on="date", direction="backward")
    merged = merged[OUTPUT_COLUMNS].sort_values(["state", "date"]).reset_index(drop=True)

    # Forward-fill missing claims_count within each state only, using
    # strictly prior weeks for that same state (no cross-state fill, no
    # future values) -- identical discipline to align_data.py.
    was_missing = merged["claims_count"].isna()
    merged["claims_count"] = merged.groupby("state")["claims_count"].ffill()

    filled_log = merged.loc[was_missing & merged["claims_count"].notna(), ["date", "state", "claims_count"]]
    filled_log = filled_log.rename(columns={"claims_count": "filled_value"})
    filled_log.to_csv(os.path.join(HERE, "filled_rows_log_ui.csv"), index=False)

    still_missing_claims = merged["claims_count"].isna().sum()
    still_missing_macro = merged[["fed_funds", "cpi"]].isna().sum().sum()

    out_path = os.path.join(HERE, "aligned_weekly_ui_claims.csv")
    merged.to_csv(out_path, index=False)

    print(f"aligned_weekly_ui_claims.csv written: {len(merged)} rows, "
          f"{merged['date'].min().date()} to {merged['date'].max().date()}")
    print(f"filled_rows_log_ui.csv written: {len(filled_log)} rows forward-filled")
    if still_missing_claims:
        print(f"WARNING: {still_missing_claims} rows still missing claims_count "
              f"(no prior-week value available within their state)")
    if still_missing_macro:
        print(f"WARNING: {still_missing_macro} macro cells (fed_funds/cpi) still missing "
              f"-- check pull_macro_ui.py's GRID_START against the earliest claims date")
    print("Missing values per column (after fill):")
    print(merged.isna().sum().to_string())
    print(merged.head(10))


if __name__ == "__main__":
    main()
