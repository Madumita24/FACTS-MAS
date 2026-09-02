"""Pull weekly initial unemployment insurance claims (DOL ETA 539, mirrored
by FRED) for 15 target states.

Access method, confirmed live against the real API before writing this
(not assumed from docs): FRED hosts each state's ETA 539 initial-claims
series under the ID pattern `{STATE}ICLAIMS` (e.g. CAICLAIMS, TXICLAIMS) --
Number, Not Seasonally Adjusted, Weekly, ending Saturday. This is a mirror
of the same DOL source oui.doleta.gov itself publishes, reached the same
way pull_macro.py already reaches FEDFUNDS/CPIAUCSL/UNRATE: no scraping,
no new credential, same FRED_API_KEY and fredapi library already in this
repo's environment.

History depth: every target state's series goes back to 1984-1986, far
deeper than this pilot needs. GRID_START below is a deliberate scope
decision (not a data-availability limit) -- see evaluation_protocol_ui.md
for the full reasoning: recent-enough to be one comparable macro regime,
long enough for a real train/test split, and it deliberately includes the
2020 COVID claims spike rather than excising it (documented there as a
known limitation of this AR+Macro-only pilot, not silently avoided).

Unlike pull_macro.py's exogenous series, this is the forecast TARGET, so
no publication-lag shift is applied here -- that machinery exists to keep
a forecast origin from seeing a macro value before it was public, and
doesn't apply to a target series read at its own true date. Weekly claims
do carry a short real-world release lag (~5 days, Thursday release for the
week ending the prior Saturday), noted for completeness, not modeled here.
"""
import os

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

GRID_START = "2013-01-01"

# Full name (matches this pilot's "state" column convention, same spirit as
# pull_zillow.py's short MSA labels) -> FRED series ID.
TARGET_STATES = {
    "California": "CAICLAIMS",
    "Texas": "TXICLAIMS",
    "New York": "NYICLAIMS",
    "Florida": "FLICLAIMS",
    "Illinois": "ILICLAIMS",
    "Michigan": "MIICLAIMS",
    "Massachusetts": "MAICLAIMS",
    "Washington": "WAICLAIMS",
    "Georgia": "GAICLAIMS",
    "Arizona": "AZICLAIMS",
    "Colorado": "COICLAIMS",
    "Louisiana": "LAICLAIMS",
    "Wisconsin": "WIICLAIMS",
    "Tennessee": "TNICLAIMS",
    "Oregon": "ORICLAIMS",
}


def load_api_key() -> str:
    load_dotenv()
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY not found in environment or .env file")
    return key


def main():
    fred = Fred(api_key=load_api_key())

    frames = []
    missing = []
    for state, series_id in TARGET_STATES.items():
        s = fred.get_series(series_id, observation_start=GRID_START).dropna()
        if s.empty:
            missing.append(state)
            continue
        df = s.rename("claims_count").to_frame()
        df.index.name = "date"
        df = df.reset_index()
        df["state"] = state
        frames.append(df)

    if missing:
        print(f"WARNING: no data returned for: {missing}")
    else:
        print("All 15 requested states found in FRED's ETA 539 mirror.")

    long_df = pd.concat(frames, ignore_index=True)
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df = long_df[["date", "state", "claims_count"]].sort_values(["state", "date"])

    out_path = os.path.join(os.path.dirname(__file__), "ui_claims_weekly.csv")
    long_df.to_csv(out_path, index=False)

    print(f"ui_claims_weekly.csv written: {len(long_df)} rows, "
          f"{long_df['date'].min().date()} to {long_df['date'].max().date()}")
    print("Rows per state:")
    print(long_df.groupby("state").size().to_string())
    print("Missing claims_count per state:")
    print(long_df.groupby("state")["claims_count"].apply(lambda s: s.isna().sum()).to_string())


if __name__ == "__main__":
    main()
