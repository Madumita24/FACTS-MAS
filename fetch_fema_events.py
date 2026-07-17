"""Fetch FEMA disaster declarations for the counties belonging to our 15
MSAs, and mark which weeks (in aligned_weekly.csv's date grid) each
declaration was active for.

Reuses the exact same Census county<->CBSA delineation crosswalk built for
hazard_index in Phase 1 (pull_intrinsic.py's fetch_county_to_msa_crosswalk),
rather than rebuilding it.

Output is long-format, one row per (msa, week, active declaration) --
sparse, since most (msa, week) pairs have zero active declarations. This
also directly supports the caching design in run_event_agent.py: a
declaration's identity (femaDeclarationString) is preserved per row, so a
years-long declaration like a COVID-19 disaster (2020-01-20 to 2023-05-11 in
the sample checked -- ~170 weeks) is trivially groupable back to a single
LLM interpretation rather than one call per week. Not filtering out
unusually long-duration declarations here; that's a downstream judgment
call, documented rather than silently made.
"""
import requests
import pandas as pd

from pull_intrinsic import fetch_county_to_msa_crosswalk

API_URL = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
PAGE_SIZE = 1000


def fetch_state_declarations(state_fips: str, start_date: str) -> list:
    records = []
    skip = 0
    while True:
        params = {
            "$filter": f"fipsStateCode eq '{state_fips}' and declarationDate ge '{start_date}'",
            "$top": PAGE_SIZE,
            "$skip": skip,
            "$format": "json",
        }
        resp = requests.get(API_URL, params=params, timeout=60)
        resp.raise_for_status()
        batch = resp.json()["DisasterDeclarationsSummaries"]
        records.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        skip += PAGE_SIZE
    return records


def build_weekly_grid(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(df["date"].unique()))


def main():
    df = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
    weekly_grid = build_weekly_grid(df)
    range_start, range_end = weekly_grid.min(), weekly_grid.max()

    county_to_msa = fetch_county_to_msa_crosswalk()
    states = sorted({fips[:2] for fips in county_to_msa})
    print(f"Fetching FEMA declarations for {len(states)} states covering {len(county_to_msa)} counties...")

    rows = []
    for state_fips in states:
        declarations = fetch_state_declarations(state_fips, str(range_start.date() - pd.Timedelta(weeks=52)))

        for rec in declarations:
            fips = f"{rec['fipsStateCode']}{rec['fipsCountyCode']}"
            msa = county_to_msa.get(fips)
            if msa is None:
                continue  # county not in one of our 15 target MSAs

            begin = pd.Timestamp(rec["incidentBeginDate"] or rec["declarationDate"]).tz_localize(None)
            end = pd.Timestamp(rec["incidentEndDate"] or rec["declarationDate"]).tz_localize(None)
            if end < begin:
                end = begin

            active_weeks = weekly_grid[(weekly_grid >= begin) & (weekly_grid <= end)
                                        & (weekly_grid >= range_start) & (weekly_grid <= range_end)]
            for week in active_weeks:
                rows.append({
                    "msa": msa,
                    "week": week.date(),
                    "declaration_id": rec["femaDeclarationString"],
                    "disaster_number": rec["disasterNumber"],
                    "incident_type": rec["incidentType"],
                    "declaration_title": rec["declarationTitle"],
                    "designated_area": rec["designatedArea"],
                    "incident_begin_date": begin.date(),
                    "incident_end_date": end.date(),
                    "declaration_date": pd.Timestamp(rec["declarationDate"]).date(),
                    # Severity-relevant fields, as actually present in the FEMA record --
                    # nothing fabricated. declarationType: DR (Major Disaster, generally
                    # more severe) vs EM (Emergency, generally less severe). The four
                    # ProgramDeclared flags indicate which FEMA assistance programs were
                    # actually activated -- a real proxy for severity, not a guess.
                    "declaration_type": rec.get("declarationType"),
                    "ih_program_declared": rec.get("ihProgramDeclared"),
                    "ia_program_declared": rec.get("iaProgramDeclared"),
                    "pa_program_declared": rec.get("paProgramDeclared"),
                    "hm_program_declared": rec.get("hmProgramDeclared"),
                })

    out = pd.DataFrame(rows).drop_duplicates(subset=["msa", "week", "declaration_id"])
    out.to_csv("fema_events_weekly.csv", index=False)

    print(f"\nfema_events_weekly.csv written: {len(out)} rows")
    print(f"Unique declarations: {out['declaration_id'].nunique()}")
    print(f"MSAs with at least one active-week declaration: {out['msa'].nunique()}/15")
    print("\nRows per MSA:")
    print(out.groupby("msa").size().sort_values(ascending=False).to_string())
    print("\nLongest-duration declarations (by active week count):")
    print(out.groupby(["msa", "declaration_id", "declaration_title"]).size()
          .sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
