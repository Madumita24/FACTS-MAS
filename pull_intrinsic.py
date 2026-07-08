"""Pull static, MSA-level intrinsic features: Census ACS demographics + a NOAA
storm-hazard index. One row per MSA (these are slow-moving structural
features, not weekly time series).

Data sources
------------
- Population, median household income, total housing units: Census ACS
  2024 5-year estimates (2020-2024 vintage, the most recent available),
  queried directly at CBSA (Core-Based Statistical Area) geography -- no
  county aggregation needed, ACS publishes these totals natively per CBSA.
- Land area (for density): Census 2023 CBSA Gazetteer file (ALAND_SQMI).
  ACS has no land-area field, so density = population / ALAND_SQMI.
- Hazard index: NOAA Storm Events Database (2016-2025, 10 years), summed
  across all counties belonging to each CBSA via the Census 2023
  metro/micro delineation file (county <-> CBSA crosswalk).

Hazard index definition & known limitation
-------------------------------------------
hazard_index = count of storm events with recorded impact (non-empty
property/crop damage, OR direct/indirect injuries, OR direct/indirect
deaths) across the MSA's constituent counties, 2016-2025.

NOAA reports each event against either a county (CZ_TYPE == "C") or an NWS
forecast zone (CZ_TYPE == "Z"), and zones don't map 1:1 to counties. This
pipeline only counts county-reported events (~57% of all records
nationally in a spot check of 2024 data) because a zone-to-county
crosswalk isn't wired up in this version. hazard_index is therefore a
real but partial signal -- likely undercounted for hazard types that are
more commonly zone-reported (e.g. flooding, wind), and this undercount
rate can vary by state/region. Treat as directional, not absolute.

FEMA's National Risk Index was considered as an alternative (it's a
pre-aggregated composite score, which would have been cleaner) but has no
stable programmatic download as of this pipeline run: the data-resources
page 403s to automated fetches and redirects to an interactive JS tool
(RAPT) rather than exposing a static file, and it isn't listed in the
OpenFEMA API catalog. Revisit if FEMA exposes a stable endpoint later.
"""
import io
import re
import zipfile

import pandas as pd
import requests
from dotenv import load_dotenv
import os

ACS_YEAR = 2024
GAZETTEER_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_cbsa_national.zip"
DELINEATION_URL = "https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2023/delineation-files/list1_2023.xlsx"
NOAA_INDEX_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
NOAA_YEARS = range(2016, 2026)  # last 10 full years relative to a 2026 pull

# Verified against the 2023 CBSA Gazetteer file / delineation file (see module
# docstring). Titles reflect the current (post-2023) official CBSA names,
# which differ slightly from older common names (e.g. Houston no longer
# includes "Sugar Land"; San Francisco's CBSA now reads "-Fremont" instead
# of "-Berkeley"); no MSA substitutions were needed, all 15 resolved cleanly.
CBSA_CODES = {
    "New York": "35620",
    "Los Angeles": "31080",
    "Chicago": "16980",
    "Dallas": "19100",
    "Houston": "26420",
    "Washington DC": "47900",
    "Miami": "33100",
    "Philadelphia": "37980",
    "Atlanta": "12060",
    "Phoenix": "38060",
    "Boston": "14460",
    "San Francisco": "41860",
    "Riverside": "40140",
    "Detroit": "19820",
    "Seattle": "42660",
}


def load_census_key() -> str:
    load_dotenv()
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        raise RuntimeError("CENSUS_API_KEY not found in environment or .env file")
    return key


def fetch_acs(key: str) -> pd.DataFrame:
    codes = ",".join(CBSA_CODES.values())
    url = (
        f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5"
        f"?get=NAME,B01003_001E,B19013_001E,B25001_001E"
        f"&for=metropolitan%20statistical%20area/micropolitan%20statistical%20area:{codes}"
        f"&key={key}"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    rows = resp.json()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df = df.rename(columns={
        "B01003_001E": "population",
        "B19013_001E": "median_income",
        "B25001_001E": "housing_units",
        "metropolitan statistical area/micropolitan statistical area": "cbsa_code",
    })
    for col in ["population", "median_income", "housing_units"]:
        df[col] = pd.to_numeric(df[col])
    return df[["cbsa_code", "population", "median_income", "housing_units"]]


def fetch_land_area() -> pd.DataFrame:
    resp = requests.get(GAZETTEER_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            df = pd.read_csv(f, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    df["GEOID"] = df["GEOID"].astype(str)
    return df[["GEOID", "ALAND_SQMI"]].rename(columns={"GEOID": "cbsa_code"})


def fetch_county_to_msa_crosswalk() -> dict:
    resp = requests.get(DELINEATION_URL, timeout=60)
    resp.raise_for_status()
    df = pd.read_excel(io.BytesIO(resp.content), skiprows=2)
    df = df.dropna(subset=["FIPS State Code", "FIPS County Code"])
    df["cbsa_code"] = df["CBSA Code"].astype(int).astype(str)

    code_to_msa = {v: k for k, v in CBSA_CODES.items()}
    df = df[df["cbsa_code"].isin(code_to_msa)]

    df["fips"] = (
        df["FIPS State Code"].astype(int).astype(str).str.zfill(2)
        + df["FIPS County Code"].astype(int).astype(str).str.zfill(3)
    )
    return dict(zip(df["fips"], df["cbsa_code"].map(code_to_msa)))


def list_noaa_detail_files() -> dict:
    resp = requests.get(NOAA_INDEX_URL, timeout=60)
    resp.raise_for_status()
    files = re.findall(r'StormEvents_details-ftp_v1\.0_d(\d{4})_c\d{8}\.csv\.gz', resp.text)
    by_year = {}
    for year in files:
        year_int = int(year)
        if year_int in NOAA_YEARS:
            match = re.search(
                rf'StormEvents_details-ftp_v1\.0_d{year}_c\d{{8}}\.csv\.gz', resp.text
            )
            by_year[year_int] = match.group(0)
    return by_year


def fetch_hazard_counts(county_to_msa: dict) -> pd.Series:
    files = list_noaa_detail_files()
    missing_years = sorted(set(NOAA_YEARS) - set(files))
    if missing_years:
        print(f"WARNING: NOAA storm events files not found for years: {missing_years}")

    counts = {msa: 0 for msa in CBSA_CODES}
    usecols = ["STATE_FIPS", "CZ_TYPE", "CZ_FIPS", "DAMAGE_PROPERTY", "DAMAGE_CROPS",
               "INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT"]

    for year, filename in sorted(files.items()):
        resp = requests.get(NOAA_INDEX_URL + filename, timeout=120)
        resp.raise_for_status()
        df = pd.read_csv(io.BytesIO(resp.content), compression="gzip", usecols=usecols,
                          low_memory=False)

        df = df[df["CZ_TYPE"] == "C"].dropna(subset=["STATE_FIPS", "CZ_FIPS"])
        df["fips"] = (
            df["STATE_FIPS"].astype(int).astype(str).str.zfill(2)
            + df["CZ_FIPS"].astype(int).astype(str).str.zfill(3)
        )
        df["msa"] = df["fips"].map(county_to_msa)
        df = df.dropna(subset=["msa"])

        has_impact = (
            df["DAMAGE_PROPERTY"].notna() & (df["DAMAGE_PROPERTY"].astype(str).str.strip() != "")
            | df["DAMAGE_CROPS"].notna() & (df["DAMAGE_CROPS"].astype(str).str.strip() != "")
            | (df["INJURIES_DIRECT"].fillna(0) > 0)
            | (df["INJURIES_INDIRECT"].fillna(0) > 0)
            | (df["DEATHS_DIRECT"].fillna(0) > 0)
            | (df["DEATHS_INDIRECT"].fillna(0) > 0)
        )
        year_counts = df[has_impact].groupby("msa").size()
        for msa, n in year_counts.items():
            counts[msa] += int(n)
        print(f"  NOAA {year}: {len(df)} county-level events in target MSAs, "
              f"{int(has_impact.sum())} with recorded impact")

    result = pd.Series(counts, name="hazard_index")
    result.index.name = "msa"
    return result


def main():
    key = load_census_key()

    print("Fetching ACS 2024 5-year estimates...")
    acs = fetch_acs(key)

    print("Fetching CBSA land area from Census Gazetteer...")
    land = fetch_land_area()

    msa_df = acs.merge(land, on="cbsa_code", how="left")
    code_to_msa = {v: k for k, v in CBSA_CODES.items()}
    msa_df["msa"] = msa_df["cbsa_code"].map(code_to_msa)
    msa_df["density"] = msa_df["population"] / msa_df["ALAND_SQMI"]

    print("Fetching county-to-MSA crosswalk from Census delineation file...")
    county_to_msa = fetch_county_to_msa_crosswalk()

    print("Fetching NOAA Storm Events data (2016-2025)...")
    hazard = fetch_hazard_counts(county_to_msa)

    out = msa_df.merge(hazard, left_on="msa", right_index=True, how="left").set_index("msa").loc[list(CBSA_CODES)]
    out = out.reset_index()[["msa", "population", "median_income", "housing_units", "density", "hazard_index"]]

    out.to_csv("intrinsic_static.csv", index=False)

    print(f"\nintrinsic_static.csv written: {len(out)} rows")
    print("Missing values per column:")
    print(out.isna().sum().to_string())
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
