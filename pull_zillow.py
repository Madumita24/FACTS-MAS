"""Pull Zillow's For-Sale Inventory (Smooth, All Homes, Weekly View) dataset
for 15 target MSAs and reshape from wide (one column per week) to long form.
"""
import io

import pandas as pd
import requests

CSV_URL = "https://files.zillowstatic.com/research/public_csvs/invt_fs/Metro_invt_fs_uc_sfrcondo_sm_week.csv"

# Zillow RegionName -> our short MSA label
TARGET_MSAS = {
    "New York, NY": "New York",
    "Los Angeles, CA": "Los Angeles",
    "Chicago, IL": "Chicago",
    "Dallas, TX": "Dallas",
    "Houston, TX": "Houston",
    "Washington, DC": "Washington DC",
    "Miami, FL": "Miami",
    "Philadelphia, PA": "Philadelphia",
    "Atlanta, GA": "Atlanta",
    "Phoenix, AZ": "Phoenix",
    "Boston, MA": "Boston",
    "San Francisco, CA": "San Francisco",
    "Riverside, CA": "Riverside",
    "Detroit, MI": "Detroit",
    "Seattle, WA": "Seattle",
}


def main():
    resp = requests.get(CSV_URL, timeout=60)
    resp.raise_for_status()
    raw = pd.read_csv(io.BytesIO(resp.content))

    df = raw[(raw["RegionType"] == "msa") & (raw["RegionName"].isin(TARGET_MSAS))]

    missing = set(TARGET_MSAS) - set(df["RegionName"])
    if missing:
        print(f"WARNING: requested MSAs not found in Zillow data: {sorted(missing)}")
    else:
        print("All 15 requested MSAs found in Zillow data.")

    date_cols = [c for c in df.columns if c[:4].isdigit()]
    long_df = df.melt(id_vars=["RegionName"], value_vars=date_cols,
                       var_name="date", value_name="inventory_count")
    long_df["msa"] = long_df["RegionName"].map(TARGET_MSAS)
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df = long_df[["date", "msa", "inventory_count"]].sort_values(["msa", "date"])

    long_df.to_csv("zillow_inventory_weekly.csv", index=False)

    print(f"zillow_inventory_weekly.csv written: {len(long_df)} rows, "
          f"{long_df['date'].min().date()} to {long_df['date'].max().date()}")
    print("Missing inventory_count per MSA:")
    print(long_df.groupby("msa")["inventory_count"].apply(lambda s: s.isna().sum()).to_string())


if __name__ == "__main__":
    main()
