"""
Phase 3 (A2): turn a downloaded Zillow wide CSV into a capped long panel.

Reads the header only, keeps date columns <= 2026-02-21, and loads just those
columns (usecols), so no value after the cap is parsed. Builds the same panel
as aligned_weekly.csv (same 15 MSAs, same date grid, same macro columns),
with inventory_count replaced by the downloaded series, forward-filled within
MSA (as align_data.py does). Logs every filled row.

Run with -I and explicit paths (downloaded data is untrusted):
  python -I audit_reports/phase3_scripts/p3_cap_zillow.py <download.csv> <aligned_weekly.csv> <out_long.csv>
"""
import csv
import hashlib
import sys

import pandas as pd

CAP = pd.Timestamp("2026-02-21")
TARGET = {
    "New York, NY": "New York", "Los Angeles, CA": "Los Angeles", "Chicago, IL": "Chicago",
    "Dallas, TX": "Dallas", "Houston, TX": "Houston", "Washington, DC": "Washington DC",
    "Miami, FL": "Miami", "Philadelphia, PA": "Philadelphia", "Atlanta, GA": "Atlanta",
    "Phoenix, AZ": "Phoenix", "Boston, MA": "Boston", "San Francisco, CA": "San Francisco",
    "Riverside, CA": "Riverside", "Detroit, MI": "Detroit", "Seattle, WA": "Seattle",
}

src, aligned_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
print("sha256", hashlib.sha256(open(src, "rb").read()).hexdigest())
with open(src, newline="") as fh:
    header = next(csv.reader(fh))
date_cols = [c for c in header if c[:4].isdigit()]
keep = [c for c in date_cols if pd.Timestamp(c) <= CAP]
dropped = len(date_cols) - len(keep)
print(f"date columns: {len(date_cols)} total, {len(keep)} kept (<= {CAP.date()}), {dropped} after cap never parsed")
print("kept range:", keep[0], "..", keep[-1])
raw = pd.read_csv(src, usecols=["RegionName", "RegionType"] + keep)
raw = raw[(raw.RegionType == "msa") & raw.RegionName.isin(TARGET)]
missing = set(TARGET) - set(raw.RegionName)
assert not missing, f"missing MSAs: {missing}"
long = raw.melt(id_vars=["RegionName"], value_vars=keep, var_name="date", value_name="inv")
long["msa"] = long.RegionName.map(TARGET)
long["date"] = pd.to_datetime(long.date)
assert long.date.max() <= CAP

al = pd.read_csv(aligned_path, parse_dates=["date"])
al = al[al.date <= CAP].drop(columns=["inventory_count"])
m = al.merge(long[["date", "msa", "inv"]], on=["date", "msa"], how="left").sort_values(["msa", "date"])
print("grid dates without a value in the download (before fill):", int(m.inv.isna().sum()))
was = m.inv.isna()
m["inventory_count"] = m.groupby("msa")["inv"].ffill()
filled = m[was & m.inventory_count.notna()]
print("forward-filled rows:", len(filled))
print(filled[["date", "msa"]].to_string(index=False) if len(filled) <= 40 else filled.groupby("msa").size())
lead = m[m.inventory_count.isna()]
print("rows still missing (leading gaps):", len(lead), lead.groupby("msa").size().to_dict() if len(lead) else "")
m = m.drop(columns=["inv"])
assert m.date.max() <= CAP
m.to_csv(out, index=False)
print("written", out, len(m), "rows; sha256", hashlib.sha256(open(out, "rb").read()).hexdigest())
