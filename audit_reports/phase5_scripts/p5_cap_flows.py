"""
Phase 5 (P5a data): cap the three Zillow flow files (smoothed, weekly, metro) at 2026-02-21.

Header read first; only date columns <= cap are loaded (usecols). Same 15 MSAs. Long format
(date, msa, new_listings, new_pending, price_cut_share). Reports the first non-missing date per
MSA and series, forward-fills interior gaps within MSA (logged), never back-fills.

Run with -I and explicit paths (downloaded data is untrusted):
  python -I audit_reports/phase5_scripts/p5_cap_flows.py <new_listings.csv> <new_pending.csv> <price_cut.csv> <out.csv>
"""
import csv
import hashlib
import sys

import pandas as pd

CAP = pd.Timestamp("2026-02-21")
FOLD1_TRAIN_START = pd.Timestamp("2019-02-02")
TARGET = {
    "New York, NY": "New York", "Los Angeles, CA": "Los Angeles", "Chicago, IL": "Chicago",
    "Dallas, TX": "Dallas", "Houston, TX": "Houston", "Washington, DC": "Washington DC",
    "Miami, FL": "Miami", "Philadelphia, PA": "Philadelphia", "Atlanta, GA": "Atlanta",
    "Phoenix, AZ": "Phoenix", "Boston, MA": "Boston", "San Francisco, CA": "San Francisco",
    "Riverside, CA": "Riverside", "Detroit, MI": "Detroit", "Seattle, WA": "Seattle",
}
names = ["new_listings", "new_pending", "price_cut_share"]
frames = []
for name, src in zip(names, sys.argv[1:4]):
    print(f"== {name}: sha256 {hashlib.sha256(open(src, 'rb').read()).hexdigest()}")
    with open(src, newline="") as fh:
        header = next(csv.reader(fh))
    dc = [c for c in header if c[:4].isdigit()]
    keep = [c for c in dc if pd.Timestamp(c) <= CAP]
    print(f"   date columns {len(dc)}, kept {len(keep)} ({keep[0]}..{keep[-1]}), {len(dc) - len(keep)} after cap never parsed")
    raw = pd.read_csv(src, usecols=["RegionName", "RegionType"] + keep)
    raw = raw[(raw.RegionType == "msa") & raw.RegionName.isin(TARGET)]
    miss = set(TARGET) - set(raw.RegionName)
    assert not miss, f"{name}: missing MSAs {miss}"
    lg = raw.melt(id_vars=["RegionName"], value_vars=keep, var_name="date", value_name=name)
    lg["msa"] = lg.RegionName.map(TARGET)
    lg["date"] = pd.to_datetime(lg.date)
    lg = lg[["date", "msa", name]].sort_values(["msa", "date"])
    first = lg.dropna(subset=[name]).groupby("msa").date.min()
    late = first[first > FOLD1_TRAIN_START]
    print(f"   first non-missing date: earliest {first.min().date()}, latest {first.max().date()}")
    print(f"   MSAs starting after {FOLD1_TRAIN_START.date()}: {late.dt.date.to_dict() if len(late) else 'none'}")
    interior = lg.groupby("msa")[name].apply(lambda s: int(s.loc[s.first_valid_index():].isna().sum()))
    print(f"   interior missing values to forward-fill: {interior[interior > 0].to_dict() or 'none'}")
    lg[name] = lg.groupby("msa")[name].ffill()
    frames.append(lg.set_index(["date", "msa"]))
out = pd.concat(frames, axis=1).reset_index()
assert out.date.max() <= CAP
out.to_csv(sys.argv[4], index=False)
print("written", sys.argv[4], len(out), "rows; sha256", hashlib.sha256(open(sys.argv[4], "rb").read()).hexdigest())
