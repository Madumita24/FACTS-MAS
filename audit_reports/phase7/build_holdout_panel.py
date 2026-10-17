"""
Phase 7 data build (FINAL_CONFIG.md section 11 item 2). Separate from the frozen runner.

Real mode (Phase 7 only; requires --phase7-go):
  * downloads fresh Zillow smoothed and raw weekly metro inventory (pull_zillow.py URL and MSA map) and fresh
    FRED series (pull_macro.py: same series, fetch start, publication lags, Monday grid, backward as-of join),
  * archives every download (and each FRED series as fetched) in a new dated directory with sha256,
  * builds the long panels exactly as align_data.py does (backward merge_asof of macro onto Zillow dates,
    forward-fill inventory within MSA, filled rows logged),
  * writes panel_smoothed.csv and panel_raw.csv into the archive directory (never over repo files),
  * REVISION CHECK: every value dated <= 2026-02-21 must equal the capped artifacts
    (inventory: audit_reports/artifacts/zillow_{sm,raw}_vintage20261015_capped20260221.csv;
     macro: aligned_weekly.csv rows <= 2026-02-21). Differences are written to revision_check.csv and
    summarised; they are never hidden or patched.
  The FRED key is read from the environment / .env by pull_macro.load_api_key when the operator runs it.

Self-test (no network, no key, no post-cap data):
  --self-test --zillow-sm <dl.csv> --zillow-raw <dl.csv>
  rebuilds panels from already-downloaded Zillow files reading only columns <= 2026-02-21, with the repo's
  macro_weekly.csv standing in for FRED, and runs the same revision check (expected: zero differences).

Usage (repo root):
  python audit_reports/phase7/build_holdout_panel.py --self-test --zillow-sm <f> --zillow-raw <f>
  python audit_reports/phase7/build_holdout_panel.py --phase7-go [--archive-dir <dir>]
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd  # noqa: E402

import pull_macro  # noqa: E402  (functions only; its main() is never called: it would overwrite macro_weekly.csv)
import pull_zillow  # noqa: E402  (constants only; same reason)

CAP = pd.Timestamp("2026-02-21")
SM_URL = pull_zillow.CSV_URL
RAW_URL = SM_URL.replace("_sm_week.csv", "_week.csv")
OUTPUT_COLUMNS = ["date", "msa", "inventory_count", "mortgage_rate", "fed_funds", "cpi", "unemployment"]
ART = os.path.join("audit_reports", "artifacts")
# First date of the frozen panel (aligned_weekly.csv). Zillow's current files start earlier (smoothed
# 2018-01-27, raw 2018-01-06); keeping those extra early weeks would change AR fits and seasonal factors in the
# holdout validation window relative to the frozen backtest, so rows before this date are dropped (recorded
# in the manifest; flagged to Umang).
GRID0 = pd.Timestamp("2018-02-03")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def zillow_long(path, cap_only):
    with open(path, newline="") as fh:
        header = next(csv.reader(fh))
    dcols = [c for c in header if c[:4].isdigit()]
    if cap_only:
        dcols = [c for c in dcols if pd.Timestamp(c) <= CAP]
    raw = pd.read_csv(path, usecols=["RegionName", "RegionType"] + dcols)
    df = raw[(raw.RegionType == "msa") & raw.RegionName.isin(pull_zillow.TARGET_MSAS)]
    missing = set(pull_zillow.TARGET_MSAS) - set(df.RegionName)
    assert not missing, f"missing MSAs {missing}"
    lg = df.melt(id_vars=["RegionName"], value_vars=dcols, var_name="date", value_name="inventory_count")
    lg["msa"] = lg.RegionName.map(pull_zillow.TARGET_MSAS)
    lg["date"] = pd.to_datetime(lg.date)
    return lg[["date", "msa", "inventory_count"]].sort_values(["msa", "date"])


def align(zillow, macro):
    """align_data.main() logic, returning (panel, filled_log) instead of writing repo files."""
    merged = pd.merge_asof(zillow.sort_values("date"), macro.sort_values("date"), on="date", direction="backward")
    merged = merged[OUTPUT_COLUMNS].sort_values(["msa", "date"]).reset_index(drop=True)
    was = merged["inventory_count"].isna()
    merged["inventory_count"] = merged.groupby("msa")["inventory_count"].ffill()
    filled = merged.loc[was & merged["inventory_count"].notna(), ["date", "msa", "inventory_count"]]
    return merged, filled.rename(columns={"inventory_count": "filled_value"})


def scrub(text, secret):
    """Remove a secret from any text before it can be printed, logged or archived."""
    return str(text).replace(secret, "***") if secret else str(text)


def assert_no_secret(directory, secret):
    """Fail if any file in `directory` contains the secret (FRED key). Never prints the secret."""
    if not secret:
        return 0
    needle = secret.encode()
    n = 0
    for root, _, files in os.walk(directory):
        for f in files:
            n += 1
            if needle in open(os.path.join(root, f), "rb").read():
                raise AssertionError(f"secret found in output file {os.path.join(root, f)} (value not shown)")
    return n


def fred_macro(archive):
    """pull_macro.main() logic, archiving each fetched series; returns (macro frame, key held in memory only).
    The key comes only from pull_macro.load_api_key; any error message is scrubbed of it before re-raising."""
    from fredapi import Fred
    key = pull_macro.load_api_key()
    try:
        fred = Fred(api_key=key)
        end = pd.Timestamp.today().normalize()
        grid = pd.DataFrame({"date": pd.date_range(pull_macro.GRID_START, end, freq=pull_macro.WEEKLY_ANCHOR)})
        out = grid.copy()
        for sid, col in pull_macro.SERIES.items():
            sdf = pull_macro.fetch_series(fred, sid)
            sdf.to_csv(os.path.join(archive, f"fred_{sid}.csv"), index=False)
            out[col] = pull_macro.asof_to_grid(grid, sdf, col)
    except Exception as e:  # never let the key reach a traceback
        raise RuntimeError(f"FRED fetch failed: {type(e).__name__}: {scrub(e, key)}") from None
    return out, key


def revision_check(panel, capped_path, label):
    ref = pd.read_csv(capped_path, parse_dates=["date"])
    new = panel[panel.date <= CAP]
    m = ref.merge(new, on=["date", "msa"], how="outer", suffixes=("_ref", "_new"), indicator=True)
    rows = []
    only = m[m._merge != "both"]
    for r in only.itertuples():
        rows.append(dict(panel=label, date=r.date.date(), msa=r.msa, column="(row)", ref=r._merge, new=""))
    both = m[m._merge == "both"]
    for col in OUTPUT_COLUMNS[2:]:
        a, b = both[f"{col}_ref"], both[f"{col}_new"]
        diff = ~((a == b) | (a.isna() & b.isna()))
        for r in both[diff].itertuples():
            rows.append(dict(panel=label, date=r.date.date(), msa=r.msa, column=col,
                             ref=getattr(r, f"{col}_ref"), new=getattr(r, f"{col}_new")))
    return pd.DataFrame(rows, columns=["panel", "date", "msa", "column", "ref", "new"])


def inventory_materiality(rc, panels):
    """Count and size of pre-cap inventory changes per panel; 'material' if more than 1% of pre-cap inventory
    cells change by more than 0.5% (FINAL_CONFIG section 11). Macro differences are reported only."""
    out = {}
    for label, path in panels.items():
        n_cells = int((pd.read_csv(path, parse_dates=["date"]).date <= CAP).sum())
        d = rc[(rc.panel == label) & (rc.column == "inventory_count")].copy()
        rel = (pd.to_numeric(d.new, errors="coerce") / pd.to_numeric(d.ref, errors="coerce") - 1).abs()
        big = int((rel > 0.005).sum())
        out[label] = {"pre_cap_inventory_cells": n_cells, "changed": int(len(d)), "changed_gt_0.5pct": big,
                      "share_changed_gt_0.5pct": big / max(n_cells, 1),
                      "median_abs_rel_change": float(rel.median()) if len(rel) else 0.0,
                      "max_abs_rel_change": float(rel.max()) if len(rel) else 0.0,
                      "MATERIAL": bool(big / max(n_cells, 1) > 0.01)}
    return out


def build(archive, sm_path, raw_path, macro, cap_only):
    manifest = {"built_at": datetime.datetime.now().isoformat(timespec="seconds"), "cap_only": cap_only,
                "grid_start_rule": f"rows before {GRID0.date()} dropped (frozen panel's first date)", "files": {}}
    outs = {}
    checks = []
    for label, path, capped in (("smoothed", sm_path, "zillow_sm_vintage20261015_capped20260221.csv"),
                                ("raw", raw_path, "zillow_raw_vintage20261015_capped20260221.csv")):
        zl = zillow_long(path, cap_only)
        manifest[f"{label}_zillow_first_date_in_file"] = str(zl.date.min().date())
        zl = zl[zl.date >= GRID0]
        panel, filled = align(zl, macro)
        p = os.path.join(archive, f"panel_{label}.csv")
        panel.to_csv(p, index=False)
        filled.to_csv(os.path.join(archive, f"filled_rows_{label}.csv"), index=False)
        manifest["files"][f"panel_{label}.csv"] = sha(p)
        manifest[f"{label}_last_date"] = str(panel.date.max().date())
        checks.append(revision_check(panel, os.path.join(ART, capped), label))
        outs[label] = p
    rc = pd.concat(checks)
    rc.to_csv(os.path.join(archive, "revision_check.csv"), index=False)
    manifest["revision_differences"] = {f"{k[0]}|{k[1]}": int(v) for k, v in rc.groupby(["panel", "column"]).size().items()} if len(rc) else {}
    manifest["revision_differences_total"] = int(len(rc))
    manifest["inventory_revision_summary"] = inventory_materiality(rc, outs)
    # FINAL_CONFIG section 11: fresh vintage is used everywhere; differences are reported, never patched.
    for f in sorted(os.listdir(archive)):
        if f.endswith(".csv") and f not in manifest["files"]:
            manifest["files"][f] = sha(os.path.join(archive, f))
    json.dump(manifest, open(os.path.join(archive, "manifest.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=1, default=str))
    print("archive:", archive)
    return outs, rc


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--self-test", action="store_true")
    g.add_argument("--phase7-go", action="store_true")
    ap.add_argument("--zillow-sm")
    ap.add_argument("--zillow-raw")
    ap.add_argument("--archive-dir", default=None)
    a = ap.parse_args()
    stamp = datetime.date.today().strftime("%Y%m%d")
    if a.self_test:
        archive = a.archive_dir or os.path.join("audit_reports", "phase7", f"selftest_{stamp}")
        os.makedirs(archive, exist_ok=True)
        macro = pd.read_csv("macro_weekly.csv", parse_dates=["date"])
        macro = macro[macro.date <= CAP]
        build(archive, a.zillow_sm, a.zillow_raw, macro, cap_only=True)
        return
    import requests
    archive = a.archive_dir or os.path.join("audit_reports", "phase7", f"archive_{stamp}")
    os.makedirs(archive, exist_ok=False)
    paths = {}
    for label, url in (("sm", SM_URL), ("raw", RAW_URL)):
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        p = os.path.join(archive, os.path.basename(url))
        open(p, "wb").write(r.content)
        paths[label] = p
    macro, key = fred_macro(archive)
    macro.to_csv(os.path.join(archive, "macro_weekly_fresh.csv"), index=False)
    build(archive, paths["sm"], paths["raw"], macro, cap_only=False)
    n = assert_no_secret(archive, key)
    del key
    print(f"secret check OK: FRED key absent from all {n} output files")


if __name__ == "__main__":
    main()
