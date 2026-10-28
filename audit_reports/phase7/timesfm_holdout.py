"""
Phase 7 TimesFM forecasts for the holdout (FINAL_CONFIG.md sections 6 and 11 item 4). Run in the facts-timesfm env.

Same specification as the backtest runs (google/timesfm-2.5-200m-pytorch, same ForecastConfig, 52-week context,
clip at 0). For each holdout origin t (2026-02-28 onward) the context is the 52 weeks ending at the
information origin t - lag (lag 0 or 2); one call with horizon 13 + lag, and the steps for targets
t + 1 .. t + h are kept. Rows are written only for horizons whose targets are complete in the panel.
Lag 0 and lag 2 go to separate files. Output columns: msa, origin, horizon, step, forecast
(the format run_holdout.py --timesfm-csv expects).

Usage:
  <facts-timesfm python> audit_reports/phase7/timesfm_holdout.py --panel <holdout panel> --lag 0 --out <csv>
  <facts-timesfm python> audit_reports/phase7/timesfm_holdout.py --dry-run --lag 2 --out <csv> [--max-msas 2]
      (dry run: capped aligned_weekly.csv, pseudo-holdout origins 2025-02-08 onward whose targets end before
       2026-01-01; no post-cap data)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "audit_reports", "phase6_scripts"))
from p6_timesfm import HISTORY_WEEKS, model  # noqa: E402  (same checkpoint and ForecastConfig)

CAP = pd.Timestamp("2026-02-21")
RESERVED = pd.Timestamp("2026-01-01")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel")
    ap.add_argument("--lag", type=int, choices=[0, 2], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-msas", type=int, default=None)
    a = ap.parse_args()
    if a.dry_run:
        panel = pd.read_csv(os.path.join(REPO, "aligned_weekly.csv"), parse_dates=["date"])
        panel = panel[panel.date <= CAP]
        start, target_limit = pd.Timestamp("2025-02-08"), RESERVED
    else:
        if not a.panel:
            raise SystemExit("--panel is required outside --dry-run")
        panel = pd.read_csv(a.panel, parse_dates=["date"])
        start, target_limit = pd.Timestamp("2026-02-28"), None
    msas = sorted(panel.msa.unique())[: a.max_msas]
    jobs = []
    for msa in msas:
        g = panel[panel.msa == msa].sort_values("date").reset_index(drop=True)
        dates, y = list(g.date), g.inventory_count.to_numpy(float)
        for i, o in enumerate(dates):
            if o < start:
                continue
            hs = [h for h in (4, 8, 13) if i + h < len(dates) and (target_limit is None or dates[i + h] < target_limit)]
            if hs:
                jobs.append((msa, o, hs, y[: i - a.lag + 1][-HISTORY_WEEKS:]))
    m = model()
    rows = []
    for k in range(0, len(jobs), 64):
        chunk = jobs[k:k + 64]
        point, _ = m.forecast(horizon=13 + a.lag, inputs=[c[3] for c in chunk])
        for (msa, o, hs, _), p in zip(chunk, point):
            p = np.maximum(np.asarray(p, float), 0.0)
            for h in hs:
                for step in range(1, h + 1):
                    rows.append((msa, o.date(), h, step, float(p[a.lag + step - 1])))
    pd.DataFrame(rows, columns=["msa", "origin", "horizon", "step", "forecast"]).to_csv(a.out, index=False)
    print(f"{'DRY RUN ' if a.dry_run else ''}lag={a.lag}: {len(jobs)} (msa, origin) contexts, {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
