"""
Phase 6c: TimesFM 2.5 forecasts on the six backtest folds (run in the separate facts-timesfm env).

Same specification as timesfm_baseline.py (google/timesfm-2.5-200m-pytorch, same ForecastConfig,
52-week context, clip at 0). Standalone: no project imports, so it runs in its own env. Reads
aligned_weekly.csv and drops everything after 2026-02-21 at read time.

  --mode timing   time a small batch (fold 1, first 5 origins x 15 MSAs) in both call styles
  --mode folds    every test origin of folds 1-6 (2,610 series contexts)
  --per-h         one call per horizon (4, 8, 13) as the original script; otherwise one h=13 call, sliced later
Output: CSV with columns msa, origin, horizon, step, forecast (scored in the base env by p6_timesfm_score.py).

Usage (repo root): C:\\Anaconda3\\envs\\facts-timesfm\\python.exe audit_reports/phase6_scripts/p6_timesfm.py --mode folds --out <csv>
"""
import argparse
import time

import numpy as np
import pandas as pd

CAP = pd.Timestamp("2026-02-21")
TEST_WINDOWS = [("2021-05-01", "2021-11-13"), ("2022-02-19", "2022-09-03"), ("2022-12-10", "2023-06-24"),
                ("2023-09-30", "2024-04-13"), ("2024-07-20", "2025-02-01"), ("2025-05-10", "2025-11-22")]
HISTORY_WEEKS = 52
REPO_ID = "google/timesfm-2.5-200m-pytorch"


def model():
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(REPO_ID)
    m.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=256, normalize_inputs=True,
                                     use_continuous_quantile_head=False, force_flip_invariance=True,
                                     infer_is_positive=True, fix_quantile_crossing=True))
    return m


def contexts(panel, origins_by_msa):
    out = []
    for msa, origins in origins_by_msa.items():
        s = panel[panel.msa == msa].sort_values("date").set_index("date")["inventory_count"]
        for o in origins:
            out.append((msa, o, s[s.index <= o].tail(HISTORY_WEEKS).to_numpy(np.float64)))
    return out


def run(m, ctx, horizons, batch=64):
    rows = []
    for H in horizons:
        for i in range(0, len(ctx), batch):
            chunk = ctx[i:i + batch]
            point, _ = m.forecast(horizon=H, inputs=[c[2] for c in chunk])
            for (msa, o, _), p in zip(chunk, point):
                for k, v in enumerate(np.maximum(p[:H], 0.0), start=1):
                    rows.append((msa, o.date(), H, k, float(v)))
    return pd.DataFrame(rows, columns=["msa", "origin", "horizon", "step", "forecast"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["timing", "folds"], required=True)
    ap.add_argument("--per-h", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    panel = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
    panel = panel[panel.date <= CAP]
    assert panel.date.max() <= CAP
    msas = sorted(panel.msa.unique())
    if a.mode == "timing":
        d = panel[panel.msa == msas[0]].date
        o5 = [x for x in d if pd.Timestamp(TEST_WINDOWS[0][0]) <= x <= pd.Timestamp(TEST_WINDOWS[0][1])][:5]
        obm = {m_: o5 for m_ in msas}
    else:
        obm = {}
        for m_ in msas:
            d = panel[panel.msa == m_].date
            obm[m_] = [x for x in d if any(pd.Timestamp(s) <= x <= pd.Timestamp(e) for s, e in TEST_WINDOWS)]
    t0 = time.time()
    m = model()
    t1 = time.time()
    ctx = contexts(panel, obm)
    res = run(m, ctx, [4, 8, 13] if a.per_h else [13])
    t2 = time.time()
    res.to_csv(a.out, index=False)
    print(f"series contexts {len(ctx)} | per_h={a.per_h} | model load {t1 - t0:.1f}s | forecasting {t2 - t1:.1f}s "
          f"| {(t2 - t1) / len(ctx):.3f}s per context | written {a.out}")


if __name__ == "__main__":
    main()
