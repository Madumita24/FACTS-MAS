"""
Phase 1 (H1.9): regime episodes (maximal runs of consecutive weeks with one
label), per validation window and per test fold. Labels come from the cache
(classify_regime at each origin). Checks the label is identical across MSAs.

Run from repo root:  python audit_reports/phase1_scripts/p1_regime_episodes.py
"""
import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "tests_audit"))
import audit_core as ac  # noqa: E402

import pandas as pd  # noqa: E402

from facts_mas.agent_cache import AgentCache  # noqa: E402
from facts_mas.run_backtest import FOLDS  # noqa: E402

cache = AgentCache.load("agent_output_cache.pkl")
lab = pd.DataFrame([(e.msa, e.origin, e.regime) for e in cache.entries.values() if e.horizon == 4],
                   columns=["msa", "origin", "regime"])
per_date = lab.groupby("origin")["regime"].nunique()
print("dates where MSAs disagree on regime:", int((per_date > 1).sum()), "of", len(per_date))
s = lab.drop_duplicates("origin").set_index("origin")["regime"].sort_index()


def episodes(a, b):
    x = s[(s.index >= a) & (s.index <= b)]
    runs, prev, start, n = [], None, None, 0
    for d, r in x.items():
        if r != prev:
            if prev is not None:
                runs.append((prev, start, last, n))
            prev, start, n = r, d, 0
        n += 1
        last = d
    if prev is not None:
        runs.append((prev, start, last, n))
    return runs


rows = []
for f in FOLDS:
    for kind, a, b in (("validation", f.train_end - datetime.timedelta(weeks=104), f.train_end),
                       ("test", f.test_start, f.test_end)):
        runs = episodes(a, b)
        rec = {"fold": f.fold, "window": kind, "start": a, "end": b}
        for r in ac.REGIMES:
            rr = [x for x in runs if x[0] == r]
            rec[f"{r}_episodes"] = len(rr)
            rec[f"{r}_weeks"] = sum(x[3] for x in rr)
        rec["runs"] = "; ".join(f"{x[0]} {x[1]}..{x[2]} ({x[3]}w)" for x in runs)
        rows.append(rec)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 200)
print(df.drop(columns="runs").to_string(index=False))
print()
for _, r in df.iterrows():
    print(f"fold {r.fold} {r.window:<10} {r.runs}")
print("\nall labelled origins, full span:")
for x in episodes(s.index.min(), s.index.max()):
    print(f"  {x[0]:<8} {x[1]} .. {x[2]}  ({x[3]} weeks)")

out = os.path.join("audit_reports", "results_tables", "phase1_regime_episodes.csv")
with open(out, "w", newline="") as fh:
    fh.write("# CONFIG: labels from agent_output_cache.pkl (classify_regime at origin, national fed funds 13-week change, +/-0.25) | "
             "validation window = origins in [train_end-104wk, train_end] | test = origins in [test_start, test_end] | episode = maximal run of consecutive weeks with one label\n")
    df.to_csv(fh, index=False)
print("written", out)
