import pandas as pd, sys
R = pd.read_csv(sys.argv[1], parse_dates=["origin"])
r = R[R.h == 4]
print(pd.crosstab(r.fold, r.regime, margins=True))
for h in (4, 8, 13):
    x = R[(R.h == h) & (R.fold == 6)]
    last_t = x.origin + pd.to_timedelta(7 * h, unit="D")
    print(h, "fold6 cells with any target >= 2026-01-01:", int((last_t >= "2026-01-01").sum()), "of", len(x), "| max target", last_t.max().date())
# imputed targets: cells whose target window contains a forward-filled date
fl = pd.read_csv("filled_rows_log.csv", parse_dates=["date"])
for h in (4, 8, 13):
    x = R[R.h == h]
    n = 0
    for _, f in fl.iterrows():
        n += int(((x.msa == f.msa) & (x.origin < f.date) & (x.origin + pd.to_timedelta(7 * h, unit="D") >= f.date)).sum())
    z = int((x.naive == 0).sum())
    print(h, "test cells touching an imputed target:", n, "| cells with naive MAPE exactly 0:", z)
