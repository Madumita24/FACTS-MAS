"""Compute 6-fold expanding-window backtest boundaries for the AR Agent,
against the confirmed January 2026 live-eval cutoff (supersedes the
January 2025 assumption baked into evaluation_protocol.md -- that doc is
NOT updated by this script; run it, review the table, then decide how to
reconcile the doc).

Design (unchanged from the original Jan-2025 version except for the
cutoff date and the resulting test-fold size):
  - Training data usable from 2019-02-02 (52-week feature warm-up past the
    2018-02-03 data start).
  - Each fold's training window ends a 13-week embargo before its own test
    window starts. This isn't a formality: a training row's 13-week-ahead
    label would otherwise reference a value that hadn't occurred yet as of
    that fold's simulated "today". The embargo is re-applied at every
    fold's own boundary, not just once at the very start.
  - Expanding window: once a fold's test period is chronologically past,
    it becomes part of the next fold's training data with no leakage.
  - Hard ceiling: no fold's test window may extend past 2025-12-31, since
    everything from 2026-01-01 onward is the reserved live-eval period.
"""
import pandas as pd

TRAIN_START = pd.Timestamp("2019-02-02")
INITIAL_TRAIN_WEEKS = 104          # ~2 years before fold 1's test starts
EMBARGO_WEEKS = 13                 # == longest forecast horizon
N_FOLDS = 6
HELD_OUT_START = pd.Timestamp("2026-01-01")  # confirmed cutoff; nothing on/after this date may appear in any fold


def compute_test_fold_weeks() -> int:
    first_train_end = TRAIN_START + pd.Timedelta(weeks=INITIAL_TRAIN_WEEKS - 1)
    first_test_start = first_train_end + pd.Timedelta(weeks=EMBARGO_WEEKS + 1)
    budget_days = (HELD_OUT_START - first_test_start).days
    budget_weeks = budget_days // 7
    # Each subsequent fold consumes (embargo + test_fold_weeks) weeks of budget;
    # fold 1 already consumed its embargo getting from first_train_end to first_test_start,
    # so total budget must fit N_FOLDS test windows plus (N_FOLDS - 1) further embargoes.
    test_fold_weeks = (budget_weeks - (N_FOLDS - 1) * EMBARGO_WEEKS) // N_FOLDS
    return test_fold_weeks


def build_fold_table() -> pd.DataFrame:
    test_fold_weeks = compute_test_fold_weeks()
    train_end = TRAIN_START + pd.Timedelta(weeks=INITIAL_TRAIN_WEEKS - 1)

    rows = []
    for i in range(1, N_FOLDS + 1):
        embargo_end = train_end + pd.Timedelta(weeks=EMBARGO_WEEKS)
        test_start = embargo_end + pd.Timedelta(weeks=1)
        test_end = test_start + pd.Timedelta(weeks=test_fold_weeks - 1)

        assert test_end < HELD_OUT_START, (
            f"Fold {i} test_end {test_end.date()} would reach the held-out period "
            f"starting {HELD_OUT_START.date()}"
        )

        train_weeks = int((train_end - TRAIN_START).days / 7) + 1
        rows.append({
            "fold": i,
            "train_start": TRAIN_START.date(),
            "train_end": train_end.date(),
            "train_weeks": train_weeks,
            "embargo_end": embargo_end.date(),
            "test_start": test_start.date(),
            "test_end": test_end.date(),
        })

        train_end = test_end  # expanding window: this fold's test period is now revealed history

    return pd.DataFrame(rows)


def main():
    test_fold_weeks = compute_test_fold_weeks()
    print(f"Computed test_fold_weeks: {test_fold_weeks}")
    print(f"Held-out live-eval period: {HELD_OUT_START.date()} onward\n")

    table = build_fold_table()
    print(table.to_string(index=False))

    slack_days = (HELD_OUT_START - pd.Timestamp(table.iloc[-1]["test_end"])).days
    print(f"\nSlack before held-out period: {slack_days} days ({slack_days // 7} weeks)")


if __name__ == "__main__":
    main()
