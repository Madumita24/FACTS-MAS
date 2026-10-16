"""
Phase 7 CoT baseline on the HOLDOUT ONLY (FINAL_CONFIG.md section 6). Prepared by the audit; to be run by
Umang or Madumita. Contains no API key: the key is read from the environment (OPENAI_API_KEY or .env) only
when the operator runs it.

What it does
  * Uses the existing NEXUS-paper CoT prompt exactly (imports cot_baseline.py: SYSTEM_PROMPT,
    USER_TEMPLATE, parsing, one conversational correction, no silent fallback).
    Note: that system prompt tells the model "Your knowledge cutoff date is January 2025" whatever model
    is used; kept unchanged for comparability and disclosed.
  * Holdout origins: 2026-02-28 onward, every week with a complete h-week target in the refreshed panel.
    One call per (MSA, origin, horizon) as in the fold-1 run; 52-week context ending at the information
    origin (origin, or origin - 2 weeks with --pub-lag 2, keeping the last h of h + 2 forecast steps).
  * Event text: "No significant events recorded." (the FEMA event cache ends in 2025; disclosed).
  * Logs to a JSON run log: model name, the operator-supplied STATED knowledge cutoff (from the provider's
    documentation), the model's own SELF-REPORTED cutoff (asked once, verbatim answer stored), run date,
    panel sha256, counts and failures. Results count as out-of-sample only if the stated cutoff is before
    the earliest holdout target date; otherwise the log marks them NOT out-of-sample.
  * Output: CSV (msa, origin, horizon, step, forecast) for run_holdout.py --cot-csv.

Usage (operator):
  python audit_reports/phase7/cot_holdout.py --panel <refreshed long panel CSV> --model gpt-4o-mini \\
         --stated-cutoff YYYY-MM-DD --out <dir> [--pub-lag 2]
  python audit_reports/phase7/cot_holdout.py --dry-run-prompts      (no API call; pre-cap pseudo origin)
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd  # noqa: E402

HOLDOUT_START = pd.Timestamp("2026-02-28")
TRAIN_END = pd.Timestamp("2026-02-21")
NO_EVENTS = "No significant events recorded."


def holdout_cells(panel, lag):
    rows = []
    for msa, g in panel.groupby("msa"):
        g = g.sort_values("date").reset_index(drop=True)
        dates = list(g.date)
        for i, o in enumerate(dates):
            if o < HOLDOUT_START:
                continue
            for h in (4, 8, 13):
                if i + h < len(dates):
                    rows.append((msa, o, h, dates[i - lag]))
    return rows


def dry_run():
    import cot_baseline as cb
    p = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
    p = p[p.date <= TRAIN_END]
    s = p[p.msa == "Phoenix"].sort_values("date").set_index("date")["inventory_count"]
    s = s[s.index <= pd.Timestamp("2025-06-07")]
    prompt, start, last = cb._build_user_prompt(s, "Phoenix", 4, NO_EVENTS)
    print("DRY RUN (no API call). Model constant in cot_baseline.py:", cb.MODEL, "| temperature", cb.TEMPERATURE)
    print("system prompt:", cb.SYSTEM_PROMPT)
    print(prompt[:600], "...")
    print("context", start, "->", last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel")
    ap.add_argument("--model")
    ap.add_argument("--stated-cutoff", help="provider-documented knowledge cutoff, YYYY-MM-DD")
    ap.add_argument("--out")
    ap.add_argument("--pub-lag", type=int, choices=[0, 2], default=0)
    ap.add_argument("--dry-run-prompts", action="store_true")
    a = ap.parse_args()
    if a.dry_run_prompts:
        return dry_run()
    if not (a.panel and a.model and a.stated_cutoff and a.out):
        raise SystemExit("--panel, --model, --stated-cutoff and --out are required")
    import cot_baseline as cb
    cb.MODEL = a.model
    client = cb._client()  # reads OPENAI_API_KEY from the operator's environment
    os.makedirs(a.out, exist_ok=False)
    panel = pd.read_csv(a.panel, parse_dates=["date"])
    cells = holdout_cells(panel, a.pub_lag)
    first_target = min(o for _, o, _, _ in cells) + pd.Timedelta(weeks=1)
    ask = client.chat.completions.create(model=a.model, temperature=0, messages=[
        {"role": "user", "content": "What is your training data knowledge cutoff date? Answer with a month and year only."}])
    log = {"model": a.model, "temperature": cb.TEMPERATURE, "stated_cutoff_operator": a.stated_cutoff,
           "self_reported_cutoff_verbatim": ask.choices[0].message.content,
           "run_date": datetime.datetime.now().isoformat(timespec="seconds"), "pub_lag": a.pub_lag,
           "panel": a.panel, "panel_sha256": hashlib.sha256(open(a.panel, "rb").read()).hexdigest(),
           "earliest_holdout_target": str(first_target.date()),
           "out_of_sample": pd.Timestamp(a.stated_cutoff) < first_target,
           "event_text": NO_EVENTS, "system_prompt": cb.SYSTEM_PROMPT, "n_requests": len(cells), "failures": []}
    rows = []
    for msa, o, h, oi in cells:
        s = panel[panel.msa == msa].sort_values("date").set_index("date")["inventory_count"]
        s = s[s.index <= oi]
        try:
            r = cb.forecast_cot_nexus(s, msa, h + a.pub_lag, NO_EVENTS, client=client)
            vals = r["point_forecast"][a.pub_lag:]
            rows += [(msa, o.date(), h, k, max(0.0, float(v))) for k, v in enumerate(vals, start=1)]
        except Exception as e:  # recorded, never filled with a fallback
            log["failures"].append({"msa": msa, "origin": str(o.date()), "h": h, "error": str(e)})
    pd.DataFrame(rows, columns=["msa", "origin", "horizon", "step", "forecast"]).to_csv(
        os.path.join(a.out, f"cot_forecasts_lag{a.pub_lag}.csv"), index=False)
    json.dump(log, open(os.path.join(a.out, f"cot_run_log_lag{a.pub_lag}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in log.items() if k != "failures"}, indent=1), "\nfailures:", len(log["failures"]))


if __name__ == "__main__":
    main()
