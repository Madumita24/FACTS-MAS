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
  * Every API call goes through ModelCheckedClient: the requested model must equal --model and
    response.model must equal --model (logged per call in cot_calls_lag*.csv); a mismatch stops the run.
  * Transport errors only (429, 5xx, timeouts) are retried, at most 3 attempts with exponential backoff
    (1 s, 2 s), every attempt logged; parsing and the one conversational correction are unchanged.
  * Failure count and share are logged; above 5% of cells the paired statistics are reported with and
    without CoT (FINAL_CONFIG section 11 item 5).
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
FAIL_SHARE_FLAG = 0.05


class ModelMismatch(RuntimeError):
    """The model requested or served differs from --model."""


class ModelCheckedClient:
    """Wraps an OpenAI client. Every chat.completions.create call must request exactly `expected` (proves the
    --model override reaches the API call) and the response's `model` field must equal `expected` (logged for
    every call). Either mismatch stops the run. Pass the exact snapshot name the API reports as --model
    (e.g. a dated gpt-4o-mini snapshot), otherwise the first call stops the run."""

    MAX_ATTEMPTS = 3
    BASE_DELAY_S = 1.0

    def __init__(self, client, expected, sleep=None):
        import time
        import types
        self._client, self.expected, self.calls = client, expected, []
        self._sleep = sleep or time.sleep
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    @staticmethod
    def is_transport_error(e):
        """Retry only rate limits (429), server errors (5xx) and timeouts (FINAL_CONFIG section 11)."""
        status = getattr(e, "status_code", None)
        if isinstance(status, int) and (status == 429 or 500 <= status <= 599):
            return True
        return "timeout" in type(e).__name__.lower()

    def _create(self, **kw):
        if kw.get("model") != self.expected:
            raise ModelMismatch(f"requested model {kw.get('model')!r} != --model {self.expected!r}")
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            at = datetime.datetime.now().isoformat(timespec="seconds")
            try:
                resp = self._client.chat.completions.create(**kw)
            except Exception as e:
                retry = self.is_transport_error(e) and attempt < self.MAX_ATTEMPTS
                self.calls.append({"requested_model": kw.get("model"), "response_model": None, "at": at,
                                   "attempt": attempt, "error": type(e).__name__,
                                   "status_code": getattr(e, "status_code", None), "retried": retry})
                if not retry:
                    raise
                self._sleep(self.BASE_DELAY_S * 2 ** (attempt - 1))
                continue
            self.calls.append({"requested_model": kw.get("model"), "response_model": getattr(resp, "model", None),
                               "at": at, "attempt": attempt, "error": None, "status_code": None, "retried": False})
            if getattr(resp, "model", None) != self.expected:
                raise ModelMismatch(f"response.model {getattr(resp, 'model', None)!r} != --model {self.expected!r}")
            return resp

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
    cb.MODEL = a.model                      # forecast_cot_nexus reads cb.MODEL at call time
    client = ModelCheckedClient(cb._client(), a.model)  # key read from the operator's environment
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
           "event_text": NO_EVENTS, "system_prompt": cb.SYSTEM_PROMPT, "n_cells": len(cells), "failures": []}
    rows = []
    for msa, o, h, oi in cells:
        s = panel[panel.msa == msa].sort_values("date").set_index("date")["inventory_count"]
        s = s[s.index <= oi]
        try:
            r = cb.forecast_cot_nexus(s, msa, h + a.pub_lag, NO_EVENTS, client=client)
            vals = r["point_forecast"][a.pub_lag:]
            rows += [(msa, o.date(), h, k, max(0.0, float(v))) for k, v in enumerate(vals, start=1)]
        except ModelMismatch:
            raise                            # a wrong model stops the run; nothing is written
        except Exception as e:  # recorded, never filled with a fallback
            log["failures"].append({"msa": msa, "origin": str(o.date()), "h": h, "error": str(e)})
    log["n_failures"] = len(log["failures"])
    log["failure_share"] = len(log["failures"]) / max(len(cells), 1)
    log["report_with_and_without_cot"] = log["failure_share"] > FAIL_SHARE_FLAG
    log["response_models_seen"] = sorted({c["response_model"] for c in client.calls if c["response_model"]})
    log["transport_retries"] = sum(1 for c in client.calls if c.get("retried"))
    pd.DataFrame(client.calls).to_csv(os.path.join(a.out, f"cot_calls_lag{a.pub_lag}.csv"), index=False)
    pd.DataFrame(rows, columns=["msa", "origin", "horizon", "step", "forecast"]).to_csv(
        os.path.join(a.out, f"cot_forecasts_lag{a.pub_lag}.csv"), index=False)
    json.dump(log, open(os.path.join(a.out, f"cot_run_log_lag{a.pub_lag}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in log.items() if k != "failures"}, indent=1), "\nfailures:", len(log["failures"]))


if __name__ == "__main__":
    main()
