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
  * Transport errors only (429, 5xx, timeouts, connection errors) are retried, at most 3 attempts with
    exponential backoff (1 s, 2 s) or at least a numeric Retry-After header if present (an HTTP-date or
    unparseable header falls back to the backoff; a request over 60 s is not waited for and the cell is
    recorded as a transport failure, eligible for the single re-run pass), every attempt logged;
    parsing and the one conversational correction are unchanged.
  * If more than 5% of cells fail, exactly one logged re-run pass over cells that failed on transport errors
    (never parse failures) is allowed: --rerun-transport-failures with the same --out, --model, --panel.
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


# gpt-5.6-luna list price per million tokens (OpenAI docs, FINAL_CONFIG 11.4); used only for the preflight estimate
PRICE_IN_PER_M, PRICE_OUT_PER_M = 0.20, 1.20


def harden_client(base):
    """FINAL_CONFIG 11.4 transport settings, applied to the OpenAI client from cot_baseline._client():
    * Accept-Encoding gzip/deflate only: httpx2 otherwise asks for brotli, and the installed brotli 1.0.9 cannot
      decode it (TypeError: process() takes no keyword arguments, first attempt 2026-10-22 00:29);
    * max_retries=0: no hidden SDK retries, so ModelCheckedClient is the only, logged retry layer (11.2 D);
    * 180 s per-request timeout (a timeout is a transport error)."""
    return base.with_options(default_headers={"Accept-Encoding": "gzip, deflate"}, max_retries=0, timeout=180.0)


class ModelMismatch(RuntimeError):
    """The model requested or served differs from --model."""


class ModelCheckedClient:
    """Wraps an OpenAI client. Every chat.completions.create call must request exactly `expected` (proves the
    --model override reaches the API call) and the response's `model` field must equal `expected` (logged for
    every call). Either mismatch stops the run. Pass the exact snapshot name the API reports as --model
    (e.g. a dated gpt-4o-mini snapshot), otherwise the first call stops the run."""

    MAX_ATTEMPTS = 3
    BASE_DELAY_S = 1.0
    RETRY_AFTER_CAP_S = 60.0   # longer requests are not waited for: the cell fails as a transport failure

    def __init__(self, client, expected, sleep=None, reasoning_effort=None):
        import time
        import types
        self._client, self.expected, self.calls = client, expected, []
        # FINAL_CONFIG 11.4: for reasoning models, send the documented reasoning_effort and omit temperature.
        # None keeps the original behaviour (temperature passed through unchanged).
        self.reasoning_effort = reasoning_effort
        self._sleep = sleep or time.sleep
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    @staticmethod
    def is_transport_error(e):
        """Transport errors only (FINAL_CONFIG section 11.2): HTTP 429, 5xx, timeouts and connection errors
        (openai.APIConnectionError and its subclass APITimeoutError). Parse failures are never transport errors."""
        status = getattr(e, "status_code", None)
        if isinstance(status, int) and (status == 429 or 500 <= status <= 599):
            return True
        names = [k.__name__ for k in type(e).__mro__]
        return any(n == "APIConnectionError" or "timeout" in n.lower() for n in names)

    @staticmethod
    def retry_after_seconds(e):
        """Seconds requested by a Retry-After header, only if it is a plain number of seconds. An HTTP date or
        anything unparseable returns None (the caller then uses the normal backoff); this never raises."""
        try:
            resp = getattr(e, "response", None)
            headers = getattr(resp, "headers", None) or {}
            val = headers.get("retry-after") or headers.get("Retry-After")
            if val is None:
                return None
            sec = float(str(val).strip())
            return sec if sec >= 0 and sec == sec and sec != float("inf") else None
        except Exception:
            return None

    def _create(self, **kw):
        if kw.get("model") != self.expected:
            raise ModelMismatch(f"requested model {kw.get('model')!r} != --model {self.expected!r}")
        if self.reasoning_effort is not None:
            kw = {k: v for k, v in kw.items() if k != "temperature"}
            kw["reasoning_effort"] = self.reasoning_effort
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            at = datetime.datetime.now().isoformat(timespec="seconds")
            try:
                resp = self._client.chat.completions.create(**kw)
            except Exception as e:
                retry = self.is_transport_error(e) and attempt < self.MAX_ATTEMPTS
                ra = self.retry_after_seconds(e)
                wait, note = None, None
                if retry and ra is not None and ra > self.RETRY_AFTER_CAP_S:
                    retry, note = False, f"retry_after {ra:g}s exceeds cap {self.RETRY_AFTER_CAP_S:g}s; not waited"
                if retry:
                    backoff = self.BASE_DELAY_S * 2 ** (attempt - 1)
                    wait = max(backoff, ra) if ra is not None else backoff
                self.calls.append({"requested_model": kw.get("model"), "response_model": None, "at": at,
                                   "attempt": attempt, "error": type(e).__name__,
                                   "status_code": getattr(e, "status_code", None), "retried": retry,
                                   "retry_after_s": ra, "wait_s": wait, "note": note})
                if not retry:
                    raise
                self._sleep(wait)
                continue
            u = getattr(resp, "usage", None)
            self.calls.append({"requested_model": kw.get("model"), "response_model": getattr(resp, "model", None),
                               "at": at, "attempt": attempt, "error": None, "status_code": None, "retried": False,
                               "retry_after_s": None, "wait_s": None, "note": None,
                               "prompt_tokens": getattr(u, "prompt_tokens", None),
                               "completion_tokens": getattr(u, "completion_tokens", None)})
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


def load_progress(path):
    """Read a JSONL progress file written by run_cells. Returns (rows, failures, calls, done_keys, bad_lines).
    A line that does not parse (e.g. cut off by a crash mid-write) is skipped, so that cell is simply redone."""
    rows, failures, calls, done, bad = [], [], [], set(), 0
    if not os.path.exists(path):
        return rows, failures, calls, done, bad
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            bad += 1
            continue
        done.add((r["msa"], r["origin"], r["h"]))
        calls += r.get("calls", [])
        if "values" in r:
            rows += [(r["msa"], r["origin"], r["h"], k, v) for k, v in enumerate(r["values"], start=1)]
        else:
            failures.append(r["failure"])
    return rows, failures, calls, done, bad


def preflight(cb, a):
    """Two real calls, pre-cap data only. Prints what the paid run depends on; writes no result file."""
    client = ModelCheckedClient(harden_client(cb._client()), a.model, reasoning_effort=a.reasoning_effort)
    ask = client.chat.completions.create(model=a.model, temperature=0, messages=[
        {"role": "user", "content": "What is your training data knowledge cutoff date? Answer with a month and year only."}])
    p = pd.read_csv("aligned_weekly.csv", parse_dates=["date"])
    s = p[(p.msa == "Phoenix") & (p.date <= pd.Timestamp("2025-06-07"))].sort_values("date").set_index("date")[
        "inventory_count"]
    r = cb.forecast_cot_nexus(s, "Phoenix", 4, NO_EVENTS, client=client)
    fc = [c for c in client.calls[1:] if c["error"] is None]
    tin = sum(c["prompt_tokens"] or 0 for c in fc) / max(len(fc), 1)
    tout = sum(c["completion_tokens"] or 0 for c in fc) / max(len(fc), 1)
    per_lag = 1110 * (tin * PRICE_IN_PER_M + tout * PRICE_OUT_PER_M) / 1e6
    print("PREFLIGHT OK")
    print(" response.model:", sorted({c["response_model"] for c in client.calls if c["response_model"]}))
    print(" self-reported cutoff:", ask.choices[0].message.content)
    print(" Phoenix 2025-06-07 h=4 forecast:", r["point_forecast"], "| attempts:", r["attempts"])
    print(f" tokens per forecast call: in {tin:.0f}, out {tout:.0f}; estimated cost about ${per_lag:.2f} per lag "
          f"(1,110 cells; more if 8/13-week answers are longer)")


def run_cells(cells, panel, client, cb, lag, progress=None):
    """Forecast every (msa, origin, h, info_origin) cell. Returns (rows, failures); each failure records whether
    it was a transport error (eligible for the single re-run pass) or a parse / other failure (never re-run).
    With `progress`, each finished cell (values or failure, plus its API calls) is appended to that JSONL file and
    flushed to disk immediately, so a crash loses at most the cell in flight (see --resume)."""
    import sys
    import time
    rows, failures = [], []
    if progress and os.path.exists(progress) and os.path.getsize(progress):
        with open(progress, "rb+") as f:      # a crash can leave a cut-off last line: start a fresh line after it
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
    t0, n = time.time(), len(cells)
    for i, (msa, o, h, oi) in enumerate(cells, start=1):
        s = panel[panel.msa == msa].sort_values("date").set_index("date")["inventory_count"]
        s = s[s.index <= oi]
        n_calls = len(getattr(client, "calls", []))
        rec = {"msa": msa, "origin": str(o.date()), "h": h}
        try:
            r = cb.forecast_cot_nexus(s, msa, h + lag, NO_EVENTS, client=client)
            vals = [max(0.0, float(v)) for v in r["point_forecast"][lag:]]
            rows += [(msa, o.date(), h, k, v) for k, v in enumerate(vals, start=1)]
            rec["values"] = vals
        except ModelMismatch:
            raise                            # a wrong model stops the run
        except Exception as e:  # recorded, never filled with a fallback
            fail = {"msa": msa, "origin": str(o.date()), "h": h, "error": f"{type(e).__name__}: {e}",
                    "transport": bool(ModelCheckedClient.is_transport_error(e))}
            failures.append(fail)
            rec["failure"] = fail
        if progress:
            rec["calls"] = getattr(client, "calls", [])[n_calls:]
            with open(progress, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
        el = time.time() - t0
        eta = el / i * (n - i)
        print(f"[{i}/{n}] {msa} {o.date()} h={h} failures={len(failures)} "
              f"elapsed={el / 60:.1f}m eta={eta / 60:.1f}m", file=sys.stderr, flush=True)
    return rows, failures


def select_rerun_cells(log, cells):
    """FINAL_CONFIG 11.2: one re-run pass, only if the failure share exceeds 5%, only over cells that failed on
    transport errors. Returns the subset of `cells` to re-run (possibly empty)."""
    if not log["failure_share"] > FAIL_SHARE_FLAG:
        return []
    keys = {(f["msa"], f["origin"], f["h"]) for f in log["failures"] if f.get("transport")}
    return [c for c in cells if (c[0], str(c[1].date()), c[2]) in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel")
    ap.add_argument("--model")
    ap.add_argument("--stated-cutoff", help="provider-documented knowledge cutoff, YYYY-MM-DD")
    ap.add_argument("--out")
    ap.add_argument("--pub-lag", type=int, choices=[0, 2], default=0)
    ap.add_argument("--dry-run-prompts", action="store_true")
    ap.add_argument("--preflight", action="store_true",
                    help="2 real API calls on PRE-CAP data (cutoff question + one Phoenix 2025-06-07 forecast); "
                         "checks connection, model name and parsing, prints a cost estimate; writes nothing")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run in the same --out (same --model, --panel, --reasoning-effort, "
                         "--pub-lag); finished cells are not called again")
    ap.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"], default=None,
                    help="reasoning models only (FINAL_CONFIG 11.4): sent as reasoning_effort; temperature is omitted")
    ap.add_argument("--rerun-transport-failures", action="store_true",
                    help="the single permitted re-run pass over transport-failed cells of an existing --out run")
    a = ap.parse_args()
    if a.dry_run_prompts:
        return dry_run()
    if a.preflight and not a.model:
        raise SystemExit("--preflight needs --model (and the same --reasoning-effort as the real run)")
    if not a.preflight and not (a.panel and a.model and a.stated_cutoff and a.out):
        raise SystemExit("--panel, --model, --stated-cutoff and --out are required")
    import cot_baseline as cb
    cb.MODEL = a.model                      # forecast_cot_nexus reads cb.MODEL at call time
    if a.preflight:
        return preflight(cb, a)
    panel = pd.read_csv(a.panel, parse_dates=["date"])
    cells = holdout_cells(panel, a.pub_lag)
    tag = f"lag{a.pub_lag}"
    if a.rerun_transport_failures:
        log_path = os.path.join(a.out, f"cot_run_log_{tag}.json")
        log = json.load(open(log_path))
        if log.get("rerun_pass_done"):
            raise SystemExit("the single re-run pass has already been done for this run")
        if log["model"] != a.model or log["panel_sha256"] != hashlib.sha256(open(a.panel, "rb").read()).hexdigest():
            raise SystemExit("re-run must use the same --model and --panel as the original run")
        todo = select_rerun_cells(log, cells)
        if not todo:
            raise SystemExit(f"no re-run allowed: failure share {log['failure_share']:.3%} (rule: > 5%) "
                             f"or no transport failures")
        if log.get("reasoning_effort") != a.reasoning_effort:
            raise SystemExit("re-run must use the same --reasoning-effort as the original run")
        client = ModelCheckedClient(harden_client(cb._client()), a.model, reasoning_effort=a.reasoning_effort)
        rows, still = run_cells(todo, panel, client, cb, a.pub_lag)
        prev = pd.read_csv(os.path.join(a.out, f"cot_forecasts_{tag}.csv"))
        merged = pd.concat([prev, pd.DataFrame(rows, columns=prev.columns)], ignore_index=True)
        merged.to_csv(os.path.join(a.out, f"cot_forecasts_{tag}_after_rerun.csv"), index=False)
        pd.DataFrame(client.calls).to_csv(os.path.join(a.out, f"cot_calls_{tag}_rerun.csv"), index=False)
        fixed = {(m, str(o), h) for m, o, h, _, _ in rows}
        after = [f for f in log["failures"] if (f["msa"], f["origin"], f["h"]) not in fixed]
        log["rerun_pass_done"] = True
        log["rerun"] = {"run_date": datetime.datetime.now().isoformat(timespec="seconds"), "cells_rerun": len(todo),
                        "recovered": len(fixed), "still_failing": still, "failures_after_rerun": after}
        log["failure_share_after_rerun"] = len(after) / max(len(cells), 1)
        log["report_with_and_without_cot"] = log["failure_share_after_rerun"] > FAIL_SHARE_FLAG
        json.dump(log, open(log_path, "w"), indent=1)
        print(json.dumps({"cells_rerun": len(todo), "recovered": len(fixed),
                          "failure_share_after_rerun": log["failure_share_after_rerun"]}, indent=1))
        return
    client = ModelCheckedClient(harden_client(cb._client()), a.model,
                                reasoning_effort=a.reasoning_effort)  # key read from the operator's environment
    header_path = os.path.join(a.out, f"cot_run_header_{tag}.json")
    progress_path = os.path.join(a.out, f"cot_progress_{tag}.jsonl")
    panel_sha = hashlib.sha256(open(a.panel, "rb").read()).hexdigest()
    if a.resume:
        if not os.path.isdir(a.out):
            raise SystemExit(f"--resume: {a.out} does not exist")
        if os.path.exists(os.path.join(a.out, f"cot_run_log_{tag}.json")):
            raise SystemExit("--resume: this run already finished")
    else:
        os.makedirs(a.out, exist_ok=False)
    first_target = min(o for _, o, _, _ in cells) + pd.Timedelta(weeks=1)
    if os.path.exists(header_path):
        log = json.load(open(header_path))
        if (log["model"], log["panel_sha256"], log.get("reasoning_effort"), log["pub_lag"]) != (
                a.model, panel_sha, a.reasoning_effort, a.pub_lag):
            raise SystemExit("--resume must use the same --model, --panel, --reasoning-effort and --pub-lag")
    else:
        ask = client.chat.completions.create(model=a.model, temperature=0, messages=[
            {"role": "user", "content": "What is your training data knowledge cutoff date? Answer with a month and year only."}])
        log = {"model": a.model, "temperature": cb.TEMPERATURE if a.reasoning_effort is None else None,
               "reasoning_effort": a.reasoning_effort, "stated_cutoff_operator": a.stated_cutoff,
               "self_reported_cutoff_verbatim": ask.choices[0].message.content,
               "run_date": datetime.datetime.now().isoformat(timespec="seconds"), "pub_lag": a.pub_lag,
               "panel": a.panel, "panel_sha256": panel_sha,
               "earliest_holdout_target": str(first_target.date()),
               "out_of_sample": pd.Timestamp(a.stated_cutoff) < first_target,
               "event_text": NO_EVENTS, "system_prompt": cb.SYSTEM_PROMPT, "n_cells": len(cells),
               "cutoff_question_calls": list(client.calls), "resumes": []}
        client.calls.clear()
        json.dump(log, open(header_path, "w"), indent=1, default=str)
    _, _, _, done, bad = load_progress(progress_path)
    if a.resume:
        log["resumes"].append({"at": datetime.datetime.now().isoformat(timespec="seconds"),
                               "cells_already_done": len(done), "unreadable_progress_lines": bad})
        json.dump(log, open(header_path, "w"), indent=1, default=str)
    todo = [c for c in cells if (c[0], str(c[1].date()), c[2]) not in done]
    run_cells(todo, panel, client, cb, a.pub_lag, progress=progress_path)
    rows, failures, calls, done, bad = load_progress(progress_path)   # the progress file is the single record
    client.calls = log.get("cutoff_question_calls", []) + calls
    log["failures"] = failures
    log["n_failures"] = len(failures)
    log["n_transport_failures"] = sum(f["transport"] for f in failures)
    log["failure_share"] = len(failures) / max(len(cells), 1)
    log["report_with_and_without_cot"] = log["failure_share"] > FAIL_SHARE_FLAG
    log["rerun_allowed"] = bool(log["failure_share"] > FAIL_SHARE_FLAG and log["n_transport_failures"] > 0)
    log["rerun_pass_done"] = False
    log["response_models_seen"] = sorted({c["response_model"] for c in client.calls if c["response_model"]})
    log["transport_retries"] = sum(1 for c in client.calls if c.get("retried"))
    log["unreadable_progress_lines"] = bad
    log["prompt_tokens"] = sum(c.get("prompt_tokens") or 0 for c in client.calls)
    log["completion_tokens"] = sum(c.get("completion_tokens") or 0 for c in client.calls)
    pd.DataFrame(client.calls).to_csv(os.path.join(a.out, f"cot_calls_{tag}.csv"), index=False)
    pd.DataFrame(rows, columns=["msa", "origin", "horizon", "step", "forecast"]).to_csv(
        os.path.join(a.out, f"cot_forecasts_{tag}.csv"), index=False)
    json.dump(log, open(os.path.join(a.out, f"cot_run_log_{tag}.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in log.items() if k != "failures"}, indent=1), "\nfailures:", len(failures))


if __name__ == "__main__":
    main()
