"""
Pre-Phase-7 checks (no network, no API key, no post-cap data).

1. build_holdout_panel.revision_check flags a single perturbed value (negative control) and passes on identity.
2. cot_holdout.ModelCheckedClient: the --model override reaches the API call, response.model is logged,
   and a response from a different model stops the run.
"""
import os
import sys
import types

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit_core as ac  # noqa: E402

sys.path.insert(0, os.path.join(ac.REPO, "audit_reports", "phase7"))
import build_holdout_panel as bhp  # noqa: E402
import cot_holdout as ch  # noqa: E402

ART = os.path.join(ac.REPO, "audit_reports", "artifacts", "zillow_sm_vintage20261015_capped20260221.csv")


def test_revision_check_identity_and_negative_control():
    ref = pd.read_csv(ART, parse_dates=["date"])
    assert len(bhp.revision_check(ref.copy(), ART, "smoothed")) == 0
    bad = ref.copy()
    bad.loc[(bad.msa == "Phoenix") & (bad.date == pd.Timestamp("2024-06-01")), "inventory_count"] += 1
    rc = bhp.revision_check(bad, ART, "smoothed")
    assert len(rc) == 1 and rc.iloc[0].msa == "Phoenix" and rc.iloc[0].column == "inventory_count"
    worse = ref.copy()
    worse.loc[worse.date == pd.Timestamp("2025-01-04"), "cpi"] += 0.1   # a macro revision on one date, all MSAs
    assert (bhp.revision_check(worse, ART, "smoothed").column == "cpi").sum() == 15


class _Resp:
    def __init__(self, model, text):
        self.model = model
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=text))]


class _FakeOpenAI:
    def __init__(self, served_model):
        self.served = served_model
        self.seen = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.seen.append(kw["model"])
        return _Resp(self.served, "ok <prediction>1,2,3,4</prediction>")


def test_model_override_reaches_call_and_response_model_logged():
    import cot_baseline as cb
    fake = _FakeOpenAI("gpt-4o-mini-2024-07-18")
    client = ch.ModelCheckedClient(fake, "gpt-4o-mini-2024-07-18")
    old = cb.MODEL
    try:
        cb.MODEL = "gpt-4o-mini-2024-07-18"
        s = pd.Series([1.0] * 60, index=pd.date_range("2024-01-06", periods=60, freq="W-SAT"))
        r = cb.forecast_cot_nexus(s, "Phoenix", 4, "No significant events recorded.", client=client)
    finally:
        cb.MODEL = old
    assert r["point_forecast"] == [1.0, 2.0, 3.0, 4.0]
    assert fake.seen == ["gpt-4o-mini-2024-07-18"]
    assert client.calls[0]["requested_model"] == client.calls[0]["response_model"] == "gpt-4o-mini-2024-07-18"


def test_response_from_other_model_stops_the_run():
    client = ch.ModelCheckedClient(_FakeOpenAI("gpt-4o-mini-2024-07-18"), "gpt-4o-mini")
    with pytest.raises(ch.ModelMismatch):
        client.chat.completions.create(model="gpt-4o-mini", messages=[])
    with pytest.raises(ch.ModelMismatch):  # override not reaching the call is also caught
        ch.ModelCheckedClient(_FakeOpenAI("x"), "x").chat.completions.create(model="y", messages=[])


# ── FRED key never reaches an output file or an error message ─────────────

FAKE_KEY = "abcdef0123456789abcdef0123456789"   # not a real key


def test_secret_scan_negative_control(tmp_path):
    (tmp_path / "ok.csv").write_text("date,value\n2026-01-01,1\n")
    assert bhp.assert_no_secret(str(tmp_path), FAKE_KEY) == 1
    (tmp_path / "leak.json").write_text('{"url": "https://x?api_key=%s"}' % FAKE_KEY)
    with pytest.raises(AssertionError) as e:
        bhp.assert_no_secret(str(tmp_path), FAKE_KEY)
    assert FAKE_KEY not in str(e.value)


def test_fred_errors_are_scrubbed(tmp_path, monkeypatch):
    monkeypatch.setattr(bhp.pull_macro, "load_api_key", lambda: FAKE_KEY)

    def boom(fred, sid):
        raise ValueError(f"HTTP 400 for https://api.stlouisfed.org/fred/series?api_key={FAKE_KEY}")
    monkeypatch.setattr(bhp.pull_macro, "fetch_series", boom)
    with pytest.raises(RuntimeError) as e:
        bhp.fred_macro(str(tmp_path))
    assert FAKE_KEY not in str(e.value) and "***" in str(e.value)
    assert bhp.assert_no_secret(str(tmp_path), FAKE_KEY) == 0


# ── CoT transport retries (429, 5xx, timeouts only; max 3 attempts) ───────

class _StatusErr(Exception):
    def __init__(self, code):
        super().__init__(f"status {code}")
        self.status_code = code


class APITimeoutError(Exception):  # same class name as openai's timeout error
    pass


class _Flaky:
    def __init__(self, errors, model="snap-1"):
        self.errors, self.model, self.n = list(errors), model, 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.n += 1
        if self.errors:
            raise self.errors.pop(0)
        return _Resp(self.model, "ok <prediction>1,2,3,4</prediction>")


def test_retry_transport_errors_then_succeed():
    slept = []
    fake = _Flaky([_StatusErr(429), APITimeoutError("t")])
    c = ch.ModelCheckedClient(fake, "snap-1", sleep=slept.append)
    c.chat.completions.create(model="snap-1", messages=[])
    assert fake.n == 3 and slept == [1.0, 2.0]
    assert [x["attempt"] for x in c.calls] == [1, 2, 3] and [x["retried"] for x in c.calls] == [True, True, False]


def test_no_retry_on_client_error_and_cap_at_three():
    c = ch.ModelCheckedClient(_Flaky([_StatusErr(400)]), "snap-1", sleep=lambda s: None)
    with pytest.raises(_StatusErr):
        c.chat.completions.create(model="snap-1", messages=[])
    assert len(c.calls) == 1 and c.calls[0]["retried"] is False
    fake = _Flaky([_StatusErr(503)] * 5)
    c = ch.ModelCheckedClient(fake, "snap-1", sleep=lambda s: None)
    with pytest.raises(_StatusErr):
        c.chat.completions.create(model="snap-1", messages=[])
    assert fake.n == 3


# ── filled-target post-hoc filter ──────────────────────────────────────────

def test_filled_target_flags():
    import filled_target_check as ftc
    D = pd.DataFrame({"msa": ["A", "A", "B"], "origin": ["2026-03-07", "2026-04-04", "2026-03-07"],
                      "last_target": ["2026-04-04", "2026-05-02", "2026-04-04"], "h": [4, 4, 4]})
    filled = pd.DataFrame({"date": ["2026-03-21"], "msa": ["A"], "filled_value": [1.0]})
    assert ftc.flag_cells(D, filled).tolist() == [True, False, False]


# ── filled-target check refuses a cut mismatch ─────────────────────────────

def test_filled_target_cut_check(tmp_path):
    import json
    import filled_target_check as ftc
    res = tmp_path / "results_x"
    res.mkdir()
    (res / "run_meta.json").write_text(json.dumps({"panel": "audit_reports/phase7/archive_20261101/panel_raw.csv"}))
    assert ftc.check_cut(str(res), "archive_20261101/filled_rows_raw.csv") == "raw"
    with pytest.raises(SystemExit):
        ftc.check_cut(str(res), "archive_20261101/filled_rows_smoothed.csv")
    with pytest.raises(SystemExit):
        ftc.check_cut(str(res), "archive_20261101/filled_rows.csv")


# ── CoT: connection errors are transport errors; Retry-After honoured ─────

class APIConnectionError(Exception):  # same class name as openai's connection error
    pass


class _RetryAfterErr(Exception):
    def __init__(self, code, retry_after):
        super().__init__(f"status {code}")
        self.status_code = code
        self.response = types.SimpleNamespace(headers={"retry-after": retry_after})


def test_connection_error_is_retried():
    slept = []
    fake = _Flaky([APIConnectionError("reset")])
    c = ch.ModelCheckedClient(fake, "snap-1", sleep=slept.append)
    c.chat.completions.create(model="snap-1", messages=[])
    assert fake.n == 2 and slept == [1.0] and c.calls[0]["error"] == "APIConnectionError"


def test_retry_after_header_sets_minimum_wait():
    slept = []
    fake = _Flaky([_RetryAfterErr(429, "7"), _RetryAfterErr(503, "0.5")])
    c = ch.ModelCheckedClient(fake, "snap-1", sleep=slept.append)
    c.chat.completions.create(model="snap-1", messages=[])
    assert slept == [7.0, 2.0]                       # max(backoff 1, 7) then max(backoff 2, 0.5)
    assert [x["retry_after_s"] for x in c.calls[:2]] == [7.0, 0.5]


# ── CoT: the single re-run pass selects transport failures only, only above 5% ─

def test_rerun_selects_transport_failures_only():
    o1, o2 = pd.Timestamp("2026-03-07"), pd.Timestamp("2026-03-14")
    cells = [("A", o1, 4, o1), ("A", o2, 4, o2), ("B", o1, 8, o1)]
    fails = [{"msa": "A", "origin": "2026-03-07", "h": 4, "transport": True},
             {"msa": "B", "origin": "2026-03-07", "h": 8, "transport": False}]
    assert ch.select_rerun_cells({"failure_share": 0.10, "failures": fails}, cells) == [cells[0]]
    assert ch.select_rerun_cells({"failure_share": 0.04, "failures": fails}, cells) == []


# ── Retry-After hardening: unparseable / HTTP-date falls back; > 60 s is not waited for ─

def test_retry_after_http_date_or_garbage_falls_back_to_backoff():
    slept = []
    fake = _Flaky([_RetryAfterErr(429, "Wed, 21 Oct 2026 07:28:00 GMT"), _RetryAfterErr(503, "soon")])
    c = ch.ModelCheckedClient(fake, "snap-1", sleep=slept.append)
    c.chat.completions.create(model="snap-1", messages=[])
    assert slept == [1.0, 2.0] and fake.n == 3
    assert [x["retry_after_s"] for x in c.calls[:2]] == [None, None]


def test_retry_after_over_cap_is_not_waited_and_counts_as_transport_failure():
    slept = []
    fake = _Flaky([_RetryAfterErr(429, "120")])
    c = ch.ModelCheckedClient(fake, "snap-1", sleep=slept.append)
    with pytest.raises(_RetryAfterErr) as e:
        c.chat.completions.create(model="snap-1", messages=[])
    assert slept == [] and fake.n == 1
    assert c.calls[0]["retried"] is False and "exceeds cap" in c.calls[0]["note"]
    assert ch.ModelCheckedClient.is_transport_error(e.value)   # recorded as transport: eligible for the re-run pass
    import cot_baseline as cb
    old = cb.MODEL
    try:
        cb.MODEL = "snap-1"
        o = pd.Timestamp("2025-06-07")
        panel = pd.DataFrame({"date": pd.date_range("2024-06-15", periods=52, freq="W-SAT"), "msa": "A",
                              "inventory_count": 100.0})
        rows, fails = ch.run_cells([("A", o, 4, o)], panel, ch.ModelCheckedClient(_Flaky([_RetryAfterErr(429, "120")]),
                                                                                 "snap-1", sleep=slept.append), cb, 0)
    finally:
        cb.MODEL = old
    assert rows == [] and len(fails) == 1 and fails[0]["transport"] is True and slept == []
