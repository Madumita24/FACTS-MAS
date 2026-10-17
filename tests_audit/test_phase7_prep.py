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
