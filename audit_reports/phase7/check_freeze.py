"""
Verify that every forecast-path file still matches the sha256 recorded in FINAL_CONFIG.md section 9.
Raises SystemExit (non-zero) on any mismatch. Called by run_holdout.py before every run.

Usage (repo root): python audit_reports/phase7/check_freeze.py
"""
import hashlib
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def recorded():
    txt = open(os.path.join(REPO, "audit_reports", "FINAL_CONFIG.md"), encoding="utf-8").read()
    sec = txt[txt.index("## 9. Freeze record"):]
    return dict((f, h) for f, h in re.findall(r"^\| `([^`]+)` \| ([0-9a-f]{64}) \|$", sec, flags=re.M))


def check(verbose=True):
    bad = []
    rec = recorded()
    assert rec, "no hashes recorded in FINAL_CONFIG.md section 9"
    for f, h in rec.items():
        got = hashlib.sha256(open(os.path.join(REPO, f), "rb").read()).hexdigest()
        if got != h:
            bad.append((f, h, got))
    if bad:
        for f, h, got in bad:
            print(f"FREEZE VIOLATION: {f}\n  recorded {h}\n  current  {got}")
        raise SystemExit(f"{len(bad)} forecast-path file(s) changed since the freeze")
    if verbose:
        print(f"freeze check OK: {len(rec)} files match FINAL_CONFIG.md section 9")
    return True


if __name__ == "__main__":
    check()
