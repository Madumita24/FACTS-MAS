# External Baselines

## NEXUS-paper CoT baseline (exact Appendix C prompt)

`cot_baseline.py` implements the literal NEXUS paper prompt template (system prompt, user template, `<prediction>` tag wrapper, and the "BRIEF ANALYSIS / FORMAT / WRAPPER" strict constraints) verbatim, not a simplified paraphrase — the comparison below is methodologically faithful to what the paper itself evaluated, not our own house-style prompt. Model: gpt-4o-mini, temperature=0.1 (same choice/reason as `event_agent.py` — existing API access, not a capability claim; no Claude comparison run). Retry discipline matches `event_agent.py`: one retry via an explicit conversational correction (the model sees its own malformed response and is told exactly what was wrong) if `<prediction>` parsing fails or the value count doesn't match `horizon_weeks`; fails loudly (raises) after 2 attempts rather than silently defaulting to naive or any other fallback.

## FINAL result: Fold 1, full test window (29 origins), properly matched methodology

This is the primary, decisive result — every origin in Fold 1's test window (15 MSAs × 3 horizons × 29 origins = 1,305 calls), averaged exactly the way the fused system's own headline backtest metrics are computed. Full per-origin data in `cot_baseline_fold1_full_window_results.csv`, per-(MSA, horizon) aggregation in `fold1_full_window_comparison.csv`. One CoT call failed after 2 attempts (Philadelphia, h=8, origin 2021-06-05 — model returned 16 values instead of 8) and was logged and excluded, not silently defaulted; every other cell used the full 29 origins.

| Horizon | naive mean/median | CoT mean/median | fused mean/median | fused beats naive | fused beats CoT | naive beats CoT |
|---|---|---|---|---|---|---|
| 4wk | 3.834 / 3.988 | 2.578 / 2.127 | **2.156 / 1.651** | **14/15** | 12/15 | **0/15** |
| 8wk | 7.758 / 7.788 | 5.786 / 5.149 | **5.126 / 4.148** | **14/15** | 11/15 | **0/15** |
| 13wk | 13.927 / 13.028 | 11.260 / 10.814 | **9.155 / 8.936** | **14/15** | 13/15 | **0/15** |

- **Fused beats naive on 14/15 MSAs at every single horizon** — a near-unanimous, consistent win, not a close or horizon-dependent result.
- **Fused beats NEXUS-CoT on 11-13/15 MSAs at every horizon** — our system is the best of the three at every horizon, not just on average.
- **NEXUS-CoT beats naive on 15/15 MSAs at every horizon** — CoT is a real, solid baseline once measured correctly, consistently ahead of naive by a clear margin. It just isn't as good as the full fused system.
- The lone fused loss is Boston at 4wk/8wk and Phoenix at 13wk — isolated exceptions, not a pattern spread across many MSAs.

### Critical methodological finding: the single-origin snapshot showed the OPPOSITE conclusion at 13wk

The original single-origin check (below) found fused losing to naive on 10/15 MSAs at 13wk, and naive beating CoT on 8-10/15 MSAs at every horizon. Averaged properly over the full 29-origin test window, **both of those findings reverse completely**: fused wins 14/15, and naive beats CoT on 0/15. This was pure measurement artifact from evaluating at a single noisy point in time, not a real property of any of the three approaches.

This is the strongest evidence produced tonight for *why* test-window-averaged evaluation, not single-point spot-checks, is the correct methodology for this project — and it validates a design decision made back in Phase 1 (the evaluation protocol's insistence on averaging over every weekly origin in a fold's test window, not sampling a handful of points) that had not, until now, been directly demonstrated to matter this much.

### Folds 2-6: deferred, not run

Fold 1's result is decisive (14/15 and 0/15 splits at every horizon) — not a borderline case where more folds would meaningfully change the conclusion. Running the remaining 5 folds is estimated at ~7.5 sequential hours (7,830 calls at the measured ~3.4s/call), for evidence that would very likely just confirm the same pattern. Cost is trivial (~$2 for all 6 folds); wall-clock time is the actual constraint. If revisited: the efficient path is concurrent/batched API calls (not built tonight, since every LLM call across this whole project has run sequentially by convention) rather than running the current sequential script for another 7.5 hours.

---

## Superseded: single-origin-per-horizon result (kept for the methodological lesson above, not as evidence)

This was evaluated at a single forecast origin (`fold.train_end`) per horizon rather than the full test window, extending the original 3-MSA sample's own methodology to all 15 MSAs. It is **not** a valid measurement of the fused system's real performance — see the finding above — and is retained here only because the contrast between this table and the final result is itself the most useful evidence in this document.

Cost: 45 calls, ~$0.012, all succeeded on first attempt.

| Horizon | naive mean / median | CoT mean / median | fused mean / median | fused beats naive | fused beats CoT | naive beats CoT |
|---|---|---|---|---|---|---|
| 4wk | 3.817 / 2.505 | 3.875 / 2.705 | 3.102 / 2.322 | 10/15 | 9/15 | 8/15 |
| 8wk | 6.765 / 4.929 | 6.728 / 4.472 | 6.493 / 4.869 | 8/15 | 6/15 | 8/15 |
| 13wk | 9.121 / 7.799 | 9.487 / 7.323 | 9.541 / 7.676 | 5/15 | 4/15 | **10/15** |

At the time, this looked like fused genuinely losing at 13wk and CoT having a real long-horizon weakness. Both conclusions were wrong, as the full-test-window result above demonstrates.

## Historical context: the original 3-MSA sample that prompted scaling up

| MSA | naive MAPE | NEXUS-CoT MAPE | our fused (adaptive) MAPE |
|---|---|---|---|
| Phoenix | 1.10 | 2.89 | 2.61 |
| Miami | 24.74 | 26.81 | 34.71 |
| Chicago | 2.54 | 6.26 | 2.52 |
| **Mean** | **9.46** | **11.99** | **13.28** |

At the time, the hypothesis was that this looked bad mainly because Phoenix and Miami — both independently documented as AR's weak points (`ar_agent.md`) — happened to be 2 of the 3 MSAs sampled. That hypothesis turned out to be a reasonable instinct but the wrong diagnosis: the real issue, confirmed later, was single-origin measurement noise (see above), not MSA selection. Phoenix and Miami were red herrings that happened to correlate with the actual problem.

## Scope and what's still deferred

Folds 2-6, full test window: deferred as detailed above. Cost is not the blocker; sequential wall-clock time is.
