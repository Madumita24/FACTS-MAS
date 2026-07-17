"""Step 3: full batch runner for the Event Agent across all unique FEMA
declarations in fema_events_weekly.csv.

Caching design, deliberately deviating from the literal "cache by (msa, week)"
spec wording -- documented in event_agent.md: some declarations span years
(COVID-19: ~172 active weeks), so caching by (msa, week) would NOT dedupe a
long-running declaration's repeated interpretation, since each week is a
distinct key even though the declaration text is identical. Caching by
(msa, declaration_id) instead is what actually achieves the stated goal --
one LLM call per declaration per MSA, then broadcast that single result to
every (msa, week) row the declaration covers.

Cost-awareness gate: prints the number of unique LLM calls and an estimated
cost (gpt-4o-mini: $0.15/1M input tokens, $0.60/1M output tokens, per
OpenAI's published pricing) before making any calls, and requires explicit
confirmation if that count exceeds 50 -- either interactively, or via --yes
for non-interactive/pre-authorized runs.
"""
import argparse
import sys

import pandas as pd

from event_agent import CONFIDENCE_GATE, _client, build_declaration_input, interpret_event

CONFIRMATION_THRESHOLD = 50
EST_INPUT_TOKENS_PER_CALL = 750   # system prompt (~600) + user prompt (~150), rough
EST_OUTPUT_TOKENS_PER_CALL = 150  # JSON incl. reasoning field, rough
INPUT_PRICE_PER_1M = 0.15
OUTPUT_PRICE_PER_1M = 0.60


def estimate_cost(n_calls: int) -> float:
    input_cost = n_calls * EST_INPUT_TOKENS_PER_CALL / 1_000_000 * INPUT_PRICE_PER_1M
    output_cost = n_calls * EST_OUTPUT_TOKENS_PER_CALL / 1_000_000 * OUTPUT_PRICE_PER_1M
    return input_cost + output_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true",
                         help="Skip the interactive confirmation gate (for pre-authorized/non-interactive runs)")
    args = parser.parse_args()

    df = pd.read_csv("fema_events_weekly.csv")
    unique_declarations = df.drop_duplicates(subset=["msa", "declaration_id"]).copy()
    n_calls = len(unique_declarations)
    est_cost = estimate_cost(n_calls)

    print(f"fema_events_weekly.csv: {len(df)} total (msa, week) rows")
    print(f"Unique (msa, declaration_id) pairs needing an LLM call: {n_calls}")
    print(f"Estimated cost (gpt-4o-mini, ~{EST_INPUT_TOKENS_PER_CALL} input + "
          f"~{EST_OUTPUT_TOKENS_PER_CALL} output tokens/call): ${est_cost:.4f}")

    if n_calls > CONFIRMATION_THRESHOLD:
        if args.yes:
            print(f"\n{n_calls} calls exceeds the {CONFIRMATION_THRESHOLD}-call confirmation threshold "
                  f"-- proceeding with --yes (pre-authorized).")
        elif sys.stdin.isatty():
            resp = input(f"\n{n_calls} calls exceeds the {CONFIRMATION_THRESHOLD}-call threshold. Proceed? [y/N] ")
            if resp.strip().lower() != "y":
                print("Aborted.")
                return
        else:
            print(f"\n{n_calls} calls exceeds the {CONFIRMATION_THRESHOLD}-call threshold and this is a "
                  f"non-interactive session -- re-run with --yes to proceed. Aborting.")
            return

    client = _client()
    print(f"\nCalling interpret_event() for {n_calls} unique declarations...")

    results = {}
    for i, row in enumerate(unique_declarations.itertuples(), 1):
        declaration_text, metadata = build_declaration_input(row)
        result = interpret_event(declaration_text, metadata, client=client)
        results[(row.msa, row.declaration_id)] = result
        if i % 20 == 0 or i == n_calls:
            print(f"  {i}/{n_calls} done")

    results_df = pd.DataFrame([
        {"msa": msa, "declaration_id": decl_id, **result}
        for (msa, decl_id), result in results.items()
    ])
    results_df.to_csv("event_agent_declarations.csv", index=False)

    # Broadcast each declaration's single interpretation back to every
    # (msa, week) row it covers.
    weekly = df.merge(results_df, on=["msa", "declaration_id"], how="left")
    weekly.to_csv("event_agent_weekly.csv", index=False)

    print(f"\nevent_agent_declarations.csv written: {len(results_df)} rows (one per unique declaration)")
    print(f"event_agent_weekly.csv written: {len(weekly)} rows (one per (msa, week, active declaration))")

    print("\nevent_type distribution (unique declarations):")
    print(results_df["event_type"].value_counts().to_string())

    print(f"\nGated (confidence<={CONFIDENCE_GATE}): "
          f"{results_df['gated'].sum()}/{len(results_df)} unique declarations")

    print("\nimpact_magnitude distribution (unique declarations, post-gating):")
    print(results_df["impact_magnitude"].value_counts().sort_index().to_string())

    print("\nSample of 10 results:")
    print(results_df.sample(min(10, len(results_df)), random_state=0)
          [["msa", "declaration_id", "event_type", "impact_magnitude", "confidence", "gated"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
