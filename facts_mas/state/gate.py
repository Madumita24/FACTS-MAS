"""
NEXUS Modification A: Smart Early-Exit Gate.

NEXUS Proposal §3, Mod A:
    "A lightweight, text-blind time-series model (Google TimesFM 2.5) runs
    continuously in the background and produces a baseline forecast for
    every timestep at very low computational cost."

    The gate decides whether the heavy multi-agent reasoning loop should
    activate, based on two trigger classes:
      1. Text-event signals (Fed announcement, employer shift, disaster, etc.)
      2. Statistical anomaly (baseline prediction error z-score)

    Critical Mod A↔B connection (§3, Mod B):
        "The Smart Early-Exit Gate does not only watch the target city's own
        data for anomalies. It also watches the target city's Granger-identified
        neighbors. A neighbor shock can itself be the trigger that wakes up
        the heavy reasoning loop for the target city."

Design:
    All math (z-score computation, threshold checks) is pure Python/NumPy.
    No LLM is involved in the gate decision.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator

from facts_mas.state.ontology import GateTriggerType


# ═══════════════════════════════════════════════════════════════════════════
# Gate Configuration
# ═══════════════════════════════════════════════════════════════════════════

class GateConfig(BaseModel):
    """
    Configurable thresholds for the Smart Early-Exit Gate.

    These are system-level parameters, not LLM-generated.
    """

    model_config = {"frozen": True}

    anomaly_zscore_threshold: float = Field(
        default=2.0,
        gt=0.0,
        description=(
            "If the baseline model's recent prediction error z-score exceeds "
            "this threshold, the statistical anomaly trigger fires."
        ),
    )
    neighbor_anomaly_zscore_threshold: float = Field(
        default=2.5,
        gt=0.0,
        description=(
            "Z-score threshold for neighbor spillover trigger (Mod A↔B). "
            "Slightly higher than self-anomaly to avoid excessive activations."
        ),
    )
    text_event_triggers: list[GateTriggerType] = Field(
        default_factory=lambda: [
            GateTriggerType.FED_RATE_ANNOUNCEMENT,
            GateTriggerType.MAJOR_EMPLOYER_RELOCATION,
            GateTriggerType.NATURAL_DISASTER,
            GateTriggerType.POLICY_CHANGE,
        ],
        description="Which text-event types should activate the heavy loop.",
    )
    lookback_weeks: int = Field(
        default=12,
        ge=4,
        description=(
            "Number of recent weeks of baseline errors to compute the "
            "error distribution for z-score calculation."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Gate Decision — the output of the gate evaluation
# ═══════════════════════════════════════════════════════════════════════════

class GateDecision(BaseModel):
    """
    Decision output from the Smart Early-Exit Gate.

    This is a pure-data model. The gate logic that *produces* this decision
    is pure Python (no LLM).

    If `should_activate_heavy_loop` is False, the system accepts the
    `baseline_forecast` and skips the expensive multi-agent reasoning.

    Prompt Leakage Isolation:
        This model carries only numerical outputs and trigger metadata.
        No LLM reasoning, prompts, or chain-of-thought.
    """

    model_config = {"frozen": True}

    should_activate_heavy_loop: bool = Field(
        ...,
        description=(
            "True → activate the full multi-agent reasoning loop. "
            "False → accept the baseline forecast and skip heavy reasoning."
        ),
    )
    trigger_type: GateTriggerType = Field(
        ...,
        description=(
            "Which trigger fired, or GateTriggerType.NONE if the market "
            "is calm and no triggers activated."
        ),
    )
    trigger_details: str = Field(
        default="",
        max_length=500,
        description=(
            "Brief factual explanation of what triggered the gate. "
            "E.g. 'Baseline error z-score 2.7 exceeded threshold 2.0' "
            "or 'Fed rate announcement detected in news feed'. "
            "Empty string when no trigger fired."
        ),
    )

    # ── Baseline forecast (always available — produced by TimesFM) ────────
    baseline_forecast: list[float] = Field(
        ...,
        description=(
            "TimesFM 2.5 baseline forecast for the full horizon. "
            "Used as the final output when the gate stays dormant, "
            "and as the Mod C fallback after 2 failed correction retries."
        ),
    )

    # ── Statistical anomaly details ───────────────────────────────────────
    baseline_error_zscore: Optional[float] = Field(
        default=None,
        description=(
            "Z-score of the most recent baseline prediction error, "
            "computed against the error distribution of the lookback window."
        ),
    )
    neighbor_anomaly_msa_id: Optional[str] = Field(
        default=None,
        description=(
            "If the trigger was NEIGHBOR_SPILLOVER, the MSA ID of the "
            "neighbor whose shock triggered this gate. None otherwise."
        ),
    )
    neighbor_error_zscore: Optional[float] = Field(
        default=None,
        description="Z-score of the neighbor's anomaly, if applicable.",
    )

    # ── Consistency validation ────────────────────────────────────────────

    @model_validator(mode="after")
    def _trigger_consistency(self) -> "GateDecision":
        """Ensure trigger_type matches should_activate."""
        if self.should_activate_heavy_loop and self.trigger_type == GateTriggerType.NONE:
            raise ValueError(
                "should_activate_heavy_loop is True but trigger_type is NONE. "
                "A trigger must be identified when activating the heavy loop."
            )
        if not self.should_activate_heavy_loop and self.trigger_type != GateTriggerType.NONE:
            raise ValueError(
                f"should_activate_heavy_loop is False but trigger_type is "
                f"'{self.trigger_type.value}'. If a trigger fired, the heavy "
                "loop should activate."
            )
        return self

    @model_validator(mode="after")
    def _neighbor_fields_consistent(self) -> "GateDecision":
        """If trigger is neighbor spillover, neighbor fields must be set."""
        if self.trigger_type == GateTriggerType.NEIGHBOR_SPILLOVER:
            if self.neighbor_anomaly_msa_id is None:
                raise ValueError(
                    "trigger_type is NEIGHBOR_SPILLOVER but "
                    "neighbor_anomaly_msa_id is None."
                )
        return self
