"""
NEXUS Modification D: Batched Micro-Reasoning & Tightened Calibration.

NEXUS Proposal §3, Mod D:
    Batched Micro-Reasoning:
        "We propose changing this agent so that it generates its full
        step-by-step reasoning and all of its forecasted values for the
        entire horizon in a single call, returning a structured array
        covering every timestep at once."

        "If the model's output fails to validate correctly as a complete,
        well-formed array, the system falls back to the original
        one-call-per-timestep method only as a backup."

    Tightened Calibration:
        "A candidate guideline must show the same direction of correction
        in at least 4 of the 5 training folds individually, not only when
        all folds are combined together, before it is allowed to be tested
        on the held-out validation fold."

Design:
    • BatchedReasoningOutput: full-horizon output from a single LLM call
    • CalibrationGuideline: a single guideline with fold-level validation
    • TightenedCalibrationResult: extends CalibrationResult with fold checks
"""

from __future__ import annotations

import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════════════════
# Batched Micro-Reasoning Output
# ═══════════════════════════════════════════════════════════════════════════

class BatchedReasoningOutput(BaseModel):
    """
    Full-horizon micro-reasoning output from a single LLM call.

    NEXUS Proposal §3, Mod D:
        Instead of 26 sequential calls for a 26-week horizon, the LLM
        produces all timestep forecasts and reasoning in one call.

    Prompt Leakage Isolation:
        `step_reasoning` contains the LLM's structured reasoning per timestep.
        This is stored in the graph state's internal traces key and is
        NEVER forwarded to the Synthesizer. Only `forecast_values` crosses
        the layer boundary.

    Fallback:
        If `is_valid_batch` is False (malformed output), the system
        falls back to sequential one-call-per-step mode.
    """

    model_config = {"frozen": True}

    forecast_values: list[float] = Field(
        ...,
        description=(
            "Forecasted values for each timestep in the horizon. "
            "Length must equal the horizon_weeks. "
            "This is the ONLY field that crosses into Layer 2."
        ),
    )
    step_reasoning: list[str] = Field(
        default_factory=list,
        description=(
            "LLM's per-timestep reasoning strings. "
            "⛔ INTERNAL ONLY — quarantined in _l1_internal_traces. "
            "Must NEVER be forwarded to the Synthesizer (prompt leakage)."
        ),
    )
    horizon_weeks: int = Field(
        ...,
        gt=0,
        description="Expected number of timesteps (must match forecast_values length).",
    )
    is_valid_batch: bool = Field(
        default=True,
        description=(
            "True if the LLM produced a complete, well-formed array. "
            "False triggers fallback to sequential one-call-per-step mode."
        ),
    )
    model_name: str = Field(
        ...,
        description="Which LLM produced this output (e.g. 'gemini-3.1-pro').",
    )

    # ── Validation ────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _forecast_length_matches_horizon(self) -> "BatchedReasoningOutput":
        if self.is_valid_batch and len(self.forecast_values) != self.horizon_weeks:
            raise ValueError(
                f"forecast_values has {len(self.forecast_values)} entries "
                f"but horizon_weeks is {self.horizon_weeks}. "
                "Batch output is malformed — should trigger sequential fallback."
            )
        return self


# ═══════════════════════════════════════════════════════════════════════════
# Calibration Guideline with Fold-Level Validation
# ═══════════════════════════════════════════════════════════════════════════

class FoldResult(BaseModel):
    """Result of applying a candidate guideline to a single training fold."""

    model_config = {"frozen": True}

    fold_index: int = Field(
        ...,
        ge=0,
        description="Which training fold (0-indexed).",
    )
    correction_direction: str = Field(
        ...,
        description=(
            "Direction of correction this guideline suggests for this fold. "
            "One of 'increase', 'decrease', or 'neutral'."
        ),
    )
    mape_before: float = Field(
        ...,
        ge=0.0,
        description="MAPE before applying the guideline on this fold.",
    )
    mape_after: float = Field(
        ...,
        ge=0.0,
        description="MAPE after applying the guideline on this fold.",
    )

    @field_validator("correction_direction")
    @classmethod
    def _valid_direction(cls, v: str) -> str:
        valid = {"increase", "decrease", "neutral"}
        if v not in valid:
            raise ValueError(
                f"correction_direction must be one of {valid}, got '{v}'"
            )
        return v


class CalibrationGuideline(BaseModel):
    """
    A candidate calibration guideline with fold-level consistency validation.

    NEXUS Proposal §3, Mod D (Tightened Calibration):
        "A candidate guideline must show the same direction of correction
        in at least 4 of the 5 training folds individually, not only when
        all folds are combined together, before it is allowed to be tested
        on the held-out validation fold."
    """

    model_config = {"frozen": True}

    guideline_text: str = Field(
        ...,
        max_length=2000,
        description="The calibration guideline in natural language.",
    )
    fold_results: list[FoldResult] = Field(
        ...,
        min_length=1,
        description="Per-fold results for this guideline.",
    )
    dominant_direction: Optional[str] = Field(
        default=None,
        description=(
            "The most common correction direction across folds. "
            "Computed automatically."
        ),
    )
    fold_consistency_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of folds showing the dominant direction. "
            "Must be >= min_consistent_folds to pass validation."
        ),
    )
    min_consistent_folds: int = Field(
        default=4,
        ge=1,
        description=(
            "Minimum number of folds that must show the same direction. "
            "Proposal specifies 4 out of 5."
        ),
    )
    passes_consistency_check: bool = Field(
        default=False,
        description=(
            "True if fold_consistency_count >= min_consistent_folds. "
            "Only guidelines that pass this check may be tested on "
            "the held-out validation fold."
        ),
    )

    @model_validator(mode="after")
    def _compute_consistency(self) -> "CalibrationGuideline":
        """
        Compute fold consistency — pure Python, no LLM.

        Counts how many folds agree on the same correction direction.
        """
        if not self.fold_results:
            return self

        from collections import Counter
        directions = [fr.correction_direction for fr in self.fold_results]
        counts = Counter(directions)
        dominant = counts.most_common(1)[0]

        # Use object.__setattr__ because model is frozen
        object.__setattr__(self, "dominant_direction", dominant[0])
        object.__setattr__(self, "fold_consistency_count", dominant[1])
        object.__setattr__(
            self,
            "passes_consistency_check",
            dominant[1] >= self.min_consistent_folds,
        )
        return self


# ═══════════════════════════════════════════════════════════════════════════
# TightenedCalibrationResult — extends CalibrationResult
# ═══════════════════════════════════════════════════════════════════════════

class TightenedCalibrationResult(BaseModel):
    """
    Post-mortem calibration with the tightened fold-consistency requirement.

    Extends the original CalibrationResult concept with:
      • Per-fold MAPE/RMSE (not just aggregate)
      • Guideline consistency validation
      • Validation-fold results (only for guidelines that passed)

    NEXUS Proposal §3, Mod D:
        This raises the bar for what counts as a trustworthy guideline,
        reducing the risk of coincidental consistency in small datasets
        (15 cities, 7 stock tickers).
    """

    model_config = {"frozen": True}

    mape: float = Field(..., ge=0.0, description="Aggregate MAPE.")
    rmse: float = Field(..., ge=0.0, description="Aggregate RMSE.")
    evaluation_window_start: str = Field(
        ..., description="ISO date for evaluation start."
    )
    evaluation_window_end: str = Field(
        ..., description="ISO date for evaluation end."
    )

    # ── Tightened calibration additions ────────────────────────────────────
    candidate_guidelines: list[CalibrationGuideline] = Field(
        default_factory=list,
        description="All candidate guidelines evaluated.",
    )
    approved_guidelines: list[CalibrationGuideline] = Field(
        default_factory=list,
        description=(
            "Guidelines that passed the fold-consistency check "
            "and were tested on the validation fold."
        ),
    )
    validation_fold_mape: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "MAPE on the held-out validation fold using approved guidelines."
        ),
    )
    feedback_priors: dict = Field(
        default_factory=dict,
        description=(
            "Prompt prior updates for the next calibration cycle. "
            "Based only on approved (fold-consistent) guidelines."
        ),
    )

    @model_validator(mode="after")
    def _approved_subset_of_candidates(self) -> "TightenedCalibrationResult":
        """Approved guidelines must be a subset of candidates."""
        for ag in self.approved_guidelines:
            if not ag.passes_consistency_check:
                raise ValueError(
                    "An approved guideline did not pass the consistency check. "
                    f"Guideline: {ag.guideline_text[:80]}..."
                )
        return self
