"""
NEXUS Honest Control Baseline (Phase 0 — Section 2).

NEXUS Proposal §2:
    "Before introducing any new architectural component, we will first
    isolate exactly how much of NEXUS's reported performance advantage
    comes from its multi-agent reasoning versus how much comes simply
    from giving it cleaner input data than its baseline received."

    Step 1: Use the IR+Contextualization layer to produce a clean timeline H.
    Step 2: Feed identical H into both the multi-agent system AND a
            single-pass Chain-of-Thought model.
    Step 3: Compare MAPE and RMSE side by side.

Design:
    • HonestBaselineConfig: which models to run and at what temperature
    • SingleModelResult: one model's forecast + metrics
    • BaselineComparison: side-by-side comparison of multi-agent vs. CoT

    All metric computation is pure Python/NumPy. The LLM is only used
    to *generate* forecasts, never to compute metrics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════════════════
# Baseline Configuration
# ═══════════════════════════════════════════════════════════════════════════

class HonestBaselineConfig(BaseModel):
    """
    Configuration for the Honest Control Baseline experiment.

    NEXUS Proposal §2.1:
        Both systems receive the identical cleaned timeline (H_1..τ).
        Temperature locked at 0.1 for determinism.
    """

    model_config = {"frozen": True}

    cot_models: list[str] = Field(
        default_factory=lambda: ["gemini-3.1-pro", "claude-4.5-sonnet"],
        description=(
            "Models to run single-pass Chain-of-Thought against. "
            "These serve as the control baseline."
        ),
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Sampling temperature for all models. "
            "Locked at 0.1 per proposal §5 for determinism."
        ),
    )
    response_format: str = Field(
        default="json_object",
        description="LLM response format. Must be json_object per spec §1.1.",
    )
    num_repetitions: int = Field(
        default=3,
        ge=1,
        le=5,
        description=(
            "Number of times to run each model (3-5 per Mod A's added value). "
            "Enables reporting variance/confidence intervals."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Single Model Result
# ═══════════════════════════════════════════════════════════════════════════

class SingleModelResult(BaseModel):
    """
    Forecast result from a single model (either CoT baseline or multi-agent).

    Metrics are computed in pure Python/NumPy — no LLM math.
    """

    model_config = {"frozen": True}

    model_name: str = Field(
        ...,
        description="Model identifier (e.g. 'gemini-3.1-pro' or 'nexus-multi-agent').",
    )
    system_type: str = Field(
        ...,
        description="One of 'cot_baseline' or 'multi_agent'.",
    )
    forecast_values: list[float] = Field(
        ...,
        description="Forecasted values for the horizon.",
    )
    actual_values: Optional[list[float]] = Field(
        default=None,
        description="Ground-truth values (filled in post-hoc for evaluation).",
    )
    mape: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Mean Absolute Percentage Error (computed post-hoc).",
    )
    rmse: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Root Mean Square Error (computed post-hoc).",
    )
    repetition_index: int = Field(
        default=0,
        ge=0,
        description="Which repetition this result is from (for variance reporting).",
    )

    @field_validator("system_type")
    @classmethod
    def _valid_system(cls, v: str) -> str:
        valid = {"cot_baseline", "multi_agent"}
        if v not in valid:
            raise ValueError(f"system_type must be one of {valid}, got '{v}'")
        return v

    # -- Metric computation (pure NumPy) -------------------------------------

    def compute_metrics(
        self, actual_values: list[float]
    ) -> "SingleModelResult":
        """
        Compute MAPE and RMSE from actuals. Returns a new (frozen) instance.
        Pure Python/NumPy — no LLM math.
        """
        forecast = np.array(self.forecast_values, dtype=np.float64)
        actual = np.array(actual_values, dtype=np.float64)

        # MAPE
        with np.errstate(divide="ignore", invalid="ignore"):
            ape = np.abs((actual - forecast) / actual)
        ape = ape[np.isfinite(ape)]
        mape = float(np.mean(ape)) if len(ape) > 0 else float("inf")

        # RMSE
        rmse = float(np.sqrt(np.mean((actual - forecast) ** 2)))

        return SingleModelResult(
            model_name=self.model_name,
            system_type=self.system_type,
            forecast_values=self.forecast_values,
            actual_values=actual_values,
            mape=mape,
            rmse=rmse,
            repetition_index=self.repetition_index,
        )



# ═══════════════════════════════════════════════════════════════════════════
# Baseline Comparison — side-by-side analysis
# ═══════════════════════════════════════════════════════════════════════════

class BaselineComparison(BaseModel):
    """
    Side-by-side comparison of multi-agent vs. CoT baseline results.

    NEXUS Proposal §2.2:
        "If the accuracy gap shrinks significantly once both receive the
        same clean input, that tells us the original paper's reported
        advantage was partly an artifact of unequal data preparation."

    This is a reportable research finding either way.
    """

    model_config = {"frozen": True}

    msa_id: str = Field(
        ...,
        description="MSA this comparison applies to.",
    )
    horizon_weeks: int = Field(
        ...,
        gt=0,
        description="Forecast horizon used.",
    )
    cot_results: list[SingleModelResult] = Field(
        ...,
        description="All CoT baseline results (across models and repetitions).",
    )
    multi_agent_results: list[SingleModelResult] = Field(
        ...,
        description="All multi-agent results (across repetitions).",
    )

    # ── Computed summary statistics ───────────────────────────────────────

    cot_mean_mape: Optional[float] = Field(
        default=None,
        description="Mean MAPE across all CoT results.",
    )
    multi_agent_mean_mape: Optional[float] = Field(
        default=None,
        description="Mean MAPE across all multi-agent results.",
    )
    gap_reduction_pct: Optional[float] = Field(
        default=None,
        description=(
            "Percentage reduction in the MAPE gap vs. original reported gap. "
            "Positive = multi-agent advantage shrinks with fair input. "
            "Negative = multi-agent advantage grows (real architectural value)."
        ),
    )
    cot_mape_std: Optional[float] = Field(
        default=None,
        description="Standard deviation of CoT MAPE (from multiple repetitions).",
    )
    multi_agent_mape_std: Optional[float] = Field(
        default=None,
        description="Standard deviation of multi-agent MAPE.",
    )

    @model_validator(mode="after")
    def _compute_summaries(self) -> "BaselineComparison":
        """Compute summary statistics from individual results. Pure NumPy."""
        cot_mapes = [r.mape for r in self.cot_results if r.mape is not None]
        ma_mapes = [r.mape for r in self.multi_agent_results if r.mape is not None]

        if cot_mapes:
            object.__setattr__(self, "cot_mean_mape", float(np.mean(cot_mapes)))
            object.__setattr__(self, "cot_mape_std", float(np.std(cot_mapes)))

        if ma_mapes:
            object.__setattr__(self, "multi_agent_mean_mape", float(np.mean(ma_mapes)))
            object.__setattr__(self, "multi_agent_mape_std", float(np.std(ma_mapes)))

        return self
