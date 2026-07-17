"""
NEXUS Modification C: Statistical Boundary Constraints.

NEXUS Proposal §3, Mod C:
    "The boundary is placed on the week-over-week percentage change between
    consecutive forecasted values. [...] We also apply an absolute physical
    floor, since inventory counts can never be negative."

    "This correction process is allowed to repeat a maximum of two times.
    If the model still produces an out-of-bounds value after two correction
    attempts, the system falls back to the lightweight TimesFM baseline
    value for that specific timestep."

Design:
    • BoundaryConfig: thresholds derived from historical data (pure Python)
    • BoundaryViolation: details of a specific violation
    • BoundaryCheckResult: the complete check output including retry tracking

    All boundary math is pure Python/NumPy. The LLM is only invoked
    *if* a violation is detected, to ask it to revise — and even then,
    the retry limit guarantees termination with a baseline fallback.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════════════════
# Boundary Configuration
# ═══════════════════════════════════════════════════════════════════════════

class BoundaryConfig(BaseModel):
    """
    Statistical boundary constraints derived from a city's historical data.

    Computed once per MSA from the historical inventory time-series.
    These are deterministic — no LLM involvement.

    NEXUS Proposal §3, Mod C:
        "If the synthesizer's output contains a week-over-week change larger
        than the largest such change ever observed historically for that city,
        or if it produces a negative inventory value..."
    """

    model_config = {"frozen": True}

    msa_id: str = Field(
        ...,
        description="MSA FIPS code these boundaries apply to.",
    )
    max_historical_pct_change: float = Field(
        ...,
        gt=0.0,
        description=(
            "The largest absolute week-over-week percentage change ever "
            "observed in this city's historical inventory series. "
            "Forecasted changes exceeding this are flagged."
        ),
    )
    physical_floor: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Absolute minimum allowed value. 0 for inventory counts "
            "(cannot be negative). Could be adjusted for other targets."
        ),
    )
    max_correction_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description=(
            "Maximum number of correction attempts before falling back "
            "to the TimesFM baseline. Proposal specifies 2."
        ),
    )

    # -- Factory helper (pure NumPy) -----------------------------------------

    @classmethod
    def from_historical_series(
        cls,
        msa_id: str,
        historical_values: list[float],
        *,
        max_retries: int = 2,
    ) -> "BoundaryConfig":
        """
        Compute boundary config from a historical inventory series.
        Pure Python/NumPy — no LLM math.
        """
        arr = np.array(historical_values, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.abs(np.diff(arr) / arr[:-1])
        # Filter out inf/nan from zero-valued weeks
        pct = pct[np.isfinite(pct)]
        max_pct = float(np.max(pct)) if len(pct) > 0 else 1.0
        return cls(
            msa_id=msa_id,
            max_historical_pct_change=max_pct,
            max_correction_retries=max_retries,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Boundary Violation — details of a single out-of-bounds forecast
# ═══════════════════════════════════════════════════════════════════════════

class BoundaryViolation(BaseModel):
    """A specific boundary violation in the forecasted series."""

    model_config = {"frozen": True}

    timestep_index: int = Field(
        ...,
        ge=0,
        description="Index into the forecast array where the violation occurred.",
    )
    violation_type: str = Field(
        ...,
        description="One of 'excessive_pct_change' or 'below_physical_floor'.",
    )
    forecasted_value: float = Field(
        ...,
        description="The violating forecast value.",
    )
    allowed_range_description: str = Field(
        ...,
        description=(
            "Human-readable description of the allowed range. "
            "E.g. 'Max allowed pct change: ±15.3%, observed: -22.1%' "
            "Sent to the LLM in the correction instruction."
        ),
    )

    @field_validator("violation_type")
    @classmethod
    def _valid_type(cls, v: str) -> str:
        valid = {"excessive_pct_change", "below_physical_floor"}
        if v not in valid:
            raise ValueError(f"violation_type must be one of {valid}, got '{v}'")
        return v


# ═══════════════════════════════════════════════════════════════════════════
# BoundaryCheckResult — the complete check output
# ═══════════════════════════════════════════════════════════════════════════

class BoundaryCheckResult(BaseModel):
    """
    Result of applying boundary constraints to a forecast.

    NEXUS Proposal §3, Mod C retry logic:
        Attempt 1 → check → if violation → send back to LLM with correction
        Attempt 2 → check → if violation → send back to LLM with correction
        Attempt 3 → check → if still violating → fallback to TimesFM baseline

    Prompt Leakage Isolation:
        This model carries numerical check results and structured violation
        details. It does NOT carry LLM reasoning or correction prompts.
        The correction instruction is generated at the graph-node level.
    """

    model_config = {"frozen": True}

    is_within_bounds: bool = Field(
        ...,
        description="True if the forecast passed all boundary checks.",
    )
    violations: list[BoundaryViolation] = Field(
        default_factory=list,
        description="List of specific violations found. Empty if within bounds.",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of correction retries already attempted.",
    )
    fell_back_to_baseline: bool = Field(
        default=False,
        description=(
            "True if the retry limit was exceeded and the system fell back "
            "to the TimesFM baseline for violating timesteps."
        ),
    )
    corrected_forecast: Optional[list[float]] = Field(
        default=None,
        description=(
            "The corrected forecast values, if corrections were applied. "
            "None if the original forecast was within bounds. "
            "Contains baseline-substituted values if fell_back_to_baseline."
        ),
    )

    # ── Validation ────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _consistency(self) -> "BoundaryCheckResult":
        if self.is_within_bounds and len(self.violations) > 0:
            raise ValueError(
                "is_within_bounds is True but violations were found."
            )
        if not self.is_within_bounds and len(self.violations) == 0:
            raise ValueError(
                "is_within_bounds is False but no violations listed."
            )
        return self

    # -- Factory: run the actual boundary check (pure Python) ---------------

    @classmethod
    def check_forecast(
        cls,
        forecast_values: list[float],
        last_historical_value: float,
        config: "BoundaryConfig",
        retry_count: int = 0,
        baseline_forecast: Optional[list[float]] = None,
    ) -> "BoundaryCheckResult":
        """
        Check a forecast against boundary constraints.
        Pure Python/NumPy — no LLM involved.

        Args:
            forecast_values: The forecast to check.
            last_historical_value: Last observed value (for computing
                the first week-over-week change).
            config: Boundary configuration for this MSA.
            retry_count: How many correction attempts have been made.
            baseline_forecast: TimesFM baseline (for fallback).

        Returns:
            BoundaryCheckResult with violations and/or corrections.
        """
        violations = []
        full_series = [last_historical_value] + list(forecast_values)

        for i, val in enumerate(forecast_values):
            # Physical floor check
            if val < config.physical_floor:
                violations.append(BoundaryViolation(
                    timestep_index=i,
                    violation_type="below_physical_floor",
                    forecasted_value=val,
                    allowed_range_description=(
                        f"Physical floor: {config.physical_floor}. "
                        f"Forecasted: {val:.2f}"
                    ),
                ))
                continue

            # Week-over-week pct change check
            prev = full_series[i]  # previous value (shifted by 1)
            if prev != 0:
                pct_change = abs((val - prev) / prev)
                if pct_change > config.max_historical_pct_change:
                    violations.append(BoundaryViolation(
                        timestep_index=i,
                        violation_type="excessive_pct_change",
                        forecasted_value=val,
                        allowed_range_description=(
                            f"Max allowed absolute pct change: "
                            f"{config.max_historical_pct_change:.4f} "
                            f"({config.max_historical_pct_change*100:.1f}%). "
                            f"Observed: {pct_change:.4f} ({pct_change*100:.1f}%)"
                        ),
                    ))

        if not violations:
            return cls(
                is_within_bounds=True,
                violations=[],
                retry_count=retry_count,
            )

        # Check if we've exceeded the retry limit
        exceeded = retry_count >= config.max_correction_retries
        corrected = None
        if exceeded and baseline_forecast is not None:
            # Fall back to baseline for violating timesteps
            corrected = list(forecast_values)
            violating_indices = {v.timestep_index for v in violations}
            for idx in violating_indices:
                if idx < len(baseline_forecast):
                    corrected[idx] = baseline_forecast[idx]

        return cls(
            is_within_bounds=False,
            violations=violations,
            retry_count=retry_count,
            fell_back_to_baseline=exceeded,
            corrected_forecast=corrected,
        )
