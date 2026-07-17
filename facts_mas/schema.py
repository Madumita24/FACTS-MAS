"""
FACTS-MAS Phase 2: Shared Agent I/O Schema.

This is the "schema before code" artifact (PDF §5): every agent — regardless
of owner — must return an AgentOutput that conforms to this contract.
Locked on Day 1-2 so that Madumita's agents and Umang's agents can be
built and tested independently without blocking each other.

Columns in the data:
    aligned_weekly.csv:
        date, msa, inventory_count, mortgage_rate, fed_funds, cpi, unemployment
    intrinsic_static.csv:
        msa, population, median_income, housing_units, density, hazard_index
"""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════════════════
# AgentOutput — universal contract for every agent
# ═══════════════════════════════════════════════════════════════════════════

class AgentOutput(BaseModel):
    """
    Universal output contract that every agent must return.

    Enforces:
        - forecast length == requested horizon
        - non-negative inventory forecasts
        - valid agent name
        - optional per-step confidence scores
    """

    model_config = {"frozen": True}

    agent_name: str = Field(
        ...,
        description=(
            "Identifier for the agent: 'ar', 'macro', 'event', "
            "'seasonality', or 'intrinsic'."
        ),
    )
    msa: str = Field(
        ...,
        description="MSA name (must match aligned_weekly.csv 'msa' column).",
    )
    forecast_origin: datetime.date = Field(
        ...,
        description=(
            "The 'as-of' date: the last date of data the agent was allowed "
            "to see when producing this forecast."
        ),
    )
    horizon_weeks: int = Field(
        ...,
        gt=0,
        description="Requested forecast horizon (4, 8, or 13 weeks).",
    )
    values: list[float] = Field(
        ...,
        description=(
            "Forecasted inventory counts for each week in the horizon. "
            "Length must exactly equal horizon_weeks."
        ),
    )
    confidence: Optional[list[float]] = Field(
        default=None,
        description=(
            "Optional per-step confidence scores in [0, 1]. "
            "If provided, length must match horizon_weeks."
        ),
    )

    # ── Validation ────────────────────────────────────────────────────────

    @field_validator("agent_name")
    @classmethod
    def _valid_agent_name(cls, v: str) -> str:
        valid = {"ar", "macro", "event", "seasonality", "intrinsic"}
        if v not in valid:
            raise ValueError(f"agent_name must be one of {valid}, got '{v}'")
        return v

    @model_validator(mode="after")
    def _forecast_length(self) -> "AgentOutput":
        if len(self.values) != self.horizon_weeks:
            raise ValueError(
                f"values has {len(self.values)} entries but "
                f"horizon_weeks is {self.horizon_weeks}."
            )
        return self

    @model_validator(mode="after")
    def _non_negative(self) -> "AgentOutput":
        for i, v in enumerate(self.values):
            if v < 0:
                raise ValueError(
                    f"Inventory forecast must be non-negative. "
                    f"Got {v} at index {i}."
                )
        return self

    @model_validator(mode="after")
    def _confidence_length(self) -> "AgentOutput":
        if self.confidence is not None:
            if len(self.confidence) != self.horizon_weeks:
                raise ValueError(
                    f"confidence has {len(self.confidence)} entries but "
                    f"horizon_weeks is {self.horizon_weeks}."
                )
            for i, c in enumerate(self.confidence):
                if not (0.0 <= c <= 1.0):
                    raise ValueError(
                        f"confidence[{i}] = {c} is outside [0, 1]."
                    )
        return self


# ═══════════════════════════════════════════════════════════════════════════
# FusionInput — a complete set of agent outputs ready for fusion
# ═══════════════════════════════════════════════════════════════════════════

class FusionInput(BaseModel):
    """
    A bundle of all agent outputs for a single (msa, forecast_origin, horizon)
    combination, ready to be passed to the fusion layer.
    """

    model_config = {"frozen": True}

    msa: str
    forecast_origin: datetime.date
    horizon_weeks: int
    agent_outputs: dict[str, AgentOutput] = Field(
        ...,
        description=(
            "Keyed by agent_name. Must contain all expected agents."
        ),
    )

    @model_validator(mode="after")
    def _all_agents_match(self) -> "FusionInput":
        for name, out in self.agent_outputs.items():
            if out.agent_name != name:
                raise ValueError(
                    f"Key '{name}' doesn't match agent_name '{out.agent_name}'"
                )
            if out.msa != self.msa:
                raise ValueError(
                    f"Agent '{name}' msa '{out.msa}' doesn't match "
                    f"FusionInput msa '{self.msa}'"
                )
            if out.horizon_weeks != self.horizon_weeks:
                raise ValueError(
                    f"Agent '{name}' horizon {out.horizon_weeks} doesn't "
                    f"match FusionInput horizon {self.horizon_weeks}"
                )
        return self


# ═══════════════════════════════════════════════════════════════════════════
# BacktestResult — per-fold, per-MSA, per-horizon metrics
# ═══════════════════════════════════════════════════════════════════════════

class BacktestResult(BaseModel):
    """
    Metrics for one (fold, msa, horizon) combination.

    Per evaluation_protocol.md:
        MAPE and RMSE per MSA per horizon, then averaged across MSAs.
        Horizons (4/8/13wk) reported separately, never averaged.
        Regime-stratified: hiking/cutting/stable.
    """

    model_config = {"frozen": True}

    fold: int = Field(..., ge=1, le=6)
    msa: str
    horizon_weeks: int = Field(..., description="4, 8, or 13")
    mape: float = Field(..., ge=0.0)
    rmse: float = Field(..., ge=0.0)
    regime: str = Field(
        ...,
        description="One of 'hiking', 'cutting', 'stable'.",
    )
    n_origins: int = Field(
        ...,
        ge=1,
        description="Number of forecast origins in this fold's test window.",
    )
    agent_weights: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Fusion weights used for this fold. Logged to confirm "
            "they change fold-to-fold (not frozen)."
        ),
    )

    @field_validator("regime")
    @classmethod
    def _valid_regime(cls, v: str) -> str:
        valid = {"hiking", "cutting", "stable"}
        if v not in valid:
            raise ValueError(f"regime must be one of {valid}, got '{v}'")
        return v

    @field_validator("horizon_weeks")
    @classmethod
    def _valid_horizon(cls, v: int) -> int:
        valid = {4, 8, 13}
        if v not in valid:
            raise ValueError(f"horizon_weeks must be one of {valid}, got {v}")
        return v
