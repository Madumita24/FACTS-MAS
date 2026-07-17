"""
FACTS-MAS Layer 0 — Shared Context Builder (Data Payload).

Spec Reference (Part 2, Layer 0):
    "Define a TypedDict or Pydantic model for the Global State. It must contain:
     msa_id, forecast_date, horizon_weeks, X_t, M_t, E_t, C_t."

Design Decisions:
    • Pydantic v2 BaseModel is used (over TypedDict) for runtime validation,
      immutability enforcement via `model_config = frozen`, and clean JSON I/O
      which the spec mandates for every LLM node (response_format="json_object").
    • Each data tensor (X_t, M_t, E_t, C_t) is its own Pydantic model so that
      state-key isolation is enforced at the type level, preventing prompt
      leakage (Spec Part 4, Risk 1).
    • NumPy arrays are wrapped as list[float] in the Pydantic schema for JSON
      serialisability; conversion helpers are provided.

Prompt Leakage Mitigation (Spec §4.1, Gulli p. 482):
    ─────────────────────────────────────────────────────────────────────────
    These models deliberately EXCLUDE any field for raw LLM thought
    trajectories, chain-of-thought logs, or internal prompts.  Layer 1 agent
    outputs are captured ONLY as numerical results and typed classifications.
    The Synthesizer (Layer 2) therefore never receives prompt internals.
    ─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import datetime
from typing import Annotated, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from facts_mas.state.ontology import EventType, FactorType


# ═══════════════════════════════════════════════════════════════════════════
# X_t — Zillow Numerical History
# ═══════════════════════════════════════════════════════════════════════════

class ZillowHistory(BaseModel):
    """
    Weekly Zillow Home Value Index (ZHVI) time-series for a single MSA.

    Fields carry only numerical data — no LLM text, no internal prompts.
    """

    model_config = {"frozen": True}

    dates: list[datetime.date] = Field(
        ...,
        description="ISO-8601 date stamps for each weekly observation.",
    )
    values: list[float] = Field(
        ...,
        description="ZHVI smoothed home values (USD), aligned 1-to-1 with `dates`.",
    )
    series_id: str = Field(
        ...,
        description="Zillow region/series identifier (e.g. 'ZHVI-SFR-394463').",
    )

    # -- Validation ----------------------------------------------------------

    @model_validator(mode="after")
    def _dates_values_aligned(self) -> "ZillowHistory":
        if len(self.dates) != len(self.values):
            raise ValueError(
                f"dates ({len(self.dates)}) and values ({len(self.values)}) "
                "must have the same length."
            )
        return self

    # -- Helpers for NumPy consumers (AR Agent / TimesFM) --------------------

    def to_numpy(self) -> np.ndarray:
        """Return values as a 1-D float64 array for TimesFM / numpy math."""
        return np.array(self.values, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# I_t — Zillow Inventory History (NEXUS primary forecast target)
# ═══════════════════════════════════════════════════════════════════════════

class InventoryHistory(BaseModel):
    """
    Weekly Zillow inventory (for-sale listing count) time-series for an MSA.

    NEXUS Proposal Context:
        The NEXUS architecture forecasts *inventory*, not price. This model
        carries the historical inventory series that TimesFM and the
        multi-agent loop consume. Boundary constraints (Mod C) enforce
        non-negativity on this series.

    Fields carry only numerical data — no LLM text, no internal prompts.
    """

    model_config = {"frozen": True}

    dates: list[datetime.date] = Field(
        ...,
        description="ISO-8601 date stamps for each weekly observation.",
    )
    values: list[float] = Field(
        ...,
        description=(
            "Weekly for-sale inventory counts, aligned 1-to-1 with `dates`. "
            "Must be non-negative (physical floor constraint)."
        ),
    )
    series_id: str = Field(
        ...,
        description="Zillow region/series identifier (e.g. 'INVEN-SFR-38060').",
    )

    # -- Validation ----------------------------------------------------------

    @model_validator(mode="after")
    def _dates_values_aligned(self) -> "InventoryHistory":
        if len(self.dates) != len(self.values):
            raise ValueError(
                f"dates ({len(self.dates)}) and values ({len(self.values)}) "
                "must have the same length."
            )
        return self

    @field_validator("values")
    @classmethod
    def _non_negative(cls, vals: list[float]) -> list[float]:
        """Inventory counts can never be negative (Mod C physical floor)."""
        for i, v in enumerate(vals):
            if v < 0:
                raise ValueError(
                    f"Inventory value at index {i} is {v}; "
                    "inventory counts cannot be negative."
                )
        return vals

    # -- Helpers -------------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        """Return values as a 1-D float64 array for TimesFM / numpy math."""
        return np.array(self.values, dtype=np.float64)

    def pct_changes(self) -> np.ndarray:
        """Week-over-week percentage changes (for Mod C boundary checks)."""
        arr = self.to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.diff(arr) / arr[:-1]
        return pct


# ═══════════════════════════════════════════════════════════════════════════
# M_t — Macro Features (FRED)
# ═══════════════════════════════════════════════════════════════════════════

class MacroFeatureColumn(BaseModel):
    """A single lagged macro feature column (e.g. 'mortgage_rate_30y')."""

    model_config = {"frozen": True}

    factor: FactorType = Field(
        ...,
        description="Must be a member of the approved static factor ontology.",
    )
    fred_series_id: str = Field(
        ...,
        description="FRED API series identifier (e.g. 'MORTGAGE30US').",
    )
    values: list[Optional[float]] = Field(
        ...,
        description=(
            "Macro observations aligned to X_t dates. "
            "None for missing values after shift/lag."
        ),
    )
    lag_weeks: int = Field(
        ...,
        ge=0,
        description=(
            "Number of weeks the raw series was shifted forward "
            "to prevent look-ahead bias. "
            "Spec §4.3: df['macro_feature'] = df['macro_feature'].shift(lag_weeks)."
        ),
    )


class MacroFeatures(BaseModel):
    """
    Collection of lagged macroeconomic feature columns for an MSA.

    Spec §4.3 (Look-Ahead Bias): Every feature MUST carry a `lag_weeks ≥ 0`
    value proving it was shifted before entering the state.
    """

    model_config = {"frozen": True}

    columns: list[MacroFeatureColumn] = Field(
        ...,
        min_length=1,
        description="One entry per macro feature included in this forecast run.",
    )

    # -- Validation ----------------------------------------------------------

    @field_validator("columns")
    @classmethod
    def _no_duplicate_factors(
        cls, cols: list[MacroFeatureColumn],
    ) -> list[MacroFeatureColumn]:
        seen = set()
        for c in cols:
            if c.factor in seen:
                raise ValueError(f"Duplicate macro factor: {c.factor.value}")
            seen.add(c.factor)
        return cols

    @field_validator("columns")
    @classmethod
    def _all_factors_in_ontology(
        cls, cols: list[MacroFeatureColumn],
    ) -> list[MacroFeatureColumn]:
        """Guard-rail: reject any factor not in the static enum."""
        for c in cols:
            if not isinstance(c.factor, FactorType):
                raise ValueError(
                    f"Factor '{c.factor}' is not in the approved ontology. "
                    "Modification requires user approval (Prompt Injection Gate)."
                )
        return cols

    # -- Helpers -------------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        """Return (T × F) float64 matrix — NaN for missing lag entries."""
        return np.column_stack(
            [
                np.array(
                    [v if v is not None else np.nan for v in col.values],
                    dtype=np.float64,
                )
                for col in self.columns
            ]
        )

    def factor_names(self) -> list[str]:
        return [col.factor.value for col in self.columns]


# ═══════════════════════════════════════════════════════════════════════════
# E_t — Unstructured News / Events
# ═══════════════════════════════════════════════════════════════════════════

class EventRecord(BaseModel):
    """
    A single event extracted by the Event Agent (Layer 1).

    Spec §3 Prompt Rules:
        "If no text matches the ontology, return an empty array []."

    Prompt Leakage Mitigation:
        This model stores ONLY the classification output — never the raw LLM
        chain-of-thought or internal reasoning trajectory.
    """

    model_config = {"frozen": True}

    event_type: EventType = Field(
        ...,
        description="Must be one of the 4 approved event types.",
    )
    description: str = Field(
        ...,
        max_length=500,
        description=(
            "Brief factual summary of the event. "
            "Must NOT contain LLM internal reasoning or prompt fragments."
        ),
    )
    source_url: Optional[str] = Field(
        default=None,
        description="URL of the original news article, if available.",
    )
    event_date: datetime.date = Field(
        ...,
        description="Date the event occurred or was reported.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Event Agent's self-assessed confidence. "
            "Spec §2 Approval Gate: if confidence < 0.60, "
            "impact_magnitude MUST be forced to 0."
        ),
    )
    impact_magnitude: float = Field(
        ...,
        description=(
            "Signed magnitude of the event's expected price impact "
            "(e.g. -0.03 = −3% shock). Zeroed when confidence < 0.60."
        ),
    )

    # -- Spec Approval Gate enforcement (§2, Layer 1) ------------------------

    @model_validator(mode="after")
    def _enforce_confidence_gate(self) -> "EventRecord":
        """
        Spec: 'Write a Python if statement validating the LLM's confidence
        score. If confidence < 0.60, force impact_magnitude = 0.'
        """
        if self.confidence < 0.60 and self.impact_magnitude != 0.0:
            raise ValueError(
                f"Confidence is {self.confidence:.2f} (< 0.60) but "
                f"impact_magnitude is {self.impact_magnitude}, must be 0.0. "
                "Spec §2 Approval Gate violated."
            )
        return self


# ═══════════════════════════════════════════════════════════════════════════
# C_t — Calendar / Seasonality Context
# ═══════════════════════════════════════════════════════════════════════════

class CalendarContext(BaseModel):
    """
    Deterministic calendar and seasonality features for the forecast window.

    These are computed in pure Python (no LLM involvement).
    """

    model_config = {"frozen": True}

    week_of_year: list[int] = Field(
        ...,
        description="ISO week numbers for the forecast horizon.",
    )
    month: list[int] = Field(
        ...,
        description="Month numbers (1-12) for the forecast horizon.",
    )
    quarter: list[int] = Field(
        ...,
        description="Quarter numbers (1-4) for the forecast horizon.",
    )
    is_spring_selling_season: list[bool] = Field(
        ...,
        description="True for weeks within the spring selling season (weeks ~13-26).",
    )
    is_holiday_period: list[bool] = Field(
        ...,
        description="True for weeks containing major US holidays (Thanksgiving–New Year, etc.).",
    )

    @model_validator(mode="after")
    def _lengths_consistent(self) -> "CalendarContext":
        lengths = {
            "week_of_year": len(self.week_of_year),
            "month": len(self.month),
            "quarter": len(self.quarter),
            "is_spring_selling_season": len(self.is_spring_selling_season),
            "is_holiday_period": len(self.is_holiday_period),
        }
        unique = set(lengths.values())
        if len(unique) != 1:
            raise ValueError(
                f"All calendar arrays must have the same length, got {lengths}."
            )
        return self


# ═══════════════════════════════════════════════════════════════════════════
# N_t — Neighbor Context (NEXUS Mod B: Spatial Spillover)
# ═══════════════════════════════════════════════════════════════════════════

class NeighborContext(BaseModel):
    """
    Lagged inventory trends from a Granger-identified neighbor MSA.

    NEXUS Proposal §3, Mod B:
        "When a forecast is being generated for a target city, this agent
        identifies that city's most strongly connected neighbors based on
        the Granger test results. It then pulls those neighboring cities'
        recent inventory trends, shifted backward in time to reflect
        realistic spillover lag (e.g. 4 to 8 weeks ago)."

    Prompt Leakage Isolation:
        This model carries only numerical inventory data from a neighbor,
        never any LLM reasoning or prompts from that neighbor's forecast.
    """

    model_config = {"frozen": True}

    neighbor_msa_id: str = Field(
        ...,
        description="CBSA/MSA FIPS code of the neighbor city.",
    )
    granger_p_value: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Granger causality test p-value (lower = stronger link).",
    )
    spillover_lag_weeks: int = Field(
        ...,
        ge=1,
        description=(
            "How many weeks to shift the neighbor's data backward. "
            "Typically 4-8 weeks per the proposal."
        ),
    )
    recent_inventory_values: list[float] = Field(
        ...,
        description=(
            "Neighbor's recent inventory values, already shifted by "
            "`spillover_lag_weeks` to prevent look-ahead bias."
        ),
    )
    recent_inventory_dates: list[datetime.date] = Field(
        ...,
        description="Dates corresponding to the shifted inventory values.",
    )
    recent_pct_changes: list[float] = Field(
        default_factory=list,
        description=(
            "Week-over-week percentage changes in the neighbor's inventory. "
            "Used by the gate to detect neighbor shocks (Mod A↔B connection)."
        ),
    )

    @model_validator(mode="after")
    def _lengths_aligned(self) -> "NeighborContext":
        if len(self.recent_inventory_values) != len(self.recent_inventory_dates):
            raise ValueError(
                "recent_inventory_values and recent_inventory_dates "
                "must have the same length."
            )
        return self


# ═══════════════════════════════════════════════════════════════════════════
# SharedContext — The complete Layer 0 payload
# ═══════════════════════════════════════════════════════════════════════════

class SharedContext(BaseModel):
    """
    Layer 0 Shared Context — the immutable data payload assembled before
    Layer 1 agents execute.

    Spec §2 (Layer 0):
        "Define a TypedDict or Pydantic model for the Global State. It must
        contain: msa_id, forecast_date, horizon_weeks, X_t, M_t, E_t, C_t."

    NEXUS Enhancements:
        • I_t: Inventory history (primary forecast target per NEXUS proposal)
        • N_t: Neighbor contexts from Granger-identified cities (Mod B)

    Prompt Leakage Isolation (Spec §4.1):
        This model contains ONLY data payloads. There are zero fields for
        system prompts, chain-of-thought traces, or LLM internals.
        Layer 1 raw reasoning is captured in separate, non-shared state keys
        (see graph_state.py) that the Synthesizer cannot access.
    """

    model_config = {"frozen": True}

    # ── Identity & Scope ──────────────────────────────────────────────────
    msa_id: str = Field(
        ...,
        description="CBSA/MSA FIPS code identifying the metro area.",
    )
    forecast_date: datetime.date = Field(
        ...,
        description="Anchor date: the first day of the forecast window.",
    )
    horizon_weeks: int = Field(
        ...,
        gt=0,
        le=52,
        description="Number of weeks to forecast (typically 4 or up to 26).",
    )

    # ── Data Tensors (Spec-mandated keys) ─────────────────────────────────
    X_t: ZillowHistory = Field(
        ...,
        description="Zillow numerical history (ZHVI weekly price series).",
    )
    M_t: MacroFeatures = Field(
        ...,
        description="Lagged macro features from FRED.",
    )
    E_t: list[EventRecord] = Field(
        default_factory=list,
        description=(
            "Extracted events from unstructured news/text. "
            "Empty list when no events match the ontology."
        ),
    )
    C_t: CalendarContext = Field(
        ...,
        description="Calendar and seasonality features for the horizon.",
    )

    # ── NEXUS Extensions ──────────────────────────────────────────────────
    I_t: Optional[InventoryHistory] = Field(
        default=None,
        description=(
            "Zillow inventory history (primary NEXUS forecast target). "
            "Optional for backward compat with price-only runs."
        ),
    )
    N_t: list[NeighborContext] = Field(
        default_factory=list,
        description=(
            "Spatial spillover contexts from Granger-identified neighbor MSAs "
            "(NEXUS Mod B). Empty when running single-city mode."
        ),
    )
