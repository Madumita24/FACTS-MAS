# FACTS-MAS / NEXUS State Module
# Contains LangGraph state definitions, Pydantic models, and the factor ontology.

# ── Ontology ──────────────────────────────────────────────────────────────
from facts_mas.state.ontology import (
    FactorType,
    EventType,
    GateTriggerType,
    APPROVED_FACTORS,
    APPROVED_EVENTS,
    APPROVED_GATE_TRIGGERS,
)

# ── Layer 0: Shared Context ──────────────────────────────────────────────
from facts_mas.state.layer0 import (
    ZillowHistory,
    InventoryHistory,
    MacroFeatureColumn,
    MacroFeatures,
    EventRecord,
    CalendarContext,
    NeighborContext,
    SharedContext,
)

# ── Mod A: Smart Early-Exit Gate ─────────────────────────────────────────
from facts_mas.state.gate import GateConfig, GateDecision

# ── Mod B: Spatial Spillover Agent ───────────────────────────────────────
from facts_mas.state.spatial import (
    GrangerEdge,
    SpatialGraph,
    SpilloverSignal,
)

# ── Mod C: Statistical Boundary Constraints ──────────────────────────────
from facts_mas.state.boundary import (
    BoundaryConfig,
    BoundaryViolation,
    BoundaryCheckResult,
)

# ── Mod D: Batched Micro-Reasoning & Calibration ─────────────────────────
from facts_mas.state.micro_reasoning import (
    BatchedReasoningOutput,
    FoldResult,
    CalibrationGuideline,
    TightenedCalibrationResult,
)

# ── Baseline: Honest Control Baseline ────────────────────────────────────
from facts_mas.state.baseline import (
    HonestBaselineConfig,
    SingleModelResult,
    BaselineComparison,
)

# ── Graph State ──────────────────────────────────────────────────────────
from facts_mas.state.graph_state import (
    ARForecast,
    MacroModifier,
    EventImpact,
    FactorWeight,
    NEXUSGraphState,
)

__all__ = [
    # Ontology
    "FactorType",
    "EventType",
    "GateTriggerType",
    "APPROVED_FACTORS",
    "APPROVED_EVENTS",
    "APPROVED_GATE_TRIGGERS",
    # Layer 0
    "ZillowHistory",
    "InventoryHistory",
    "MacroFeatureColumn",
    "MacroFeatures",
    "EventRecord",
    "CalendarContext",
    "NeighborContext",
    "SharedContext",
    # Mod A
    "GateConfig",
    "GateDecision",
    # Mod B
    "GrangerEdge",
    "SpatialGraph",
    "SpilloverSignal",
    # Mod C
    "BoundaryConfig",
    "BoundaryViolation",
    "BoundaryCheckResult",
    # Mod D
    "BatchedReasoningOutput",
    "FoldResult",
    "CalibrationGuideline",
    "TightenedCalibrationResult",
    # Baseline
    "HonestBaselineConfig",
    "SingleModelResult",
    "BaselineComparison",
    # Graph State
    "ARForecast",
    "MacroModifier",
    "EventImpact",
    "FactorWeight",
    "NEXUSGraphState",
]
