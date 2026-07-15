"""
NEXUS LangGraph Graph State (TypedDict).

This TypedDict is the central State object passed between LangGraph nodes.
It implements the full NEXUS optimized architecture with all 4 modifications:

    Mod A: Smart Early-Exit Gate     → gate_decision
    Mod B: Spatial Spillover Agent   → spatial_graph, spillover_signal
    Mod C: Boundary Constraints      → boundary_config, boundary_check
    Mod D: Batched Micro-Reasoning   → batched_reasoning, calibration

Architecture:
    ┌──────────────────────────────────────────────────────────────────────┐
    │                       NEXUSGraphState                               │
    │                                                                      │
    │  ┌─── Layer 0 ───────────┐   Shared, immutable data payload.       │
    │  │  shared_ctx            │   ALL agents may read this.             │
    │  │  (includes I_t, N_t)   │   Includes inventory + neighbor data.  │
    │  └────────────────────────┘                                         │
    │                                                                      │
    │  ┌─── Gate (Mod A) ──────┐   Runs BEFORE Layer 1.                  │
    │  │  gate_decision         │   Decides: skip to baseline or enter   │
    │  │                        │   the heavy multi-agent loop.           │
    │  └────────────────────────┘                                         │
    │                                                                      │
    │  ┌─── Spatial (Mod B) ───┐   Pre-computed Granger graph.           │
    │  │  spatial_graph         │   System-level (loaded at startup).     │
    │  │  spillover_signal      │   Per-forecast spillover context.       │
    │  └────────────────────────┘                                         │
    │                                                                      │
    │  ┌─── Layer 1 (ISOLATED) ────────────────────────────────────┐     │
    │  │  ar_forecast          ← AR Agent (TimesFM) numerical out  │     │
    │  │  macro_modifiers      ← Macro Agent numerical out         │     │
    │  │  event_impacts        ← Event Agent classifications       │     │
    │  │  batched_reasoning    ← Mod D single-call output          │     │
    │  │                                                            │     │
    │  │  _l1_internal_traces  ← RAW LLM thought logs             │     │
    │  │     ⛔ NEVER forwarded to Synthesizer                     │     │
    │  └────────────────────────────────────────────────────────────┘     │
    │                                                                      │
    │  ┌─── Boundary (Mod C) ──┐   Post-synthesis constraint check.      │
    │  │  boundary_config       │   Historical pct-change limits.         │
    │  │  boundary_check        │   Check result + retry tracking.        │
    │  └────────────────────────┘                                         │
    │                                                                      │
    │  ┌─── Layer 2 ───────────┐   Synthesis & calibration.              │
    │  │  weights               │   Factor attribution weights (w_i).    │
    │  │  fused_forecast        │   Pure-Python ŷ = Σ(w_i · ŷ_i).      │
    │  │  explanation           │   LLM natural-language explanation.     │
    │  │  calibration           │   Tightened calibration (Mod D).       │
    │  └────────────────────────┘                                         │
    │                                                                      │
    │  ┌─── Baseline (Phase 0) ┐   Honest Control Baseline.             │
    │  │  baseline_comparison   │   Multi-agent vs. CoT side-by-side.    │
    │  └────────────────────────┘                                         │
    └──────────────────────────────────────────────────────────────────────┘

Prompt Leakage Prevention:
    The `_l1_internal_traces` key (and `batched_reasoning.step_reasoning`)
    store raw chain-of-thought / reasoning from Layer 1 LLM calls.
    By convention and by code, these are NEVER read by the Synthesizer.
    The Synthesizer only receives numerical outputs and factor weights.
    Enforced at graph-edge level when the StateGraph is constructed.
"""

from __future__ import annotations

import warnings
from typing import Optional, TypedDict

from facts_mas.state.layer0 import SharedContext
from facts_mas.state.gate import GateConfig, GateDecision
from facts_mas.state.spatial import SpatialGraph, SpilloverSignal
from facts_mas.state.boundary import BoundaryConfig, BoundaryCheckResult
from facts_mas.state.micro_reasoning import (
    BatchedReasoningOutput,
    TightenedCalibrationResult,
)
from facts_mas.state.baseline import BaselineComparison


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1 output models (numerical-only — no LLM text leaks to Layer 2)
# ═══════════════════════════════════════════════════════════════════════════

class ARForecast(TypedDict):
    """Output of the AR Agent (TimesFM). Pure numerical — no LLM involvement."""

    model_name: str                          # e.g. "timesfm-2.5"
    point_forecast: list[float]              # ŷ_ar for each horizon week
    prediction_intervals: Optional[list[dict]]  # optional PI bounds


class MacroModifier(TypedDict):
    """Single macro factor modifier produced by the Macro Agent."""

    factor: str                              # FactorType.value
    modifier: float                          # e.g. +0.012 = +1.2% pressure
    lag_weeks_used: int                      # traceability for audit


class EventImpact(TypedDict):
    """Filtered event impact from the Event Agent (post-confidence gate)."""

    event_type: str                          # EventType.value
    impact_magnitude: float                  # 0.0 if confidence < 0.60
    confidence: float                        # original confidence score


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2 output models
# ═══════════════════════════════════════════════════════════════════════════

class FactorWeight(TypedDict):
    """Weight assignment from the Factor Attribution Agent (Router)."""

    factor: str                              # FactorType.value or agent name
    weight: float                            # w_i ∈ [0, 1], Σ = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# NEXUSGraphState — The LangGraph State (TypedDict)
# ═══════════════════════════════════════════════════════════════════════════

class NEXUSGraphState(TypedDict, total=False):
    """
    Central LangGraph state for the NEXUS optimized architecture.

    Implements all 4 modifications from the NEXUS Architecture Proposal:
      Mod A: gate_decision (Smart Early-Exit Gate)
      Mod B: spatial_graph + spillover_signal (Spatial Spillover)
      Mod C: boundary_config + boundary_check (Statistical Boundaries)
      Mod D: batched_reasoning + calibration (Batched Reasoning + Calibration)

    Plus the Honest Control Baseline (Phase 0): baseline_comparison.

    `total=False` allows keys to be absent before their producing node runs,
    which is required for LangGraph's incremental state-update pattern.
    """

    # ── Layer 0: Shared Context (populated by data pipeline) ──────────────
    shared_ctx: SharedContext

    # ── Gate (Mod A): Runs before Layer 1 ─────────────────────────────────
    # If should_activate_heavy_loop is False, the graph edges skip Layer 1
    # and go directly to Layer 2 with the baseline forecast.
    gate_config: GateConfig
    gate_decision: GateDecision

    # ── Spatial (Mod B): Granger graph + per-forecast spillover ───────────
    # spatial_graph is loaded once at startup (system-level).
    # spillover_signal is computed per-forecast from the graph.
    spatial_graph: SpatialGraph
    spillover_signal: SpilloverSignal

    # ── Layer 1: Specialised Agent Outputs (numerical only) ───────────────
    ar_forecast: ARForecast
    macro_modifiers: list[MacroModifier]
    event_impacts: list[EventImpact]
    batched_reasoning: BatchedReasoningOutput  # Mod D: full-horizon output

    # ── Layer 1: INTERNAL (⛔ prompt-leakage quarantine) ──────────────────
    # These store raw LLM reasoning logs for debugging/auditing.
    # The Synthesizer node MUST NOT read these keys.
    # Enforced at graph-edge level: Synthesizer edges exclude this key.
    _l1_internal_traces: list[dict]

    # ── Boundary (Mod C): Post-synthesis constraint check ─────────────────
    boundary_config: BoundaryConfig
    boundary_check: BoundaryCheckResult

    # ── Layer 2: Synthesis & Calibration ──────────────────────────────────
    weights: list[FactorWeight]
    fused_forecast: list[float]              # ŷ = Σ(w_i · ŷ_i)  (pure Python)
    explanation: str                          # LLM-generated NL explanation
    calibration: TightenedCalibrationResult  # Mod D tightened calibration

    # ── Baseline (Phase 0): Honest Control Baseline ───────────────────────
    baseline_comparison: BaselineComparison


# ═══════════════════════════════════════════════════════════════════════════
# Backward Compatibility Alias
# ═══════════════════════════════════════════════════════════════════════════

def __getattr__(name: str):
    """Module-level __getattr__ for deprecated FACTSGraphState alias."""
    if name == "FACTSGraphState":
        warnings.warn(
            "FACTSGraphState is deprecated. Use NEXUSGraphState instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return NEXUSGraphState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
