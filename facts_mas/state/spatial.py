"""
NEXUS Modification B: Cross-Regional Spatial Spillover Agent.

NEXUS Proposal §3, Mod B:
    "Each of the 15 metropolitan areas is treated as a node in a network.
    The strength of the connection between any two cities (the edge weight)
    is calculated using Granger causality, run on the historical Zillow
    inventory time series for every possible pair of the 15 cities."

    "Granger causality specifically tests whether knowing City A's past
    values actually helps predict City B's future values better than just
    knowing City B's own past values alone."

Design:
    • GrangerEdge: single pairwise test result
    • SpatialGraph: the full adjacency structure across all MSAs
    • SpilloverSignal: the actual lagged context fed to forecasting agents

    All Granger tests use Python's statsmodels — no LLM involvement.
    Edge weights are deterministic, computed once from historical data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════════════════
# GrangerEdge — a single pairwise Granger causality test result
# ═══════════════════════════════════════════════════════════════════════════

class GrangerEdge(BaseModel):
    """
    Result of a pairwise Granger causality test between two MSAs.

    Interpretation: "Does knowing source_msa's past inventory help predict
    target_msa's future inventory beyond target_msa's own history?"
    A low p_value (e.g. < 0.05) means yes → strong directional spillover.

    Computed via statsmodels.tsa.stattools.grangercausalitytests.
    Pure statistical — no LLM involved.
    """

    model_config = {"frozen": True}

    source_msa_id: str = Field(
        ...,
        description="MSA whose past values are tested as a predictor.",
    )
    target_msa_id: str = Field(
        ...,
        description="MSA whose future values we're trying to predict.",
    )
    optimal_lag_weeks: int = Field(
        ...,
        ge=1,
        description=(
            "Lag (in weeks) that produced the strongest Granger signal. "
            "Proposal suggests 4-8 weeks for housing spillover."
        ),
    )
    p_value: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "P-value from the Granger F-test at the optimal lag. "
            "Lower = stronger evidence of causal spillover."
        ),
    )
    f_statistic: float = Field(
        ...,
        ge=0.0,
        description="F-statistic from the Granger test.",
    )
    is_significant: bool = Field(
        ...,
        description="True if p_value < significance threshold (typically 0.05).",
    )

    @model_validator(mode="after")
    def _not_self_loop(self) -> "GrangerEdge":
        if self.source_msa_id == self.target_msa_id:
            raise ValueError(
                "source_msa_id and target_msa_id must differ; "
                "self-loops are not valid Granger tests."
            )
        return self


# ═══════════════════════════════════════════════════════════════════════════
# SpatialGraph — the full inter-MSA adjacency structure
# ═══════════════════════════════════════════════════════════════════════════

class SpatialGraph(BaseModel):
    """
    The complete Granger-causality graph across all MSAs in the study.

    NEXUS Proposal §3, Mod B:
        Built once from historical inventory data. Edges represent
        statistically significant directional spillover relationships.

    This is a system-level artifact, not per-forecast. It is computed
    offline and loaded at pipeline startup.
    """

    model_config = {"frozen": True}

    msa_ids: list[str] = Field(
        ...,
        min_length=2,
        description="All MSA FIPS codes in the study (the graph nodes).",
    )
    edges: list[GrangerEdge] = Field(
        ...,
        description=(
            "All pairwise Granger test results. Only significant edges "
            "are typically retained, but all tests can be stored for audit."
        ),
    )
    significance_threshold: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="P-value cutoff used to determine significance.",
    )
    max_lag_tested: int = Field(
        default=8,
        ge=1,
        description="Maximum lag (weeks) tested in the Granger procedure.",
    )

    # -- Query helpers -------------------------------------------------------

    def neighbors_of(
        self, target_msa_id: str, *, top_k: Optional[int] = None
    ) -> list[GrangerEdge]:
        """
        Return significant edges where `target_msa_id` is the target,
        sorted by p-value ascending (strongest neighbors first).

        Args:
            target_msa_id: The city we're forecasting.
            top_k: If set, return only the top-k strongest neighbors.
        """
        sig_edges = [
            e for e in self.edges
            if e.target_msa_id == target_msa_id and e.is_significant
        ]
        sig_edges.sort(key=lambda e: e.p_value)
        if top_k is not None:
            sig_edges = sig_edges[:top_k]
        return sig_edges

    def adjacency_matrix(self) -> tuple[list[str], np.ndarray]:
        """
        Return (msa_id_list, NxN adjacency matrix) where entry [i,j] is
        1/p_value for significant edges from msa_ids[j] → msa_ids[i],
        and 0 otherwise.

        Pure NumPy — no LLM math.
        """
        n = len(self.msa_ids)
        idx = {msa: i for i, msa in enumerate(self.msa_ids)}
        mat = np.zeros((n, n), dtype=np.float64)
        for e in self.edges:
            if e.is_significant:
                i = idx.get(e.target_msa_id)
                j = idx.get(e.source_msa_id)
                if i is not None and j is not None:
                    mat[i, j] = 1.0 / max(e.p_value, 1e-10)
        return list(self.msa_ids), mat


# ═══════════════════════════════════════════════════════════════════════════
# SpilloverSignal — context fed to forecasting agents for a target city
# ═══════════════════════════════════════════════════════════════════════════

class SpilloverSignal(BaseModel):
    """
    Aggregated spillover signal for a single target MSA, derived from
    its Granger-identified neighbors' recent inventory trends.

    This is the actionable output that the Spatial Spillover Agent feeds
    into the forecasting agents and the Early-Exit Gate.

    Prompt Leakage Isolation:
        Contains only numerical data from neighbors — no LLM reasoning,
        prompts, or chain-of-thought from any agent.
    """

    model_config = {"frozen": True}

    target_msa_id: str = Field(
        ...,
        description="The city being forecasted.",
    )
    neighbor_count: int = Field(
        ...,
        ge=0,
        description="Number of significant Granger neighbors used.",
    )
    aggregate_neighbor_trend: float = Field(
        ...,
        description=(
            "Weighted average of neighbors' recent inventory percentage "
            "changes, weighted by 1/p_value (strength of Granger link). "
            "Positive = neighbors' inventories rising, negative = falling."
        ),
    )
    max_neighbor_shock: float = Field(
        ...,
        description=(
            "Largest absolute percentage change among all neighbors' "
            "recent data. Used by the gate's neighbor anomaly check."
        ),
    )
    contributing_msa_ids: list[str] = Field(
        default_factory=list,
        description="MSA IDs of the neighbors contributing to this signal.",
    )
