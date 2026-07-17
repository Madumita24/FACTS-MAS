"""
FACTS-MAS Factor & Event Ontology (Static Enum — Test Path A, Path 1).

Spec Reference (Part 5, Test A):
    The approved ontology is hardcoded as a Python Enum. No agent may generate
    factor names outside this set. Any modification to these enums requires
    explicit user approval via the Prompt Injection Gate (Phase 0, Rule 3).

    Static ontology is *strongly preferred* per the specification.
"""

from enum import Enum


# ---------------------------------------------------------------------------
# Factor Ontology — 20 approved factors (Spec §5, Test A Path 1)
# Sourced from Housing_Price_Variables.docx (FRED / Zillow / Census categories)
# ---------------------------------------------------------------------------

class FactorType(str, Enum):
    """
    Static set of 20 approved macroeconomic / structural factors.

    ⚠️ PROMPT INJECTION GATE: Adding or removing entries requires user
    approval. Present A/B options and halt execution until approved.
    """

    # ── Macro / Monetary ──────────────────────────────────────────────────
    GDP_GROWTH           = "gdp_growth"
    INFLATION_CPI        = "inflation_cpi"
    UNEMPLOYMENT_RATE    = "unemployment_rate"
    CONSUMER_CONFIDENCE  = "consumer_confidence"
    FED_FUNDS_RATE       = "fed_funds_rate"
    MORTGAGE_RATE_30Y    = "mortgage_rate_30y"
    MONEY_SUPPLY_M2      = "money_supply_m2"
    CREDIT_AVAILABILITY  = "credit_availability"

    # ── Housing-Market Specific ───────────────────────────────────────────
    HOUSING_STARTS       = "housing_starts"
    BUILDING_PERMITS     = "building_permits"
    EXISTING_HOME_SALES  = "existing_home_sales"
    INVENTORY_SUPPLY     = "inventory_supply"
    MEDIAN_DAYS_ON_MARKET = "median_days_on_market"
    PRICE_TO_INCOME      = "price_to_income"

    # ── Demographic / Structural ──────────────────────────────────────────
    POPULATION_GROWTH    = "population_growth"
    NET_MIGRATION        = "net_migration"
    HOUSEHOLD_FORMATION  = "household_formation"
    RENTAL_VACANCY_RATE  = "rental_vacancy_rate"

    # ── Composite / External ──────────────────────────────────────────────
    SP500_RETURN         = "sp500_return"
    CONSTRUCTION_COSTS   = "construction_costs"


# ---------------------------------------------------------------------------
# Event Ontology — approved unstructured-event classifications
# Spec §3: "Approved event types: [disaster, zoning_change,
#            major_employer_shift, housing_initiative]."
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """
    Closed set of event categories the Event Agent (Layer 1) may output.

    The LLM prompt must include the constraint:
        "If no text matches the ontology, return an empty array []."
    """

    DISASTER              = "disaster"
    ZONING_CHANGE         = "zoning_change"
    MAJOR_EMPLOYER_SHIFT  = "major_employer_shift"
    HOUSING_INITIATIVE    = "housing_initiative"


# ---------------------------------------------------------------------------
# Gate Trigger Ontology — Smart Early-Exit Gate (NEXUS Mod A)
# ---------------------------------------------------------------------------

class GateTriggerType(str, Enum):
    """
    Trigger categories for the Smart Early-Exit Gate (NEXUS §3, Mod A).

    The gate checks two classes of triggers before deciding whether to
    activate the heavy multi-agent reasoning loop:
      1. Text-event signals — defined events that indicate market disruption
      2. Statistical anomaly — baseline prediction error deviates from its
         own historical distribution

    The gate also watches Granger-identified neighbors (Mod A↔B connection):
    a neighbor shock can itself be a trigger for the target city.
    """

    # ── Text-Event Triggers ───────────────────────────────────────────────
    FED_RATE_ANNOUNCEMENT    = "fed_rate_announcement"
    MAJOR_EMPLOYER_RELOCATION = "major_employer_relocation"
    NATURAL_DISASTER         = "natural_disaster"
    POLICY_CHANGE            = "policy_change"

    # ── Statistical Triggers ──────────────────────────────────────────────
    BASELINE_ANOMALY         = "baseline_anomaly"
    NEIGHBOR_SPILLOVER       = "neighbor_spillover"

    # ── No trigger fired — gate stays dormant ─────────────────────────────
    NONE                     = "none"


# Convenience exports for runtime lookups
APPROVED_FACTORS: frozenset[str] = frozenset(f.value for f in FactorType)
APPROVED_EVENTS: frozenset[str]  = frozenset(e.value for e in EventType)
APPROVED_GATE_TRIGGERS: frozenset[str] = frozenset(
    t.value for t in GateTriggerType if t != GateTriggerType.NONE
)
