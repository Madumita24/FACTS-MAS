"""
Executable nodes for the NEXUS optimized architecture.

`facts_mas/state/` holds the Pydantic models describing what each NEXUS
modification produces. This package holds the code that actually produces
them. The split is deliberate: the models were locked first so both owners
could build against a fixed contract, exactly like `schema.py` did for the
Layer-1 agents.

    Mod A  gate_node.py        Smart Early-Exit Gate
    Mod B  spatial_node.py     Cross-Regional Spatial Spillover
    Mod C  boundary_node.py    Statistical Boundary Constraints
    Mod D  batched_reasoning.py  Batched Micro-Reasoning

Build order matters and is not arbitrary: the gate watches its neighbours'
anomalies as well as its own, so Mod A depends on Mod B's graph.
"""
