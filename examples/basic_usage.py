"""
basic_usage.py
--------------
A minimal end-to-end example of temporal-belief-graph.

Scenario
--------
Three events from a fictional lore timeline:
    Kain Incident → Rift Opening → Archive Fall

The ordering is initially uncertain. Two sources provide evidence.
We update belief, validate the graph, and export a report.

Run
---
    python examples/basic_usage.py
"""

from tbg import (
    BeliefGraph,
    BeliefEdge,
    EventNode,
    PriorConfig,
    BayesianUpdater,
    Evidence,
    Validator,
    graph_to_json,
    graph_from_json,
)
from tbg.report import graph_to_markdown, graph_to_mermaid


# ---------------------------------------------------------------------------
# 1. Configure the graph
# ---------------------------------------------------------------------------

config = PriorConfig(
    default_p=0.5,
    source_weights={
        "official_lore": 1.5,   # trusted source
        "fan_wiki": 0.7,        # less trusted source
    },
)

graph = BeliefGraph(prior_config=config)


# ---------------------------------------------------------------------------
# 2. Add events
# ---------------------------------------------------------------------------

graph.add_node(EventNode(
    id="kain_incident",
    label="Kain Incident",
    era="First Rift War",
    description="The event that triggered the first dimensional rift.",
    sources=["official_lore/chapter_1"],
))

graph.add_node(EventNode(
    id="rift_opening",
    label="Rift Opening",
    era="First Rift War",
    description="The first recorded opening of a dimensional rift.",
    sources=["official_lore/chapter_2"],
))

graph.add_node(EventNode(
    id="archive_fall",
    label="Archive Fall",
    era="Second Age",
    description="Destruction of the Great Archive.",
    sources=["fan_wiki/archive_entry"],
))


# ---------------------------------------------------------------------------
# 3. Add edges with uniform prior (we do not know the order yet)
# ---------------------------------------------------------------------------

graph.init_uniform("kain_incident", "rift_opening")
graph.init_uniform("rift_opening", "archive_fall")


# ---------------------------------------------------------------------------
# 4. Update beliefs with evidence
# ---------------------------------------------------------------------------

updater = BayesianUpdater(graph)

# Official lore strongly supports: Kain Incident → Rift Opening
updater.update_edge(
    "kain_incident", "rift_opening",
    Evidence(
        key="official_001",
        supports_forward=True,
        strength=0.9,
        source="official_lore",
        note="Chapter 1 explicitly states Kain incident preceded the rift.",
    ),
)

# Fan wiki weakly disputes the same ordering
updater.update_edge(
    "kain_incident", "rift_opening",
    Evidence(
        key="fan_001",
        supports_forward=False,
        strength=0.4,
        source="fan_wiki",
        note="Fan wiki suggests the rift may have opened independently.",
    ),
)

# Official lore supports: Rift Opening → Archive Fall
updater.update_edge(
    "rift_opening", "archive_fall",
    Evidence(
        key="official_002",
        supports_forward=True,
        strength=0.8,
        source="official_lore",
        note="Archive fell during the aftermath of the first rift.",
    ),
)

# Ensemble update: three sources on Kain Incident → Rift Opening
updater.ensemble_update(
    "kain_incident", "rift_opening",
    claims=[
        (0.85, 1.5),   # official source
        (0.70, 1.0),   # secondary source
        (0.30, 0.5),   # weak dissenting source
    ],
)


# ---------------------------------------------------------------------------
# 5. Inspect results
# ---------------------------------------------------------------------------

edge = graph.get_edge("kain_incident", "rift_opening")
print("=== Edge: Kain Incident → Rift Opening ===")
print(f"  p_forward      : {edge.p_forward:.4f}")
print(f"  p_backward     : {edge.p_backward:.4f}")
print(f"  update count   : {edge.update_count}")
print(f"  evidence keys  : {edge.evidence_keys}")
print()

explanation = updater.explain_edge("kain_incident", "rift_opening")
print("=== Explanation ===")
for key, value in explanation.items():
    print(f"  {key}: {value}")
print()


# ---------------------------------------------------------------------------
# 6. Validate
# ---------------------------------------------------------------------------

result = Validator().validate(graph)
print("=== Validation ===")
print(f"  valid    : {result.is_valid}")
print(f"  errors   : {result.errors}")
print(f"  warnings : {result.warnings}")
print()


# ---------------------------------------------------------------------------
# 7. Export
# ---------------------------------------------------------------------------

# JSON round-trip
graph_to_json(graph, "output/graph.json")
restored = graph_from_json("output/graph.json")
print(f"=== JSON Export ===")
print(f"  Saved and restored: {len(restored.nodes)} nodes, {len(restored.edges)} edges")
print()

# Markdown report
md = graph_to_markdown(graph, title="Lore Timeline Report")
with open("output/report.md", "w", encoding="utf-8") as f:
    f.write(md)
print("=== Markdown Report ===")
print("  Saved to output/report.md")
print()

# Mermaid diagram
mermaid = graph_to_mermaid(graph)
with open("output/diagram.mmd", "w", encoding="utf-8") as f:
    f.write(mermaid)
print("=== Mermaid Diagram ===")
print(mermaid)
