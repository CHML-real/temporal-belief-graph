"""
io.py
-----
JSON serialization and deserialization for BeliefGraph.

Functions
---------
graph_to_dict(graph)         Convert a BeliefGraph to a plain dictionary.
graph_from_dict(data)        Reconstruct a BeliefGraph from a dictionary.
graph_to_json(graph, path)   Write a BeliefGraph to a JSON file.
graph_from_json(path)        Load a BeliefGraph from a JSON file.

Contributors
------------
lajjadred  https://github.com/lajjadred   project lead
이채문      https://github.com/CHML-real   mathematical algorithm development
CUBE       https://github.com/90cube      idea proposal and data collection
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .graph import BeliefGraph
from .schema import BeliefEdge, EvidenceRecord, EventNode, PriorConfig


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def graph_to_dict(graph: BeliefGraph) -> dict[str, Any]:
    """
    Convert a BeliefGraph to a plain JSON-serializable dictionary.

    The returned dictionary can be passed to graph_from_dict() to reconstruct
    an identical graph including all evidence history.
    """
    return {
        "version": "0.1.1",
        "config": _config_to_dict(graph.config),
        "nodes": [_node_to_dict(node) for node in graph.nodes],
        "edges": [_edge_to_dict(edge) for edge in graph.edges],
    }


def graph_to_json(graph: BeliefGraph, path: str | Path, indent: int = 2) -> None:
    """
    Write a BeliefGraph to a JSON file.

    Parameters
    ----------
    graph:
        The graph to serialize.
    path:
        Destination file path.
    indent:
        JSON indentation level. Default 2.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(graph_to_dict(graph), f, indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def graph_from_dict(data: dict[str, Any]) -> BeliefGraph:
    """
    Reconstruct a BeliefGraph from a plain dictionary.

    Parameters
    ----------
    data:
        Dictionary produced by graph_to_dict().
    """
    config = _config_from_dict(data.get("config", {}))
    graph = BeliefGraph(prior_config=config)

    for node_data in data.get("nodes", []):
        graph.add_node(_node_from_dict(node_data))

    for edge_data in data.get("edges", []):
        graph.add_edge(_edge_from_dict(edge_data))

    return graph


def graph_from_json(path: str | Path) -> BeliefGraph:
    """
    Load a BeliefGraph from a JSON file.

    Parameters
    ----------
    path:
        Source file path.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return graph_from_dict(data)


# ---------------------------------------------------------------------------
# Internal helpers — export
# ---------------------------------------------------------------------------

def _config_to_dict(config: PriorConfig) -> dict[str, Any]:
    return {
        "default_p": config.default_p,
        "source_weights": dict(config.source_weights),
        "pseudo_era_list": list(config.pseudo_era_list),
        "weak_evidence_threshold": config.weak_evidence_threshold,
        "overconfidence_threshold": config.overconfidence_threshold,
    }


def _node_to_dict(node: EventNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "label": node.label,
        "era": node.era,
        "description": node.description,
        "sources": list(node.sources),
        "metadata": dict(node.metadata),
    }


def _edge_to_dict(edge: BeliefEdge) -> dict[str, Any]:
    return {
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "p_forward": edge.p_forward,
        "weight": edge.weight,
        "relation_type": edge.relation_type,
        "evidence_keys": list(edge.evidence_keys),
        "evidence_history": [_record_to_dict(r) for r in edge.evidence_history],
        "metadata": dict(edge.metadata),
    }


def _record_to_dict(record: EvidenceRecord) -> dict[str, Any]:
    return {
        "key": record.key,
        "source": record.source,
        "supports_forward": record.supports_forward,
        "strength": record.strength,
        "source_weight": record.source_weight,
        "prior_before": record.prior_before,
        "posterior_after": record.posterior_after,
        "update_method": record.update_method,
        "note": record.note,
        "metadata": dict(record.metadata),
    }


# ---------------------------------------------------------------------------
# Internal helpers — import
# ---------------------------------------------------------------------------

def _config_from_dict(data: dict[str, Any]) -> PriorConfig:
    return PriorConfig(
        default_p=data.get("default_p", 0.5),
        source_weights=data.get("source_weights", {}),
        pseudo_era_list=data.get("pseudo_era_list", []),
        weak_evidence_threshold=data.get("weak_evidence_threshold", 1),
        overconfidence_threshold=data.get("overconfidence_threshold", 0.95),
    )


def _node_from_dict(data: dict[str, Any]) -> EventNode:
    return EventNode(
        id=data["id"],
        label=data["label"],
        era=data.get("era"),
        description=data.get("description", ""),
        sources=data.get("sources", []),
        metadata=data.get("metadata", {}),
    )


def _edge_from_dict(data: dict[str, Any]) -> BeliefEdge:
    edge = BeliefEdge(
        source_id=data["source_id"],
        target_id=data["target_id"],
        p_forward=data.get("p_forward", 0.5),
        weight=data.get("weight", 1.0),
        relation_type=data.get("relation_type", "temporal"),
        evidence_keys=data.get("evidence_keys", []),
        metadata=data.get("metadata", {}),
    )
    for record_data in data.get("evidence_history", []):
        edge.evidence_history.append(_record_from_dict(record_data))
    return edge


def _record_from_dict(data: dict[str, Any]) -> EvidenceRecord:
    return EvidenceRecord(
        key=data["key"],
        source=data.get("source", ""),
        supports_forward=data["supports_forward"],
        strength=data["strength"],
        source_weight=data.get("source_weight", 1.0),
        prior_before=data["prior_before"],
        posterior_after=data["posterior_after"],
        update_method=data.get("update_method", "log_odds"),
        note=data.get("note", ""),
        metadata=data.get("metadata", {}),
    )
