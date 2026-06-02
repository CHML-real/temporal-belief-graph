"""
graph.py
--------
Probabilistic directed belief graph for uncertain event orderings.

The graph itself may contain uncertain or contradictory edges. A confirmed DAG
can be derived later by applying a probability threshold.
"""

from __future__ import annotations

from typing import Optional

from .schema import BeliefEdge, EventNode, PriorConfig


class BeliefGraph:
    """
    Directed graph whose edges store temporal belief probabilities.

    Parameters
    ----------
    prior_config:
        Initial probability and source-weight configuration.
    """

    def __init__(self, prior_config: Optional[PriorConfig] = None):
        self._config: PriorConfig = prior_config or PriorConfig()
        self._nodes: dict[str, EventNode] = {}
        self._edges: dict[tuple[str, str], BeliefEdge] = {}

    @property
    def config(self) -> PriorConfig:
        """Return the graph configuration."""
        return self._config

    def add_node(self, node: EventNode) -> None:
        """Add or replace a node by id."""
        self._nodes[node.id] = node

    def get_node(self, node_id: str) -> EventNode:
        """Return a node by id."""
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id!r} does not exist in the graph.")
        return self._nodes[node_id]

    def has_node(self, node_id: str) -> bool:
        """Return True when the node exists."""
        return node_id in self._nodes

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all incident edges."""
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id!r} does not exist in the graph.")
        del self._nodes[node_id]
        for key in [key for key in self._edges if node_id in key]:
            del self._edges[key]

    @property
    def nodes(self) -> list[EventNode]:
        """Return all nodes."""
        return list(self._nodes.values())

    @property
    def node_ids(self) -> list[str]:
        """Return all node ids."""
        return list(self._nodes.keys())

    def add_edge(self, edge: BeliefEdge) -> None:
        """Add or replace an edge after checking that both endpoint nodes exist."""
        if edge.source_id not in self._nodes:
            raise KeyError(f"Source node {edge.source_id!r} does not exist in the graph.")
        if edge.target_id not in self._nodes:
            raise KeyError(f"Target node {edge.target_id!r} does not exist in the graph.")
        self._edges[(edge.source_id, edge.target_id)] = edge

    def get_edge(self, source_id: str, target_id: str) -> BeliefEdge:
        """Return an edge by endpoint ids."""
        key = (source_id, target_id)
        if key not in self._edges:
            raise KeyError(f"Edge {source_id!r} -> {target_id!r} does not exist in the graph.")
        return self._edges[key]

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """Return True when the directed edge exists."""
        return (source_id, target_id) in self._edges

    def remove_edge(self, source_id: str, target_id: str) -> None:
        """Remove a directed edge."""
        key = (source_id, target_id)
        if key not in self._edges:
            raise KeyError(f"Edge {source_id!r} -> {target_id!r} does not exist in the graph.")
        del self._edges[key]

    @property
    def edges(self) -> list[BeliefEdge]:
        """Return all edges."""
        return list(self._edges.values())

    def successors(self, node_id: str) -> list[EventNode]:
        """Return nodes directly reachable from node_id."""
        self.get_node(node_id)
        return [self._nodes[target] for (source, target) in self._edges if source == node_id]

    def predecessors(self, node_id: str) -> list[EventNode]:
        """Return nodes that directly point to node_id."""
        self.get_node(node_id)
        return [self._nodes[source] for (source, target) in self._edges if target == node_id]

    def uncertain_edges(self, lower: float = 0.4, upper: float = 0.6) -> list[BeliefEdge]:
        """Return edges whose forward probability lies within an uncertainty band."""
        if not (0.0 <= lower <= upper <= 1.0):
            raise ValueError("Uncertainty bounds must satisfy 0 <= lower <= upper <= 1.")
        return [edge for edge in self._edges.values() if lower <= edge.p_forward <= upper]

    def confirmed_edges(self, threshold: float = 0.75) -> list[BeliefEdge]:
        """Return edges whose forward probability is greater than or equal to threshold."""
        if not (0.0 < threshold < 1.0):
            raise ValueError("threshold must be in the open interval (0, 1).")
        return [edge for edge in self._edges.values() if edge.p_forward >= threshold]

    def contradiction_edges(self, threshold: float = 0.75) -> list[tuple[BeliefEdge, BeliefEdge]]:
        """Return pairs of opposing edges that are both highly confident."""
        if not (0.5 < threshold < 1.0):
            raise ValueError("threshold must be in the open interval (0.5, 1).")
        pairs: list[tuple[BeliefEdge, BeliefEdge]] = []
        seen: set[frozenset[tuple[str, str]]] = set()
        for (source, target), edge in self._edges.items():
            reverse_key = (target, source)
            if reverse_key not in self._edges:
                continue
            pair_key = frozenset({(source, target), reverse_key})
            if pair_key in seen:
                continue
            reverse_edge = self._edges[reverse_key]
            if edge.p_forward >= threshold and reverse_edge.p_forward >= threshold:
                pairs.append((edge, reverse_edge))
            seen.add(pair_key)
        return pairs

    def init_uniform(self, source_id: str, target_id: str) -> BeliefEdge:
        """Create a uniform-prior edge if it does not already exist."""
        key = (source_id, target_id)
        if key in self._edges:
            return self._edges[key]
        edge = BeliefEdge(source_id=source_id, target_id=target_id, p_forward=self._config.default_p)
        self.add_edge(edge)
        return edge

    def to_adjacency_dict(self, threshold: float = 0.75) -> dict[str, list[str]]:
        """Return an adjacency dictionary containing only confirmed forward edges."""
        if not (0.0 < threshold < 1.0):
            raise ValueError("threshold must be in the open interval (0, 1).")
        adjacency: dict[str, list[str]] = {node_id: [] for node_id in self._nodes}
        for edge in self.confirmed_edges(threshold=threshold):
            adjacency[edge.source_id].append(edge.target_id)
        return adjacency

    def __repr__(self) -> str:
        return (
            "BeliefGraph("
            f"nodes={len(self._nodes)}, "
            f"edges={len(self._edges)}, "
            f"uncertain={len(self.uncertain_edges())})"
        )
