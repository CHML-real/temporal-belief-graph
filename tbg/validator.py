"""
validator.py
------------
Validation utilities for Temporal Belief Graph.

The validator reports errors and warnings without raising exceptions by default.
Call ValidationResult.raise_if_errors() when exception-based validation is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .graph import BeliefGraph


@dataclass(slots=True)
class ValidationResult:
    """Container for graph validation errors and warnings."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return True when there are no validation errors."""
        return not self.errors

    def raise_if_errors(self) -> None:
        """Raise ValueError when errors exist."""
        if self.errors:
            raise ValueError("BeliefGraph validation failed:\n" + "\n".join(self.errors))

    def __repr__(self) -> str:
        return f"ValidationResult(errors={len(self.errors)}, warnings={len(self.warnings)}, valid={self.is_valid})"


class Validator:
    """
    Validate structural and probabilistic consistency of a BeliefGraph.

    Parameters
    ----------
    cycle_threshold:
        Edges above this probability are treated as confirmed for cycle checks.
    contradiction_threshold:
        Opposing edges above this probability are treated as contradictions.
    """

    def __init__(self, cycle_threshold: float = 0.75, contradiction_threshold: float = 0.75):
        if not (0.0 < cycle_threshold < 1.0):
            raise ValueError("cycle_threshold must be in the open interval (0, 1).")
        if not (0.5 < contradiction_threshold < 1.0):
            raise ValueError("contradiction_threshold must be in the open interval (0.5, 1).")
        self.cycle_threshold = cycle_threshold
        self.contradiction_threshold = contradiction_threshold

    def validate(self, graph: BeliefGraph) -> ValidationResult:
        """Validate the graph and return a ValidationResult."""
        result = ValidationResult()
        self._check_pseudo_era(graph, result)
        self._check_missing_era(graph, result)
        self._check_edge_evidence(graph, result)
        self._check_overconfident_edges(graph, result)
        self._check_contradictions(graph, result)
        self._check_cycles(graph, result)
        self._check_isolated_nodes(graph, result)
        return result

    def _check_pseudo_era(self, graph: BeliefGraph, result: ValidationResult) -> None:
        pseudo_set = {value.casefold() for value in graph.config.all_pseudo_eras}
        for node in graph.nodes:
            if node.era is not None and node.era.casefold() in pseudo_set:
                result.errors.append(
                    f"[pseudo_era] Node {node.id!r} uses pseudo era {node.era!r}. Use a concrete era, arc, chapter, or event segment."
                )

    def _check_missing_era(self, graph: BeliefGraph, result: ValidationResult) -> None:
        for node in graph.nodes:
            if node.era is None:
                result.warnings.append(f"[missing_era] Node {node.id!r} has no era value.")

    def _check_edge_evidence(self, graph: BeliefGraph, result: ValidationResult) -> None:
        for edge in graph.edges:
            if not edge.evidence_history and edge.p_forward != graph.config.default_p:
                result.warnings.append(
                    f"[missing_evidence] Edge {edge.source_id!r} -> {edge.target_id!r} has probability {edge.p_forward:.4f} but no evidence history."
                )

    def _check_overconfident_edges(self, graph: BeliefGraph, result: ValidationResult) -> None:
        threshold = graph.config.overconfidence_threshold
        weak_count = graph.config.weak_evidence_threshold
        for edge in graph.edges:
            is_overconfident = edge.p_forward >= threshold or edge.p_forward <= 1.0 - threshold
            if is_overconfident and edge.update_count <= weak_count:
                result.warnings.append(
                    f"[overconfident_edge] Edge {edge.source_id!r} -> {edge.target_id!r} has probability {edge.p_forward:.4f} with only {edge.update_count} update(s)."
                )

    def _check_contradictions(self, graph: BeliefGraph, result: ValidationResult) -> None:
        for forward, reverse in graph.contradiction_edges(threshold=self.contradiction_threshold):
            result.errors.append(
                f"[contradiction] Opposing edges are both highly confident: "
                f"{forward.source_id!r} -> {forward.target_id!r} ({forward.p_forward:.4f}) and "
                f"{reverse.source_id!r} -> {reverse.target_id!r} ({reverse.p_forward:.4f})."
            )

    def _check_cycles(self, graph: BeliefGraph, result: ValidationResult) -> None:
        adjacency = graph.to_adjacency_dict(threshold=self.cycle_threshold)
        state: dict[str, int] = {node_id: 0 for node_id in adjacency}
        path: list[str] = []

        def dfs(node_id: str) -> bool:
            state[node_id] = 1
            path.append(node_id)
            for neighbor in adjacency[node_id]:
                if state[neighbor] == 1:
                    start_index = path.index(neighbor)
                    cycle = path[start_index:] + [neighbor]
                    result.errors.append(f"[cycle] Confirmed temporal cycle detected: {' -> '.join(cycle)}")
                    return True
                if state[neighbor] == 0 and dfs(neighbor):
                    return True
            path.pop()
            state[node_id] = 2
            return False

        for node_id in adjacency:
            if state[node_id] == 0:
                dfs(node_id)

    def _check_isolated_nodes(self, graph: BeliefGraph, result: ValidationResult) -> None:
        connected: set[str] = set()
        for edge in graph.edges:
            connected.add(edge.source_id)
            connected.add(edge.target_id)
        for node in graph.nodes:
            if node.id not in connected:
                result.warnings.append(f"[isolated_node] Node {node.id!r} has no incident edges.")
