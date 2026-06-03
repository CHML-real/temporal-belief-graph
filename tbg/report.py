"""
report.py
---------
Markdown and Mermaid report export for BeliefGraph.

Functions
---------
graph_to_markdown(graph)   Generate a Markdown report of the graph state.
graph_to_mermaid(graph)    Generate a Mermaid diagram of the graph.

Contributors
------------
lajjadred  https://github.com/lajjadred   project lead
이채문      https://github.com/CHML-real   mathematical algorithm development
CUBE       https://github.com/90cube      idea proposal and data collection
"""

from __future__ import annotations

from pathlib import Path

from .graph import BeliefGraph


def graph_to_markdown(graph: BeliefGraph, title: str = "Temporal Belief Graph Report") -> str:
    """
    Generate a Markdown report summarizing the current state of a BeliefGraph.

    Sections
    --------
    - Summary
    - Confirmed Edges   (p_forward >= 0.75)
    - Uncertain Edges   (0.4 <= p_forward <= 0.6)
    - All Nodes
    - Evidence Summary

    Parameters
    ----------
    graph:
        The graph to report on.
    title:
        Report heading.

    Returns
    -------
    str
        Full Markdown string.
    """
    lines: list[str] = []

    # Title
    lines.append(f"# {title}")
    lines.append("")

    # Summary
    confirmed = graph.confirmed_edges(threshold=0.75)
    uncertain = graph.uncertain_edges()
    contradictions = graph.contradiction_edges()
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Item | Count |")
    lines.append(f"|------|-------|")
    lines.append(f"| Nodes | {len(graph.nodes)} |")
    lines.append(f"| Edges | {len(graph.edges)} |")
    lines.append(f"| Confirmed edges (p ≥ 0.75) | {len(confirmed)} |")
    lines.append(f"| Uncertain edges | {len(uncertain)} |")
    lines.append(f"| Contradictions | {len(contradictions)} |")
    lines.append("")

    # Confirmed edges
    lines.append("## Confirmed Edges")
    lines.append("")
    if confirmed:
        lines.append("| Source | Target | P(source before target) | Evidence Count |")
        lines.append("|--------|--------|-------------------------|----------------|")
        for edge in sorted(confirmed, key=lambda e: e.p_forward, reverse=True):
            src_label = _node_label(graph, edge.source_id)
            tgt_label = _node_label(graph, edge.target_id)
            lines.append(
                f"| {src_label} | {tgt_label} | {edge.p_forward:.4f} | {edge.update_count} |"
            )
    else:
        lines.append("_No confirmed edges._")
    lines.append("")

    # Uncertain edges
    lines.append("## Uncertain Edges")
    lines.append("")
    if uncertain:
        lines.append("| Source | Target | P(source before target) | Evidence Count |")
        lines.append("|--------|--------|-------------------------|----------------|")
        for edge in uncertain:
            src_label = _node_label(graph, edge.source_id)
            tgt_label = _node_label(graph, edge.target_id)
            lines.append(
                f"| {src_label} | {tgt_label} | {edge.p_forward:.4f} | {edge.update_count} |"
            )
    else:
        lines.append("_No uncertain edges._")
    lines.append("")

    # Contradictions
    if contradictions:
        lines.append("## Contradictions")
        lines.append("")
        lines.append("| Edge A | Edge B | P(A) | P(B) |")
        lines.append("|--------|--------|------|------|")
        for fwd, rev in contradictions:
            lines.append(
                f"| {fwd.source_id} → {fwd.target_id} "
                f"| {rev.source_id} → {rev.target_id} "
                f"| {fwd.p_forward:.4f} | {rev.p_forward:.4f} |"
            )
        lines.append("")

    # All nodes
    lines.append("## Nodes")
    lines.append("")
    lines.append("| ID | Label | Era |")
    lines.append("|----|-------|-----|")
    for node in graph.nodes:
        era = node.era if node.era is not None else "_unset_"
        lines.append(f"| `{node.id}` | {node.label} | {era} |")
    lines.append("")

    # Evidence summary
    lines.append("## Evidence Summary")
    lines.append("")
    total_updates = sum(e.update_count for e in graph.edges)
    if total_updates > 0:
        lines.append("| Edge | Evidence Keys |")
        lines.append("|------|---------------|")
        for edge in graph.edges:
            if edge.evidence_keys:
                keys = ", ".join(f"`{k}`" for k in edge.evidence_keys)
                lines.append(f"| {edge.source_id} → {edge.target_id} | {keys} |")
    else:
        lines.append("_No evidence recorded._")
    lines.append("")

    return "\n".join(lines)


def graph_to_mermaid(
    graph: BeliefGraph,
    threshold: float = 0.75,
    show_uncertain: bool = True,
) -> str:
    """
    Generate a Mermaid flowchart diagram of the graph.

    Confirmed edges (p >= threshold) are shown as solid arrows.
    Uncertain edges are shown as dashed arrows when show_uncertain=True.

    Parameters
    ----------
    graph:
        The graph to diagram.
    threshold:
        Probability threshold for confirmed edges.
    show_uncertain:
        Whether to include uncertain edges in the diagram.

    Returns
    -------
    str
        Mermaid diagram string. Wrap in ```mermaid``` fences for rendering.
    """
    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be in the open interval (0, 1).")

    lines: list[str] = ["graph TD"]

    # Node definitions
    for node in graph.nodes:
        safe_id = _safe_id(node.id)
        lines.append(f'    {safe_id}["{node.label}"]')

    lines.append("")

    # Confirmed edges — solid arrow
    for edge in graph.confirmed_edges(threshold=threshold):
        src = _safe_id(edge.source_id)
        tgt = _safe_id(edge.target_id)
        label = f"{edge.p_forward:.2f}"
        lines.append(f"    {src} -->|{label}| {tgt}")

    # Uncertain edges — dashed arrow
    if show_uncertain:
        for edge in graph.uncertain_edges():
            src = _safe_id(edge.source_id)
            tgt = _safe_id(edge.target_id)
            label = f"{edge.p_forward:.2f}"
            lines.append(f"    {src} -.->|{label}| {tgt}")

    return "\n".join(lines)


def save_markdown(
    graph: BeliefGraph,
    path: str | Path,
    title: str = "Temporal Belief Graph Report",
) -> None:
    """Write the Markdown report to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph_to_markdown(graph, title=title), encoding="utf-8")


def save_mermaid(
    graph: BeliefGraph,
    path: str | Path,
    threshold: float = 0.75,
    show_uncertain: bool = True,
) -> None:
    """Write the Mermaid diagram to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph_to_mermaid(graph, threshold=threshold, show_uncertain=show_uncertain), encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_label(graph: BeliefGraph, node_id: str) -> str:
    """Return node label if available, otherwise the id."""
    try:
        return graph.get_node(node_id).label
    except KeyError:
        return node_id


def _safe_id(node_id: str) -> str:
    """Convert a node id to a Mermaid-safe identifier."""
    return node_id.replace("-", "_").replace(" ", "_").replace(".", "_")
