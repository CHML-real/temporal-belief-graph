"""tests/test_report.py"""
import pytest
from tbg.schema import EventNode, BeliefEdge
from tbg.graph import BeliefGraph
from tbg.bayesian import BayesianUpdater, Evidence
from tbg.report import graph_to_markdown, graph_to_mermaid, save_markdown, save_mermaid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def graph():
    g = BeliefGraph()
    g.add_node(EventNode(id="kain_incident", label="Kain Incident", era="First Rift War"))
    g.add_node(EventNode(id="rift_opening",  label="Rift Opening",  era="First Rift War"))
    g.add_node(EventNode(id="archive_fall",  label="Archive Fall",  era="Second Age"))
    g.add_edge(BeliefEdge(source_id="kain_incident", target_id="rift_opening",  p_forward=0.9))
    g.add_edge(BeliefEdge(source_id="rift_opening",  target_id="archive_fall",  p_forward=0.5))
    updater = BayesianUpdater(g)
    updater.update_edge(
        "kain_incident", "rift_opening",
        Evidence(key="official_001", supports_forward=True, strength=0.8, source="official"),
    )
    return g


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

class TestMarkdown:
    def test_contains_title(self, graph):
        md = graph_to_markdown(graph)
        assert "# Temporal Belief Graph Report" in md

    def test_custom_title(self, graph):
        md = graph_to_markdown(graph, title="My Report")
        assert "# My Report" in md

    def test_contains_summary_section(self, graph):
        md = graph_to_markdown(graph)
        assert "## Summary" in md

    def test_node_count_in_summary(self, graph):
        md = graph_to_markdown(graph)
        assert "| Nodes | 3 |" in md

    def test_confirmed_edges_section(self, graph):
        md = graph_to_markdown(graph)
        assert "## Confirmed Edges" in md
        assert "Kain Incident" in md

    def test_uncertain_edges_section(self, graph):
        md = graph_to_markdown(graph)
        assert "## Uncertain Edges" in md
        assert "Rift Opening" in md

    def test_nodes_section(self, graph):
        md = graph_to_markdown(graph)
        assert "## Nodes" in md
        assert "kain_incident" in md

    def test_evidence_summary_section(self, graph):
        md = graph_to_markdown(graph)
        assert "## Evidence Summary" in md
        assert "official_001" in md

    def test_empty_graph(self):
        g = BeliefGraph()
        md = graph_to_markdown(g)
        assert "| Nodes | 0 |" in md
        assert "_No confirmed edges._" in md
        assert "_No evidence recorded._" in md


# ---------------------------------------------------------------------------
# Mermaid
# ---------------------------------------------------------------------------

class TestMermaid:
    def test_starts_with_graph_td(self, graph):
        diagram = graph_to_mermaid(graph)
        assert diagram.startswith("graph TD")

    def test_contains_node_definitions(self, graph):
        diagram = graph_to_mermaid(graph)
        assert "Kain Incident" in diagram
        assert "Rift Opening" in diagram

    def test_confirmed_edge_solid_arrow(self, graph):
        diagram = graph_to_mermaid(graph)
        assert "-->" in diagram

    def test_uncertain_edge_dashed_arrow(self, graph):
        diagram = graph_to_mermaid(graph)
        assert "-.->" in diagram

    def test_hide_uncertain_edges(self, graph):
        diagram = graph_to_mermaid(graph, show_uncertain=False)
        assert "-.->" not in diagram

    def test_invalid_threshold_raises(self, graph):
        with pytest.raises(ValueError, match="threshold"):
            graph_to_mermaid(graph, threshold=0.0)

    def test_empty_graph(self):
        g = BeliefGraph()
        diagram = graph_to_mermaid(g)
        assert "graph TD" in diagram


# ---------------------------------------------------------------------------
# File save
# ---------------------------------------------------------------------------

class TestFileSave:
    def test_save_markdown(self, graph, tmp_path):
        path = tmp_path / "report.md"
        save_markdown(graph, path)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "# Temporal Belief Graph Report" in content

    def test_save_mermaid(self, graph, tmp_path):
        path = tmp_path / "diagram.mmd"
        save_mermaid(graph, path)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "graph TD" in content

    def test_parent_dirs_created(self, graph, tmp_path):
        path = tmp_path / "nested" / "dir" / "report.md"
        save_markdown(graph, path)
        assert path.exists()
