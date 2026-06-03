"""tests/test_io.py"""
import json
import pytest
from pathlib import Path
from tbg.schema import EventNode, BeliefEdge, PriorConfig, EvidenceRecord
from tbg.graph import BeliefGraph
from tbg.bayesian import BayesianUpdater, Evidence
from tbg.io import graph_to_dict, graph_from_dict, graph_to_json, graph_from_json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_graph():
    config = PriorConfig(source_weights={"official": 1.5})
    g = BeliefGraph(prior_config=config)
    g.add_node(EventNode(id="a", label="Event A", era="First War"))
    g.add_node(EventNode(id="b", label="Event B", era="Second War"))
    g.add_edge(BeliefEdge(source_id="a", target_id="b", p_forward=0.8))
    return g


@pytest.fixture
def graph_with_evidence():
    g = BeliefGraph()
    g.add_node(EventNode(id="a", label="Event A", era="First War"))
    g.add_node(EventNode(id="b", label="Event B", era="Second War"))
    g.add_edge(BeliefEdge(source_id="a", target_id="b", p_forward=0.5))
    updater = BayesianUpdater(g)
    updater.update_edge("a", "b", Evidence(key="ev1", supports_forward=True, strength=0.9, source="official"))
    updater.update_edge("a", "b", Evidence(key="ev2", supports_forward=False, strength=0.5, source="fan"))
    return g


# ---------------------------------------------------------------------------
# graph_to_dict / graph_from_dict
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_nodes_preserved(self, simple_graph):
        data = graph_to_dict(simple_graph)
        restored = graph_from_dict(data)
        assert {n.id for n in restored.nodes} == {"a", "b"}

    def test_node_fields_preserved(self, simple_graph):
        data = graph_to_dict(simple_graph)
        restored = graph_from_dict(data)
        node = restored.get_node("a")
        assert node.label == "Event A"
        assert node.era == "First War"

    def test_edge_p_forward_preserved(self, simple_graph):
        data = graph_to_dict(simple_graph)
        restored = graph_from_dict(data)
        edge = restored.get_edge("a", "b")
        assert edge.p_forward == pytest.approx(0.8)

    def test_config_source_weights_preserved(self, simple_graph):
        data = graph_to_dict(simple_graph)
        restored = graph_from_dict(data)
        assert restored.config.source_weights.get("official") == 1.5

    def test_evidence_history_preserved(self, graph_with_evidence):
        data = graph_to_dict(graph_with_evidence)
        restored = graph_from_dict(data)
        edge = restored.get_edge("a", "b")
        assert edge.update_count == 2
        assert edge.evidence_history[0].key == "ev1"
        assert edge.evidence_history[1].key == "ev2"

    def test_evidence_keys_preserved(self, graph_with_evidence):
        data = graph_to_dict(graph_with_evidence)
        restored = graph_from_dict(data)
        edge = restored.get_edge("a", "b")
        assert "ev1" in edge.evidence_keys
        assert "ev2" in edge.evidence_keys

    def test_none_era_preserved(self):
        g = BeliefGraph()
        g.add_node(EventNode(id="x", label="X"))
        data = graph_to_dict(g)
        restored = graph_from_dict(data)
        assert restored.get_node("x").era is None

    def test_version_field_present(self, simple_graph):
        data = graph_to_dict(simple_graph)
        assert "version" in data

    def test_empty_graph_roundtrip(self):
        g = BeliefGraph()
        data = graph_to_dict(g)
        restored = graph_from_dict(data)
        assert len(restored.nodes) == 0
        assert len(restored.edges) == 0


# ---------------------------------------------------------------------------
# graph_to_json / graph_from_json
# ---------------------------------------------------------------------------

class TestJsonFile:
    def test_write_and_read(self, simple_graph, tmp_path):
        path = tmp_path / "graph.json"
        graph_to_json(simple_graph, path)
        restored = graph_from_json(path)
        assert {n.id for n in restored.nodes} == {"a", "b"}
        assert restored.get_edge("a", "b").p_forward == pytest.approx(0.8)

    def test_file_is_valid_json(self, simple_graph, tmp_path):
        path = tmp_path / "graph.json"
        graph_to_json(simple_graph, path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "nodes" in data
        assert "edges" in data

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            graph_from_json(tmp_path / "nonexistent.json")

    def test_parent_dirs_created(self, simple_graph, tmp_path):
        path = tmp_path / "nested" / "dir" / "graph.json"
        graph_to_json(simple_graph, path)
        assert path.exists()

    def test_unicode_preserved_in_file(self, tmp_path):
        g = BeliefGraph()
        g.add_node(EventNode(id="kain", label="카인 사건", era="제1시대"))
        path = tmp_path / "graph.json"
        graph_to_json(g, path)
        restored = graph_from_json(path)
        assert restored.get_node("kain").label == "카인 사건"
