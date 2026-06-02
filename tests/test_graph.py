"""tests/test_graph.py"""
import pytest
from tbg.schema import EventNode, BeliefEdge, PriorConfig
from tbg.graph import BeliefGraph


@pytest.fixture
def graph():
    g = BeliefGraph()
    g.add_node(EventNode(id="a", label="사건 A", era="1차 전쟁"))
    g.add_node(EventNode(id="b", label="사건 B", era="2차 전쟁"))
    g.add_node(EventNode(id="c", label="사건 C", era="3차 전쟁"))
    return g


class TestNodes:
    def test_add_and_get(self, graph):
        node = graph.get_node("a")
        assert node.label == "사건 A"

    def test_get_missing_raises(self, graph):
        with pytest.raises(KeyError):
            graph.get_node("없는노드")

    def test_nodes_list(self, graph):
        assert len(graph.nodes) == 3

    def test_remove_node(self, graph):
        graph.remove_node("a")
        assert len(graph.nodes) == 2

    def test_remove_node_cascades_edges(self, graph):
        graph.add_edge(BeliefEdge(source_id="a", target_id="b", p_forward=0.8))
        graph.remove_node("a")
        assert len(graph.edges) == 0

    def test_overwrite_node(self, graph):
        graph.add_node(EventNode(id="a", label="수정된 사건 A", era="1차 전쟁"))
        assert graph.get_node("a").label == "수정된 사건 A"


class TestEdges:
    def test_add_and_get(self, graph):
        graph.add_edge(BeliefEdge(source_id="a", target_id="b", p_forward=0.9))
        e = graph.get_edge("a", "b")
        assert e.p_forward == 0.9

    def test_missing_source_raises(self, graph):
        # 새 graph.py 는 "Source node" 메시지를 사용
        with pytest.raises(KeyError):
            graph.add_edge(BeliefEdge(source_id="없음", target_id="b"))

    def test_missing_target_raises(self, graph):
        with pytest.raises(KeyError):
            graph.add_edge(BeliefEdge(source_id="a", target_id="없음"))

    def test_remove_edge(self, graph):
        graph.add_edge(BeliefEdge(source_id="a", target_id="b"))
        graph.remove_edge("a", "b")
        assert len(graph.edges) == 0

    def test_get_missing_edge_raises(self, graph):
        with pytest.raises(KeyError):
            graph.get_edge("a", "b")


class TestTraversal:
    def test_successors(self, graph):
        graph.add_edge(BeliefEdge(source_id="a", target_id="b"))
        graph.add_edge(BeliefEdge(source_id="a", target_id="c"))
        succ = graph.successors("a")
        assert {n.id for n in succ} == {"b", "c"}

    def test_predecessors(self, graph):
        graph.add_edge(BeliefEdge(source_id="a", target_id="c"))
        graph.add_edge(BeliefEdge(source_id="b", target_id="c"))
        pred = graph.predecessors("c")
        assert {n.id for n in pred} == {"a", "b"}

    def test_uncertain_edges(self, graph):
        graph.add_edge(BeliefEdge(source_id="a", target_id="b", p_forward=0.5))
        graph.add_edge(BeliefEdge(source_id="b", target_id="c", p_forward=0.9))
        uncertain = graph.uncertain_edges()
        assert len(uncertain) == 1
        assert uncertain[0].source_id == "a"


class TestInitUniform:
    def test_creates_edge(self, graph):
        e = graph.init_uniform("a", "b")
        assert e.p_forward == 0.5

    def test_existing_edge_not_overwritten(self, graph):
        graph.add_edge(BeliefEdge(source_id="a", target_id="b", p_forward=0.8))
        e = graph.init_uniform("a", "b")
        assert e.p_forward == 0.8

    def test_custom_prior_config(self):
        config = PriorConfig(default_p=0.7)
        g = BeliefGraph(prior_config=config)
        g.add_node(EventNode(id="x", label="X", era="사건X"))
        g.add_node(EventNode(id="y", label="Y", era="사건Y"))
        e = g.init_uniform("x", "y")
        assert e.p_forward == 0.7


class TestRepr:
    def test_repr(self, graph):
        graph.add_edge(BeliefEdge(source_id="a", target_id="b", p_forward=0.5))
        r = repr(graph)
        assert "nodes=3" in r
        assert "edges=1" in r
        assert "uncertain=1" in r
