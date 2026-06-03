"""tests/test_bayesian.py"""
import pytest
from tbg.schema import EventNode, BeliefEdge, PriorConfig
from tbg.graph import BeliefGraph
from tbg.bayesian import Evidence, BayesianUpdater


@pytest.fixture
def graph():
    g = BeliefGraph()
    g.add_node(EventNode(id="a", label="Event A", era="First War"))
    g.add_node(EventNode(id="b", label="Event B", era="Second War"))
    g.add_edge(BeliefEdge(source_id="a", target_id="b", p_forward=0.5))
    return g


@pytest.fixture
def updater(graph):
    return BayesianUpdater(graph)


class TestEvidence:
    def test_normal(self):
        ev = Evidence(key="ev1", supports_forward=True, strength=0.8)
        assert ev.strength == 0.8

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="key"):
            Evidence(key="", supports_forward=True)

    def test_strength_out_of_range_raises(self):
        with pytest.raises(ValueError, match="strength"):
            Evidence(key="ev1", supports_forward=True, strength=0.0)

    def test_strength_exactly_one(self):
        ev = Evidence(key="ev1", supports_forward=True, strength=1.0)
        assert ev.strength == 1.0


class TestUpdateEdge:
    def test_forward_evidence_increases_p(self, graph, updater):
        ev = Evidence(key="ev1", supports_forward=True, strength=0.9)
        edge = updater.update_edge("a", "b", ev)
        assert edge.p_forward > 0.5

    def test_backward_evidence_decreases_p(self, graph, updater):
        ev = Evidence(key="ev1", supports_forward=False, strength=0.9)
        edge = updater.update_edge("a", "b", ev)
        assert edge.p_forward < 0.5

    def test_weak_evidence_small_change(self, graph, updater):
        ev = Evidence(key="ev1", supports_forward=True, strength=0.1)
        edge = updater.update_edge("a", "b", ev)
        assert 0.5 < edge.p_forward < 0.6

    def test_evidence_key_accumulated(self, graph, updater):
        ev = Evidence(key="unique_key_001", supports_forward=True, strength=0.8)
        edge = updater.update_edge("a", "b", ev)
        assert "unique_key_001" in edge.evidence_keys

    def test_duplicate_key_not_duplicated(self, graph, updater):
        ev = Evidence(key="ev1", supports_forward=True, strength=0.8)
        updater.update_edge("a", "b", ev)
        updater.update_edge("a", "b", ev)
        edge = graph.get_edge("a", "b")
        assert edge.evidence_keys.count("ev1") == 1

    def test_evidence_history_recorded(self, graph, updater):
        ev = Evidence(key="ev1", supports_forward=True, strength=0.8)
        updater.update_edge("a", "b", ev)
        edge = graph.get_edge("a", "b")
        assert edge.update_count == 1
        assert edge.evidence_history[0].prior_before == 0.5

    def test_weak_source_does_not_reverse_direction(self):
        """A weak source supporting forward must still increase p_forward."""
        config = PriorConfig(source_weights={"weak_source": 0.1})
        g = BeliefGraph(prior_config=config)
        g.add_node(EventNode(id="a", label="Event A", era="Age A"))
        g.add_node(EventNode(id="b", label="Event B", era="Age B"))
        g.add_edge(BeliefEdge(source_id="a", target_id="b", p_forward=0.5))
        ev = Evidence(key="ev1", supports_forward=True, strength=0.5, source="weak_source")
        BayesianUpdater(g).update_edge("a", "b", ev)
        assert g.get_edge("a", "b").p_forward > 0.5

    def test_clip_prevents_zero(self, graph, updater):
        for i in range(20):
            ev = Evidence(key=f"ev{i}", supports_forward=False, strength=1.0)
            updater.update_edge("a", "b", ev)
        edge = graph.get_edge("a", "b")
        assert edge.p_forward >= updater.clip

    def test_clip_prevents_one(self, graph, updater):
        for i in range(20):
            ev = Evidence(key=f"ev{i}", supports_forward=True, strength=1.0)
            updater.update_edge("a", "b", ev)
        edge = graph.get_edge("a", "b")
        assert edge.p_forward <= 1.0 - updater.clip

    def test_source_weight_applied(self):
        config = PriorConfig(source_weights={"high_trust": 2.0, "low_trust": 0.5})
        g1 = BeliefGraph(prior_config=config)
        g2 = BeliefGraph(prior_config=config)
        for g in [g1, g2]:
            g.add_node(EventNode(id="a", label="Event A", era="Age A"))
            g.add_node(EventNode(id="b", label="Event B", era="Age B"))
            g.add_edge(BeliefEdge(source_id="a", target_id="b", p_forward=0.5))
        ev_high = Evidence(key="ev1", supports_forward=True, strength=0.7, source="high_trust")
        ev_low  = Evidence(key="ev1", supports_forward=True, strength=0.7, source="low_trust")
        BayesianUpdater(g1).update_edge("a", "b", ev_high)
        BayesianUpdater(g2).update_edge("a", "b", ev_low)
        assert g1.get_edge("a", "b").p_forward > g2.get_edge("a", "b").p_forward


class TestUpdateEdgeBatch:
    def test_batch_stronger_than_single(self, graph, updater):
        ev_single = Evidence(key="s1", supports_forward=True, strength=0.8)
        g2 = BeliefGraph()
        g2.add_node(EventNode(id="a", label="Event A", era="Age A"))
        g2.add_node(EventNode(id="b", label="Event B", era="Age B"))
        g2.add_edge(BeliefEdge(source_id="a", target_id="b", p_forward=0.5))
        u2 = BayesianUpdater(g2)
        updater.update_edge("a", "b", ev_single)
        evs = [
            Evidence(key="b1", supports_forward=True, strength=0.8),
            Evidence(key="b2", supports_forward=True, strength=0.8),
            Evidence(key="b3", supports_forward=True, strength=0.8),
        ]
        u2.update_edge_batch("a", "b", evs)
        assert g2.get_edge("a", "b").p_forward > graph.get_edge("a", "b").p_forward

    def test_empty_batch_no_change(self, graph, updater):
        before = graph.get_edge("a", "b").p_forward
        updater.update_edge_batch("a", "b", [])
        after = graph.get_edge("a", "b").p_forward
        assert before == after


class TestEnsembleUpdate:
    def test_majority_forward_increases_p(self, graph, updater):
        claims = [(0.9, 1.0), (0.8, 1.0), (0.3, 0.5)]
        edge = updater.ensemble_update("a", "b", claims)
        assert edge.p_forward > 0.5

    def test_majority_backward_decreases_p(self, graph, updater):
        claims = [(0.1, 1.0), (0.2, 1.0), (0.7, 0.3)]
        edge = updater.ensemble_update("a", "b", claims)
        assert edge.p_forward < 0.5

    def test_balanced_claims_near_prior(self, graph, updater):
        claims = [(0.9, 1.0), (0.1, 1.0)]
        edge = updater.ensemble_update("a", "b", claims)
        assert 0.3 < edge.p_forward < 0.7

    def test_empty_claims_raises(self, graph, updater):
        with pytest.raises(ValueError, match="claims"):
            updater.ensemble_update("a", "b", [])

    def test_invalid_claim_p_raises(self, graph, updater):
        with pytest.raises(ValueError):
            updater.ensemble_update("a", "b", [(1.5, 1.0)])

    def test_invalid_claim_weight_raises(self, graph, updater):
        with pytest.raises(ValueError):
            updater.ensemble_update("a", "b", [(0.8, -1.0)])

    def test_higher_weight_source_dominates(self, graph, updater):
        claims_high = [(0.9, 10.0), (0.1, 1.0)]
        claims_low  = [(0.9, 1.0),  (0.1, 10.0)]
        g2 = BeliefGraph()
        g2.add_node(EventNode(id="a", label="Event A", era="Age A"))
        g2.add_node(EventNode(id="b", label="Event B", era="Age B"))
        g2.add_edge(BeliefEdge(source_id="a", target_id="b", p_forward=0.5))
        updater.ensemble_update("a", "b", claims_high)
        BayesianUpdater(g2).ensemble_update("a", "b", claims_low)
        assert graph.get_edge("a", "b").p_forward > g2.get_edge("a", "b").p_forward


class TestUpdaterInit:
    def test_invalid_clip_raises(self, graph):
        with pytest.raises(ValueError, match="clip"):
            BayesianUpdater(graph, clip=0.0)

    def test_custom_clip(self, graph):
        u = BayesianUpdater(graph, clip=0.05)
        assert u.clip == 0.05
