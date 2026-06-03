"""tests/test_validator.py"""
import pytest
from tbg.schema import EventNode, BeliefEdge, PriorConfig
from tbg.graph import BeliefGraph
from tbg.validator import Validator, ValidationResult


@pytest.fixture
def clean_graph():
    g = BeliefGraph()
    g.add_node(EventNode(id="a", label="Event A", era="First War"))
    g.add_node(EventNode(id="b", label="Event B", era="Second War"))
    g.add_node(EventNode(id="c", label="Event C", era="Third War"))
    g.add_edge(BeliefEdge(source_id="a", target_id="b", p_forward=0.9))
    g.add_edge(BeliefEdge(source_id="b", target_id="c", p_forward=0.8))
    return g


@pytest.fixture
def validator():
    return Validator()


class TestValidationResult:
    def test_valid_when_no_errors(self):
        r = ValidationResult()
        assert r.is_valid is True

    def test_invalid_when_errors(self):
        r = ValidationResult(errors=["error one"])
        assert r.is_valid is False

    def test_raise_if_errors(self):
        r = ValidationResult(errors=["critical error"])
        with pytest.raises(ValueError, match="critical error"):
            r.raise_if_errors()

    def test_no_raise_when_valid(self):
        r = ValidationResult()
        r.raise_if_errors()

    def test_repr(self):
        r = ValidationResult(errors=["e1"], warnings=["w1"])
        assert "errors=1" in repr(r)
        assert "warnings=1" in repr(r)


class TestCleanGraph:
    def test_clean_graph_is_valid(self, clean_graph, validator):
        result = validator.validate(clean_graph)
        assert result.is_valid is True


class TestPseudoEra:
    def test_detects_pseudo_era(self, validator):
        g = BeliefGraph()
        node = EventNode(id="x", label="X")
        node.era = "present"
        g.add_node(node)
        result = validator.validate(g)
        assert not result.is_valid
        assert any("pseudo_era" in e for e in result.errors)

    def test_custom_pseudo_era(self, validator):
        config = PriorConfig(pseudo_era_list=["the beginning"])
        g = BeliefGraph(prior_config=config)
        node = EventNode(id="x", label="X")
        node.era = "the beginning"
        g.add_node(node)
        result = validator.validate(g)
        assert not result.is_valid


class TestMissingEra:
    def test_warns_missing_era(self, validator):
        g = BeliefGraph()
        g.add_node(EventNode(id="x", label="X"))
        result = validator.validate(g)
        assert result.is_valid
        assert any("missing_era" in w for w in result.warnings)


class TestCycleDetection:
    def test_no_cycle(self, clean_graph, validator):
        result = validator.validate(clean_graph)
        assert not any("cycle" in e for e in result.errors)

    def test_simple_cycle(self, validator):
        g = BeliefGraph()
        g.add_node(EventNode(id="a", label="Event A", era="Age A"))
        g.add_node(EventNode(id="b", label="Event B", era="Age B"))
        g.add_edge(BeliefEdge(source_id="a", target_id="b", p_forward=0.9))
        g.add_edge(BeliefEdge(source_id="b", target_id="a", p_forward=0.9))
        result = validator.validate(g)
        assert not result.is_valid
        assert any("cycle" in e for e in result.errors)

    def test_three_node_cycle(self, validator):
        g = BeliefGraph()
        for nid in ["a", "b", "c"]:
            g.add_node(EventNode(id=nid, label=f"Event {nid.upper()}", era=f"Age {nid.upper()}"))
        g.add_edge(BeliefEdge(source_id="a", target_id="b", p_forward=0.9))
        g.add_edge(BeliefEdge(source_id="b", target_id="c", p_forward=0.9))
        g.add_edge(BeliefEdge(source_id="c", target_id="a", p_forward=0.9))
        result = validator.validate(g)
        assert not result.is_valid
        assert any("cycle" in e for e in result.errors)

    def test_uncertain_edge_ignored_in_cycle(self, validator):
        g = BeliefGraph()
        g.add_node(EventNode(id="a", label="Event A", era="Age A"))
        g.add_node(EventNode(id="b", label="Event B", era="Age B"))
        g.add_edge(BeliefEdge(source_id="a", target_id="b", p_forward=0.9))
        g.add_edge(BeliefEdge(source_id="b", target_id="a", p_forward=0.4))
        result = validator.validate(g)
        assert result.is_valid


class TestIsolatedNodes:
    def test_isolated_node_warning(self, validator):
        g = BeliefGraph()
        g.add_node(EventNode(id="alone", label="Alone Event", era="Age X"))
        result = validator.validate(g)
        assert result.is_valid
        assert any("isolated" in w for w in result.warnings)

    def test_connected_nodes_no_warning(self, clean_graph, validator):
        result = validator.validate(clean_graph)
        assert not any("isolated" in w for w in result.warnings)


class TestValidatorInit:
    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="cycle_threshold"):
            Validator(cycle_threshold=0.0)

    def test_custom_threshold(self):
        v = Validator(cycle_threshold=0.8)
        assert v.cycle_threshold == 0.8
