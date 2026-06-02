"""tests/test_schema.py"""
import pytest
from tbg.schema import EventNode, BeliefEdge, PriorConfig


class TestEventNode:
    def test_normal(self):
        n = EventNode(id="kain_incident", label="카인 사건", era="카인 사건 직후")
        assert n.id == "kain_incident"
        assert n.era == "카인 사건 직후"

    def test_empty_id_raises(self):
        with pytest.raises(ValueError, match="id"):
            EventNode(id="", label="테스트")

    def test_empty_label_raises(self):
        with pytest.raises(ValueError, match="label"):
            EventNode(id="test", label="")

    def test_pseudo_era_not_blocked_in_schema(self):
        # 새 schema 는 pseudo era 를 schema 단에서 막지 않고 validator 에서 처리
        node = EventNode(id="x", label="x", era="present")
        assert node.era == "present"

    def test_no_era_is_allowed(self):
        n = EventNode(id="x", label="x")
        assert n.era is None

    def test_sources_default_empty(self):
        n = EventNode(id="x", label="x")
        assert n.sources == []


class TestBeliefEdge:
    def test_normal(self):
        e = BeliefEdge(source_id="a", target_id="b", p_forward=0.8)
        assert e.p_backward == pytest.approx(0.2)

    def test_uniform_prior(self):
        e = BeliefEdge(source_id="a", target_id="b")
        assert e.p_forward == 0.5
        assert e.is_uncertain is True

    def test_certain_forward(self):
        e = BeliefEdge(source_id="a", target_id="b", p_forward=0.95)
        assert e.is_uncertain is False

    def test_p_out_of_range_raises(self):
        with pytest.raises(ValueError, match="p_forward"):
            BeliefEdge(source_id="a", target_id="b", p_forward=1.5)

    def test_self_loop_raises(self):
        with pytest.raises(ValueError):
            BeliefEdge(source_id="a", target_id="a")

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="weight"):
            BeliefEdge(source_id="a", target_id="b", weight=-1.0)

    def test_evidence_history_default_empty(self):
        e = BeliefEdge(source_id="a", target_id="b")
        assert e.evidence_history == []

    def test_update_count(self):
        e = BeliefEdge(source_id="a", target_id="b")
        assert e.update_count == 0


class TestPriorConfig:
    def test_defaults(self):
        pc = PriorConfig()
        assert pc.default_p == 0.5
        # 새 schema 는 영문 pseudo era 기본 목록 사용
        assert "present" in pc.all_pseudo_eras
        assert "past" in pc.all_pseudo_eras

    def test_custom_pseudo_era(self):
        pc = PriorConfig(pseudo_era_list=["태초에"])
        assert "태초에" in pc.all_pseudo_eras
        assert "present" in pc.all_pseudo_eras  # 기본 목록도 유지

    def test_source_weights(self):
        pc = PriorConfig(source_weights={"event_chain.json": 1.5})
        assert pc.source_weights["event_chain.json"] == 1.5

    def test_invalid_default_p(self):
        with pytest.raises(ValueError, match="default_p"):
            PriorConfig(default_p=0.0)

    def test_invalid_source_weight(self):
        with pytest.raises(ValueError):
            PriorConfig(source_weights={"bad_source": -0.5})
