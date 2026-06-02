"""
graph.py
--------
BeliefGraph : 확률 엣지 기반 DAG 구조.

- EventNode 를 노드로, BeliefEdge 를 엣지로 관리한다.
- 엣지의 p_forward 는 이후 베이즈 업데이터(bayesian.py)가 갱신한다.
- 이 파일은 구조(추가/조회/삭제)만 담당한다.
"""

from __future__ import annotations
from typing import Optional
from .schema import EventNode, BeliefEdge, PriorConfig


class BeliefGraph:
    """
    확률 엣지 기반 방향 그래프.

    Parameters
    ----------
    prior_config : PriorConfig
        초기 조건. 생략하면 default PriorConfig() 사용.
    """

    def __init__(self, prior_config: Optional[PriorConfig] = None):
        self._config: PriorConfig = prior_config or PriorConfig()
        self._nodes: dict[str, EventNode] = {}
        # (source_id, target_id) → BeliefEdge
        self._edges: dict[tuple[str, str], BeliefEdge] = {}

    # ------------------------------------------------------------------
    # 노드
    # ------------------------------------------------------------------

    def add_node(self, node: EventNode) -> None:
        """노드를 추가한다. 같은 id 가 이미 있으면 덮어쓴다."""
        self._nodes[node.id] = node

    def get_node(self, node_id: str) -> EventNode:
        if node_id not in self._nodes:
            raise KeyError(f"노드 '{node_id}' 가 그래프에 없습니다.")
        return self._nodes[node_id]

    def remove_node(self, node_id: str) -> None:
        """노드와 해당 노드에 연결된 모든 엣지를 함께 제거한다."""
        if node_id not in self._nodes:
            raise KeyError(f"노드 '{node_id}' 가 그래프에 없습니다.")
        del self._nodes[node_id]
        to_remove = [
            key for key in self._edges
            if key[0] == node_id or key[1] == node_id
        ]
        for key in to_remove:
            del self._edges[key]

    @property
    def nodes(self) -> list[EventNode]:
        return list(self._nodes.values())

    # ------------------------------------------------------------------
    # 엣지
    # ------------------------------------------------------------------

    def add_edge(self, edge: BeliefEdge) -> None:
        """
        엣지를 추가한다.
        - source/target 노드가 그래프에 없으면 KeyError.
        - 같은 (source, target) 쌍이 이미 있으면 덮어쓴다.
        """
        if edge.source_id not in self._nodes:
            raise KeyError(f"source 노드 '{edge.source_id}' 가 그래프에 없습니다.")
        if edge.target_id not in self._nodes:
            raise KeyError(f"target 노드 '{edge.target_id}' 가 그래프에 없습니다.")
        self._edges[(edge.source_id, edge.target_id)] = edge

    def get_edge(self, source_id: str, target_id: str) -> BeliefEdge:
        key = (source_id, target_id)
        if key not in self._edges:
            raise KeyError(f"엣지 '{source_id} → {target_id}' 가 그래프에 없습니다.")
        return self._edges[key]

    def remove_edge(self, source_id: str, target_id: str) -> None:
        key = (source_id, target_id)
        if key not in self._edges:
            raise KeyError(f"엣지 '{source_id} → {target_id}' 가 그래프에 없습니다.")
        del self._edges[key]

    @property
    def edges(self) -> list[BeliefEdge]:
        return list(self._edges.values())

    # ------------------------------------------------------------------
    # 조회 유틸
    # ------------------------------------------------------------------

    def successors(self, node_id: str) -> list[EventNode]:
        """node_id 에서 출발하는 엣지의 target 노드 목록."""
        return [
            self._nodes[t]
            for (s, t) in self._edges
            if s == node_id and t in self._nodes
        ]

    def predecessors(self, node_id: str) -> list[EventNode]:
        """node_id 로 들어오는 엣지의 source 노드 목록."""
        return [
            self._nodes[s]
            for (s, t) in self._edges
            if t == node_id and s in self._nodes
        ]

    def uncertain_edges(self) -> list[BeliefEdge]:
        """p_forward 가 [0.4, 0.6] 인 불확실 엣지 목록."""
        return [e for e in self._edges.values() if e.is_uncertain]

    # ------------------------------------------------------------------
    # 초기화 헬퍼
    # ------------------------------------------------------------------

    def init_uniform(self, source_id: str, target_id: str) -> BeliefEdge:
        """
        두 노드 사이에 uniform prior(p=0.5) 엣지를 생성하고 추가한다.
        이미 엣지가 있으면 기존 엣지를 반환한다.
        """
        key = (source_id, target_id)
        if key in self._edges:
            return self._edges[key]
        edge = BeliefEdge(
            source_id=source_id,
            target_id=target_id,
            p_forward=self._config.default_p,
        )
        self.add_edge(edge)
        return edge

    # ------------------------------------------------------------------
    # 표현
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BeliefGraph("
            f"nodes={len(self._nodes)}, "
            f"edges={len(self._edges)}, "
            f"uncertain={len(self.uncertain_edges())})"
        )
