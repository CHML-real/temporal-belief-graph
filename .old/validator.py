"""
validator.py
------------
Validator : BeliefGraph 의 유효성을 검사한다.

검사 항목
1. pseudo era 탐지   - EventNode.era 가 금지 키워드면 오류
2. era 미지정 탐지   - EventNode.era 가 None 이면 경고
3. 사이클 탐지       - p_forward > threshold 인 엣지만 확정 엣지로 보고
                       DFS 로 사이클 존재 여부를 확인한다
4. 고립 노드 탐지    - 엣지가 하나도 없는 노드 목록 반환

ValidationResult 로 결과를 돌려주며 예외를 던지지 않는다.
예외가 필요한 경우 .raise_if_errors() 를 호출한다.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from .graph import BeliefGraph


# ---------------------------------------------------------------------------
# 결과 컨테이너
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def raise_if_errors(self) -> None:
        if self.errors:
            msg = "\n".join(self.errors)
            raise ValueError(f"BeliefGraph 유효성 오류:\n{msg}")

    def __repr__(self) -> str:
        return (
            f"ValidationResult("
            f"errors={len(self.errors)}, "
            f"warnings={len(self.warnings)}, "
            f"valid={self.is_valid})"
        )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class Validator:
    """
    Parameters
    ----------
    cycle_threshold : float
        이 값보다 p_forward 가 크면 '확정 엣지'로 간주하고 사이클을 탐지한다.
        기본 0.5 (불확실 엣지는 사이클 탐지에서 제외).
    """

    def __init__(self, cycle_threshold: float = 0.5):
        if not (0.0 < cycle_threshold < 1.0):
            raise ValueError(
                f"cycle_threshold={cycle_threshold} 는 (0, 1) 범위여야 합니다."
            )
        self.cycle_threshold = cycle_threshold

    # ------------------------------------------------------------------
    # 퍼블릭 API
    # ------------------------------------------------------------------

    def validate(self, graph: BeliefGraph) -> ValidationResult:
        """그래프 전체를 검사하고 ValidationResult 를 반환한다."""
        result = ValidationResult()
        self._check_pseudo_era(graph, result)
        self._check_missing_era(graph, result)
        self._check_cycles(graph, result)
        self._check_isolated_nodes(graph, result)
        return result

    # ------------------------------------------------------------------
    # 개별 검사
    # ------------------------------------------------------------------

    def _check_pseudo_era(self, graph: BeliefGraph, result: ValidationResult) -> None:
        pseudo_set = set(graph._config.all_pseudo_eras)
        for node in graph.nodes:
            if node.era in pseudo_set:
                result.errors.append(
                    f"[pseudo_era] 노드 '{node.id}' 의 era='{node.era}' 는 "
                    "금지된 pseudo era 입니다."
                )

    def _check_missing_era(self, graph: BeliefGraph, result: ValidationResult) -> None:
        for node in graph.nodes:
            if node.era is None:
                result.warnings.append(
                    f"[missing_era] 노드 '{node.id}' 에 era 가 설정되지 않았습니다."
                )

    def _check_cycles(self, graph: BeliefGraph, result: ValidationResult) -> None:
        """
        p_forward > cycle_threshold 인 엣지만 '확정 방향'으로 보고
        DFS 로 사이클을 탐지한다.
        """
        # 인접 리스트 구성
        adj: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
        for edge in graph.edges:
            if edge.p_forward > self.cycle_threshold:
                adj[edge.source_id].append(edge.target_id)

        # DFS 상태: 0=미방문, 1=방문중, 2=완료
        state: dict[str, int] = {node_id: 0 for node_id in adj}
        cycle_path: list[str] = []

        def dfs(node_id: str) -> bool:
            state[node_id] = 1
            cycle_path.append(node_id)
            for neighbor in adj[node_id]:
                if state[neighbor] == 1:
                    # 사이클 발견 - 경로 추출
                    idx = cycle_path.index(neighbor)
                    cycle = cycle_path[idx:] + [neighbor]
                    result.errors.append(
                        f"[cycle] 사이클 탐지: {' → '.join(cycle)}"
                    )
                    return True
                if state[neighbor] == 0:
                    if dfs(neighbor):
                        return True
            cycle_path.pop()
            state[node_id] = 2
            return False

        for node_id in adj:
            if state[node_id] == 0:
                dfs(node_id)

    def _check_isolated_nodes(self, graph: BeliefGraph, result: ValidationResult) -> None:
        connected = set()
        for edge in graph.edges:
            connected.add(edge.source_id)
            connected.add(edge.target_id)
        for node in graph.nodes:
            if node.id not in connected:
                result.warnings.append(
                    f"[isolated] 노드 '{node.id}' 는 연결된 엣지가 없습니다."
                )
