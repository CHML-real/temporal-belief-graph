"""
bayesian.py
-----------
BayesianUpdater : evidence 가 누적될 때마다 BeliefEdge.p_forward 를 갱신한다.

핵심 수식
---------
P(A→B | evidence) ∝ P(evidence | A→B) × P(A→B)

likelihood 는 evidence 의 방향성(supports_forward)과
소스 신뢰도 가중치(weight)로 결정한다.

업데이트 방식
-------------
1. 단일 evidence  : update_edge()
2. 복수 evidence  : update_edge_batch()
3. 앙상블 통합    : ensemble_update()
   - 여러 소스가 서로 다른 p_forward 를 주장할 때
     가중 평균으로 통합한 뒤 베이즈 업데이트 1회 적용
"""

from __future__ import annotations
from dataclasses import dataclass
from .graph import BeliefGraph
from .schema import BeliefEdge


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """
    하나의 증거 단위.

    Parameters
    ----------
    key : str
        증거 식별자. (예: "corpus/event_chain.json#kain_001")
    supports_forward : bool
        True  → source → target 순서를 지지
        False → target → source 순서를 지지
    strength : float
        증거의 강도. (0.0, 1.0] 범위.
        1.0 = 완전히 확실한 증거, 0.1 = 매우 약한 힌트.
    source : str
        증거 출처 키. PriorConfig.source_weights 와 매핑된다.
    """

    key: str
    supports_forward: bool
    strength: float = 0.5
    source: str = ""

    def __post_init__(self):
        if not (0.0 < self.strength <= 1.0):
            raise ValueError(
                f"Evidence.strength={self.strength} 는 (0, 1] 범위여야 합니다."
            )
        if not self.key:
            raise ValueError("Evidence.key 는 빈 문자열일 수 없습니다.")


# ---------------------------------------------------------------------------
# BayesianUpdater
# ---------------------------------------------------------------------------

class BayesianUpdater:
    """
    Parameters
    ----------
    graph : BeliefGraph
        업데이트 대상 그래프. 내부 엣지를 직접 수정한다.
    clip : float
        p_forward 의 최솟값/최댓값 클리핑. 기본 0.01.
        확률이 0 또는 1 로 수렴해 버리는 것을 방지한다.
    """

    def __init__(self, graph: BeliefGraph, clip: float = 0.01):
        if not (0.0 < clip < 0.5):
            raise ValueError(
                f"clip={clip} 는 (0, 0.5) 범위여야 합니다."
            )
        self.graph = graph
        self.clip = clip

    # ------------------------------------------------------------------
    # 단일 업데이트
    # ------------------------------------------------------------------

    def update_edge(
        self,
        source_id: str,
        target_id: str,
        evidence: Evidence,
    ) -> BeliefEdge:
        """
        단일 evidence 로 엣지 하나를 베이즈 업데이트한다.

        Returns
        -------
        BeliefEdge
            업데이트된 엣지 (그래프 내 객체를 직접 수정 후 반환).
        """
        edge = self.graph.get_edge(source_id, target_id)
        source_weight = self.graph._config.source_weights.get(evidence.source, 1.0)

        # likelihood 계산
        # strength=0.1 → lf=0.55, lb=0.45 (약한 신호)
        # strength=1.0 → lf=1.0,  lb=0.0  (강한 신호)
        # source_weight 는 supports 방향 likelihood 에 스케일로 적용
        base_lf = 0.5 + evidence.strength * 0.5
        base_lb = 0.5 - evidence.strength * 0.5
        if evidence.supports_forward:
            lf = base_lf * source_weight
            lb = base_lb
        else:
            lf = base_lb
            lb = base_lf * source_weight

        # 베이즈 업데이트
        prior_f = edge.p_forward
        prior_b = edge.p_backward

        posterior_f_unnorm = lf * prior_f
        posterior_b_unnorm = lb * prior_b
        total = posterior_f_unnorm + posterior_b_unnorm

        if total == 0:
            return edge  # 업데이트 불가 (수치 안정성)

        new_p = posterior_f_unnorm / total
        edge.p_forward = max(self.clip, min(1.0 - self.clip, new_p))

        # evidence key 누적
        if evidence.key not in edge.evidence_keys:
            edge.evidence_keys.append(evidence.key)

        return edge

    # ------------------------------------------------------------------
    # 배치 업데이트
    # ------------------------------------------------------------------

    def update_edge_batch(
        self,
        source_id: str,
        target_id: str,
        evidences: list[Evidence],
    ) -> BeliefEdge:
        """
        복수의 evidence 를 순서대로 적용한다.
        각 evidence 가 이전 posterior 를 다음 prior 로 사용한다.
        """
        for ev in evidences:
            self.update_edge(source_id, target_id, ev)
        return self.graph.get_edge(source_id, target_id)

    # ------------------------------------------------------------------
    # 앙상블 업데이트
    # ------------------------------------------------------------------

    def ensemble_update(
        self,
        source_id: str,
        target_id: str,
        claims: list[tuple[float, float]],
    ) -> BeliefEdge:
        """
        여러 소스의 p_forward 주장을 가중 평균으로 통합한 뒤
        베이즈 업데이트 1회 적용한다.

        Parameters
        ----------
        claims : list[tuple[float, float]]
            (p_forward_주장, 소스_가중치) 쌍의 목록.
            예: [(0.9, 1.5), (0.6, 0.8), (0.4, 1.0)]

        Returns
        -------
        BeliefEdge
            업데이트된 엣지.
        """
        if not claims:
            raise ValueError("claims 가 비어 있습니다.")

        for p, w in claims:
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"claim p_forward={p} 는 [0, 1] 범위여야 합니다.")
            if w <= 0:
                raise ValueError(f"claim weight={w} 는 0 보다 커야 합니다.")

        total_weight = sum(w for _, w in claims)
        ensemble_p = sum(p * w for p, w in claims) / total_weight

        # 앙상블 결과를 단일 evidence 로 변환해 베이즈 업데이트
        # ensemble_p=0.5 (중립) → strength≈0 (prior 유지)
        # ensemble_p=1.0 → strength=1.0 (강한 순방향)
        supports_forward = ensemble_p >= 0.5
        strength = abs(ensemble_p - 0.5) * 2.0   # [0, 1]
        strength = max(0.01, min(1.0, strength))

        ev = Evidence(
            key=f"ensemble:{source_id}→{target_id}",
            supports_forward=supports_forward,
            strength=strength,
            source="ensemble",
        )
        return self.update_edge(source_id, target_id, ev)
