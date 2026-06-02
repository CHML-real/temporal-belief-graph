"""
schema.py
---------
tbg의 기본 데이터 스키마.

EventNode   : 이벤트 하나를 표현하는 노드
BeliefEdge  : A → B 순서 관계의 확률을 담는 엣지
PriorConfig : 사람이 처음 한 번만 설정하는 초기 조건
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# EventNode
# ---------------------------------------------------------------------------

@dataclass
class EventNode:
    """
    스토리/세계관에서 하나의 사건(이벤트)을 표현한다.

    Parameters
    ----------
    id : str
        고유 식별자. 영문 소문자 + 언더스코어 권장. (예: "kain_incident")
    label : str
        사람이 읽을 이름. (예: "카인 사건")
    era : Optional[str]
        이 노드가 속한 시대/구간 이름.
        None 이면 era 미지정 상태 → validator 가 경고.
    description : str
        사건에 대한 자유 텍스트 설명.
    sources : list[str]
        이 사건을 뒷받침하는 소스 키 목록.
        (예: ["corpus/event_chain.json#001"])
    """

    id: str
    label: str
    era: Optional[str] = None
    description: str = ""
    sources: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            raise ValueError("EventNode.id 는 빈 문자열일 수 없습니다.")
        if not self.label:
            raise ValueError("EventNode.label 은 빈 문자열일 수 없습니다.")
        # pseudo era 금지 목록 (validator 에서도 재검사, 여기서는 즉시 차단)
        _PSEUDO = {"현재", "과거", "미래", "언젠가", "나중에", "예전에"}
        if self.era in _PSEUDO:
            raise ValueError(
                f"EventNode.era='{self.era}' 는 pseudo era 입니다. "
                "구체적인 사건명/시대명을 사용하세요."
            )


# ---------------------------------------------------------------------------
# BeliefEdge
# ---------------------------------------------------------------------------

@dataclass
class BeliefEdge:
    """
    두 EventNode 사이의 순서 관계를 확률로 표현하는 엣지.

    'source → target' 순서일 확률을 p_forward 로 저장한다.
    p_forward = 0.5  → 순서를 모른다 (uniform prior)
    p_forward = 1.0  → source 가 반드시 먼저
    p_forward = 0.0  → target 이 반드시 먼저

    Parameters
    ----------
    source_id : str
        선행 이벤트 후보 노드 id.
    target_id : str
        후행 이벤트 후보 노드 id.
    p_forward : float
        P(source → target). 0.0 이상 1.0 이하.
    evidence_keys : list[str]
        이 엣지의 확률을 뒷받침하는 증거 키 목록.
    weight : float
        앙상블에서 이 엣지 소스의 신뢰도 가중치. 기본 1.0.
    """

    source_id: str
    target_id: str
    p_forward: float = 0.5
    evidence_keys: list[str] = field(default_factory=list)
    weight: float = 1.0

    def __post_init__(self):
        if not (0.0 <= self.p_forward <= 1.0):
            raise ValueError(
                f"BeliefEdge.p_forward={self.p_forward} 는 [0, 1] 범위여야 합니다."
            )
        if self.weight <= 0:
            raise ValueError(
                f"BeliefEdge.weight={self.weight} 는 0 보다 커야 합니다."
            )
        if self.source_id == self.target_id:
            raise ValueError(
                "BeliefEdge 의 source_id 와 target_id 는 같을 수 없습니다. "
                "(self-loop 금지)"
            )

    @property
    def p_backward(self) -> float:
        """P(target → source) = 1 - p_forward"""
        return 1.0 - self.p_forward

    @property
    def is_uncertain(self) -> bool:
        """p_forward 가 [0.4, 0.6] 구간이면 '불확실' 상태로 간주."""
        return 0.4 <= self.p_forward <= 0.6


# ---------------------------------------------------------------------------
# PriorConfig
# ---------------------------------------------------------------------------

@dataclass
class PriorConfig:
    """
    사람이 처음 한 번만 설정하는 초기 조건 스키마.

    이후 evidence 가 누적되면 BeliefEdge.p_forward 가 자동 갱신되므로
    PriorConfig 는 '시작점'만 정의한다.

    Parameters
    ----------
    default_p : float
        아무 정보 없을 때 기본 prior. 보통 0.5 (uniform).
    source_weights : dict[str, float]
        소스별 초기 신뢰도 가중치.
        (예: {"corpus/event_chain.json": 1.5, "fan_wiki": 0.7})
    pseudo_era_list : list[str]
        추가로 막을 pseudo era 키워드. 기본 목록에 합산된다.
    """

    default_p: float = 0.5
    source_weights: dict[str, float] = field(default_factory=dict)
    pseudo_era_list: list[str] = field(default_factory=list)

    # 기본 pseudo era 목록 (EventNode.__post_init__ 과 동일하게 유지)
    _BASE_PSEUDO: list[str] = field(
        default_factory=lambda: ["현재", "과거", "미래", "언젠가", "나중에", "예전에"],
        init=False,
        repr=False,
    )

    def __post_init__(self):
        if not (0.0 < self.default_p < 1.0):
            raise ValueError(
                f"PriorConfig.default_p={self.default_p} 는 (0, 1) 열린 구간이어야 합니다."
            )
        for src, w in self.source_weights.items():
            if w <= 0:
                raise ValueError(
                    f"source_weights['{src}']={w} 는 0 보다 커야 합니다."
                )

    @property
    def all_pseudo_eras(self) -> list[str]:
        """기본 목록 + 사용자 추가 목록을 합친 전체 pseudo era 목록."""
        return list(set(self._BASE_PSEUDO + self.pseudo_era_list))
