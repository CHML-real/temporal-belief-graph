"""
schema.py
---------
Core data schemas for Temporal Belief Graph.

EventNode      : A single event in a narrative, lore, historical, or world model.
BeliefEdge     : A probabilistic temporal relation between two events.
EvidenceRecord : A stored audit trail for probability updates.
PriorConfig    : Initial probability and source-weight configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


_DEFAULT_PSEUDO_ERAS = [
    "present",
    "past",
    "future",
    "someday",
    "later",
    "before",
    "after",
    "unknown era",
]


@dataclass(slots=True)
class EventNode:
    """
    Represents one event in a temporal belief graph.

    Parameters
    ----------
    id:
        Stable unique event identifier.
    label:
        Human-readable event name.
    era:
        Optional concrete era, arc, chapter, period, or timeline segment.
    description:
        Free-form event description.
    sources:
        Source keys supporting the existence or definition of this event.
    metadata:
        Optional user-defined metadata.
    """

    id: str
    label: str
    era: Optional[str] = None
    description: str = ""
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("EventNode.id cannot be empty.")
        if not self.label or not self.label.strip():
            raise ValueError("EventNode.label cannot be empty.")
        self.id = self.id.strip()
        self.label = self.label.strip()
        if self.era is not None:
            self.era = self.era.strip()
            if not self.era:
                self.era = None


@dataclass(slots=True)
class EvidenceRecord:
    """
    Audit record for a probability update applied to a BeliefEdge.

    This object is intentionally stored on the edge so every posterior value can
    be explained after multiple updates.
    """

    key: str
    source: str
    supports_forward: bool
    strength: float
    source_weight: float
    prior_before: float
    posterior_after: float
    update_method: str = "log_odds"
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("EvidenceRecord.key cannot be empty.")
        if not (0.0 <= self.prior_before <= 1.0):
            raise ValueError("EvidenceRecord.prior_before must be in [0, 1].")
        if not (0.0 <= self.posterior_after <= 1.0):
            raise ValueError("EvidenceRecord.posterior_after must be in [0, 1].")
        if not (0.0 < self.strength <= 1.0):
            raise ValueError("EvidenceRecord.strength must be in (0, 1].")
        if self.source_weight <= 0.0:
            raise ValueError("EvidenceRecord.source_weight must be greater than 0.")


@dataclass(slots=True)
class BeliefEdge:
    """
    Probabilistic temporal relation between two events.

    p_forward is interpreted as P(source occurs before target).
    p_backward is derived as 1 - p_forward.
    """

    source_id: str
    target_id: str
    p_forward: float = 0.5
    evidence_keys: list[str] = field(default_factory=list)
    evidence_history: list[EvidenceRecord] = field(default_factory=list)
    weight: float = 1.0
    relation_type: str = "temporal"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id or not self.source_id.strip():
            raise ValueError("BeliefEdge.source_id cannot be empty.")
        if not self.target_id or not self.target_id.strip():
            raise ValueError("BeliefEdge.target_id cannot be empty.")
        self.source_id = self.source_id.strip()
        self.target_id = self.target_id.strip()
        if self.source_id == self.target_id:
            raise ValueError("BeliefEdge cannot connect a node to itself.")
        if not (0.0 <= self.p_forward <= 1.0):
            raise ValueError("BeliefEdge.p_forward must be in [0, 1].")
        if self.weight <= 0.0:
            raise ValueError("BeliefEdge.weight must be greater than 0.")
        if not self.relation_type or not self.relation_type.strip():
            raise ValueError("BeliefEdge.relation_type cannot be empty.")
        self.relation_type = self.relation_type.strip()

    @property
    def p_backward(self) -> float:
        """Return P(target occurs before source)."""
        return 1.0 - self.p_forward

    @property
    def is_uncertain(self) -> bool:
        """Return True when the edge remains close to a uniform prior."""
        return 0.4 <= self.p_forward <= 0.6

    @property
    def update_count(self) -> int:
        """Return the number of recorded probability updates."""
        return len(self.evidence_history)

    def add_evidence_record(self, record: EvidenceRecord) -> None:
        """Attach an evidence record and keep the legacy evidence key list in sync."""
        self.evidence_history.append(record)
        if record.key not in self.evidence_keys:
            self.evidence_keys.append(record.key)


@dataclass(slots=True)
class PriorConfig:
    """
    Initial configuration for a temporal belief graph.

    source_weights should encode the initial trust assigned to sources. A value
    below 1.0 reduces update magnitude. It must not reverse the evidence
    direction.
    """

    default_p: float = 0.5
    source_weights: dict[str, float] = field(default_factory=dict)
    pseudo_era_list: list[str] = field(default_factory=list)
    weak_evidence_threshold: int = 1
    overconfidence_threshold: float = 0.95

    def __post_init__(self) -> None:
        if not (0.0 < self.default_p < 1.0):
            raise ValueError("PriorConfig.default_p must be in the open interval (0, 1).")
        for source, weight in self.source_weights.items():
            if not source or not source.strip():
                raise ValueError("PriorConfig.source_weights cannot contain an empty source key.")
            if weight <= 0.0:
                raise ValueError(f"Source weight for {source!r} must be greater than 0.")
        if self.weak_evidence_threshold < 0:
            raise ValueError("PriorConfig.weak_evidence_threshold cannot be negative.")
        if not (0.5 < self.overconfidence_threshold < 1.0):
            raise ValueError("PriorConfig.overconfidence_threshold must be in (0.5, 1.0).")

    @property
    def all_pseudo_eras(self) -> list[str]:
        """Return default and user-defined pseudo-era labels."""
        return sorted(set(_DEFAULT_PSEUDO_ERAS + self.pseudo_era_list))
