"""
bayesian.py
-----------
Probability update utilities for Temporal Belief Graph.

The default updater uses log-odds updates instead of multiplying likelihoods by
source weights directly. This prevents weak supporting evidence from moving the
belief in the opposite direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .graph import BeliefGraph
from .schema import BeliefEdge, EvidenceRecord


@dataclass(slots=True)
class Evidence:
    """
    A single evidence item supporting one temporal direction.

    Parameters
    ----------
    key:
        Stable evidence identifier.
    supports_forward:
        True supports source before target. False supports target before source.
    strength:
        Evidence strength in (0, 1]. Higher values create larger updates.
    source:
        Source key used to look up PriorConfig.source_weights.
    note:
        Optional human-readable note.
    metadata:
        Optional user-defined metadata.
    """

    key: str
    supports_forward: bool
    strength: float = 0.5
    source: str = ""
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise ValueError("Evidence.key cannot be empty.")
        if not (0.0 < self.strength <= 1.0):
            raise ValueError("Evidence.strength must be in the open-closed interval (0, 1].")
        self.key = self.key.strip()
        self.source = self.source.strip() if self.source else ""


class BayesianUpdater:
    """
    Applies probability updates to BeliefEdge objects inside a BeliefGraph.

    Parameters
    ----------
    graph:
        Target graph. Updates mutate edge objects in place.
    clip:
        Probability clipping bound used to avoid exact 0 or 1.
    learning_rate:
        Global multiplier for log-odds update magnitude.
    """

    def __init__(self, graph: BeliefGraph, clip: float = 0.01, learning_rate: float = 1.0):
        if not (0.0 < clip < 0.5):
            raise ValueError("clip must be in the open interval (0, 0.5).")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0.")
        self.graph = graph
        self.clip = clip
        self.learning_rate = learning_rate

    def update_edge(self, source_id: str, target_id: str, evidence: Evidence) -> BeliefEdge:
        """Update one edge with one evidence item using a log-odds step."""
        edge = self.graph.get_edge(source_id, target_id)
        prior = edge.p_forward
        source_weight = self.graph.config.source_weights.get(evidence.source, 1.0)
        posterior = self._update_probability_log_odds(
            prior=prior,
            supports_forward=evidence.supports_forward,
            strength=evidence.strength,
            source_weight=source_weight,
        )
        edge.p_forward = posterior
        edge.add_evidence_record(
            EvidenceRecord(
                key=evidence.key,
                source=evidence.source,
                supports_forward=evidence.supports_forward,
                strength=evidence.strength,
                source_weight=source_weight,
                prior_before=prior,
                posterior_after=posterior,
                update_method="log_odds",
                note=evidence.note,
                metadata=dict(evidence.metadata),
            )
        )
        return edge

    def update_edge_batch(self, source_id: str, target_id: str, evidences: list[Evidence]) -> BeliefEdge:
        """Apply multiple evidence items sequentially to one edge."""
        for evidence in evidences:
            self.update_edge(source_id, target_id, evidence)
        return self.graph.get_edge(source_id, target_id)

    def ensemble_update(self, source_id: str, target_id: str, claims: list[tuple[float, float]]) -> BeliefEdge:
        """
        Combine multiple probability claims into one evidence-like update.

        claims is a list of (claimed_p_forward, claim_weight) pairs.
        """
        if not claims:
            raise ValueError("claims cannot be empty.")
        for probability, weight in claims:
            if not (0.0 <= probability <= 1.0):
                raise ValueError("Each claimed probability must be in [0, 1].")
            if weight <= 0.0:
                raise ValueError("Each claim weight must be greater than 0.")

        total_weight = sum(weight for _, weight in claims)
        ensemble_p = sum(probability * weight for probability, weight in claims) / total_weight
        supports_forward = ensemble_p >= 0.5
        strength = max(0.01, min(1.0, abs(ensemble_p - 0.5) * 2.0))

        evidence = Evidence(
            key=f"ensemble:{source_id}->{target_id}:{len(claims)}",
            supports_forward=supports_forward,
            strength=strength,
            source="ensemble",
            metadata={"claims": claims, "ensemble_p_forward": ensemble_p},
        )
        return self.update_edge(source_id, target_id, evidence)

    def explain_edge(self, source_id: str, target_id: str) -> dict[str, Any]:
        """Return a compact explanation dictionary for an edge."""
        edge = self.graph.get_edge(source_id, target_id)
        forward_records = [record for record in edge.evidence_history if record.supports_forward]
        backward_records = [record for record in edge.evidence_history if not record.supports_forward]
        if edge.p_forward > 0.6:
            dominant_direction = "forward"
        elif edge.p_forward < 0.4:
            dominant_direction = "backward"
        else:
            dominant_direction = "uncertain"

        return {
            "edge": f"{source_id} -> {target_id}",
            "p_forward": edge.p_forward,
            "p_backward": edge.p_backward,
            "dominant_direction": dominant_direction,
            "evidence_count": len(edge.evidence_history),
            "forward_evidence_count": len(forward_records),
            "backward_evidence_count": len(backward_records),
            "evidence_keys": list(edge.evidence_keys),
            "last_update": self._record_to_dict(edge.evidence_history[-1]) if edge.evidence_history else None,
        }

    def _update_probability_log_odds(
        self,
        prior: float,
        supports_forward: bool,
        strength: float,
        source_weight: float,
    ) -> float:
        if not (0.0 <= prior <= 1.0):
            raise ValueError("prior must be in [0, 1].")
        if not (0.0 < strength <= 1.0):
            raise ValueError("strength must be in (0, 1].")
        if source_weight <= 0.0:
            raise ValueError("source_weight must be greater than 0.")

        clipped_prior = self._clip_probability(prior)
        old_logit = math.log(clipped_prior / (1.0 - clipped_prior))
        direction = 1.0 if supports_forward else -1.0
        delta = direction * strength * source_weight * self.learning_rate
        new_logit = old_logit + delta
        posterior = 1.0 / (1.0 + math.exp(-new_logit))
        return self._clip_probability(posterior)

    def _clip_probability(self, value: float) -> float:
        return max(self.clip, min(1.0 - self.clip, value))

    @staticmethod
    def _record_to_dict(record: EvidenceRecord) -> dict[str, Any]:
        return {
            "key": record.key,
            "source": record.source,
            "supports_forward": record.supports_forward,
            "strength": record.strength,
            "source_weight": record.source_weight,
            "prior_before": record.prior_before,
            "posterior_after": record.posterior_after,
            "update_method": record.update_method,
            "note": record.note,
            "metadata": dict(record.metadata),
        }
