"""
Temporal Belief Graph
---------------------
A lightweight Python package for modeling uncertain event orderings in
narrative timelines, lore databases, historical reconstructions, and
worldbuilding systems.

Instead of assuming every event relation is fixed, temporal-belief-graph
represents temporal relations as probabilistic belief edges and updates
them as new evidence is added using log-odds Bayesian updates.

Contributors
------------
lajjadred  https://github.com/lajjadred   project lead
이채문      https://github.com/CHML-real   mathematical algorithm development
CUBE       https://github.com/90cube      idea proposal and data collection
"""

from .schema import BeliefEdge, EvidenceRecord, EventNode, PriorConfig
from .graph import BeliefGraph
from .validator import ValidationResult, Validator
from .bayesian import BayesianUpdater, Evidence
from .io import graph_to_dict, graph_from_dict, graph_to_json, graph_from_json

__version__ = "0.1.1"
__all__ = [
    "EventNode",
    "BeliefEdge",
    "EvidenceRecord",
    "PriorConfig",
    "BeliefGraph",
    "ValidationResult",
    "Validator",
    "Evidence",
    "BayesianUpdater",
    "graph_to_dict",
    "graph_from_dict",
    "graph_to_json",
    "graph_from_json",
    "graph_to_markdown",
    "graph_to_mermaid",
    "save_markdown",
    "save_mermaid",
]

from .report import graph_to_markdown, graph_to_mermaid, save_markdown, save_mermaid
