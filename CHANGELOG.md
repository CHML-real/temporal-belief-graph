# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-06-03

### Added

- Initial release of `temporal-belief-graph`.
- `EventNode` schema for representing single events with era, description, and source tracking.
- `BeliefEdge` schema for probabilistic temporal relations between events, including full evidence history via `EvidenceRecord`.
- `PriorConfig` for initial graph configuration including source weights and pseudo-era definitions.
- `BeliefGraph` for managing nodes and edges, with support for confirmed edge extraction, contradiction detection, uncertain edge queries, and adjacency dict export.
- `BayesianUpdater` implementing log-odds Bayesian updates that preserve evidence direction regardless of source weight magnitude.
- `Evidence` dataclass for structured evidence input.
- `ensemble_update()` for combining multiple source claims via weighted average before applying a single Bayesian step.
- `explain_edge()` for human-readable explanation of the current belief state of an edge.
- `Validator` with seven checks: pseudo era, missing era, missing evidence, overconfident edges, contradictions, directed cycles, and isolated nodes.
- `ValidationResult` container with `errors`, `warnings`, `is_valid`, and `raise_if_errors()`.
- Full test suite with 80 tests across all modules.
- PyPI distribution via `hatchling`.
