# temporal-belief-graph

> A lightweight Python package for modeling **uncertain event orderings** in narrative timelines, lore databases, historical reconstructions, and worldbuilding systems.

Instead of assuming every temporal relation is fixed, `temporal-belief-graph` represents event ordering as **probabilistic belief edges** and updates them as new evidence accumulates — using log-odds Bayesian updates that never reverse direction due to weak sources.

---

## Why this exists

Most graph libraries treat edges as binary: A comes before B, or it does not.  
Real-world narrative and lore data is messier. Multiple sources disagree. Evidence is partial. Some orderings are guesses.

`temporal-belief-graph` was built to handle exactly this:

- A story timeline where fan sources and official sources conflict
- A historical reconstruction where some events have uncertain sequencing
- A worldbuilding database where causality and temporal order must be separated
- Any system where **"we think A happened before B, but we are not sure"** needs to be expressed and updated over time

---

## Installation

```bash
pip install temporal-belief-graph
```

Requires Python 3.10 or later. No external dependencies.

---

## Quick start

```python
from tbg import BeliefGraph, EventNode, BeliefEdge, PriorConfig
from tbg import BayesianUpdater, Evidence, Validator

# 1. Define a graph with initial config
config = PriorConfig(
    default_p=0.5,
    source_weights={"official_lore": 1.5, "fan_wiki": 0.7},
)
graph = BeliefGraph(prior_config=config)

# 2. Add events
graph.add_node(EventNode(id="kain_incident", label="Kain Incident", era="First Rift War"))
graph.add_node(EventNode(id="rift_opening",  label="Rift Opening",  era="First Rift War"))

# 3. Add an uncertain edge (default p=0.5 means we do not know the order yet)
graph.init_uniform("kain_incident", "rift_opening")

# 4. Update belief with evidence
updater = BayesianUpdater(graph)

updater.update_edge(
    "kain_incident", "rift_opening",
    Evidence(key="official_001", supports_forward=True, strength=0.9, source="official_lore"),
)
updater.update_edge(
    "kain_incident", "rift_opening",
    Evidence(key="fan_002", supports_forward=False, strength=0.6, source="fan_wiki"),
)

edge = graph.get_edge("kain_incident", "rift_opening")
print(edge.p_forward)   # updated belief: P(kain_incident → rift_opening)

# 5. Explain the current belief state
print(updater.explain_edge("kain_incident", "rift_opening"))

# 6. Validate the graph
result = Validator().validate(graph)
print(result.is_valid)
print(result.warnings)
```

---

## Core concepts

### EventNode

A single event in the graph. Every node carries an `era` — a concrete arc, chapter, or period name. Vague era labels like `"present"` or `"past"` are flagged by the validator.

```python
EventNode(id="my_event", label="The Great Collapse", era="Third Age")
```

### BeliefEdge

A directed edge from `source_id` to `target_id` carrying `p_forward`: the probability that the source event occurs before the target event.

| `p_forward` | Meaning |
|-------------|---------|
| `0.5` | No information — uniform prior |
| `0.9` | Strong belief that source comes first |
| `0.1` | Strong belief that target comes first |

### BayesianUpdater

Updates edge probabilities using **log-odds steps**:

```
new_logit = old_logit + direction × strength × source_weight × learning_rate
p_new     = sigmoid(new_logit)
```

Key property: a source with `source_weight < 1.0` produces a **smaller** update in the same direction — it never flips the direction of evidence.

### Ensemble update

When multiple sources each propose a `p_forward` value, `ensemble_update` combines them via weighted average before applying a single Bayesian step:

```python
updater.ensemble_update(
    "event_a", "event_b",
    claims=[
        (0.9, 1.5),   # official source claims 0.9, weight 1.5
        (0.6, 0.8),   # secondary source claims 0.6, weight 0.8
        (0.3, 0.5),   # weak source claims 0.3, weight 0.5
    ],
)
```

### Validator

Runs seven checks and returns a `ValidationResult` with `errors` and `warnings`:

| Check | Level |
|-------|-------|
| `pseudo_era` — vague era label used | error |
| `cycle` — confirmed edges form a directed cycle | error |
| `contradiction` — two opposing edges both highly confident | error |
| `missing_era` — node has no era value | warning |
| `missing_evidence` — edge probability changed without evidence | warning |
| `overconfident_edge` — high probability with very few updates | warning |
| `isolated_node` — node has no incident edges | warning |

```python
result = Validator().validate(graph)
result.raise_if_errors()   # raises ValueError if any errors exist
```

---

## Package structure

```
tbg/
├── schema.py      EventNode, BeliefEdge, EvidenceRecord, PriorConfig
├── graph.py       BeliefGraph
├── bayesian.py    BayesianUpdater, Evidence
└── validator.py   Validator, ValidationResult
```

---

## Design principles

**Probabilistic by default.** Every edge starts uncertain. Certainty is earned through accumulated evidence, not assumed.

**Direction-safe updates.** A weak source supporting forward always moves `p_forward` up — never down. Source weight controls magnitude, not direction.

**Explainable history.** Every probability update is recorded as an `EvidenceRecord` on the edge. You can always reconstruct why the current value is what it is.

**No hard deletes.** The graph is append-friendly. Contradictions are surfaced by the validator, not silently overwritten.

**No external dependencies.** Pure Python standard library only.

---

## Development

Install the package in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

Run tests across all supported Python versions (requires [tox](https://tox.wiki/)):

```bash
tox
```

Build the package:

```bash
python -m build
```

> This package is currently in alpha. APIs may change before version 1.0.0.

---

## Contributing

Contributions, issues, and feature requests are welcome.  
Please open an issue before submitting a pull request.

---

## Contributors

| Handle | GitHub | Role |
|--------|--------|------|
| lajjadred | [@lajjadred](https://github.com/lajjadred) | Project lead |
| Chae Mun Lee | [@CHML-real](https://github.com/CHML-real) | Mathematical algorithm development |
| CUBE | [@90cube](https://github.com/90cube) | Idea proposal and data collection |

---

## License
 
MIT License. See [LICENSE](LICENSE) for details.
