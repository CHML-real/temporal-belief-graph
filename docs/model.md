# Model Documentation

## Why probabilistic edges?

Most graph libraries represent temporal relations as binary: either A comes before B, or it does not. This works well when the ordering is known and fixed.

Real narrative and lore data rarely works this way. Multiple sources may disagree. Evidence is partial, contradictory, or arrives incrementally. Some orderings are inferences rather than established facts.

`temporal-belief-graph` represents each temporal relation as a probability:

```
P(source occurs before target) = p_forward ∈ [0, 1]
```

A value of `0.5` means no information — the ordering is completely uncertain. A value of `0.9` means strong belief that the source comes first. This allows the graph to hold uncertain states explicitly rather than forcing a premature binary decision.

This structure is called a **Temporal Partial Order**: a partial order where the ordering between some pairs of events is not yet determined, and where the degree of belief in each ordering is tracked as a continuous value.

---

## Why log-odds updates instead of likelihood multiplication?

### The problem with direct likelihood multiplication

A naive Bayesian update multiplies prior and likelihood directly:

```
P(A→B | evidence) ∝ P(evidence | A→B) × P(A→B)
```

When source weights are applied by scaling the likelihood, a source weight below 1.0 causes a counterintuitive failure: a weak source that supports forward ordering can actually decrease `p_forward`.

Example:

```
prior     = 0.5
source    supports forward
strength  = 0.5
weight    = 0.1   ← weak source

lf = 0.5 × 0.1 = 0.05
lb = 0.5 × 1.0 = 0.50

posterior = 0.05 / (0.05 + 0.50) ≈ 0.09   ← moved backward
```

A source that supports forward ordering moved belief in the opposite direction. This is mathematically inconsistent for a system that treats source weight as a measure of trust, not direction.

### The log-odds solution

Log-odds updates avoid this by separating magnitude from direction:

```
logit(p) = log(p / (1 - p))

new_logit = old_logit + direction × strength × source_weight × learning_rate

p_new = sigmoid(new_logit) = 1 / (1 + exp(-new_logit))
```

Where:

- `direction = +1` if evidence supports forward, `-1` if backward
- `strength ∈ (0, 1]` controls how much this evidence moves the belief
- `source_weight > 0` scales the magnitude — a weak source (weight < 1) produces a smaller update in the same direction, never a reversal

This guarantees the direction-safety property:

> A source with weight < 1.0 that supports forward ordering always increases `p_forward`, regardless of how small the weight is.

---

## Why separate temporal and causal relations?

Temporal and causal relations are often conflated but are logically distinct:

```
Temporal : A occurred before B
Causal   : A caused B
```

An event can precede another without causing it. An event can cause another that was recorded earlier. Mixing these in a single edge type leads to ambiguous queries and incorrect propagation.

`temporal-belief-graph` uses a `relation_type` field on `BeliefEdge` to distinguish these. The default is `"temporal"`. Users building causal models can set `relation_type="causal"` and filter edges accordingly. Future versions will add explicit propagation rules per relation type.

---

## Why handle contradictions in the validator rather than the updater?

Contradictions — two opposing edges that are both highly confident — are treated as a validation concern, not an update failure.

This is intentional. The updater's job is to apply evidence faithfully, one step at a time. If two sources genuinely disagree, suppressing one at update time would lose information. The contradiction should be visible and explicitly flagged.

The validator surfaces contradictions as errors, giving the user the opportunity to investigate the conflicting sources rather than silently resolving them.

---

## Why detect cycles separately from the graph structure?

The graph is not assumed to be a DAG at construction time. Cycles may appear as evidence accumulates — for example, when two sources propose orderings that together imply a loop.

The validator detects cycles only among **confirmed edges** (those above a probability threshold). Uncertain edges are excluded, because a cycle involving uncertain relations does not necessarily imply a logical contradiction — it may simply reflect insufficient evidence.

```
Confirmed edge  : p_forward > cycle_threshold (default 0.75)
Uncertain edge  : 0.4 ≤ p_forward ≤ 0.6
```

A confirmed DAG can be extracted at any time:

```python
graph.to_adjacency_dict(threshold=0.75)
graph.confirmed_edges(threshold=0.75)
```

---

## Where does this model apply?

The design was motivated by narrative timeline reconstruction, but the structure generalizes:

**Narrative lore databases**
Game lore, fantasy worldbuilding, and extended universes often have conflicting sources — official canon, developer statements, fan wikis, in-game documents. Each source has different reliability, and the ordering of events is frequently disputed or unclear.

**Historical reconstruction**
Archaeologists and historians work with fragmentary evidence. Some events are well-dated; others are placed only relative to other events. The ordering of some pairs remains genuinely uncertain even after extensive research.

**Event log analysis**
In distributed systems, event logs from multiple nodes may disagree on ordering due to clock skew. Probabilistic temporal relations can represent the degree of confidence in each ordering based on timestamp reliability and causal dependencies.

**Knowledge graph construction**
When building knowledge graphs from multiple documents, temporal claims often conflict. Rather than picking one, a belief graph can hold all claims and update continuously as new documents are processed.

---

## Probability clipping

All posterior values are clipped to `[clip, 1 - clip]` where `clip = 0.01` by default.

This prevents exact 0 or 1 values, which would cause the log-odds update to diverge (logit of 0 or 1 is undefined). It also reflects epistemic humility: no finite amount of evidence should make us completely certain.

The clip bound is configurable:

```python
BayesianUpdater(graph, clip=0.05)
```

---

## Ensemble updates

When multiple sources each propose a `p_forward` value, the ensemble update combines them via weighted average before applying a single log-odds step:

```
ensemble_p = Σ (p_i × w_i) / Σ w_i

strength   = |ensemble_p - 0.5| × 2     # maps [0.5, 1.0] → [0.0, 1.0]
direction  = forward if ensemble_p ≥ 0.5 else backward
```

The weighted average respects source weights without amplifying any single source disproportionately. The resulting `strength` value captures how much the ensemble consensus deviates from uncertainty.

---

## Evidence history and explainability

Every probability update is recorded as an `EvidenceRecord` on the edge:

```python
@dataclass
class EvidenceRecord:
    key: str
    source: str
    supports_forward: bool
    strength: float
    source_weight: float
    prior_before: float
    posterior_after: float
    update_method: str
    note: str
    metadata: dict
```

This means every current `p_forward` value can be fully reconstructed from its history. The `BayesianUpdater.explain_edge()` method returns a human-readable summary of the current belief state, including the count of supporting and opposing evidence items and the last recorded update.
