# Morphology Metrics Planning Notes

## Purpose
This document is a companion to `notes/contrastive-learning-plan.md`.

Its purpose is to track the morphology-metrics part of the project separately from the core contrastive-learning refactor, while still keeping the two efforts conceptually connected.

The main question is how graph morphology information can help produce more discriminative embeddings for AML and other downstream financial tasks.

---

## Motivation

Potential uses of morphology metrics include:
- predicting morphology metrics directly as auxiliary targets
- adding morphology metrics as node / edge / transaction features
- using morphology-derived signals to regularize the embedding space
- incorporating morphology into multi-task or self-supervised objectives

This could help the encoder capture structural roles that are not obvious from local message passing alone.

---

## Candidate Metrics

Initial examples to consider:
- betweenness centrality
- triangle counts
- clustering coefficient
- degree and degree-derived measures
- k-core / shell index
- local motif counts

This list will likely evolve as we learn which metrics are feasible and meaningful at project scale.

---

## Core Design Questions

### 1. What object should receive the target?
Possible choices:
- node-level targets
- edge / transaction-level targets
- local subgraph-level targets

This is an important decision because the current downstream AML task is edge / transaction centered, while many morphology metrics are naturally node-level.

### 2. Which graph representation should define the metric?
Possible choices:
- homogeneous graph
- heterogeneous graph with reverse message passing
- both, for comparison

If reverse edges are synthetic, we need to decide whether they should affect the metric definition or only the model architecture.

### 3. When and how should the metrics be computed?
Possible choices:
- exact offline precomputation
- approximate offline precomputation
- split-specific precomputation per temporal partition
- sampled or local approximations

Scale and leakage are likely to be major constraints here.

### 4. How should morphology enter the learning objective?
Possible choices:
- auxiliary regression head
- auxiliary classification head
- extra feature channels
- multi-task loss
- regularization term
- morphology-aware positive/negative design for contrastive learning

---

## Integration Options

### Option A: Auxiliary Prediction Task
Train the encoder to predict one or more morphology metrics alongside the main objective.

Pros:
- conceptually clean
- easy to interpret
- fits naturally into a multi-task setup

Cons:
- requires deciding target scale and normalization
- may add substantial preprocessing cost

### Option B: Feature Augmentation
Precompute morphology metrics and feed them in as additional features.

Pros:
- simple to reason about
- may be easy to integrate into existing pipelines

Cons:
- less clearly “self-supervised”
- may introduce leakage or stale features if computed carelessly

### Option C: Morphology-Aware Regularization
Encourage embeddings to preserve morphology similarity or ordering.

Pros:
- flexible
- may align well with representation learning goals

Cons:
- more design ambiguity
- harder to debug than direct targets

### Option D: Morphology-Aware Contrastive Extension
Use morphology-derived similarity as part of positive-pair or auxiliary contrastive structure.

Pros:
- conceptually aligned with the current contrastive-learning work

Cons:
- easy to make too complex too early
- risks reintroducing expensive pair construction if not designed carefully

---

## Practical Risks

- temporal leakage from precomputing graph-wide metrics across train/val/test periods
- high compute or memory cost for exact centrality-style metrics at large scale
- mismatch between node-level metrics and edge-level AML prediction
- difficulty interpreting improvements if too many objectives are added at once
- risk of overcomplicating the contrastive refactor before the current evaluation path is stable

---

## Recommended Near-Term Position

For now:
1. keep morphology metrics visible in project planning
2. do not let them block the current contrastive/homo/hetero/evaluation refactor
3. revisit implementation after:
   - objective / graph-form / evaluation decoupling
   - restored AML evaluation
   - a clearer hetero contrastive path

That sequencing should make it easier to tell whether morphology adds value on top of a stable baseline.

---

## First Questions To Resolve Later

1. Which morphology metrics are most meaningful for AML-style transaction graphs?
2. Which of those can be computed feasibly at the graph sizes we care about?
3. Should the first morphology experiment target nodes, edges, or transaction neighborhoods?
4. Is the first integration better framed as:
   - auxiliary prediction
   - additional features
   - or contrastive regularization?
5. Should the first morphology-aware experiment happen on the homogeneous pipeline or wait for hetero contrastive support?

---

## Relationship To Contrastive Plan

The contrastive-learning plan should continue to own:
- objective/control-flow refactoring
- homo vs hetero contrastive support
- AML evaluation and fine-tuning

This document should own:
- morphology metric selection
- computation strategy
- leakage/scaling considerations
- integration choices for morphology-aware learning
