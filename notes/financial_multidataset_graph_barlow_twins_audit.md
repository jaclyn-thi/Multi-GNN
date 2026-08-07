# Graph Barlow Twins adaptation audit (read-only)

**Date:** 2026-08-03  
**Scope:** Official Graph Barlow Twins (paper + code) vs financial multi-dataset
Phase-4B edge / R198 protocol.  
**Safety:** Source/config inspection only. No `dvc pull`, no dataset downloads,
no env install into Multi-GNN, no Multi-GNN code changes, no Slurm, no test
split access.

Official clone (sparse source only): commit
`ec62580aa89bf3f0d20c92e7549031deedc105ab`
(`https://github.com/pbielak/graph-barlow-twins`).

Paper: [arXiv:2106.02466](https://arxiv.org/abs/2106.02466).

---

## Recommendation (one initial implementation)

**A. Faithful direct-R198 GBT** — apply the official `gssl.loss.barlow_twins_loss`
to aligned seed-edge R198 pairs, **projection off**, **both views with gradients**,
locked LONG protocol otherwise.

**Not B** for the first run: the Graph Barlow Twins paper and `gssl` training
path explicitly omit a projector (“GNN embeddings are already low-dimensional”).
**Not C** as the GBT experiment itself (C remains the InfoNCE reference arm).

---

## Official answers (1–12)

### 1. Cross-correlation normalization

Canonical implementation: [`gssl/loss.py`](https://github.com/pbielak/graph-barlow-twins/blob/master/gssl/loss.py)
`barlow_twins_loss`.

1. Per-feature batch standardization (along dim 0 = batch/nodes in batch):
   \(Z' = (Z - \mathrm{mean}_0(Z)) / (\mathrm{std}_0(Z) + \varepsilon)\).
2. Empirical cross-correlation:
   \(C = (Z_a'{}^\top Z_b') / B \in \mathbb{R}^{D\times D}\).

Vendored twin: [`GCL/losses/barlow_twins.py`](https://github.com/pbielak/graph-barlow-twins/blob/master/GCL/losses/barlow_twins.py)
`bt_loss` (same math when `batch_norm=True`). Training scripts call **`gssl.loss`**,
not GCL.

Paper Eq. (1) is written as a Pearson-style ratio; after the code’s explicit
z-scoring, \(C = Z_a'{}^\top Z_b'/B\) is the operational definition used in all
`gssl` trainers.

### 2. Invariance and redundancy terms

\[
\mathcal{L}_{\mathrm{BT}}
= \sum_{i=1}^{D}(1 - C_{ii})^2
+ \lambda \sum_{i\neq j} C_{ij}^2
\]

Code:

```python
(1 - c.diagonal()).pow(2).sum() + _lambda * c[off_diagonal_mask].pow(2).sum()
```

### 3. Lambda

\(\lambda = 1/D\) with \(D =\) feature dim (`feature_dim = z_a.size(1)`).
Depends on representation dimension. For R198: \(\lambda = 1/198\).

### 4. Epsilon and variance

- `EPS = 1e-15` in `gssl/loss.py`.
- `torch.std(dim=0)` **default `unbiased=True`** (divide by \(B-1\)).
- Note: image Barlow Twins often uses BN (biased \(N\)); Graph BT code uses
  unbiased `std`. Faithful Graph-BT adaptation should keep **unbiased + 1e-15**.

### 5. Batch-statistic standardization

**Yes.** Representations are standardized with **batch** mean/std along the
sample axis before forming \(C\). Not \(\ell_2\) row-normalize (that is InfoNCE).

### 6. Both branches retain gradients

**Yes.** `gssl/full_batch/model.py` and `gssl/batched/model.py`:

- same encoder for both views;
- `loss = barlow_twins_loss(z_a, z_b)`;
- `loss.backward()` with **no** `detach` / stop-grad / EMA twin.

Paper: gradients “symmetrically backpropagated”; no momentum encoder, stop-grad,
or predictor.

**Critical Multi-GNN delta:** current `INFONCE_ONLY` /
`ablation_mixed_step` runs view2 under `torch.no_grad()` and
`z2_seed.detach()`. A faithful GBT step **must not** do that.

### 7. Projector in official graph method

**No.** Paper §3: image BT used an MLP projector; Graph BT “eliminates that
step” because GNN embeddings are already low-d.  
`gssl` encoders return node embeddings straight into `barlow_twins_loss`.
No projection module in full-batch or batched G-BT trainers.

### 8. Full-batch vs batched

| Mode | Encoder inputs | \(Z\) shape into loss | \(C\) shape |
|------|----------------|------------------------|-------------|
| Full-batch | whole graph | \(N\times D\) (all nodes) | \(D\times D\) |
| Batched (NeighborSampler) | subgraph around `batch_size` targets | \(B\times D\) (targets only) | \(D\times D\) |

Loss math is identical; only who enters the batch axis changes.  
**Largest correlation tensor is always \(D\times D\)**, never \(N\times N\).

### 9. How OGB-Products avoids graph-size \(N\times N\)

[`gssl/batched/model.py`](https://github.com/pbielak/graph-barlow-twins/blob/master/gssl/batched/model.py)
+ [`gssl/batched/encoders.py`](https://github.com/pbielak/graph-barlow-twins/blob/master/gssl/batched/encoders.py):

- `NeighborSampler` with `batch_size` (Products config: 512);
- augment only the sampled `x[n_id]` / layer `adjs`;
- encoder slices to target `size` each layer → **\(Z\in\mathbb{R}^{B\times D}\)**;
- `barlow_twins_loss` builds **\(C\in\mathbb{R}^{D\times D}\)** only.

There is **no** all-pairs node similarity and **no** dense \(|V|\times|V|\)
adjacency in the loss. Full-graph work appears only in **inference**
(`encoder.inference` over nodes for eval) — out of scope for our train loss.

### 10. Batch-size assumptions

- Official batched configs: \(B\in\{256,512,1024,2048\}\) (Products: 512).
- Need \(B\ge 2\) for `std`; larger \(B\) stabilizes \(C\).
- \(B < D\) is allowed (rank\((C)\le B\)); still used in practice.
- Our LONG seed batch: `BATCH_SIZE = 8192` (`mixed_ssl_phase4a`) — **more than
  enough** for \(D=198\); loss memory stays \(O(D^2)\).

### 11. Collapse / dead-dimension diagnostics

Official logging (weak): mean \(\|z\|_2\) via TensorBoard (`norm` scalar).

For our adaptation, propose stronger gates (unit tests + smoke):

- mean \(|C_{ii}-1|\), mean \(C_{ij}^2\) off-diag;
- fraction of dims with pre-norm `std < 1e-4` (dead / unstable);
- effective rank of \(Z\) (already in Phase-4B step helpers);
- both-view encoder grad norms \(> 0\).

BT’s off-diagonal term is the primary anti-collapse mechanism (vs InfoNCE
negatives / stop-grad).

### 12. Augmentations vs transaction edges

| Official Graph BT | Our Phase-4B protocol |
|-------------------|------------------------|
| Edge drop on graph `edge_index` | Edge drop on transaction / hetero edges (`edge_drop_rate=0.1`) |
| **Node** feature mask, **same** mask across all nodes (`size=(1,F)`) | **Edge-attribute** Bernoulli mask per entry (`edge_attr_mask_rate=0.1`) |
| Node embeddings as SSL targets | **Seed-edge R198** as SSL targets |
| Homogeneous citation/product graphs | Multi-domain financial hetero graphs + domain BN |

**Transfers:** two independently corrupted views; topology noise via edge drop;
attribute corruption.  
**Does not transfer 1:1:** node-global feature mask ≠ per-edge attr mask;
node SSL ≠ edge SSL; no official multi-domain BN. Keep our existing
`generate_views` (locked protocol); only replace the **contrastive objective**.

---

## Memory audit (loss tensors; \(D=198\))

Let \(B\) = number of **aligned seed edges** in the step (≤ 8192).

| Tensor | Shape | ~bytes (fp32, \(B=8192\)) | Notes |
|--------|-------|---------------------------:|-------|
| `z_a`, `z_b` | \(B\times D\) | 2 × 3.09 MiB | inputs |
| `mean`, `std` | \(D\) | negligible | |
| `z_a_norm`, `z_b_norm` | \(B\times D\) | 2 × 3.09 MiB | |
| **`c` cross-corr** | **\(D\times D\)** | **~0.15 MiB** | **largest corr tensor** |
| `eye` / off-diag mask | \(D\times D\) bool | ~0.04 MiB | |
| scalars | — | — | invariance + redundancy |

**Confirmed:** largest correlation tensor is **\(D\times D = 198\times 198\)**,
independent of graph size and of \(B\).

### Reject / do not port

| Official or related op | Why reject for us |
|------------------------|-------------------|
| Full-batch \(Z\in\mathbb{R}^{N\times D}\) over entire AML graph | Graph-size memory; we already seed-batch |
| Dense adjacency / full \(|V|\times|V|\) | Not in BT loss; never allocate |
| All-pairs node/edge similarity \(B\times B\) or \(N\times N\) | InfoNCE-style; **not** part of BT |
| Full-dataset feature matrix materialization | Products uses NeighborSampler instead |
| Image-BT large projector + BN projector path | Graph BT paper/code omit projector |
| Asymmetric view2 `no_grad`/`detach` from our InfoNCE path | Breaks faithful GBT |

Safe pattern for us: **aligned seed pairs → `barlow_twins_loss(z1, z2)`** with
both views live in the graph (same as official batched axis = “batch of
entities”).

---

## Compare A / B / C

| | A direct-R198 GBT | B GBT + projector | C identity InfoNCE (current) |
|--|-------------------|-------------------|------------------------------|
| SSL target | R198 | H = MLP(R198) | R198 (INFONCE_ONLY) |
| Loss | \(\mathcal{L}_{BT}\) on \(D=198\) | \(\mathcal{L}_{BT}\) on \(D_h\) (e.g. 128) | InfoNCE + negs |
| Corr / logits | \(D\times D\) | \(D_h\times D_h\) | up to \(B\times B\) or \(B\times K\) |
| View2 grads | **required** | required | **detached** today |
| Official support | **Yes** (paper + `gssl`) | Optional follow-up only | Different family |
| First experiment | **Yes** | No | Reference only |

---

## Exact proposed loss (initial)

On aligned seed embeddings \(Z_a, Z_b \in \mathbb{R}^{B\times 198}\)
(same edge-id row order; both require grad):

```text
λ = 1/198
ε = 1e-15
Z'_v = (Z_v - mean_0(Z_v)) / (std_0(Z_v; unbiased=True) + ε)   for v in {a,b}
C = (Z'_a^T @ Z'_b) / B                                         # 198×198
L_GBT = Σ_i (1 - C_ii)^2 + λ Σ_{i≠j} C_ij^2
```

**Supporting sources:**

- `gssl/loss.py` :: `barlow_twins_loss` (primary)
- `gssl/full_batch/model.py` / `gssl/batched/model.py` :: twin forward + backward
- Paper §3 Eqs. (1)–(2); projector omission paragraph
- `GCL/losses/barlow_twins.py` :: confirmatory duplicate (not used by `gssl` train)

**Protocol lock (unchanged except objective + view2 grads):**

- domains Small-HI / SAML-D / Small-LI; contract `financial_multidataset_shared_core_v1`
- `edge_dim=6`; shared GIN → R198
- views: edge drop + edge-attr mask @ 0.1; domain BN
- projection **disabled**
- 3000 steps / 1000 per domain; Phase-4B init + seed streams + linear LR
- validation-only probes; no test

---

## Unavoidable differences from the paper

1. **Edge** representations (transaction seeds), not **node** embeddings.
2. Edge-attribute masking vs node-global feature masking.
3. Multi-domain shared encoder + domain-specific BN / TF caches (paper is
   single-graph node SSL).
4. NeighborLoader / hetero message passing vs PyG `NeighborSampler` node batches
   (same *idea*: batch axis = sampled entities; different plumbing).
5. Paper HTML Eq. (1) typesetting vs code’s explicit z-score + \(Z'{}^\top Z'/B\)
   (follow **code**).
6. Eval: frozen PaperStyleMLP on val AML labels vs paper’s logistic regression
   node classification (and paper reports test; we stay val-only).

---

## Proposed unit tests (no jobs)

1. **Shape:** `C.shape == (198, 198)` for random \(B\in\{8,64,512\}\).
2. **Lambda:** `_lambda == 1/198`.
3. **Perfect copies:** \(Z_b=Z_a\) finite non-degenerate → near-zero invariance term;
   off-diag small after decorrelated synthetic features.
4. **Both grads:** `z_a.requires_grad` and `z_b.requires_grad`; after backward,
   both `.grad` nonzero.
5. **No \(B\times B\):** monkeypatch / assert loss path never allocates
   `(B,B)` similarity (compare to InfoNCE).
6. **Std convention:** match `torch.std(..., unbiased=True)` reference numpy.
7. **Empty / B=1 guards:** raise or skip safely (std undefined / unstable).
8. **Alignment:** permuting view2 rows without realigning edge ids must **not**
   be treated as matched pairs (reuse align helper tests).

---

## Proposed 20–50-step smoke (human approval later)

| Knob | Value |
|------|------:|
| Arm | `GBT_DIRECT_R198` (new; contrast-only like INFONCE_ONLY but BT + dual grads) |
| Steps | 30 (or 20–50) |
| Domains | all three, round-robin as LONG |
| Init | Phase-3/4B shared init |
| Projection | off |
| TF-MoE | off (isolate SSL objective) |
| Batch | same 8192 / accum as LONG if memory allows dual-view grads; else document reduced smoke batch |
| Augment | edge_drop=0.1, attr_mask=0.1 |
| Checks | finite loss; `C` diag mean ↑ toward 1; enc grad>0 both views; no test loader |
| Artifacts | unique `results/diagnostics/..._gbt_smoke/` only |

Stop here for review — **no implementation or submission in this pass**.

---

## Twin JSON

See `results/diagnostics/financial_multidataset_graph_barlow_twins_audit.json`.
