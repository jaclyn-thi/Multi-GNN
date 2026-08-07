# Financial multidataset SSL augmentation & identity-shortcut audit

> Twin: `results/diagnostics/financial_multidataset_ssl_augmentation_audit.json`  
> Scope: **read-only** static code tracing (+ CPU checkpoint metadata).  
> No training, no Slurm, no full graph/CSV/embedding loads, no test access, no source/checkpoint modification.

## Verdict (short)

**Attribute masking is active** in Phase-3 / Phase-4B / LONG via argparse defaults (`edge_attr_mask_rate=0.1`), applied **independently per view** to **all six** shared-core edge features after edge-drop. Seed edges are **not** force-kept (`preserve_seed_edges=false`). InfoNCE positives are **same EdgeID across views only**; negatives are in-batch subsampled (8192) with **no** FNF / soft-positives / queue. Overall same-transaction identity-shortcut risk: **HIGH** (endpoints + residual unmasked attributes dominate; masking alone is weak).

---

## Trace path (Phase-4B MIXED / LONG — identical `mixed_step`)

```
scripts/train_mixed_ssl_phase4b_scout.py::run_arm
  → arm_schedule / LinkNeighborLoader (per-domain RNG)
  → mixed_step
      attach_edge_id_from_batch  (EdgeID on stores; stripped from edge_attr)
      generate_views(...)        # graph_augmentations.py
      forward_seed_r198_hetero(view1)          # grads ON
      torch.no_grad(): forward_seed_r198_hetero(view2)  # stop-grad
      align_seed_r198_pair by EdgeID
      edge_identity_infonce_loss(z1, z2.detach(), … asymmetric)
      TF-MoE MAE on z1 only (targets, not encoder inputs)
```

Phase-3 MIXED_1TO1 uses the same `generate_views` + `_contrastive_view_kwargs` + seed R198 + asymmetric InfoNCE pattern in `scripts/train_mixed_ssl_phase3_scout.py`.

LONG is the same trainer entrypoint with arm `MIXED_3DOMAIN_LONG` (longer schedule only).

---

## Recipe comparison table

| Field | Phase-4B MIXED_3DOMAIN | Phase-4B MIXED_3DOMAIN_LONG | Phase-3 MIXED_1TO1 | Historical corrected projected (reference) |
|---|---|---|---|---|
| recipe/run | `financial_md_phase4b_mixed_3domain_seed2` | `…_mixed_3domain_long_3000_seed2` | `smallhi_samld_mixed_ssl_phase3` MIXED_1TO1 | Final corrected/no-preserve multiseed (AMLWorld) |
| source entrypoint | `scripts/train_mixed_ssl_phase4b_scout.py` | same (`--arm MIXED_3DOMAIN_LONG`) | `scripts/train_mixed_ssl_phase3_scout.py` | Main contrastive `training.py` hetero path + projection (unique names `*_asym_proj_*`) |
| feature contract | `financial_multidataset_shared_core_v1` | same | `smallhi_samld_shared_core_v1` (geometry-equivalent base2+ports2+TDS2) | Historical AMLWorld ports+TDS (`edge_dim=8` typical) |
| edge_dim / ordered names | 6: Timestamp, Amount Received, in_port, out_port, in_td, out_td | same | same six names | Amount/Currency/PaymentFormat + ports + TDS (not this contract) |
| projection enabled | **false** (hard-refused if true) | **false** | **false** | **true** (asym proj → 128; InfoNCE on H) |
| tensor receiving InfoNCE | seed R198 (198=3×66) | same | same | projected H128 (not R198) |
| view-1 gradient | ON | ON | ON | ON (asymmetric) |
| view-2 gradient | **stop-grad** (`no_grad` + `.detach()`) | same | same | typically stop-grad when `--contrastive_asymmetric` |
| edge-drop augmentation | **yes**, policy `random` | same | same | yes (default random 0.1; ablations vary) |
| edge-drop probability | **0.1** (`--edge_drop_target_rate` default; not overridden in `make_ns`) | same | same | **0.1** default in documented recipes |
| feature/attribute masking | **yes** | **yes** | **yes** | **yes** (same CLI default; not Phase-4-specific omission) |
| masking probability | **0.1** per eligible cell (`edge_attr_mask_rate`) | same | same | **0.1** default |
| exact masked feature slots | **all 6** (`mask_cols=None`, `exclude_last_column=False` for GIN) | same | same | all eligible columns of that schema (GIN) |
| masks independent across views | **yes** (separate `mask_edge_attr` calls) | same | same | same `generate_views` |
| seed-edge features eligible | **yes** (no seed exception in `mask_edge_attr`) | same | same | same |
| context-edge features eligible | **yes** (entire `edge_attr` tensor) | same | same | same |
| seed edge preserved/dropped | **conditionally present** — may be dropped (`preserve_seed_edges=false`); only EdgeIDs present in **both** views enter InfoNCE | same | same | preserve **OFF** in final corrected family |
| shared-anchor construction | intersect seed EdgeIDs surviving both views (`align_seed_r198_pair`) | same | same | EdgeID alignment in `edge_identity_infonce_loss` |
| positive definition | **same transaction EdgeID** across views only | same | same | same identity positive (plus optional disabled tiers) |
| additional positives | **none** (no morph / KNN soft / multi_positive / endpoint weak) | none | none | historically optional FNF/multi-pos/KNN; **not** in final primary corrected recipe defaults |
| negative definition | other batch seeds (asymmetric InfoNCE denom) | same | same | same family |
| # / source of negatives | **8192** GPU-subsampled in-batch | 8192 | 8192 | 8192 typical |
| memory bank/queue | **0 / None** | 0 | 0 | 0 in corrected/no-preserve |
| endpoint/same-pair FNF | **none** (`false_neg_filter_mode` default) | none | none | optional historical; final corrected primary = none |
| reverse-edge treatment | dropped **in sync** with forward by EdgeID; attr mask applied independently on rev store | same | same | same `generate_views` hetero path |
| TF targets enter encoder inputs? | **No** — MoE MAE on R198 only | No | No | N/A / separate experts when used |
| direct txn identity signal unchanged? | **Partially** — Bernoulli mask leaves ≈53% of seeds with all 6 attrs intact **per view**; endpoints always present | same | same | similar mask + endpoints; plus projection bottleneck |
| evidence | `mixed_step` L385–410; `util.py` L501–524; `graph_augmentations.generate_views`; `direct_r198/seed_readout.py`; ckpt `edge_emb (66,6)` | same trainer | `train_mixed_ssl_phase3_scout.py` `make_ns` | `notes/final_corrected_no_preserve_multiseed.md`; `training.py` `_contrastive_view_kwargs` |

---

## Phase-4 / LONG — explicit Q&A

### 1. Is feature masking actually active at runtime?
**Yes.** `make_ns` does not override `--edge_attr_mask_rate`; `create_parser` default is `0.1`. `_contrastive_view_kwargs` passes that into `generate_views`, which runs `mask_edge_attr` when `edge_attr_mask_rate > 0`. Resolved-run JSON omits the key but does not disable it.

### 2. Which of the six features are masked?
**All six** when present: Timestamp, Amount Received, in_port, out_port, in_td, out_td.  
(`mask_cols=None` → all columns eligible; GIN does not set `exclude_last_column`.)

Semantic group masking (`--semantic_group_mask`) is **off** (default). Even if on, it only targets currency/payment-format columns — **absent** from this 6-feature contract.

### 3. Is masking independent between views?
**Yes.** Separate Bernoulli draws for view1 and view2 (and separately on forward vs reverse stores within a view).

### 4. Expected fraction retaining all six attributes in both views
Per-feature Bernoulli \(p=0.1\), \(d=6\), independent across features and views:

| Event | Formula | Value |
|---|---|---:|
| All 6 survive in **one** view | \((1-p)^d = 0.9^6\) | **0.5314** |
| All 6 survive in **both** views | \((0.9^6)^2\) | **0.2824** |
| ≥1 attribute differs across views | \(1-[(1-p)^2+p^2]^d = 1-0.82^6\) | **0.6960** |

(Masking is **per-cell / per-feature**, not vector-level dropout of the whole edge.)

Separately, seed **edge-drop** with \(q=0.1\) and no preserve: \(P(\text{seed in both views})=(1-q)^2=0.81\). InfoNCE only scores the intersection.

### 5. Seed vs context edge attributes?
**No special treatment.** Both receive the same `mask_edge_attr` on the surviving edge stores. Seeds are not exempt from masking or (by default) from dropping.

### 6. Can the same edge still be identified through…
| Channel | Assessment |
|---|---|
| Unchanged direct edge attributes | **Yes, often** — ~28% keep all six in both views; many more keep a subset |
| Source/destination endpoints | **Yes** — node IDs / ego features shared; MP sees same endpoints if seed kept |
| Port/TDS values | **Yes when unmasked** — they are first-class edge_attr dims |
| Loader/order identity | **Not as encoder input** — shuffle + EdgeID pairing; row index not used for positives |
| Reverse-edge copies | **Synchronized drop**; attrs masked independently — not an unmasked duplicate identity channel of the forward seed readout (seed R198 uses **forward** `ea_f` only) |
| Explicit EdgeID / positional | EdgeID is **alignment-only** after `attach_edge_id_from_batch` strips the synthetic id column from `edge_attr` |

### 7. Is EdgeID supplied to the encoder?
**No.** Stored as `edge_id` on the store for pairing/drop sync; stripped from `edge_attr` before encoding (`train_util.attach_edge_id_from_batch` / `_strip_synthetic_edge_attr_id_column`).

### 8. Does R198 contain direct seed-edge attributes via `cat(relu(src‖dst), edge_attr)`?
**Formally yes** (`models.GINe._legacy_readout_forward` / `direct_r198.seed_readout.r198_readout`):  
`R198 = cat(relu(src‖dst), ea_processed)` with `n_hidden=66` → dim **198**.  
`ea_processed` is the **edge embedding stream** (`Linear(6→66)` then EMLP updates), initialized from the **view’s (masked) raw** `edge_attr` — not a separate raw-6 concat after an unmasked bypass.

### 9. Masked vs original edge_attr in R198?
**Masked (view) tensor** → `edge_emb` → MP/EMLP → readout. There is **no** parallel unmasked raw copy into R198 for InfoNCE.

### 10. Does edge dropping remove seed readout information?
**Yes, it can.** If the seed EdgeID is dropped from a view, that seed is absent from that view’s forward store → no R198 row → dropped from `align_seed_r198_pair`. Dropping changes **both** neighborhood context **and** whether the seed’s own readout exists. Context-only interpretation is incorrect when `preserve_seed_edges=false`.

### 11. Are the two views meaningfully different for the seed representation?
**Partially.** Expected differences: independent edge-drop neighborhoods, independent attr masks (~70% chance ≥1 attr differs), and stop-grad on view2. But endpoints + frequently unmasked attrs remain strong shared cues.

### 12. Could InfoNCE solve instance matching without strong graph context?
**Plausibly yes (HIGH risk).** Asymmetric identity InfoNCE can exploit (i) near-unique endpoint pairs, (ii) residual amount/time/port/TDS equality when unmasked, and (iii) seed presence itself. Graph context helps but is not forced to be necessary.

---

## Positive / negative semantics

**Enabled in Phase-3/4 mixed trainers**
- Positive: **identity only** — same `edge_id` in view1 and view2.
- Negatives: other aligned batch seeds; `num_neg_samples=8192`; `symmetric=False`; `memory_queue=None`.

**Not enabled** (code exists, defaults off / not passed)
- `--multi_positive_mode` (same_sender/receiver/endpoint/pair weak positives)
- `--false_neg_filter_mode` (exclusion-only FNF)
- `--enable_knn_soft_positives`
- `--enable_edge_neighbor_positives` (GCPAL-inspired multipositive; notes mark **not exact GCPAL**)
- `--morph_contrast` soft positives
- Memory bank / queue

**Historical notes (outcomes, not re-run here)** — from `notes/cli-reference.md`, `notes/contrastive-learning-plan.md`, `notes/results.md`:
- `same_pair` **FNF** was a leading F1/recall candidate on Small-HI; stacking with degree-aware edge-drop hurt.
- Endpoint **multi-positive** InfoNCE underperformed exclusion-only FNF.
- Feature-KNN negative exclusion / soft positives: archive reports little help / hurt.

**Exact GCPAL KNN-view:** **Not run as an exact matched GCPAL implementation.** In-repo GCPAL work is explicitly labeled approximate / challenge / txn-node scout (`notes/gcpal_challenge_fullstack_eval.md`, `notes/gcpal_txn_node_poscomplete_scout_B_gcpal_20ep_seed2.md`: “Not an exact GCPAL reproduction”). Do **not** claim exact GCPAL failure.

---

## Identity-shortcut risk

| Axis | Risk | Evidence |
|---|---|---|
| Direct edge attributes | **MODERATE→HIGH** | \(p=0.1\) leaves \(0.9^6≈53\%\) fully intact per view; both-view full intact ≈28% |
| Endpoints / graph position | **HIGH** | Seed R198 always uses `x[src], x[dst]`; ego node feats; no endpoint scrambling |
| Port/TDS structural features | **MODERATE** | Eligible for mask but often unmasked; highly discriminative when equal |
| Batch/order/EdgeID leakage into encoder | **LOW** | EdgeID stripped from `edge_attr`; positives by id after readout |
| Unchanged reverse-edge copies | **LOW–MODERATE** | Sync drop; independent mask; seed readout uses forward `ea_f` only |
| **Overall same-txn instance discrimination** | **HIGH** | Weak attr noise + strong endpoint identity + identity-only InfoNCE |

---

## Mechanistic comparison vs historical projected recipe

| Lever | Historical corrected/no-preserve | Phase-4 shared-core mixed |
|---|---|---|
| Attribute masking | Default **0.1** (same mechanism) | Default **0.1** — **not omitted** |
| InfoNCE tensor | **Projection MLP → H128** | **Raw R198 (198)** |
| preserve_seed_edges | OFF | OFF |
| Edge drop | ~0.1 random | 0.1 random |
| Negatives / queue | 8192 / 0 | 8192 / 0 |
| Feature contract | AMLWorld dim-8-style | financial shared-core dim-6 |

**Plausible quality impacts (mechanistic, not causal):** removing the projection may make InfoNCE optimize R198 geometry more directly (including identity shortcuts into the downstream representation); masking strength is **not** the primary Phase-3/4 vs historical delta.

---

## Ranked minimal follow-ups (do not implement)

**A. Matched projection-on ablation (most defensible single next lever)**  
- Change **only**: enable contrast projection MLP; InfoNCE on projected H; experts stay on R198; frozen eval still on R198.  
- Keep matched: contract, mask/drop rates, preserve=false, asym, neg=8192, queue=0, TF-MoE, domains/schedule, init.

**B. Stronger / seed-aware attribute masking**  
- Only if wanting to stress-test shortcuts given weak \(p=0.1\).  
- Change **only** mask rate and/or seed-targeted mask; keep projection off and all else matched.

**C. No-update gradient-norm / cosine diagnostic**  
- Change **only**: diagnostic instrumentation at fixed checkpoints (no train).  
- Keep models frozen.

**D. VICReg / negative-free objective (later)**  
- Change **only** the SSL objective family; requires careful matching of aug + readout; defer.

---

## End-state answers

1. **Is attribute masking active in Phase 4?** Yes (`edge_attr_mask_rate=0.1` default).
2. **Exact masked features and probability:** all six shared-core dims; Bernoulli **p=0.1** per cell.
3. **Are seed attributes masked?** Yes (no seed exemption).
4. **Does R198 see masked or original attrs?** **Masked** view attrs → `edge_emb` → processed `ea` in `cat(relu(src‖dst), ea)`.
5. **Current positive definition:** same EdgeID across two views only.
6. **Current negative definition:** other in-batch seeds; 8192 subsampled; no queue; no FNF.
7. **Was an exact GCPAL KNN view run?** **No** (only approximate / labeled non-exact scouts).
8. **Overall identity-shortcut risk:** **HIGH**.
9. **Most defensible single next ablation:** **A — matched projection-on** (InfoNCE on proj; eval on R198).
10. **Files read / written:** see JSON twin.
11. **Confirmation:** no training, no full data loads, no test access, no Slurm, no source/checkpoint modification.

Stop for human review.
