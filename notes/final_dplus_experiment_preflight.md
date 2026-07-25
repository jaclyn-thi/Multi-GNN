# Final D+ experiment preflight (read-only)

**No code changes. No Slurm submissions.** This note locks provenance for the proposed edge-centric D+ system and resolves seed-1/3 + fine-tune configs before any new jobs.

Companion: [`results/diagnostics/final_dplus_experiment_preflight.json`](../results/diagnostics/final_dplus_experiment_preflight.json)

---

## 1. Family distinctions (do not conflate)

| Family | Graph | Objective | Role vs final D+ |
|--------|-------|-----------|------------------|
| **Paper-faithful supervised Multi-GIN+EU** | Account nodes / transaction edges | Supervised, `--supervised_head legacy`, **ports ON, TDS OFF**, 50ep | Formal supervised baseline only (seeds 1–3 aggregate F1≈0.660). **Not** the proposed SSL system. |
| **D+ edge-centric contrastive** | Same Multi-GIN edge graph | Contrastive InfoNCE, asym proj, 8192 neg, queue 0, accum 4, **corrected reverse + preserve_seed**, ports+TDS+emlps+ego+reverse_mp | **Winning encoder** for the proposed system (job **18514684**, seed 2, 40ep). |
| **Txn-node GCPAL diagnostics** | Transaction-as-node | Neighbor/SupCon scouts, expanding-window H | Diagnostic only; **must not** be used for final D+ experiments. |
| **Poscomplete neighbor scout** | Edge graph, but poscomplete batching | Identity vs neighbor SupCon, 10ep | Separate protocol (jobs 18719614/15/18787415). **Not** a substitute for D+ multiseed. |

Final experiments must stay **edge-centric** with Multi-GIN adaptations and **must not** use the transaction-node graph.

---

## 2. Winning proposed system provenance

### Downstream selection (job **18678029**, no GNN train)

| Field | Value |
|-------|-------|
| Selected tag | `edge_pre3h \| H+X+TF \| mlp \| none` |
| Temporal val AUPRC | **0.550** |
| Temporal val AUROC | 0.986 |
| Temporal val F1 @ val-thr | 0.575 |
| Temporal test AUPRC @0.5 | **0.674** |
| Temporal test AUROC | 0.988 |
| Temporal test F1 @0.5 | **0.656** |
| Feature dim | **227** = 198 (H) + 24 (X) + 5 (TF) |
| Downstream seed | **2** (fullstack eval) |
| Artifacts | `notes/gcpal_challenge_fullstack_eval.md`, `results/diagnostics/gcpal_challenge_fullstack_eval.json` |

### Source encoder (job **18514684**)

| Field | Locked value |
|-------|----------------|
| Unique name | `gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2` |
| Slurm | `slurm/comparison_gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2.sh` |
| Checkpoint | `saved-models/checkpoint_gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2.tar` |
| Checkpoint sha256 | `a320920141f585c5825cbd63ce760a845fb434a9b162d4c87270dc72b0442b87` |
| Checkpoint epoch | **40** (best-by-`loss/train` equals final; no separate `_last`) |
| Training seed | **2** |
| git_head (train log) | `ed7a15c18ab75f3d8d2e4600113c32a7f25046c2` |
| Pre-3h extract job | **18558352** (later forensic extract; same checkpoint) |
| Pre-3h dir | `embeddings/.../pre_embedding_3h/` (`meta.json` checkpoint_epoch=40) |

### Architecture and Multi-GIN flags

| Setting | Value |
|---------|-------|
| Model | `gin` (`GINe`), `n_gnn_layers=2`, `n_hidden≈66` (rounded 66) |
| Edge MLP updates | `--emlps` → `edge_updates=True` |
| Reverse MP | `--reverse_mp` → hetero `node__to__node` + `node__rev_to__node` |
| Ego IDs | `--ego` |
| Ports | **ON** |
| TDS | **ON** |
| Corrected reverse semantics | `--correct_reverse_edge_features` → `reverse_edge_feature_semantics=corrected` |
| Preserve seed edges | `--preserve_seed_edges` **ON** |
| Edge feature dim | **8** = base 4 + in/out port + in/out td |
| Schema names | `base_0..3`, `in_port`, `out_port`, `in_td`, `out_td` |
| Swap pairs (corrected) | `(in_port,out_port)`, `(in_td,out_td)` |
| TF as encoder input | **OFF** (`include_temporal_flow_edge_features=False`) |
| Embedding dim (post head) | **128** |
| Pre-3h dim | **198** = `3 * n_hidden` = cat(src, dst, edge_attr) **into** `embedding_head` |
| Projection (contrastive only) | `--contrast_projection_head` 128→128→128; **not** used at extract/probe |

### Contrastive / optimization

| Setting | Value |
|---------|-------|
| Objective | `contrastive` |
| Asymmetric | `--contrastive_asymmetric` |
| Negatives | 8192 |
| Memory queue | 0 |
| Accum steps | 4 |
| Batch size | 8192 |
| Fanout | `100 100` |
| Temperature | **0.5** |
| Optimizer | Adam |
| LR | **0.006213…** (`model_settings.json` → `gin.params.lr`) |
| Dropout / final_dropout | ≈0.00983 / ≈0.105 |
| Checkpoint policy | `best` on SSL `loss/train` (no morph val) |
| Train loader workers | 16 (script) |

### Representations and stacks

| Piece | Definition |
|-------|------------|
| **H (pre-3h)** | Hook on `embedding_head` input; seed-edge-only extract via `LinkNeighborLoader` + coverage checks |
| **X** | `edge_native` one-hot from `build_full_feature_matrix(..., ("edge_native",), categorical_encoding="one_hot")` (24-d in winning stack) |
| **TF** | Causal cache `results/cache/temporal_flow_causal/Small-HI/` v1; 5 features (interarrival, 7d count, amount-vs-mean, pair-repeat); past-only; no labels |
| **MLP** | `PaperStyleMLP`: Linear→ReLU→Dropout(0.1)→Linear(1); Adam lr=1e-3; 15 epochs; bs=8192; StandardScaler on train; weight=`none`; BCE-with-logits |

### Temporal split provenance (checksums)

From pre-3h `meta.json` split_checksums (extracted rows) and TF cache split files:

| Split | Extracted rows (pre-3h) | edge_id_sum | TF split file sha256 (prefix) |
|-------|------------------------:|------------:|------------------------------|
| train | 3,248,255 | 5276763386951 | `e98296e866395e09…` (`split_train_edge_id.npy`) |
| val | 965,462 | 3602792697738 | `7ffbcff035a92432…` |
| test | 863,054 | 4010180066870 | `fd901f951853f052…` |

CSV sha256 (TF meta): `14177d91096af6007de3bc3645fc8252441bdb323fe36a0378325e2e31f78c20`.

### Why this encoder (A/B/C/D)

Among contrastive arms on seed 2, **D (corrected + preserve)** was strongest on embedding AUROC/AUPRC/F1 among semantically valid corrected runs (morning analysis / ablation notes). Fullstack then selected **pre-3h H+X+TF + MLP** by **temporal val AUPRC** (never test). Inherited malformed reverse is **not** the proposed system despite competitive emb+raw logistic F1.

---

## 3. Epoch trajectory audit (seed-2 D+)

### What exists

| Artifact | Epochs | Metric type | Notes |
|----------|--------|-------------|-------|
| Job 18514684 train log | 1–40 | SSL `loss/train` “best” score | Monotone decrease 7.254→**7.044**; **best=epoch 40=final** |
| Intermediate D+ checkpoints | — | — | **Not saved** (only final best tar) |
| Intermediate pre-3h H+X+TF MLP | — | — | **Missing** for this run |
| 10ep `allneg` diagnostic | 10 | Logistic emb / emb+raw | **Different** objective (`allneg`), not a clean D+ epoch cut |
| Neighbor poscomplete scout | 1/3/5/10 | MLP H+X+TF val | **Different batching**; cannot justify D+ horizon |

### SSL train-loss trajectory (selection score; **not** representation quality)

Best score updates continue through epoch 40 with **diminishing** steps (e.g. ep38→40: 7.0441→7.0437). No SSL plateau before 40; also no SSL validation AUPRC.

### Available frozen-probe comparison (imperfect)

Logistic ablation **val F1** (max-F1-on-val), post-128 stacks:

| Run | emb val F1 | emb+raw val F1 |
|-----|----------:|---------------:|
| D+ 10ep allneg (diagnostic) | 0.281 | **0.305** |
| D+ 40ep (winning encoder) | **0.291** | 0.303 |

Embedding-only val F1 improves 10→40; emb+raw is flat/slightly down. **No** matched pre-3h H+X+TF val AUPRC curve vs epoch for D+.

### Classification: continuation **>40 epochs**

**UNRESOLVED** → **retain 40 epochs** for exact multiseed replication.

Rationale: SSL loss still inches down at 40, but there is **no** validation evidence on the winning frozen stack (pre-3h H+X+TF MLP val AUPRC) that later epochs improve representation quality. Do not invent a longer horizon.

---

## 4. Hetero module map (fine-tune boundary)

After `to_hetero`, D+ checkpoint tensors include (68 model tensors):

- `node_emb.node.*`
- `edge_emb.node__to__node.*` and `edge_emb.node__rev_to__node.*`
- `convs.{0,1}.node__to__node.*` and `convs.{0,1}.node__rev_to__node.*`
- `emlps.{0,1}.*` (edge updates; present when `--emlps`)
- `batch_norms.{0,1}.node.module.*` (+ running stats)
- `embedding_head.node__to__node.*` and `embedding_head.node__rev_to__node.*`
- Contrastive projection is **separate** (`contrast_projection_state_dict`) and should stay **frozen / unused** in supervised fine-tune

**Pre-3h H** is produced **before** `embedding_head`. Unfreezing only `embedding_head` does **not** adapt H for the winning stack.

**BatchNorm / dropout:** keep BN in train mode for unfrozen blocks (update running stats carefully; prefer eval-BN for frozen earlier layers). Classifier uses Dropout(0.1). GNN dropout from config is low (~0.01).

**Reverse-relation parameters:** any stage that unfreezes `convs.1` / `emlps.1` / `edge_emb` / `embedding_head` **must** include `node__rev_to__node` twins in the optimizer or reverse MP is silently frozen.

---

## 5. Recommended staged fine-tune protocol (one protocol, not a sweep)

**Init:** load D+ seed-2 checkpoint (epoch 40). Keep graph flags identical (corrected reverse + preserve_seed + ports/tds/emlps/ego/reverse_mp). Stack remains **H_pre3h ∥ X ∥ TF**. Downstream seed **2**. Max **20** supervised epochs with **temporal val AUPRC early stopping** (patience suggested 5; not a sweep).

| Stage | Epochs (budget) | Trainable | Frozen | LR |
|-------|-----------------|-----------|--------|-----|
| **1 — classifier warmup** | 3–5 (short) | `PaperStyleMLP` only | Entire GNN + embedding_head + projection | clf **1e-3** |
| **2 — partial encoder** | remaining ≤20 total | Stage-1 MLP **plus** final block: `convs.1` (**to + rev_to**), `emlps.1` (**to + rev_to**), `batch_norms.1` | `convs.0`, `emlps.0`, `batch_norms.0`, `node_emb`, `edge_emb`, `embedding_head`, projection | clf **1e-3**; encoder **1e-4** (≪ clf; ~60× below pretrain 6e-3) |

**Why this boundary:** smallest set that actually moves **pre-3h H** while keeping early message-passing and input embeddings stable. Reverse relation params in `convs.1` / `emlps.1` **enter the optimizer** in stage 2.

**Out of scope for this protocol:** full-encoder FT, projection-head FT, txn-node graph, changing reverse/preserve semantics, morphology stacks, random-40/60 selection.

**Implementation gap (blocker for execution, not for design):** `--finetune` today loads full checkpoint into a full training objective; a **partial-unfreeze + H+X+TF head** path is not yet a dedicated script. Design is locked; code is not written in this preflight.

---

## 6. Exact seed-1 / seed-3 replication template

Copy the seed-2 Advanced-GPU envelope and **only** change `SEED` / `RUN_NAME` / job name. Do **not** change graph/contrastive flags.

```bash
# Template (do not submit from this preflight)
RUN_NAME="gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed${SEED}"
# SEED in {1,3}
# Same as seed-2 script body:
COMMON="--data Small-HI --model gin --tqdm --batch_size 8192 --num_neighs 100 100 --loader_num_workers 16 --save_model --seed ${SEED}"
GRAPH="--reverse_mp --ego --ports --emlps --tds --correct_reverse_edge_features"
PROJECTION="--contrast_projection_head --contrast_projection_hidden 128 --contrast_projection_dim 128"
CONTRASTIVE_EXTRA="--contrastive_asymmetric --contrastive_num_neg_samples 8192 --contrastive_memory_bank_size 0 --contrastive_accum_steps 4 --preserve_seed_edges"

python main.py $COMMON $GRAPH \
  --unique_name "$RUN_NAME" \
  --n_epochs 40 \
  --objective contrastive \
  $PROJECTION --checkpoint_policy best $CONTRASTIVE_EXTRA \
  --testing

# Then extract pre-3h (workers=0 recommended for hang-safety) and evaluate
# frozen H / H+X+TF MLP with the same fullstack recipe; select on val AUPRC only.
```

Refuse overwrite of existing `checkpoint_${RUN_NAME}.tar` / embeddings / probe JSON.

---

## 7. Unresolved blockers

1. **No intermediate D+ epoch checkpoints** → cannot plot true H+X+TF val AUPRC vs epoch for seed 2.
2. **No dedicated partial-unfreeze fine-tune script** yet (`--finetune` ≠ full weights into standard train loop).
3. **Seed-1/3 D+ runs not yet executed**; only the command template is locked.
4. **10ep allneg diagnostic ≠ D+ epoch cut** — do not treat it as horizon evidence.
5. Neighbor poscomplete scout remains a **separate** protocol (do not mix into multiseed D+).

---

## 8. Preflight confirmations

1. **Seed-1/3 template:** §6 (40ep, identical D+ flags, unique names with seed suffix).
2. **Fine-tune protocol:** §5 (MLP warmup → unfreeze final GNN block incl. reverse; dual LR; ≤20ep; val AUPRC early stop).
3. **>40 pretrain:** **UNRESOLVED** → keep **40**.
4. **Blockers:** §7.
5. **No code changed; no jobs submitted** in this preflight.
