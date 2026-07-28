# Sequential AMLWorld → PaySim SSL continuation — preflight audit

> Read-only audit of **CURRENT HEAD** (`7eb1975b6ea5b0505b278e0e839ff66391126d63`).  
> No training, Slurm, or test-split inspection was performed.  
> Machine-readable twin: `results/diagnostics/sequential_aml_to_paysim_ssl_preflight.json`

**Verdict:** **PARTIAL — runnable via generic CLI, not a turnkey protocol.**  
PaySim contrastive encoder training with `--finetune` weight continuation already works through `main.py → get_data → train_gnn → train_hetero_contrastive`. There is **no** dedicated sequential AML→PaySim SSL script, **no** BN freeze policy, **no** strict contract/flag asserts on load, and **no** built-in matched random control or val-only dual-domain eval harness. Those gaps are protocol/orchestration — not a hard block on training itself.

Intended experiment class: `exploratory_posthoc=true`, `table_eligible=false`, validation only.

---

## Intended protocol (target)

1. Init from frozen corrected/no-preserve AML seed-2:
   `saved-models/checkpoint_gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2.tar`  
   SHA256 `18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6`
2. Continue InfoNCE on **PaySim TRAIN** edges only; encoder sees **no fraud labels**.
3. Optimizer **reset** (`checkpoint_weight_continuation_with_optimizer_reset`).
4. Evaluate **frozen** encoder on PaySim **val** and AMLWorld **val** only.
5. Compare vs original frozen AML ckpt + matched PaySim-only random-init SSL control.

---

## Answers (1–12)

### 1. Does `main.py --data PaySim --finetune` support contrastive encoder **training**?

**YES.**

- `main.py` always calls `get_data` then `train_gnn` unless `--inference`.
- Default `--objective` is `contrastive`.
- PaySim is a registered adapter (`dataset_specs.PAYSIM_SPEC`); `data_loading.get_data` has no PaySim early-exit that disables training.
- With `--reverse_mp`, training uses `train_hetero_contrastive`.

Existing PaySim Slurm/scripts are mostly extract/probe/transfer. That is usage history, not a code prohibition.

### 2. What does `--finetune` restore?

| Component | Restored on `--finetune`? | Notes |
|-----------|---------------------------|-------|
| Encoder `model_state_dict` (GIN + edge_emb + embedding head) | **YES** | `load_model` → `load_state_dict` |
| BatchNorm running buffers | **YES** | Inside `model_state_dict` |
| Contrast projection head | **YES** if built + key present | `load_checkpoint_auxiliary_modules` |
| Morph / TF aux / masked decoder | **YES** if present | Same helper; source seed-2 has **no** morph |
| Optimizer Adam state | **NO** | `load_optimizer=False`, then **rebuild** Adam over model+proj(+aux) |
| LR scheduler | **N/A** | Contrastive path has none |
| Epoch / step resume index | **NO** | Epochs restart at 0 |

Provenance log string: `checkpoint_weight_continuation_with_optimizer_reset`.

**Caveat:** CLI help text still says weights *and optimizer* are restored — implementation resets Adam. Load path is always `checkpoint_{unique_name}.tar` (not `_finetuned`); saves under `--finetune` write `checkpoint_{unique_name}_finetuned.tar`.

### 3. Can the corrected/no-preserve edge-dim-8 ckpt load under PaySim feature-contract path?

**YES for tensor geometry / TF flag; PARTIAL for semantic strictness.**

Inspected source checkpoint metadata:

- `edge_feature_schema.edge_dim=8`, ports+tds, `correct_reverse_edge_features=True`, `preserve_seed_edges=False`
- `edge_emb.*.weight` shape `(66, 8)`, `embedding_dim=128`
- Has `contrast_projection_state_dict`; no morph head
- `include_temporal_flow_edge_features=False`

PaySim with `--ports --tds` also builds edge_dim **8**. Omitting `--feature_contract` or using `paysim_legacy_duplicate_v1` keeps base semantic values on the historical duplicate path (legacy is a no-op on values).

Load-time checks today:

- **Strict** `model.load_state_dict` (shape mismatch fails)
- **Assert** TF edge-feature flag consistency
- **No** assert of `edge_dim`, `correct_reverse`, `preserve_seed`, ports/tds, or contract ID vs checkpoint

So a misconfigured PaySim run can still load if shapes happen to match.

### 4. Which versioned PaySim contract is used during **TRAINING**?

**Same path as extraction.**

- Omit `--feature_contract` → no transform (bit-exact historical / legacy duplicate semantics).
- Explicit `--feature_contract paysim_legacy_duplicate_v1` → same values + recorded summary.
- Other IDs (`paysim_type_only_v1`, `paysim_structure_only_v1`) alter base slots before ports/TDS/z-norm and would **diverge** from the AML encoder’s training geometry/semantics.

**Recommended for this continuation:** omit flag **or** set `paysim_legacy_duplicate_v1` explicitly for provenance.

### 5. Is normalization fit using PaySim train only?

**Only if `--train_fit_edge_znorm` is set.**

- Flag **on**: mean/std from `tr_data.edge_attr`; applied to val/test (`data_loading.py`).
- Flag **off** (default): independent per-split z-norm (transductive; **not** allowed for inductive claims).

**Required for this protocol:** `--train_fit_edge_znorm`.

### 6. Can PaySim labels enter contrastive SSL?

**NO for loss, sampling identity, checkpoint selection, or stopping — with a presence caveat.**

| Stage | Labels drive behavior? |
|-------|------------------------|
| Temporal split | **No** — timestamps/`hourly_step` fractions |
| Train seed set | **No** — all train forward edges are seeds |
| InfoNCE | **No** — identity / optional morph/KNN; no fraud |
| Best-ckpt score | **No** — morph val or `loss/train` |
| Early stop on AUPRC/F1 | **No** — none in contrastive |
| Val CE/F1 in contrastive loop | **No** — commented out |

**Caveat:** `y` / `isFraud` is loaded onto the graph and passed as `edge_label` into `LinkNeighborLoader` for API compatibility. Contrastive training does not read those labels for the objective. Hygiene improvement (optional): omit `edge_label` on contrastive loaders.

Downstream **evaluation** probes necessarily use labels on **val** only; that is outside encoder SSL.

### 7. BatchNorm running statistics during PaySim continuation?

**They update** (modules stay in train mode).

- GIN uses `BatchNorm`.
- Contrastive loops do not freeze BN / call `model.eval()` for BN-only.
- No `--freeze_bn` flag.

Loaded AML BN buffers are the **initialization**, then PaySim forwards overwrite `running_*`. Frozen-AML-BN transfer protocols used at **extract** time are **not** what continuation training does.

**Protocol choice required before claiming results:** document “PaySim-adapted BN” vs add a freeze/AML-only-BN mode (code change).

### 8. Shared-live-tree / artifact collision risk?

**YES if careless with `unique_name`.**

| Risk | Detail |
|------|--------|
| Overwrite locked AML `.tar` | Avoided if using `--finetune --save_model` (writes `_finetuned`) **and** never training without `--finetune` on the source unique |
| Load collision | `--finetune` loads `checkpoint_{unique}.tar` — staging a **copy** under a new unique is safest (scout pattern) |
| Embeddings | `embeddings/{unique}/` (or broken symlink → use results-side dirs as scout did) |
| Parallel arms | C0-style continuation vs random control need **distinct** uniques |

**Do not** set `--unique_name` to the locked source name for training saves without staging.

### 9. Smallest 2-step smoke

**Design (not executed):**

1. Stage source weights → `checkpoint_<smoke_unique>.tar` (strip optimizer optional; `load_optimizer=False` already).
2. GPU (or CPU for compile-only) run:

```bash
python main.py --data PaySim --model gin --testing --tqdm \
  --objective contrastive --finetune --save_model \
  --unique_name <smoke_unique> --seed 2 \
  --n_epochs 1 --max_optimizer_steps 2 \
  --batch_size 8192 --contrastive_accum_steps 1 \
  --num_neighs 100 100 --loader_num_workers 0 \
  --reverse_mp --ego --ports --emlps --tds \
  --correct_reverse_edge_features \
  --train_fit_edge_znorm \
  --contrast_projection_head --contrast_projection_hidden 128 \
  --contrast_projection_dim 128 --contrastive_asymmetric \
  --contrastive_num_neg_samples 8192 --contrastive_memory_bank_size 0 \
  --contrastive_temperature 0.5 \
  --checkpoint_policy last
```

**Proves:** finite Train Loss; optimizer steps=2; provenance log; `_finetuned` save/reload; no morph/label-driven selection.

**Does not prove cheaply:** full featurize is still a full PaySim `get_data` (~large). Prefer Advanced GPU; login-node GPU discouraged.

**Must use `--reverse_mp`:** `--max_optimizer_steps` is wired in `train_hetero_contrastive` only.

### 10. Runtime estimate (5 epochs or 500 steps)

PaySim split sizes (from existing transfer diagnostics metadata): train **3,792,821** / val **1,276,276** edges.

| Quantity | Estimate |
|----------|----------|
| Microbatches / epoch @ bs 8192 | ≈ 3,792,821 / 8192 ≈ **463** |
| Opt steps / epoch @ accum 4 | ≈ **116** |
| 5 epochs | ≈ **580** opt steps |
| 500 opt steps | ≈ **4.3** epoch-equivalents |

Wallclock (Advanced GPU, `loader_num_workers=0`, matched D+ flags), extrapolated from AML C0 ~0.5 h / 500 steps and slower PaySim subgraphs:

| Budget | Train-only est. | + val-only AML+PaySim extract/probe |
|--------|----------------:|------------------------------------:|
| 500 opt steps | **~1.0–2.5 h** | **+0.5–1.5 h** |
| 5 epochs (~580 steps) | **~1.2–3.0 h** | same add-on |

Still expected **≪ 6 h** per arm if extract is val-only. Featurize/cache dominates smoke more than the 2 steps.

### 11. Exact minimal commands

Shared flags (matched recipe):

```text
--data PaySim --model gin --objective contrastive --testing --tqdm
--reverse_mp --ego --ports --emlps --tds --correct_reverse_edge_features
--train_fit_edge_znorm --loader_num_workers 0
--batch_size 8192 --num_neighs 100 100
--contrast_projection_head --contrast_projection_hidden 128 --contrast_projection_dim 128
--contrastive_asymmetric --contrastive_num_neg_samples 8192
--contrastive_memory_bank_size 0 --contrastive_accum_steps 4
--contrastive_temperature 0.5 --checkpoint_policy last --save_model
# omit --preserve_seed_edges; omit TF edge features
# optional: --feature_contract paysim_legacy_duplicate_v1
```

**A. AML-initialized PaySim continuation**

```bash
# stage (once): copy locked weights to a NEW unique (do not overwrite source)
cp saved-models/checkpoint_gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2.tar \
   saved-models/checkpoint_seq_aml2ps_c_seed2.tar
# optionally strip optimizer_state_dict in a small staging script

python main.py <shared flags> \
  --unique_name seq_aml2ps_c_seed2 --seed 2 \
  --finetune --n_epochs 5 --max_optimizer_steps 500
```

**B. Matched random-init PaySim-only control**

```bash
python main.py <shared flags> \
  --unique_name seq_aml2ps_rand_seed2 --seed 2 \
  --n_epochs 5 --max_optimizer_steps 500
# NO --finetune → freshly constructed encoder+proj; same PaySim batches/steps/augs/compute
```

**Eval (both arms, val only — not part of `main.py` train):** extract PaySim + Small-HI with `--extract_splits train,val`, frozen encoder (`--finetune` on extract if loading `_finetuned`), `--train_fit_edge_znorm` on PaySim, then val-only downstream probe. Forbid `test.npz` / test metrics.

### 12. Code changes required (do **not** implement yet)

| Priority | Change | Why |
|----------|--------|-----|
| P0 orchestration | Dedicated script/slurm staging + unique policy + afterok DAG | Collision safety; matched A vs B; artifacts |
| P0 protocol | Document BN policy; optionally implement freeze / AML-BN-only | Unambiguous forgetting vs adaptation claims |
| P1 | Strict load asserts: edge_dim, correct_reverse, preserve, ports/tds, contract | Prevent silent wrong-geometry runs |
| P1 | Fix `--finetune` help text (optimizer not restored) | Operator clarity |
| P1 | Val-only eval harness (PaySim + AMLWorld); refuse test | Gate requirements |
| P2 | Optional: contrastive loaders without `edge_label` | Label-hygiene |
| P2 | Wire `max_optimizer_steps` into homo path if ever used | Completeness |
| Not needed for this protocol | Alternating J/JM/JC dual-domain trainer | Separate future experiment |

---

## Predeclared validation gate (for a future run)

Write **before** reading full-arm val scores:

1. AML→PaySim continuation must improve PaySim **validation** AUPRC over **both**:
   - original frozen AML seed-2 encoder (same extract/probe recipe), and
   - matched PaySim-only random SSL control,
   by a **predeclared meaningful margin** (suggest ≥ 0.003 AUPRC **or** document alternative).
2. Must beat the existing **X-only** validation control under the same downstream recipe.  
   **Note:** `paysim_dplus_transfer_final.json` currently stores X-only **test** metrics only — a **val-only** X-only baseline must be produced or located before gating; do not use test.
3. Report AMLWorld **validation** AUPRC after continuation (forgetting). Soft budget optional (e.g. regress ≤ 0.02 vs frozen original) — predeclare if used.
4. **No test** evaluation or inspection for selection/gate.
5. `exploratory_posthoc=true`, `table_eligible=false`.

Downstream recipe should match the locked transfer stack where possible (e.g. pre-3h H+X, train-fit z-norm, PaperStyleMLP) so comparisons are protocol-compatible — exact probe choice must be frozen in the run script before scores are read.

---

## Related artifacts (navigation)

- Locked AML family: `results/diagnostics/final_corrected_no_preserve_multiseed.json`
- Frozen PaySim D+ transfer: `results/diagnostics/paysim_dplus_transfer_final.json` / `notes/paysim_dplus_transfer_final.md`
- Prior AML-only continuation scout: `results/diagnostics/final_exploratory_ssl_scout.json` / `notes/final_exploratory_ssl_scout.md`
- J/BN discussion (future alternating): `notes/final_exploratory_ssl_scout_preflight.md` §§9–10

---

## Bottom line

Sequential **PaySim-train SSL continuation from the corrected seed-2 checkpoint is already supported** by the generic training stack with `--finetune` + matching D+ flags + `--train_fit_edge_znorm`, and the encoder path is **label-free** for InfoNCE. What is **missing** is a safe, reproducible experiment harness (staging, BN policy, strict asserts, matched random arm, val-only dual eval, predeclared gate) — not the ability to take two optimizer steps on PaySim.
