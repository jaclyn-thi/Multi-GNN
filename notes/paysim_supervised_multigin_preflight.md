# PaySim supervised Multi-GIN preflight (read-only)

> Twin: `results/diagnostics/paysim_supervised_multigin_preflight.json`  
> Scope: inventory + code audit only. No training, Slurm, or new test inspection.  
> **Locked primary (amendment):** Candidate A — paper-faithful Multi-GIN+EU config used for AMLWorld supervised parity.

## Executive answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Valid supervised PaySim Multi-GIN baseline already exists? | **No.** Explicitly deferred/missing; no PaySim supervised checkpoints, evals, or registry rows. |
| 2 | Current training path runnable? | **Yes** — `main.py --data PaySim --model gin --objective supervised --supervised_head legacy …` is supported end-to-end. |
| 3 | Primary architecture? | **Candidate A** (ports, emlps, reverse_mp, ego; **omit** tds / preserve / correct_reverse / TF encoder inputs). |
| 4 | Feature contract + norm? | **`paysim_legacy_duplicate_v1`** (omit flag or set explicitly); **legacy per-graph edge z_norm** matching AML supervised parity (not transfer `train_fit_edge_znorm`). |
| 5 | Estimated runtime? | AML HI 50ep ≈ **3.96 h**; PaySim ~25% more edges → **~5 h / seed** at same recipe; **fits 6 h with slim margin** — smoke first; `--resume_supervised` if near limit. |
| 6 | Minimal seeds? | **1 development seed** → validation gate → then seeds **1 and/or 3** only if valid (AML formal used 1–3). |
| 7 | Smallest next Slurm job? | **GPU smoke:** 1–2 epochs, Candidate A flags, unique_name under `paysim_legacy_supervised_…`, refuse overwrite. |
| 8 | Files modified / jobs submitted this turn? | **No** (this audit MD/JSON only). |

**Thesis label (locked):**  
“Target-supervised PaySim Multi-GIN+EU baseline using the paper-faithful architectural configuration.”  
Do **not** call it an exact published PaySim reproduction (no protocol-compatible published PaySim Multi-GIN numbers in-repo).

---

## 1. Inventory — does a baseline already exist?

**Verdict: no usable end-to-end supervised PaySim Multi-GIN baseline.**

| Artifact class | Found for PaySim supervised E2E GNN? | Notes |
|----------------|--------------------------------------|-------|
| `saved-models/*paysim*supervised*` / legacy supervised ckpts | **No** | Supervised ckpts are Small-HI / Small-LI only |
| `results/diagnostics/eval_*paysim*supervised*` | **No** | |
| `notes/supervised_*PaySim*` | **No** | |
| Thesis registry PaySim rows | 3 rows; **0 supervised** | Objectives: `None` (norm ablation) or `contrastive_frozen_transfer` |
| Slurm `train_*paysim*supervised*` | **No** | PaySim Slurm is extract/probe/transfer/SSL |
| Explicit deferrals | **Yes** | `notes/paysim_frozen_transfer_preflight.md` §9: “Supervised PaySim-from-scratch — **Missing**”; `notes/downstream-eval-plan.md` lists it as TODO |

### Confusion table (not this baseline)

| Run family | Why not a supervised PaySim GNN ceiling |
|------------|----------------------------------------|
| Frozen AML → PaySim logistic / MLP probes | Encoder frozen; labels train probe only |
| Random-encoder PaySim probes | Architecture probe, not target-supervised training |
| Contrastive PaySim-only / sequential AML→PaySim SSL | Self-supervised encoder updates; not CE on isFraud |
| Historical ports-only frozen transfer | Wrong encoder family / not E2E supervised |
| AMLWorld legacy supervised Multi-GIN+EU | Correct recipe, **wrong dataset** |

### Usable-as-baseline table (empty for PaySim E2E)

| run/job/ckpt | arch/flags | contract | norm | split | selection | seeds | val/test | path | usable? |
|--------------|------------|----------|------|-------|-----------|-------|----------|------|---------|
| — | — | — | — | — | — | — | — | — | **None found** |

Closest **reference** (AMLWorld, not PaySim):  
`small_hi_legacy_supervised_gin_emlps_ports_50ep_seed{1,2,3}` — ports, TDS-off, legacy head, 50ep, `checkpoint_best_val_f1.tar`, paper_argmax; aggregate test F1 **0.660 ± 0.060**. Protocol clone target for Candidate A.

---

## 2. Supervised PaySim support (code path)

### Runnable command (Candidate A)

```bash
python main.py \
  --data PaySim --model gin --objective supervised --supervised_head legacy \
  --unique_name paysim_legacy_supervised_gin_emlps_ports_50ep_seed2 \
  --save_model --n_epochs 50 --batch_size 8192 --num_neighs 100 100 \
  --loader_num_workers 0 --reverse_mp --ego --ports --emlps --seed 2 --tqdm
# deliberately omit: --tds --preserve_seed_edges --correct_reverse_edge_features
# TF encoder inputs; optionally set --feature_contract paysim_legacy_duplicate_v1
```

| Item | Status / location |
|------|-------------------|
| Model | `models.GINe` + `to_hetero` when `--reverse_mp` |
| Head | `--supervised_head legacy` → Egressy `3h→50→25→2` (`models._legacy_edge_classifier`) |
| Train loop | `training.train_gnn` → `train_hetero_supervised` |
| Labels | CSV `Is Laundering` ← PaySim `isFraud` at format time (`dataset_specs.PAYSIM_SPEC`) |
| Seed-edge loss | **Yes.** CE only on loader seed edges via `mask = isin(edge_id, batch_seed_ids)`; neighbor/context edges participate in MP but **not** as CE examples (`training.py` ~891–909) |
| Loss | `CrossEntropyLoss(weight=[w_ce1, w_ce2])` |
| Class weights | From `model_settings.json` → gin `w_ce1≈1.0`, `w_ce2≈6.275` — **fixed AML Bayesian settings, not recomputed from PaySim train** |
| Split | `temporal_edge_split`, PaySim `hourly_step`, 60/20/20 |
| Checkpoint selection | **Best validation minority-class F1** (argmax F1); ties keep earliest epoch (`SupervisedCheckpointer`) |
| Test in selection? | **No** for epoch choice (test F1 may be *logged* at selected epoch only) |
| Decision rule | **paper_argmax** (two-class logit argmax); eval via `scripts/evaluate_supervised_gnn.py` |
| Fixed-0.5 | Not the AML supervised primary; paper_argmax is primary. Still report AUROC/AUPRC + ranking separately |
| Early stopping | Fixed horizon (50ep); best-val ckpt retained — training does not halt early |
| Resume | `--resume_supervised` loads `checkpoint_last.tar` |
| Coverage | Eval reports expected vs scored seed edges (AML formal ~0.999+) |
| Memory/runtime | 128G / 1 GPU / batch 8192 / neighs 100,100 historically; HI 50ep ≈ 4 h |

**Labels on context edges:** not treated as supervised training targets (mask restricts CE). Good.

---

## 3. Candidate A vs B

### Candidate A — PRIMARY (locked)

| | |
|--|--|
| Flags | gin, supervised, legacy head, reverse_mp, ego, ports, emlps; **omit** tds, preserve_seed_edges, correct_reverse, TF-in-encoder |
| edge_dim | **6** = base 4 + ports 2 |
| Contract | `paysim_legacy_duplicate_v1` → edge_dim 6 achievable |
| Scientific question | Target-supervised ceiling under **paper-faithful Multi-GIN+EU architecture** (same as AML supervised parity) |
| Confounds | Schema placeholders; AML-tuned CE weights; PaySim hourly split ≠ AML days; test MP on full timeline |
| Thesis label | Target-supervised PaySim Multi-GIN+EU baseline using the paper-faithful architectural configuration |
| Runtime | ~5 h / 50ep seed (est.) |

### Candidate B — optional future ablation only

| | |
|--|--|
| Flags | Match corrected/no-preserve SSL encoder: ports+**tds**+emlps+reverse+ego+**correct_reverse**, preserve off |
| edge_dim | **8** |
| Question | Architecture-matched supervised control vs SSL encoder family |
| Status | **Not required** for current thesis baseline |

**Recommendation:** run **Candidate A only** now. B answers a different question and is deferred.

---

## 4. Feature / normalization policy (locked)

| Choice | Lock | Rationale |
|--------|------|-----------|
| Contract | `paysim_legacy_duplicate_v1` | Compatibility with AML edge geometry; type duplicated into currency + payment slots — **compatibility mapping, not semantic AML equivalence** |
| vs type_only | Do not select via test; type_only = sensitivity later | |
| Edge z_norm | **Default per-graph z_norm** (AML supervised parity scripts omit `--train_fit_edge_znorm`) | Clone AML protocol; document as partially transductive on attrs (same caveat as Multi-GNN supervised) |
| BN | Standard training BN (supervised updates running stats) | Not frozen-transfer BN |
| ports | ON | Paper Multi-GIN+EU |
| TDS | **OFF** | Paper-faithful |
| TF in GNN input | **OFF** | Remains separate downstream control if needed |
| Val/test in feature formulas | Ports/TDS construction follows existing `GraphData` path on each split graph; default z_norm is per-graph (val/test attrs enter their own graph stats). Do **not** enable TF encoder inputs |

---

## 5. Metrics and selection (clone AML)

**Primary selection:** best PaySim **validation minority-class F1** under **paper_argmax**.

At `checkpoint_best_val_f1.tar`, report (val + test; test never for selection):

- AUROC, AUPRC  
- F1 / precision / recall @ paper_argmax  
- positive prediction rate, confusion counts  
- P@100 / P@500 / P@1000  

Document extreme imbalance (PaySim val pos rate ~6e-4; test higher ~3e-3 on frozen-transfer cohort).

**Class weights:** current path uses **global gin settings**, not PaySim-train-estimated inverse frequency. Accept for parity with AML supervised recipe; document as limitation (not “PaySim-train-only weights”).

---

## 6. Required comparisons (eventual)

| System | Role |
|--------|------|
| X-only target-trained classifier | Feature-only floor |
| Frozen AML encoder transfer (P1 / D+) | Label-free / frozen transfer |
| Label-free BN adaptation (P2) | Adaptation without CE on encoder |
| Optional AML→PaySim sequential SSL | SSL continuation |
| **Target-supervised PaySim Multi-GIN+EU (A)** | **Upper/reference ceiling** |

Supervised model **sees fraud labels** end-to-end — not a fair competitor to label-free transfer.

---

## 7. Runtime and seed plan

| Item | Estimate / plan |
|------|-----------------|
| AML HI 50ep evidence | Job `18473402`: start 18:41 → end 22:39 ≈ **3.96 h** (`slurm-logs/hi_ports_50ep_s1_18473402.out`) |
| PaySim size | ~6.36M edges vs HI ~5.08M (~**+25%**) |
| One-seed 50ep | **~4.5–5.5 h** → likely fits 6 h GPU MaxTime; margin thin |
| Smoke | 1–2 epochs, ~10–30 min + data load/ports |
| Continuation | `--resume_supervised` supported via `checkpoint_last.tar` |
| 3 seeds on 3 GPUs | Independent jobs possible; **not** needed before gate |
| Horizon | **50 epochs** (AML paper-parity transfer); no test-based retuning |
| Early stop | Keep full 50ep; select best-val ckpt (do not shorten protocol using test) |

**Plan:** (1) smoke 1–2ep → (2) seed-2 (or seed-1) full 50ep → (3) validation gate (finite metrics, coverage, best-val ckpt exists) → (4) optional seeds 1/3 + formal eval.

---

## 8. Integrity requirements (for implementation)

Assert:

- `dataset=PaySim`, `objective=supervised`, `supervised_head=legacy`  
- Labels enter CE; encoder params receive gradients  
- Seed-edge mask; coverage + ID alignment  
- Flags: ports+emlps+reverse_mp+ego; tds/preserve/correct_reverse/TF-in **false**  
- Contract legacy_duplicate; edge_dim=6  
- Best-val F1 selection; test not used for epoch/arch/feature/weight selection  
- Unique run names; no overwrite of AML or frozen-transfer artifacts  
- Checkpoint + config + code hashes in summary/eval JSON  

---

## Unavoidable PaySim vs AMLWorld differences

| Aspect | AMLWorld supervised parity | PaySim Candidate A |
|--------|----------------------------|--------------------|
| Split buckets | calendar days | hourly steps (`Timestamp=step*3600`) |
| Categorical slots | real currency / payment format | **type code duplicated** (legacy contract) |
| Pattern metadata | available | unsupported |
| Label source | AML laundering | `isFraud` → `Is Laundering` |
| Prevalence | HI-like | PaySim test enrichment differs |
| Published numeric target | Multi-GIN+EU F1 on AML | **None** for PaySim — reference ceiling only |
| Class weights | gin Bayesian on AML settings | **same fixed weights** (not PaySim-refit) |

---

## Exact end-state checklist

1. **Valid baseline exists?** No.  
2. **Training path runnable?** Yes.  
3. **Primary architecture?** Candidate A (paper-faithful Multi-GIN+EU).  
4. **Contract + norm?** `paysim_legacy_duplicate_v1` + legacy per-graph z_norm.  
5. **Runtime?** ~5 h / 50ep seed; smoke ≪ 1 h.  
6. **Minimal seeds?** 1 dev → gate → optional 1/3.  
7. **Smallest next job?** GPU smoke 1–2 epochs Candidate A.  
8. **Modified files / submitted jobs?** **No.**
