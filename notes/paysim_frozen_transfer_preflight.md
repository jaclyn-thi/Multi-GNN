# PaySim frozen transfer preflight (read-only)

**Date:** 2026-07-25  
**Scope:** Audit whether existing AMLWorld→PaySim frozen-transfer infrastructure can evaluate the verified edge-centric D+ encoders (seeds 1–3) and, secondarily, the AML-supervised partial-FT seed-2 encoder.  
**Actions taken:** none (no code changes, no extraction, no Slurm).

Companion: [`results/diagnostics/paysim_frozen_transfer_preflight.json`](../results/diagnostics/paysim_frozen_transfer_preflight.json)

---

## Executive verdict

**Existing PaySim transfer infra is valid as a skeleton, but not valid as-is for the locked D+ checkpoints.**

Jun 2026 PaySim runs used an older GIN stack (`reverse_mp + ego + ports`, **no** `emlps` / `tds` / `correct_reverse`) and **post-128** embeddings + sklearn logistic H-only. Verified D+ checkpoints require **edge_dim=8** (`ports+tds`), `emlps`, and **corrected reverse** semantics. Loading D+ into the current PaySim extract script would fail on `edge_emb` shape `(66, 8)` vs `(66, 6)`.

No D+-compatible PaySim embeddings exist. Prior PaySim results are **prior diagnostics**, not primary transfer.

---

## 1. Source encoders under audit

| Role | Train job | Epoch | Checkpoint | edge_emb in | Flags in ckpt |
|------|-----------|------:|------------|-------------|---------------|
| Primary seed 1 | 18801429 | 34 | `checkpoint_edge_dplus_corrected_preserve_40ep_seed1_final.tar` | **8** | ports+tds+emlps+correct_reverse+preserve_seed |
| Primary seed 2 | 18514684 | 40 | `checkpoint_gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2.tar` | **8** | same |
| Primary seed 3 | 18802579 | 29 | `checkpoint_edge_dplus_corrected_preserve_40ep_seed3_final.tar` | **8** | same |
| Secondary FT seed 2 | 18801435 | 18 | `dplus_partial_finetune_hxxtf_seed2/checkpoint_best_val_auprc.tar` | **8** | same encoder geometry; stores `encoder_state_dict` + `classifier_state_dict` separately |

FT classifier must **not** transfer. Only frozen `encoder_state_dict` is eligible as a secondary sensitivity vs frozen seed 2.

---

## 2. Infrastructure inventory

### Data / format
| Path | Role |
|------|------|
| `format_paysim.py` | Raw PaySim → shared `formatted_transactions.csv` |
| `aml-data/PaySim/PS_*.csv` | Raw (~471M) |
| `aml-data/PaySim/formatted_transactions.csv` | Present (~348M) |
| `dataset_specs.py` (`PAYSIM_SPEC`) | `hourly_step`, 0.6/0.2/0.2, 4-col edge contract |
| `dataset_splits.py` | Temporal bucket split |
| `data_loading.py` | `--data PaySim` graph build + ports/tds/z_norm/hetero |

### Extract / probe / Slurm
| Path | Role |
|------|------|
| `embedding_extraction.py` | Frozen extract; `--embeddings_dir`, `--random_init`, `--representation_source`, `--checkpoint_suffix` |
| `linear_probe.py` | Sklearn logistic on H (`Z`) only |
| `slurm/run_paysim_extract_probe.sh` | GPU extract+probe (**legacy flags**) |
| `slurm/run_paysim_linear_probe.sh` | CPU probe-only |
| `slurm/submit_paysim_transfer.sh` | Queues **old** UNIQUE_NAME tiers |
| `slurm/run_paysim_load_smoke.sh` | Load smoke |
| `scripts/validate_paysim_data.py` | Format/stats/load smoke |
| `tests/test_format_paysim.py` | Formatter unit tests |

### Notes / registry
| Path | Role |
|------|------|
| `notes/datasets.md` | Canonical PaySim setup |
| `notes/downstream-eval-plan.md` | Jun 2026 status + results table |
| `notes/results-archive.md` | Archived PaySim AUROC summary |
| Thesis registry | **No PaySim rows** |

### Residual artifacts
`embeddings/paysim/{hi_contrastive_proj_sym_20ep_bestckpt,random_init_gin}/` — **meta + probe JSONs only**; `.npz` caches **missing**.

---

## 3. Graph construction (PaySim)

| Item | Value |
|------|--------|
| Nodes | Accounts (`nameOrig`/`nameDest` factorized → `from_id`/`to_id`); constant node feature |
| Edges | One directed edge per transaction; multiedges allowed |
| Edge attrs (base) | `Timestamp`, `Amount Received`, `Received Currency`, `Payment Format` (4-d) |
| Split | **Temporal** `hourly_step` (bucket = `Timestamp // 3600` = PaySim `step`); ~60/20/20 |
| Train graph | Train edges only |
| Val graph | Train ∪ val edges |
| Test graph | **All** edges (full timeline) |
| Seed extract | Temporal index sets via `LinkNeighborLoader` + `preserve_seed`-compatible seed-edge extract |
| Scale (documented) | ~6.36M edges, ~743 hourly buckets; test ~1.29M edges, ~0.33% fraud |

### Multi-GIN flags

| Flag | Legacy PaySim scripts | Required for D+ load |
|------|----------------------|----------------------|
| `reverse_mp` | ON | ON |
| `ego` | ON | ON |
| `ports` | ON | ON |
| `emlps` | **OFF** | **ON** |
| `tds` | **OFF** | **ON** (edge_dim 6→**8**) |
| `correct_reverse_edge_features` | **OFF** | **ON** |
| `preserve_seed_edges` | N/A at extract | Training-only; D+ ckpts carry the flag |

---

## 4. Feature compatibility

### AMLWorld D+ input
Base 4 + `in_port,out_port` + `in_td,out_td` → **8**, with corrected reverse swap of port/td pairs. Pre-3h H = **198** = `3 * n_hidden` (n_hidden≈66).

### PaySim mapping (`format_paysim.py` / `PAYSIM_FEATURE_MAPPING`)
| Formatted col | PaySim source | Semantic fidelity |
|---------------|---------------|-------------------|
| Timestamp | `step * 3600` | Synthetic seconds for hourly buckets |
| Amount Received | `amount` | OK |
| Received Currency | `factorize(type)` | **Placeholder** (not currency) |
| Payment Format | same type code | **Duplicate placeholder** |
| Is Laundering | `isFraud` | OK for labels |
| Excluded | balances, `isFlaggedFraud` | Correct (leakage) |

**No dedicated AML→PaySim linear adapter / reinitialized input projection.** Compatibility is schema-level: same 4 base columns so a matching-flag GIN can load. Categorical codes are **not** aligned to AML currency/payment vocabularies.

### What must match at load time
D+ `edge_emb`: `(66, 8)`. PaySim graph under legacy scripts: `(66, 6)`. **Mismatch → `load_state_dict` failure.** Fix is flags (`--ports --tds`), not weight surgery.

---

## 5. Preprocessing / “what is zero-shot?”

| Step | Fit / scope | Label use |
|------|-------------|-----------|
| Format-time type/account codes | Full PaySim CSV (unlabeled+labeled) | Labels unused |
| Ports / TDS | Per graph object (train / train∪val / all) | Labels unused |
| Node `x` z-norm | Fit on **train** graph; val/test clone | Labels unused |
| Edge `edge_attr` z-norm | **Independent** z_norm on each graph’s edge_attr | Labels unused |
| AML source scalers | **Not** reused | — |

**Implications for zero-shot wording**

- Encoder weights stay frozen; no target-supervised update.
- Unlabeled PaySim **does** enter encoder inputs via ports/TDS construction and per-graph edge z-norm.
- Test graph includes **all** edges before z_norm → test-edge statistics influence test (and full-graph) edge normalization. This is the repo’s standard Multi-GNN induce-per-split pattern, but it is **not** “encoder sees only train-normalized features.”
- Recommend phrase: **frozen encoder transfer with target-graph structural featurization and split-graph z-norm** (not pure feature-space zero-shot).

AML preprocessing is **not** transferred.

---

## 6. Representation / freeze semantics

| Item | Legacy PaySim | Required for D+ parity with AML primary |
|------|---------------|----------------------------------------|
| Extract point | Default **post-128** (`embedding_dim=128`) | **`pre_embedding_3h` (198-d)** |
| Cached D+ PaySim H | None | Must extract |
| Freeze | Checkpoint load + no train | Same; `model.eval()` / inference in extract |
| Augmentation | Off at extract | Off |
| BN | Eval-mode running stats from AML pretrain | Keep frozen (do not update on PaySim) |

---

## 7. Downstream evaluation

### What exists
- H-only sklearn **logistic** (`linear_probe.py`): AUROC/AUPRC + F1 at val-max-F1 and optional fixed 0.5; class_weight `model|balanced|none`.
- No PaySim H+X / X-only / H+X+TF MLP path matching the locked AML 18678029 recipe.
- No PaySim causal TF cache under `results/cache/temporal_flow_causal/` (only Small-HI / Small-LI).

### Locked primary protocol (recommended for D+ transfer)
1. **Primary:** mean ± sample SD over three frozen AML D+ seeds of **pre-3h H** (198) + **PaySim X** (24 one-hot edge-native stack as in AML probe) → **H+X**, supervised **MLP** with AML-locked recipe (15 ep, lr 1e-3, BCE, downstream seed 2, val F1 threshold + report fixed 0.5). Report AUROC/AUPRC/F1 and P@K.
2. **Controls:** PaySim **X-only** MLP; frozen **H-only**; **random-init** GIN with **identical D+ flags** (re-extract; old random_init is ports-only — stale).
3. **H+X+TF:** **defer** until PaySim TF is implemented + leakage-audited (not available).
4. **Secondary:** frozen FT encoder seed 2 vs frozen D+ seed 2 only (same H / H+X protocol). Not in primary aggregate.

Imbalance: prefer AUROC/AUPRC + ranking; document test fraud ~4× train (later steps). Do not select models on test.

---

## 8. Leakage audit

| Risk | Status |
|------|--------|
| Target labels in encoder inputs | **PASS** (excluded balances / isFlaggedFraud; labels only for probe) |
| Target labels update encoder | **PASS** if extract-only / no `--finetune` train |
| FT AML classifier transferred | Must **refuse** (`classifier_state_dict` unused) |
| Future edges in test MP graph | **Inherent** to Multi-GNN test=all-edges scope — document, do not call “inductive” |
| Edge z_norm on full test graph | **PARTIAL** — unlabeled test attrs enter scaling |
| Causal TF leakage on PaySim | **N/A** (no TF) — do not enable TF yet |
| Joint transductive random protocol | Not used in proposed temporal primary |

---

## 9. Existing PaySim results classification

| Artifact | Classification | Why |
|----------|----------------|-----|
| Jun 2026 `hi_contrastive_proj_sym_20ep_bestckpt` AUROC 0.866 / 0.864 | **Prior diagnostic** | Wrong encoder family; ports-only; post-128; logistic H-only |
| Jun 2026 `random_init_gin` AUROC 0.730 | **Prior diagnostic** (random control for old stack) | Flag-mismatched vs D+ |
| `embeddings/paysim/*/probe_results*.json` | **Stale/incompatible** for D+ primary | Same |
| In-domain Small-HI AUROC 0.929 on sym ckpt | **In-domain reference for old encoder**, not D+ | Different checkpoint |
| Supervised PaySim-from-scratch | **Missing** (deferred in notes) | — |
| D+ → PaySim anything | **Missing** | — |
| Thesis registry PaySim rows | **Absent** | — |

Do **not** promote 0.866 as D+ transfer evidence.

---

## 10. Required controls (smallest defensible set)

1. PaySim **X-only** downstream MLP (no H).
2. Frozen AML D+ **H-only** (pre-3h).
3. Frozen AML D+ **H+X** (primary transfer stack).
4. **H+X+TF** — only after TF audit (currently **blocked**).
5. **Random-init** encoder with **D+-matched flags** (new extract), not the Jun 2026 ports-only random run.
6. Optional secondary: frozen **partial-FT** encoder seed 2 vs frozen seed 2.

Primary statistic: **mean ± sample SD of test metric(s) over seeds 1–3** (best-score D+ ckpts). FT excluded from that mean.

---

## 11. Published comparison audit

| Source | Motivates transfer? | Protocol-compatible PaySim baseline? | Verdict |
|--------|---------------------|--------------------------------------|---------|
| Egressy Multi-GNN (2306.11586) | Indirect (AML multigraph GNN) | No PaySim numbers in repo lit notes | **FAIL** for numerical PaySim compare |
| Papagei-style frozen probe (lit-review framing) | Yes (frozen H → linear probe) | Motivates protocol; no PaySim table | **PARTIAL** (methodological only) |
| Downstream-eval-plan GFM→PaySim narrative | Yes | Internal only | **PARTIAL** (thesis framing, not published baseline) |
| Any repo PDF with aligned PaySim split/label/metric | Not identified in lit-review index | — | **FAIL** until located |

**Do not quote a published PaySim F1/AUROC as a peer baseline** unless dataset version, split, label (`isFraud`), and metric are verified aligned.

---

## 12. Runtime / implementation assessment

| Question | Answer |
|----------|--------|
| Do D+ ckpts load without arch changes? | **Yes**, if PaySim extract uses matching flags (`ports+tds+emlps+reverse_mp+ego+correct_reverse`). **No** with current Slurm script. |
| Cached PaySim D+ embeddings? | **No** |
| FT load via stock extract? | **No** — FT blob uses `encoder_state_dict`, not `model_state_dict`; needs thin load adapter and **must drop classifier** |
| Expected extract/probe per seed | Historical PaySim extract fit in **≤6 h** on ports-only @ bs 4096 / 128G. D+ (+emlps+tds+pre-3h) likely similar or slightly slower; use Advanced GPU 6 h, `loader_num_workers=0` for hang-safety, bs 4096 (tune if OOM). |
| Three seeds in Advanced GPU envelope? | **Yes** as **3 separate 6 h jobs** (or serial if needed). Not one job for all three if extract is multi-hour. |
| Memory | Scripts historically **128G**; keep 128G for PaySim (larger than HI’s 64G). |
| Smallest smoke | (1) `validate_paysim_data.py` already OK; (2) 1-batch or short extract with D+ flags + `load_state_dict` on seed2; (3) refuse overwrite of legacy `embeddings/paysim/hi_contrastive_*`. |

### Code changes required (not done in this preflight)
1. New Slurm/extract recipe: D+ flags + `--representation_source pre_embedding_3h` + unique `embeddings/paysim/<dplus_unique>/pre_embedding_3h/`.
2. Checkpoint path override (D+ filenames ≠ PaySim UNIQUE_NAME) and/or `--checkpoint_suffix` / explicit ckpt arg.
3. FT: load `encoder_state_dict` only into GIN; ignore `classifier_state_dict` / scalers.
4. Downstream PaySim H / X / H+X MLP evaluator (reuse AML recipe; fit scaler on PaySim train only).
5. Random-init control with D+ flags.
6. Registry ingest for formal transfer rows (after runs).
7. **Do not** enable TF until PaySim causal TF + leakage audit exist.

---

## 13. Exact proposed commands (not submitted)

```bash
# Smoke (after code wires D+ flags) — DO NOT RUN in this preflight
# sbatch --export=ALL,UNIQUE_NAME=edge_dplus_..._seed2,CKPT=saved-models/checkpoint_....tar \
#   slurm/run_paysim_dplus_frozen_transfer_smoke.sh

# Formal (illustrative; scripts not yet written)
# for seed unique in seed1 seed2 seed3; do
#   sbatch slurm/run_paysim_dplus_frozen_pre3h_extract_probe.sh  # extract pre-3h
#   sbatch --dependency=afterok:... slurm/run_paysim_dplus_hx_mlp_eval.sh
# done
# Secondary FT encoder-only extract+eval vs seed2 — separate job, not in mean
```

**Confirmation: no jobs were submitted in this audit.**

---

## Final answers

1. **Is existing transfer infra valid for these D+ checkpoints?**  
   **PARTIAL / not as-is.** Loader+formatter+extract skeleton yes; current PaySim Slurm flag set and post-128 logistic protocol **no**.

2. **What exactly is transferred?**  
   Frozen AML D+ **GIN encoder weights** (and BN running stats) under matched Multi-GIN flags. Not AML scalers, not AML labels, not the FT AML classifier.

3. **Are any input layers reinitialized or adapted?**  
   **No weight reinit** if flags match (edge_dim 8). Schema placeholders remap PaySim `type`→currency/format codes; that is data mapping, not a learned adapter.

4. **What PaySim information is used before downstream classifier training?**  
   Full formatted graph for ports/TDS; per-split-graph edge z-norm; train-fit node z-norm; neighbor sampling at extract. **Not** AML-fitted preprocessors. Labels unused until probe/MLP.

5. **Locked primary feature/evaluation protocol?**  
   Frozen D+ **pre-3h H (198) + PaySim X → MLP**; primary = **mean±SD over seeds 1–3**. Controls: X-only, H-only, D+-flag random-init. TF deferred.

6. **Can AML-fine-tuned encoder be tested as secondary frozen transfer?**  
   **Yes, with a load adapter** (`encoder_state_dict` only). Compare only to frozen seed 2; **exclude from primary mean**.

7. **Required controls?**  
   X-only; H-only; H+X (primary); D+-matched random-init; optional FT-vs-seed2. No TF until audited.

8. **Defensible published comparisons?**  
   **None numerical (FAIL).** Methodological **PARTIAL** (Papagei frozen-probe / GFM narrative). Egressy is AMLWorld, not PaySim.

9. **Code changes and smoke tests required?**  
   D+ extract flags + pre-3h path + ckpt path wiring + FT encoder-only load + PaySim H/X/H+X MLP eval + D+-flag random control; smoke = successful `load_state_dict` + short extract on seed 2.

10. **Jobs submitted?**  
    **None.** Proposed commands above are illustrative only.
