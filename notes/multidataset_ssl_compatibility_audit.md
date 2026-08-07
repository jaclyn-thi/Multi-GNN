# Multi-dataset graph SSL compatibility audit

> Twin: [`results/diagnostics/multidataset_ssl_compatibility_audit.json`](../results/diagnostics/multidataset_ssl_compatibility_audit.json)  
> **AUDIT ONLY** — exploratory / planning. No trainers, adapters, or jobs implemented.  
> Git: `48bcb512415b45a2d1922fc2313bbc3cb065947e`  
> Future compute: standard `mit_preemptable` / `mit_general` / `qos=normal` (MIT advanced account expired).

---

## Executive answers (required)

| # | Question | Answer |
|---|----------|--------|
| 1 | Can AMLWorld and SAML-D use the same raw/model adapter without semantic mismatch? | **Geometry yes; categorical semantics no.** Same CSV schema and base-4→ports+TDS→`edge_dim=8` path. Slots **Received Currency** and **Payment Format** are **local integer codes** (AMLWorld vs SAML vocabularies are not aligned). Do not claim semantic currency/payment transfer. Safe joint path: **shared GIN on dim-8 + dataset-specific train-fit z-norm + domain BN**, with categorical non-equivalence disclosed. |
| 2 | Can all three TF targets be computed causally on both? | **Yes in principle** (definitions need only Timestamp / from_id / amount). **SAML-D cache builder not wired** (`SUPPORTED_DATASETS` = Small-HI, Small-LI, PaySim only). |
| 3 | Is a shared TF expert head valid? | **Conditionally yes** if targets are **per-dataset z-scored** (current MoE pattern) and heads predict standardized residuals. Absolute amount scales differ; shared head on raw amounts would be invalid. |
| 4 | What must remain dataset-specific? | Train-fit edge z-norm; BN running stats; TF target standardization; categorical code interpretation; split bucketing (`calendar_day` for both AML/SAML); data loaders/caches; domain loss calibration. |
| 5 | Can existing domain-BN support this? | **Pattern yes; N-domain no.** `scripts/joint_replay_scout.py` implements dual-buffer swap for hardcoded `aml`/`paysim`. Needs generalization to a `dict[domain→bn_bundle]`. Evidence: domain_bn ≫ shared_bn on PaySim retention. |
| 6 | Paired-domain loss vs alternating steps? | **“Mix the data” ≈ paired** (one batch/domain, averaged normalized losses, one step). **First smoke ≈ alternating** (already proven in joint replay; lower peak memory). |
| 7 | Loss normalization to avoid domination? | **Per-domain `LossNormState`** (epoch-1 means for InfoNCE and each TF MAE) then **equal average of domain totals** (or configured domain weights). Do not use one global μ across domains. |
| 8 | Smallest correct code change? | (a) Add SAML-D to TF causal cache builder; (b) generalize joint dual-BN + dual-loader schedule to `{Small-HI,SAML-D}`; (c) per-domain loss calib + domain BN; **no** universal column-union adapter. |
| 9 | First recommended full pair? | **AMLWorld Small-HI + SAML-D** |
| 10 | What blocks PaySim? | Semantic mismatch of `paysim_legacy_duplicate_v1` (type duplicated into currency **and** payment); hourly vs day splits; needs **dataset adapter** + restricted leakage-safe protocol (no balances / `isFlaggedFraud`). |
| 11 | AMLSim role? | **Future only** — raw `.7z` under `aml-data/aml-sim/`, no formatter/registry/loader. |
| 12 | Sources modified / jobs / test? | **No** source training code modified for this audit; **only** the two authorized artifacts written; **no** jobs; **no** test access; **no** NPZ/full-matrix loads. |

---

## Candidate representation (locked intent)

Shared GIN · R198 · InfoNCE on R198 · causal TF expert heads · adaptive TFMOE · projection off · TF as **targets not inputs** · dataset BN · train from scratch · labels out of SSL.

---

## A–B. Dataset compatibility cards (summary)

Canonical root: `data_config.json` → `aml-data/` → `{name}/formatted_transactions.csv`.  
Loader: `data_loading.get_data` · specs: `dataset_specs.get_dataset_spec` · splits: `dataset_splits.temporal_edge_split`.

Shared formatted columns: `EdgeID, from_id, to_id, Timestamp, Amount Sent, Sent Currency, Amount Received, Received Currency, Payment Format, Is Laundering`.  
Default encoder base (order): `[Timestamp, Amount Received, Received Currency, Payment Format]` → +`[in_port,out_port]` → +`[in_td,out_td]` ⇒ **edge_dim=8**.

| Dataset | Path | ~edges | Split | Adapter class | Notes |
|---------|------|-------:|-------|---------------|-------|
| **Small-HI** | `aml-data/Small-HI/` | ~5.08M | calendar_day 60/20/20 | **SAME_ADAPTER** (reference) | Train≈3.25M; SSL recipe ports+TDS+correct_reverse |
| **Small-LI** | `aml-data/Small-LI/` | ~6.92M | calendar_day | **SAME_ADAPTER** vs HI | Same IBM schema/vocab family |
| **Medium-HI/LI** | present | ~32M | calendar_day (assumed) | **SAME_ADAPTER** vs HI | No dedicated split audit found; scale risk |
| **SAML-D** | `aml-data/SAML-D/` | 9,504,852 | calendar_day; integrity train/val/test ≈5.71M/1.90M/1.90M | **SAME_ADAPTER geometry; categorical caveat** | Local currency/payment codes; protocol `samld_ssl_domain_v1` |
| **PaySim** | `aml-data/PaySim/` | 6,362,620 | hourly_step; 3.79M/1.28M/1.29M | **DATASET_ADAPTER_REQUIRED** | `paysim_legacy_duplicate_v1`; exclude balances + `isFlaggedFraud` |
| **AMLSim** | `aml-data/aml-sim/*.7z` | — | — | **CURRENTLY_BLOCKED** | No formatter |
| Elliptic/Ethereum | absent | — | — | future only | No loader |

**Labels in SSL:** must not enter loss, sampling, augmentations, normalization, or TF targets (current contrastive + TFMOE paths are label-free by construction).

### Encoder feature contract (dim-8) — do not equate by width alone

| Idx | Name | Small-HI / AMLWorld | SAML-D | PaySim legacy_duplicate |
|-----|------|---------------------|--------|-------------------------|
| 0 | Timestamp | seconds from start; z-norm | same units policy; z-norm | step×3600; z-norm |
| 1 | Amount Received | AML amount; z-norm | SAML amount; z-norm | PaySim amount; z-norm |
| 2 | Received Currency | AML currency code | **SAML-local** code | **PaySim type code (dup)** |
| 3 | Payment Format | AML payment code | **SAML-local** code | **same type code (dup)** |
| 4–5 | in/out port | structural | structural | structural |
| 6–7 | in/out td | structural | structural | structural |

Normalization: **`--train_fit_edge_znorm`** (inductive) must be **fit per dataset**.  
Flags for candidate SSL: `reverse_mp, ego, ports, emlps, tds, correct_reverse`; `preserve_seed_edges=false`; TF **not** concatenated into encoder inputs.

**PaySim `paysim_legacy_duplicate_v1` limitations:** places transaction **type** into both categorical AML slots; not real currency/payment; compatibility shim only. Thesis-facing mixed training needs an explicit adapter (e.g. type_only / structure_only or small MLP into common latent) and must keep balances / `isFlaggedFraud` out.

---

## C. Temporal expert compatibility (3 MoE targets)

Definitions: `morphology/temporal_flow_causal.py` · MoE subset: `direct_r198.TF_MOE_TARGET_NAMES`:

1. `log1p_sender_interarrival`  
2. `log1p_sender_past_7d_count` (W=604800 s)  
3. `log1p_amount_vs_sender_past_mean`

| Property | Status |
|----------|--------|
| Causal / past-only | Yes (tie policy B) |
| Label-free | Yes |
| Per-dataset generation | HI/LI/PaySim caches exist; **SAML-D blocked until builder allowlist + cache job** |
| Shared expert head | Valid **after per-dataset target z-score** (`load_tf_moe_context`) |
| Dataset-specific heads | Only if refusing shared standardized space |
| Amount/currency concerns | Relative log-ratio mitigates units; still standardize per domain |
| Missing history | Defined → 0 |

---

## D. Augmentation compatibility

`graph_augmentations.generate_views`: independent edge-drop (default rate 0.1); hetero reverse synced by `edge_id`; `preserve_seed_edges=false` for locked SSL; corrected reverse swaps port/TDS pairs.

Valid for all formatted graph datasets with hetero reverse_mp. No label use. Dataset differences are in base attrs only—augmentation mechanics transfer.

---

## E. BN and adapter policy

**Implementation:** `scripts/joint_replay_scout.py` — `extract_bn` / `apply_bn_` / `collect_bn_bundle` / `apply_bn_bundle_` / `save_joint_ckpt` with `bn_aml`/`bn_paysim` sidecars.  
**Evidence:** domain_bn PaySim H AUPRC≈0.041 vs shared_bn≈0.009 (`notes/joint_replay_scout_domain_bn.md`, `…_shared_bn.md`).  
**From-scratch safe:** yes if each domain’s BN is updated only on that domain’s batches.  
**N-domain minimal change:** `bn_bundles: Dict[str, Bundle]`; swap by `domain_id`; checkpoint all bundles.

| Dataset | Class | Adapter note |
|---------|-------|--------------|
| Small-HI / LI / Medium-* | SAME_ADAPTER | Shared AMLWorld schema |
| SAML-D | SAME_ADAPTER + categorical caveat | No MLP required for Stage 1–3 if domain BN + per-dataset z-norm; disclose non-aligned codes |
| PaySim | DATASET_ADAPTER_REQUIRED | Restricted features; linear/MLP into dim-8 or native→common latent; never balances/`isFlaggedFraud` |
| AMLSim | CURRENTLY_BLOCKED | Need formatter |

Avoid universal raw-column union with zero-fill.

---

## F. Mixed-training mechanics

| Design | Pros | Cons |
|--------|------|------|
| **Paired-domain update** | Closest to “mix”; simultaneous gradient from both | ~2× activations; need careful AMP/accum |
| **Alternating steps** | Matches `domain_schedule` in joint replay; lower peak mem | Weaker instantaneous mix; schedule bias |

**Recommendation:** Stage 2 smoke = **alternating 1:1** (reuse proven BN swap). Stage 3 scout prefer **paired normalized loss average** if memory allows.

**α/β:** currently **global per run** (`LearnedAlphaBeta`, `LossNormState`). For multi-domain: keep **global α/β initially** but **per-domain loss calibration**; monitor domain loss shares; escalate to per-domain α/β only if one domain’s TF/InfoNCE ratio systematically dominates.

**Minimum calibration policy:** for each domain \(d\), freeze epoch-1 means \(\mu_d^{\text{NCE}}, \mu_d^{\text{TF},k}\); optimize \(\frac{1}{|D|}\sum_d L_d^{\text{norm}}\) (or configured weights summing to 1).

---

## G. Infrastructure reuse

| Asset | Reuse | Hardcoded | Do not reuse blindly |
|-------|-------|-----------|----------------------|
| `joint_replay_scout.py` BN swap | Pattern | `aml`/`paysim` only; continuation from AML ckpt | Source-SHA lock; 500-step schedule; proj head (candidate wants proj off) |
| `sequential_aml_to_paysim_ssl_scout.py` | Finetune lessons | Two-stage forgetting study | Not mixed training |
| `direct_r198/*` TFMOE | Loss, heads, calib | Single-dataset; global α/β | — |
| `build_temporal_flow_causal_cache.py` | Builder | Allowlist excludes SAML-D | — |
| `feature_contracts.py` | PaySim contracts | PaySim-only gate | Do not apply AML contracts to SAML |
| `training.train_gnn` | Single-domain SSL | One `--data` | Needs outer multi-domain loop |
| Frozen transfer scout | Eval protocol | Extract/probe only | Not a trainer |

---

## H. Pairwise verdicts

| Pair | Verdict | Shared contract | Adapter | BN | TF | Risk | Blocker | Complexity |
|------|---------|-----------------|---------|----|----|------|---------|------------|
| **Small-HI + SAML-D** | **GO** (TF cache REPAIRABLE) | dim-8 ports+TDS+correct_rev; train-fit | Geometry shared; categoricals disclosed | Domain BN N=2 | Build SAML-D cache; per-domain z-score; shared heads OK | Silent categorical equating | SAML-D not in TF allowlist | **medium** |
| Small-HI + PaySim | **REPAIRABLE** | dim-8 geometry | **PaySim adapter required** | Domain BN | PaySim cache exists | Type-duplication semantics | Adapter design + protocol lock | **medium–large** |
| Small-HI + Small-LI | **GO** | identical family | SAME | Domain BN optional but recommended | Both caches supported | Prevalence / size imbalance | None material | **small** |
| Small-HI + Medium-HI/LI | **REPAIRABLE** | same schema | SAME | Domain BN | Need Medium TF caches | Memory / epoch cost | Scale + cache | **large** (ops) |
| Small-HI + AMLSim | **BLOCKED** | — | — | — | — | No data path | Formatter/registry | — |
| HI + SAML-D + PaySim | **BLOCKED→later** | dim-8 + PaySim adapter | PaySim adapter + SAML caveat | Domain BN N=3 | All three caches | Three-way domination | Pass 2-domain gate first | **large** |

Primary scientific risk for HI+SAML: treating SAML currency/payment codes as AML codes. Mitigate with disclosure + domain BN + per-dataset z-norm (not by inventing a fake shared vocab).

---

## I. Ranked staged plan

### Stage 1 — unit test (CPU, no full train)
- Pair: **Small-HI + SAML-D**
- Build tiny synthetic or `max_batches=1` dual loaders
- Assert: `edge_dim==8` both; BN swap changes `running_mean` keys; losses finite; **no label** in SSL graph; TF targets shape `[N,3]` after SAML cache stub or mock
- Expected: shared `GINe` parameter IDs updated by both domains’ backward

### Stage 2 — one GPU smoke (≤1 job, 20–50 steps)
- Alternating 1:1; domain BN; proj **off**; InfoNCE±TFMOE if SAML cache ready else InfoNCE-only smoke first
- Checks: finite loss/grad; both domains nonzero grad norm into shared encoder; BN bundles diverge; checkpoint save/reload restores both BNs
- Train/val only; **no** full extract; standard account

### Stage 3 — seed-2 validation scout
- After smoke: ~few epochs or fixed step budget with paired or alternating mix
- Per-domain val ranking probes (or proxy SSL metrics); controls: single-domain Small-HI, single-domain SAML-D, matched random
- Stop if one domain’s normalized loss share >~0.85 sustained or BN collapse

### Stage 4 — PaySim extension
- Restricted protocol; no balances/`isFlaggedFraud`; explicit adapter; domain BN N=3 prep as N=2+1

### Stage 5 — three-domain
- Only after HI+SAML gate passes

No large DAG in this audit.

---

## Evidence read (non-exhaustive)

`dataset_specs.py`, `feature_contracts.py`, `data_loading.py`, `data_util.py`, `dataset_splits.py`, `morphology/temporal_flow_causal.py`, `scripts/build_temporal_flow_causal_cache.py`, `scripts/joint_replay_scout.py`, `direct_r198/__init__.py`, `graph_augmentations.py`, `format_*.py`, `notes/multidataset_next_steps_preflight.md`, `notes/samld_protocol_and_integrity.md`, `notes/joint_replay_scout_{domain,shared}_bn.md`, `notes/expert_only_frozen_transfer_samld_paysim_preflight.md`, `results/diagnostics/joint_replay_scout/domain_bn.json`, `data_config.json`.

---

## Stop

Human review gate. **Do not implement adapters or multi-dataset trainer until reviewed.**
