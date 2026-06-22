# Development results (not frozen benchmarks)

Quick-run numbers for internal comparison while configs and code still change. **Not** a formal evaluation protocol — use fixed recipes for papers and PI updates once the stack stabilizes.

**Outputs:** `embeddings/{unique_name}/probe_results.json` · PaySim: `embeddings/paysim/{unique_name}/probe_results*.json`

---

## Recommended configs (Jun 2026)

**Full-label frozen probe (AUROC):** **8192neg + projection with queue disabled** is the current strongest asym AUROC recipe (**0.951 AUROC**, **0.233 F1**). Across seeds, `same_pair` false-negative filtering is the leading F1/recall variant (**0.236 mean F1**, **0.359 mean recall**) with a small AUROC tradeoff.

**Label-efficiency:** sym+proj best @ **10%** labels (0.924 AUROC); 8192neg+proj best @ **50–100%** (0.931). See [label-efficiency](#label-efficiency-small-hi) below.

**Morphology expert:** **M1b** @ 20 ep — best morph-only (0.920 AUROC). MAE expert loss did not beat MSE. Stacking BC on M1b hurts; M5a grouped heads did not fix interference (0.887 AUROC).

**Projection ablation:** [`projection-head-ablation-jun2026.md`](projection-head-ablation-jun2026.md). Earlier best contrastive+proj AUROC before the queue sweep: asym + 8192 negs (**0.930**). Earlier best F1: sym + 1024 @ `bs=16384` (**0.222**).

**Queue / negative ablations:** for asym + projection + 8192 negatives + `bs=8192 accum=4`, disabling the queue is best. Larger queues reduced AUROC/F1; increasing negatives beyond 8192 did not improve AUROC, even with false-negative filtering. See [queue and negative ablations](#queue-and-negative-ablations-small-hi) below.

**Temperature:** lower InfoNCE temperatures (`0.05`, `0.10`, `0.20`) underperformed the prior default `0.5`; keep `--contrastive_temperature 0.5` for the current recipe.

**Multi-positive InfoNCE:** endpoint weak-positive runs underperformed exclusion-only filtering. `same_pair` weak positives at weight `0.1` averaged **0.938 AUROC / 0.153 F1**; lowering the weight to `0.05` helped but still did not beat no-filter or `same_pair` exclusion.

---

## Small-HI SSL benchmark (linear probe, val-tuned F1, GIN hetero)

| Run | Config | Epochs | Test AUROC | Test F1 |
|-----|--------|--------|------------|---------|
| Contrastive + proj, **8192 negs, no queue** | asym; `queue=0`; `bs=8192 accum=4` | 20 → **ep 19** | **0.951** | **0.233** |
| Contrastive + proj, **same-pair filter** | asym; 8192 negs; `queue=0`; mean over 3 seeds | 20 | 0.942 | **0.236** |
| Contrastive + proj, **same-endpoint filter** | asym; 8192 negs; `queue=0`; seed 1 only | 20 | 0.940 | 0.277 |
| Contrastive + proj, **8192 negs, queued** | asym; 8192 negs + queue; `bs=8192 accum=4` | 20 | **0.930** | 0.191 |
| **M1b + clustering + projection** | M1b + 11 local + `--contrast_projection_head` | 20 | **0.929** | 0.156 |
| M1b + sym + projection | M1b + 14 local + sym @ `bs=16384` | 20 | **0.930** | 0.134 |
| M1b + clustering + triangles + proj | 14 local (`--morph_local_subset all`) | 20 | 0.912 | 0.145 |
| Contrastive + proj, **symmetric** | sym InfoNCE + proj; `bs=16384 accum=2` | 20 | **0.929** | **0.222** |
| Contrastive + proj, asym @ 16384 | confound control; 1024 negs | 20 | 0.920 | 0.206 |
| Contrastive + projection | asym InfoNCE + proj; `bs=32768` | 20 | 0.927 | 0.144 |
| M1b + projection | M1b + projection head | 20 → **ep 15** | 0.924 | 0.096 |
| M1b | `--morph_expert`, `local+global` | 20 | 0.920 | 0.108 |
| M1b + MAE expert | `--morph_expert_loss mae` | 20 | 0.898 | 0.145 |
| M1b + clustering (MSE) | M1b, 11 local dims | 20 | 0.903 | 0.117 |
| M3 BC-only | `local+tier2` | 20 | 0.904 | 0.093 |
| M2 expert + contrast | + `--checkpoint_policy best` | 10 → **ep 9** | 0.906 | 0.058 |
| M2 expert + contrast | + `--checkpoint_policy best` | 20 | 0.891 | 0.107 |
| M1 | `--morph_expert`, `local` | 20 | 0.910 | 0.079 |
| M3 M1b + BC (4 lift cols) | `local+global+tier2`, best ckpt | 20 → **ep 14** | 0.896 | 0.033 |
| M3 M1b + bc_max | `lift max` | 20 | 0.889 | 0.086 |
| M5a grouped BC | `layout=grouped`, `w_tier2=1` | 20 | 0.887 | 0.028 |
| M2 expert + contrast | `morph_expert_weight=0.5` | 10 | 0.876 | 0.027 |
| M3 M1b + BC (last epoch) | same, no M4 | 20 | 0.861 | 0.029 |
| Contrastive baseline | identity InfoNCE | 20 | 0.839 | 0.076 |
| M2 expert + contrast | M1b + `--morph_contrast` | 20 (last) | 0.864 | 0.025 |
| M2 contrast only | `--morph_contrast` (no expert) | 10 | 0.680 | 0.012 |
| Supervised CE (in-GNN) | `--objective supervised` | — | ~0.972 | ~0.493 |

More detail: [`morphology-metrics-plan.md`](morphology-metrics-plan.md) · typology diagnostics: [`downstream-eval-plan.md`](downstream-eval-plan.md).

---

## Queue and negative ablations (Small-HI)

Frozen linear probe after asym contrastive + projection, 8192 negatives, `bs=8192`, `accum=4`, 20 epochs.

| Queue size | Test AUROC | Test F1 | Precision | Recall |
|------------|------------|---------|-----------|--------|
| `0` | **0.951** | **0.233** | **0.209** | 0.263 |
| `8192` | 0.923 | 0.167 | 0.110 | **0.348** |
| `16384` | 0.930 | 0.165 | 0.117 | 0.282 |
| `32768` | 0.929 | 0.179 | 0.123 | 0.330 |
| `65536` | 0.922 | 0.143 | 0.102 | 0.238 |
| `131072` | 0.915 | 0.148 | 0.119 | 0.194 |

Interpretation: the queue increases contrastive difficulty but does not improve downstream probe quality in this setup. Prefer `--contrastive_memory_bank_size 0` for the current asym + projection recipe unless explicitly testing queue behavior.

### No-queue seed averages

| Mode | Mean AUROC | Mean F1 | Mean precision | Mean recall | Note |
|------|------------|---------|----------------|-------------|------|
| no filter | **0.946** | 0.216 | 0.168 | 0.319 | AUROC baseline |
| `same_pair` | 0.942 | **0.236** | **0.176** | **0.359** | leading filter candidate |
| `same_receiver` | 0.944 | 0.193 | 0.147 | 0.349 | unstable across seeds |
| `same_endpoint` | 0.938 | 0.240 | 0.203 | 0.297 | mixed replication; seed 1 was high |

`same_pair` is the cleanest false-negative filter so far: it improves mean F1/recall over no-filter while preserving most AUROC. `same_receiver` produced one strong AUROC run but was unstable; `same_endpoint` improved precision/F1 on average but did not cleanly replicate the seed-1 jump.

### Larger negative / queue scouts with filtering

| Run | Test AUROC | Test F1 | Precision | Recall | Interpretation |
|-----|------------|---------|-----------|--------|----------------|
| `same_pair`, `queue=32768` | 0.935 | 0.185 | 0.142 | 0.266 | queue still hurts |
| `same_receiver`, `queue=32768` | 0.916 | 0.108 | 0.066 | 0.284 | queue hurts strongly |
| `same_pair`, `10240` negs, `queue=0` | 0.938 | 0.156 | 0.105 | 0.309 | more negatives hurt |
| `same_receiver`, `10240` negs, `queue=0` | 0.929 | 0.213 | 0.192 | 0.239 | no clear benefit |

The direct `12288` run at `bs=8192 accum=4` OOMed even with `queue=0`; reducing to `bs=4096 accum=8` made it fit but did not improve metrics. Current evidence favors `8192` negatives, `queue=0`, and optional `same_pair` filtering.

### Temperature sweep

All runs use asym contrastive + projection, 8192 negatives, `queue=0`,
`bs=8192`, `accum=4`, seed 1.

| Temperature | Test AUROC | Test F1 | Precision | Recall | Interpretation |
|-------------|------------|---------|-----------|--------|----------------|
| `0.05` | 0.898 | 0.103 | 0.070 | 0.189 | too sharp |
| `0.10` | 0.933 | 0.139 | 0.092 | 0.289 | below baseline |
| `0.20` | 0.942 | 0.198 | 0.149 | 0.299 | closest lower-temp run |
| `0.50` | **0.951** | **0.233** | **0.209** | 0.263 | current default/best |

Lower temperatures improved neither AUROC nor F1. Retain
`--contrastive_temperature 0.5` for the current best recipe.

### Symmetric memory rescue

Symmetric + projection with `8192` negatives, `queue=0`, `bs=4096`, and
`accum=8` fit without OOM, but underperformed the asym baseline
(0.931 AUROC / 0.161 F1). This is a viable fit recipe, not a current winner.

### Morphology group diagnostics

`morph_group_diag_full_10ep` verified the new semantic group loss logging with
the shared M1b expert and no M2/FNF/multi-positive changes. Final group MSEs
were lowest for temporal/other targets and highest for `local_motif`; downstream
probe quality was diagnostic-only (0.934 AUROC / 0.079 F1). Next targeted scouts
use `--morph_target_groups degree_fan`, `local_motif`, and
`degree_fan,local_motif`.

### Multi-positive InfoNCE scouts

Multi-positive runs use weak endpoint/pair positives in the InfoNCE numerator;
identity positives remain weight `1.0`.

| Run | Test AUROC | Test F1 | Precision | Recall | Interpretation |
|-----|------------|---------|-----------|--------|----------------|
| `same_pair`, weight `0.1`, mean seeds 1–3 | 0.938 | 0.153 | 0.114 | 0.236 | weak |
| `same_pair`, weight `0.02`, seed 1 | 0.943 | 0.108 | 0.066 | 0.292 | too weak/noisy |
| `same_pair`, weight `0.05`, seed 1 | 0.944 | 0.224 | 0.183 | 0.287 | improved, still below filter |
| `same_receiver`, weight `0.1`, mean seeds 1–3 | 0.940 | 0.203 | 0.156 | 0.298 | below no-filter |

Endpoint/pair relationships appear more useful as **negative-exclusion rules**
than as weak positives in the numerator. If revisiting multi-positive InfoNCE,
start from a smaller and more targeted design rather than combining it with
queues or larger negative counts.

---

## Label-efficiency (Small-HI)

Stratified train-label fractions; threshold tuned on full val. Source: `embeddings/label_efficiency_summary.json`.

| Train labels | sym+proj | 8192neg+proj | clustering+proj | M1b+proj | Contrastive+proj |
|--------------|----------|--------------|-----------------|----------|------------------|
| 10% | **0.924** | 0.917 | 0.916 | 0.918 | 0.906 |
| 25% | **0.926** | 0.922 | **0.926** | 0.922 | 0.918 |
| 50% | 0.929 | **0.931** | 0.930 | 0.919 | 0.925 |
| 100% | 0.929 | **0.931** | 0.929 | 0.922 | 0.928 |

Full tables: [`morphology-metrics-plan.md` § Label-efficiency](morphology-metrics-plan.md).

---

## PaySim transfer (external fraud)

Frozen GIN extract + linear probe on ~6.36M edges. Canonical probe: `--class_weight model`.

| Encoder | Threshold | Test AUROC | Test F1 |
| ------- | --------- | ---------- | ------- |
| `hi_contrastive_proj_sym_20ep_bestckpt` | val-tuned | **0.866** | 0.089 |
| `hi_contrastive_proj_sym_20ep_bestckpt` | fixed 0.5 | **0.864** | **0.127** |
| `random_init_gin` | val-tuned / 0.5 | 0.730 | 0.135–0.143 |

In-domain Small-HI (same encoder): AUROC **0.929**, F1 **0.222** @ val-tuned.

Full tables and probe variants: [`downstream-eval-plan.md` § PaySim](downstream-eval-plan.md#paysim--status-jun-2026).
