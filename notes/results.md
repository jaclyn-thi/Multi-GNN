# Development results (not frozen benchmarks)

Quick-run numbers for internal comparison while configs and code still change. **Not** a formal evaluation protocol — use fixed recipes for papers and PI updates once the stack stabilizes.

**Outputs:** `embeddings/{unique_name}/probe_results.json` · PaySim: `embeddings/paysim/{unique_name}/probe_results*.json`

---

## Recommended configs (Jun 2026)

**Full-label frozen probe (AUROC):** **8192neg** or **M1b + sym + proj** (**0.930**); clustering+proj **0.929**. **Best F1 among SSL:** sym contrastive+proj only (**0.222**) — stacking morph on sym drops F1 to **0.134**.

**Label-efficiency:** sym+proj best @ **10%** labels (0.924 AUROC); 8192neg+proj best @ **50–100%** (0.931). See [label-efficiency](#label-efficiency-small-hi) below.

**Morphology expert:** **M1b** @ 20 ep — best morph-only (0.920 AUROC). MAE expert loss did not beat MSE. Stacking BC on M1b hurts; M5a grouped heads did not fix interference (0.887 AUROC).

**Projection ablation:** [`projection-head-ablation-jun2026.md`](projection-head-ablation-jun2026.md). Best contrastive+proj AUROC: asym + 8192 negs (**0.930**). Best F1: sym + 1024 @ `bs=16384` (**0.222**).

---

## Small-HI SSL benchmark (linear probe, val-tuned F1, GIN hetero)

| Run | Config | Epochs | Test AUROC | Test F1 |
|-----|--------|--------|------------|---------|
| Contrastive + proj, **8192 negs** | asym; 8192 negs + queue; `bs=8192 accum=4` | 20 | **0.930** | 0.191 |
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
