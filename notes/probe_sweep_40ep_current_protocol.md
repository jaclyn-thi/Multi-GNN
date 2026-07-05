# 40 ep targeted probe sweep (Jul 2)

**Context:** checkpointed CPU probe grid on **frozen** GIN 40 ep embeddings (seeds 1–4). Tests feature modes × class weights × regularization — no SSL retrain.

**Start here for takeaways:** [`notes/results.md` § Jul 2](results.md#40-ep-targeted-probe-sweep-jul-2). **Fair comparisons:** prefer **`cw=model`, C=1.0** (shared GIN class weights). Tables below include all grid cells; **`cw=none` “best per seed” rows can inflate val-tuned F1** via extreme thresholds — check F1@0.5.

**Reproduce:** `sbatch slurm/run_probe_sweep_40ep_seeds_checkpointed.sh` · summarize: `python scripts/summarize_probe_sweep_40ep_current_protocol.py`

---

Completed cells merged: **78**

## Best per seed (test F1, any cw/C — check F1@0.5 before trusting)

| Run | Features | cw | C | AUROC | AUPRC | F1 | F1@0.5 | Thr |
|-----|----------|----|---|------:|------:|---:|-------:|----:|
| GINe emlps+tds seed1 (40ep) | `embedding` | none | 0.1 | 0.9487 | 0.2324 | 0.3175 | 0.1274 | 0.1191 |
| GINe emlps+tds seed2 (40ep) | `embedding+raw` | none | 1.0 | 0.9551 | 0.3314 | 0.3677 | 0.3627 | 0.1987 |
| GINe emlps+tds seed3 (40ep) | `embedding+raw+morph` | none | 0.1 | 0.9513 | 0.2440 | 0.2642 | 0.2862 | 0.2856 |
| GINe emlps+tds seed4 (40ep) | `embedding+raw+morph` | none | 1.0 | 0.9381 | 0.2674 | 0.3073 | 0.2880 | 0.1226 |

## Shared setting @ cw=model, C=1.0 (mean ± std over seeds)

- **`embedding`** (n=4): F1 0.2642 ± 0.0322, AUPRC 0.1921 ± 0.0346, F1@0.5 0.2652 ± 0.0337
- **`embedding+raw`** (n=4): F1 0.2731 ± 0.0475, AUPRC 0.2672 ± 0.0365, F1@0.5 0.1899 ± 0.0892
- **`embedding+raw+morph`** (n=4): F1 0.2478 ± 0.0359, AUPRC 0.2292 ± 0.0264, F1@0.5 0.2389 ± 0.0646

## `embedding+raw` vs `embedding+raw+morph` win counts (F1)

| Seed | pairs | raw wins F1 | morph wins F1 | raw wins AUPRC |
|------|------:|------------:|--------------:|---------------:|
| gin_40ep_seed1 | 6 | 2 | 4 | 6 |
| gin_40ep_seed2 | 6 | 6 | 0 | 6 |
| gin_40ep_seed3 | 6 | 4 | 2 | 6 |
| gin_40ep_seed4 | 6 | 0 | 6 | 3 |

## Reference comparisons (prior ablations)

| Label | F1 | AUPRC | F1@0.5 |
|-------|---:|------:|-------:|
| GIN 20ep seed1 embedding | 0.2593 | 0.2133 | 0.2573 |
| GIN 20ep seed1 embedding+raw+morph | 0.2982 | 0.2755 | 0.3271 |
| FNF seed1 embedding+raw+morph | 0.3188 | 0.2763 | 0.3031 |
| FNF seed2 embedding+raw+morph | 0.2624 | 0.2425 | 0.1700 |

## Top 10 test F1

| Run | Features | cw | C | AUROC | AUPRC | F1 | F1@0.5 | Thr |
|-----|----------|----|---|------:|------:|---:|-------:|----:|
| GINe emlps+tds seed2 (40ep) | `embedding+raw` | none | 1.0 | 0.9551 | 0.3314 | 0.3677 | 0.3627 | 0.1987 |
| GINe emlps+tds seed2 (40ep) | `embedding+raw` | none | 0.1 | 0.9563 | 0.3263 | 0.3646 | 0.3612 | 0.2392 |
| GINe emlps+tds seed2 (40ep) | `embedding+raw` | none | 10.0 | 0.9535 | 0.3278 | 0.3591 | 0.3494 | 0.1792 |
| GINe emlps+tds seed2 (40ep) | `embedding+raw` | model | 0.1 | 0.9570 | 0.2984 | 0.3521 | 0.3284 | 0.6101 |
| GINe emlps+tds seed2 (40ep) | `embedding+raw` | model | 1.0 | 0.9553 | 0.2883 | 0.3464 | 0.3392 | 0.5544 |
| GINe emlps+tds seed2 (40ep) | `embedding+raw` | model | 10.0 | 0.9557 | 0.2860 | 0.3456 | 0.3320 | 0.5515 |
| GINe emlps+tds seed2 (40ep) | `embedding` | none | 10.0 | 0.9459 | 0.2773 | 0.3207 | 0.2087 | 0.0995 |
| GINe emlps+tds seed1 (40ep) | `embedding` | none | 0.1 | 0.9487 | 0.2324 | 0.3175 | 0.1274 | 0.1191 |
| GINe emlps+tds seed2 (40ep) | `embedding` | none | 1.0 | 0.9456 | 0.2717 | 0.3167 | 0.1896 | 0.0995 |
| GINe emlps+tds seed1 (40ep) | `embedding+raw+morph` | none | 0.1 | 0.9470 | 0.2962 | 0.3124 | 0.3482 | 0.2126 |
