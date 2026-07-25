# Batch-size ablation E vs F — preflight (seed 2)

**Status (2026-07-22):** scripts ready; Slurm controller unreachable at submit time. Auto-retry armed.

## Scientific question

Does reducing the seed batch from 8192 → 2048 improve representation quality when optimizer-update frequency and all-in-batch negative semantics are controlled?

## Matched configs (D recipe + all-neg)

| | E | F |
|---|---|---|
| `batch_size` | 8192 | 2048 |
| `contrastive_accum_steps` | 4 | 16 |
| requested anchors / opt update | 32768 | 32768 |
| `contrastive_num_neg_samples` | **0** (all aligned current-batch) | **0** |
| queue | 0 | 0 |
| epochs | 10 | 10 |
| seed | 2 | 2 |
| corrected reverse TDS | on | on |
| preserve_seed_edges | on | on |
| asymmetric + projection | on | on |
| KNN / morph / structural / 3rd view | off | off |

**Diff E vs F:** batch size and accumulation only.

## Preflight checks

- Artifact paths unique (no collisions) for both run names
- `8192×4 = 2048×16 = 32768`
- All-neg mode: `num_neg=0` → `num_neg_samples=None` → all aligned current-batch candidates (chunked logsumexp); no duplicate resampling
- Projected optimizer steps/epoch ≈ 100 for both; examples/epoch equal (~3.25M train seeds)
- Shared seeds ≈ batch size under preserve_seed (D observed 8191/8192)
- Runtime: D was ~4.3 min/epoch at bs8192; 10ep + extract/probe within 6h Advanced GPU budget
- Advanced GPU Slurm flags preserved (`mit_normal_gpu` / `mit_amf_advanced_gpu`, 64G, 16c, 1 GPU, 6h)

## Submit

```bash
sbatch --export=ALL,VARIANT=E --job-name=bs_allneg_E \
  slurm/ablation_batchsize_corrected_preserve_allneg_10ep_seed2.sh
sbatch --export=ALL,VARIANT=F --job-name=bs_allneg_F \
  slurm/ablation_batchsize_corrected_preserve_allneg_10ep_seed2.sh
```

## After both complete

```bash
python scripts/summarize_batchsize_ablation_E_vs_F.py
```

Writes `notes/batchsize_ablation_E_vs_F_seed2.md` and matching JSON. Do **not** auto-resume to 40 epochs.
