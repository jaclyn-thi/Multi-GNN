# Small-LI Legacy Supervised GINe Scout

**Status:** run log. **Superseded for reproduction claims** by the formal 100-epoch run — see [`supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.md`](supervised_Small-LI_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.md) (test F1 **0.357**, AUPRC **0.292** @ best-val ep 35). This scout remains useful as the integration check that preceded the formal run.

**Not paper-comparable** (20-epoch `--testing` scout, single seed). Integration check for the restored legacy supervised head before a full reproduction.

## What this run tested

The `--supervised_head` lever added two supervised heads that share the same encoder, graph flags, and data handling:

| `--supervised_head` | Classifier | Meaning |
|---------------------|-----------|---------|
| `embedding` (default) | edge rep → 128-d embedding → `Linear(128,50)→Linear(50,2)` | current embedding-head control (project modification) |
| `legacy` | edge rep → `Linear(3·h,50)→Linear(50,25)→Linear(25,2)` | IBM Multi-GNN / Egressy et al. head (fork-point `fc751e8`), numerically validated for GINe |

This scout ran the **legacy** head on Small-LI to confirm it works end-to-end and produces sensible numbers.

- **Run:** `small_li_legacy_supervised_gin_emlps_tds_seed1_scout` (Slurm job `17248077`, COMPLETED, 2h13m)
- **Config:** GINe, `--reverse_mp --ego --ports --emlps --tds`, seed 1, 20 epochs, batch 8192, CE weights (1.0, 6.275), `--testing`
- **Selection:** best validation minority-class F1, argmax over two-class logits (no threshold tuning)

## Result (paper-compatible `paper_argmax`)

Reproduction metric = test F1 at the **best-validation** epoch (11), read from `checkpoint_best_val_f1.tar`:

| | Val | Test |
|-|----:|-----:|
| F1 (argmax) @ best-val epoch 11 | **0.2424** | **0.1773** |
| F1 (argmax), post-hoc eval-mode | 0.2406 | **0.2018** |
| AUROC | 0.976 | 0.944 |
| AUPRC | 0.181 | 0.191 |

- **Final epoch (20)** test argmax F1 = **0.2293**; best vs final differ substantially (|Δ|≈0.052), so reproduction eval must use the best-val checkpoint, not the flat/last one.
- **19/20 epochs** had nonzero test argmax F1 (peak 0.262 @ epoch 6, but val there was lower, so epoch 11 wins on the selection rule).

## Legacy vs the old embedding-head "baseline"

Same flags/seed/epochs; only the head differs. The old embedding-head run (`small_li_supervised_gin_emlps_tds_seed1`) is **not** the Egressy baseline (see [`small_li_supervised_baseline_comparison.md`](small_li_supervised_baseline_comparison.md)).

| Run | Head | Checkpoint | Test F1 (argmax) | Test AUPRC | P@500 |
|-----|------|-----------|-----------------:|-----------:|------:|
| Prior embedding baseline | embedding | final epoch | 0.0000 | 0.006 | 0.002 |
| **This scout** | legacy | best-val (ep 11) | **0.18–0.20** | **0.19** | **0.316** |

The embedding head collapsed to all-zero argmax F1 on its final checkpoint; the legacy head keeps usable predictions and much stronger ranking/alert metrics, i.e. closer to the expected upstream supervised behavior.

## Reproduce / extend

```bash
python main.py --data Small-LI --model gin \
  --objective supervised --supervised_head legacy \
  --unique_name <run> --save_model \
  --reverse_mp --ego --ports --emlps --tds \
  --n_epochs 20 --batch_size 8192 --num_neighs 100 100 --seed 1 --testing

python scripts/evaluate_supervised_gnn.py --data Small-LI --model gin \
  --objective supervised --supervised_head legacy --unique_name <run> \
  --reverse_mp --ego --ports --emlps --tds \
  --output_json results/diagnostics/eval_<run>.json --output_md notes/eval_<run>.md
```

Artifacts: `results/diagnostics/supervised_Small-LI_<run>_{epoch_history,summary}.json`, `notes/supervised_Small-LI_<run>_summary.md`, checkpoints under `saved-models/<run>/`.

## Before a full reproduction claim

Required to move from "scout" to "configured to reproduce the corresponding Egressy et al. setup": drop `--testing`, use the upstream epoch count (100) and seed policy, confirm the data split matches upstream, and compare against the specific Egressy et al. Small-LI GIN row. Operational note: ~6.5 min/epoch → ~11h for 100 epochs, which exceeds `mit_normal_gpu`'s 6h cap (needs a longer QOS or checkpoint-resume).
