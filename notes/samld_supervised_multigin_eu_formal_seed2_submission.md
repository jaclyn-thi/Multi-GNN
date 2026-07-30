# SAML-D formal seed-2 submission

**Status: SUBMITTED** on `mit_preemptable` (user-authorized; MaxTime 48h).

## Jobs

| Role | Job ID | Dependency |
|------|--------|------------|
| Train (50 ep, skip_test_eval) | **19117881** | — |
| Eval (best-val ckpt; val+test once) | **19117882** | `afterok:19117881` |

Duplicate train **19117883** was submitted by a broken heredoc and **cancelled immediately**.

## Runtime

- Projection ~**8.82 h**; request **14:00:00** (≥15% margin vs ~10.14 h required)
- Peak CPU RSS ~16.6 GiB (smoke); `--mem=128G`, `--gres=gpu:1`

## Commands

```bash
sbatch slurm/run_samld_supervised_multigin_eu_v1_formal_seed2.sh
# -> 19117881
sbatch --dependency=afterok:19117881 --export=ALL,TRAIN_JOB_ID=19117881 \
  slurm/eval_samld_supervised_multigin_eu_v1_formal_seed2.sh
# -> 19117882
```

## Training flags (test locked)

```
python main.py --data SAML-D --model gin --objective supervised \
  --supervised_head legacy --unique_name samld_supervised_multigin_eu_v1_formal_seed2 \
  --seed 2 --n_epochs 50 --batch_size 4096 --num_neighs 100 100 --loader_num_workers 16 \
  --reverse_mp --ego --ports --emlps --save_model --skip_test_eval --tqdm
```

## Paths

- best: `saved-models/samld_supervised_multigin_eu_v1_formal_seed2/checkpoint_best_val_f1.tar`
- last: `saved-models/samld_supervised_multigin_eu_v1_formal_seed2/checkpoint_last.tar`
- history: `results/diagnostics/supervised_SAML-D_samld_supervised_multigin_eu_v1_formal_seed2_epoch_history.json`
- formal pack (written by eval): `notes/samld_supervised_multigin_eu_formal_seed2.md`, `results/diagnostics/samld_supervised_multigin_eu_formal_seed2.json`

## Confirmations

- Test inaccessible to **training** job (`--skip_test_eval`)
- Eval blocked by `afterok` if train fails
- Active jobs: exactly train **19117881** + eval **19117882** (duplicate 19117883 cancelled)
- Gates not rerun
- git HEAD: 9298d2d1ea03d6121c4311d6b8a67a9be97caced
