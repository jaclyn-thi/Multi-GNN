# PaySim native Multi-GIN formal seed-2 — submission

Authorized by passed smoke (`results/diagnostics/paysim_native_multigin_core_v1_smoke.json`, `gate_pass=true`, max val AUPRC 0.6665).

## Walltime projection

| Quantity | Value |
|----------|------:|
| Smoke wall (2 ep, job 19123387) | 10.5 min |
| Approx fixed overhead (ports/load) | ~8.5 min |
| Approx per-epoch train+val | ~0.5 min |
| 50-ep projection | ~33.5 min |
| +15% / +50% / 3× safety | ~38.5 / 50 / 100 min |
| Requested wall | **06:00:00** (`mit_normal_gpu` MaxTime) |
| Fits? | **Yes** (≫3× margin) |

## Jobs (exactly two)

| Role | Job ID | Dependency |
|------|--------|------------|
| Train 50 ep (`skip_test_eval`) | `19124783` | — |
| Eval (best-val ckpt; test once) | `19124784` | `afterok:19124783` |

## Protocol (locked to smoke)

- run: `paysim_native_multigin_core_v1_formal_seed2`
- seed 2; legacy head; `paysim_native_multigin_core_v1` (edge_dim=13)
- train-fit continuous z-norm; scaler SHA `45ce032c08ae0f3ef73f11f3a778bbc351da7bd43b3316ab583c941d4bcbae27`
- ports + emlps + reverse_mp + ego; TDS/preserve/correct_reverse off; inherited legacy reverse
- Adam / lr / class weights ~(1.0, 6.275); batch 8192
- save best-val-F1 + last; record max-val-AUPRC epoch diagnostically
- no balance deltas / isFlaggedFraud; no HP sweep; no seeds 1/3

## Exact train command

```bash
python main.py \
  --data PaySim --model gin --objective supervised --supervised_head legacy \
  --feature_contract paysim_native_multigin_core_v1 --train_fit_edge_znorm \
  --unique_name paysim_native_multigin_core_v1_formal_seed2 \
  --seed 2 --n_epochs 50 --batch_size 8192 --num_neighs 100 100 \
  --loader_num_workers 16 --reverse_mp --ego --ports --emlps \
  --save_model --skip_test_eval --tqdm
```

## Hashes verified pre-submit

- train/val index SHA match prior audits
- smoke scaler SHA `45ce032c…`
- unique run name ⇒ no overwrite of smoke / tabular / balance-free / failure-audit artifacts
- focused tests: NeighborLoader −2 sentinel + skip_test NaN placeholders + contract (7 passed)

## Outputs (on completion)

- `notes/paysim_native_multigin_core_v1_formal_seed2.md`
- `results/diagnostics/paysim_native_multigin_core_v1_formal_seed2.json`
- `results/diagnostics/paysim_native_multigin_core_v1_formal_seed2/train_finalize.json`
- `saved-models/paysim_native_multigin_core_v1_formal_seed2/`
- registry append-only

Twin JSON: `results/diagnostics/paysim_native_multigin_core_v1_formal_seed2_submission.json`
