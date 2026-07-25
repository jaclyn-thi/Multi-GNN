# Edge D+ neighbor-positive 10ep scout

**NOT an exact GCPAL reproduction.** Matched identity poscomplete control required.

## Training (no retrain in this gate)

| Arm | Job | Elapsed | Status |
|-----|-----|---------|--------|
| Identity poscomplete control | **18719614** | 21m42s | COMPLETED |
| Neighbor SupCon | **18719615** | 36m49s | COMPLETED |
| Val-gate extract+eval | **18787415** | 1h14m | COMPLETED |

Smoke (passed): job **18719182**. Failed evaluator: job **18719616** TIMEOUT (6h).

## Timeout root cause (job 18719616)

Job 18719616 never finished a single extraction: after ~8 min ports/TDS load and checkpoint load, `extract hetero` stayed at 0/397 for ~5h50m until the 6h TIME_LIMIT. The child used `--loader_num_workers 8` (`persistent_workers=True`). `embedding_extraction.py` previously called `next(iter(tr_loader))` for model construction before CUDA init, then re-iterated the same multi-worker loader — a classic CUDA+fork / persistent-worker deadlock on the first batch.

Healthy seed-edge pre-3h extraction (job **18558352**) completes train (~397 batches) in ~2 minutes at ~3 it/s; the timeout was not “too many checkpoints” alone. Secondary waste: parent eval reloaded data without ports, then each subprocess reloaded ports/TDS (~8 min) and planned arms×epochs×(post+pre)×(train/val/test). Empty dir `embeddings/edge_dplus_identity_poscomplete_10ep_seed2_ep01/` has no valid cache.

**Fix (job 18787415):** single-process data load, `loader_num_workers=0`, fresh loaders after sample batch, pre-3h train+val for ep 5/10 first (then 1/3 after arm crossover), join X+TF after one H extract, unique `edge_dplus_nb_valgate_*` cache dirs (D+ embeddings untouched). Profile: 0.48 s/batch → ~0.28 h projected for four train+val extracts.

## SSL train loss (do **not** use for checkpoint selection)

| Epoch | Identity loss | Neighbor loss |
|------:|-------------:|--------------:|
| 1 | 5.940 | 6.049 |
| 3 | 5.778 | 5.923 |
| 5 | 5.733 | 5.909 |
| 10 | 5.708 | 5.901 |

Epoch 10 is the **latest / lowest training-loss** checkpoint for both arms — not a validation-selected best.

## Validation decision table (pre-3h, MLP, temporal train→val)

| arm | epoch | H val AUPRC | H+X+TF val AUPRC | val F1 |
|-----|------:|------------:|-----------------:|-------:|
| identity | 1 | 0.3335 | 0.5075 | 0.5472 |
| identity | 3 | 0.3798 | **0.5370** | 0.5878 |
| identity | 5 | 0.3611 | 0.5099 | 0.5461 |
| identity | 10 | 0.3527 | 0.4952 | 0.5370 |
| neighbor | 1 | 0.3017 | 0.4981 | 0.5525 |
| neighbor | 3 | 0.3096 | 0.5098 | 0.5610 |
| neighbor | 5 | 0.3643 | 0.5032 | 0.5489 |
| neighbor | 10 | 0.3577 | **0.5128** | 0.5515 |

Selection rule: max **H+X+TF** temporal validation AUPRC; validation F1 secondary; never SSL loss; never test. Episodes 1/3 evaluated after ep5↔ep10 arm crossover.

- Identity selected: `identity|pre3h|ep03|H+X+TF|mlp|none` (val AUPRC **0.537**)
- Neighbor selected: `neighbor|pre3h|ep10|H+X+TF|mlp|none` (val AUPRC **0.513**)
- Winner (paired comparison): identity ep3
- Neighbor beats matched identity: **False** (−0.024 vs identity on H+X+TF val AUPRC)
- Reference D+ fullstack val AUPRC: **0.550** (40ep horizon; unequal batching — contextual only)
- Recommend 40ep continuation: **False**
- Automatic 40ep submitted: **False**
- GNN retrained in this gate: **False**

## Winner-only paired test (locked after validation)

### `identity|pre3h|ep03|H+X+TF|mlp|none` (validation winner / control)
- AUROC/AUPRC: 0.9856 / 0.6389
- F1@0.5 P/R: 0.6296 / 0.6133 / 0.6468
- F1@val-thr P/R: 0.6230 / 0.5961 / 0.6524
- PPR@0.5: 0.0020; TP/FP/FN/TN: 1042/657/569/860781
- P@100/500/1000: 0.940/0.924/0.838

### `neighbor|pre3h|ep10|H+X+TF|mlp|none` (neighbor’s own val-selected ckpt)
- AUROC/AUPRC: 0.9883 / 0.6273
- F1@0.5 P/R: 0.6376 / 0.7198 / 0.5723
- F1@val-thr P/R: 0.6208 / 0.6066 / 0.6356
- PPR@0.5: 0.0015; TP/FP/FN/TN: 922/359/689/861079
- P@100/500/1000: 0.930/0.902/0.794

## Verdict

1. **Timeout root cause:** multi-worker `LinkNeighborLoader` deadlock after CUDA init (`workers=8` + `persistent_workers` + sample `next(iter)` before extract), not slow-but-progressing extraction.
2. **Val-selected checkpoints:** identity **ep3**; neighbor **ep10** (by H+X+TF val AUPRC).
3. **Neighbor vs identity:** neighbor does **not** beat the matched identity control (0.513 < 0.537).
4. **vs D+ 0.550:** identity ep3 reaches 0.537 at 10ep poscomplete horizon — close but below the 40ep D+ fullstack reference; unequal batching/horizon caveat applies.
5. **40ep justified?** **No** on this scout — no material neighbor gain over matched identity.
6. **No GNN training / no automatic 40ep continuation** was submitted.
