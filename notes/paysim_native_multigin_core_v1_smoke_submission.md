# paysim_native_multigin_core_v1_smoke submission / status

- **Job ID:** `19123387` (exactly one smoke GPU job)
- **Partition / account / QoS:** `mit_normal_gpu` / `mit_amf_advanced_gpu` / `mit_amf_advanced_gpu`
- **Script:** `slurm/run_paysim_native_multigin_core_v1_smoke.sh`
- **Run name:** `paysim_native_multigin_core_v1_smoke_seed2`
- **Training:** succeeded (2 epochs); Slurm exit 2 was from an over-strict gate false-positive
- **Gate (re-run offline on same artifacts):** **PASS** (`gate_pass=true`)
- **Formal 50-epoch:** NOT submitted

## Exact train command

```bash
/home/jthi/.conda/envs/multignn/bin/python main.py \
  --data PaySim \
  --model gin \
  --objective supervised \
  --supervised_head legacy \
  --feature_contract paysim_native_multigin_core_v1 \
  --train_fit_edge_znorm \
  --unique_name paysim_native_multigin_core_v1_smoke_seed2 \
  --seed 2 \
  --n_epochs 2 \
  --batch_size 8192 \
  --num_neighs 100 100 \
  --loader_num_workers 16 \
  --reverse_mp --ego --ports --emlps \
  --save_model \
  --skip_test_eval \
  --tqdm
```

## Code provenance

- **Git HEAD:** `9298d2d1ea03d6121c4311d6b8a67a9be97caced`
- **Git describe:** `9298d2d-dirty`
- **Branch:** `main`

## Dataset / scaler hashes

- **Formatted SHA256:** `03c2fa07b95d145e754b74a5e646c2d71cd4fed051210d6292a0bbab90112c93`
- **Raw SHA256:** `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`
- **Scaler SHA256:** `45ce032c08ae0f3ef73f11f3a778bbc351da7bd43b3316ab583c941d4bcbae27`
  (train-fit continuous-only; indices `[0,1,7,8,9,10,11,12]`; one-hots unchanged)

## Runtime / memory (Slurm)

| Field | Value |
|-------|------:|
| Elapsed | 00:10:27 |
| MaxRSS | 12098316K (~11.5 GiB) |
| MaxVMSize | 12054676K |
| Batch exit | 2:0 (gate false-positive; training OK) |

## Preflight

- `tests/test_paysim_native_multigin_core_v1.py` — 7 passed (in-job + local)

## Outputs

- `notes/paysim_native_multigin_core_v1_smoke.md`
- `results/diagnostics/paysim_native_multigin_core_v1_smoke.json`
- `results/diagnostics/paysim_native_multigin_core_v1_smoke_submission.json`
- `saved-models/paysim_native_multigin_core_v1_smoke_seed2/`
- Registry append/upsert only (no historical rewrite)

## Smoke metrics (validation-only)

| Epoch | Train loss | Val AUROC | Val AUPRC | Val F1 | P / R | PPR | TP/FP/TN/FN |
|------:|-----------:|----------:|----------:|-------:|------:|----:|-------------|
| 1 | 0.01850 | 0.9718 | 0.6126 | 0.6007 | 0.633 / 0.572 | 5.5e-4 | 446/259/1275237/334 |
| 2 | 0.01105 | 0.9816 | **0.6665** | **0.6754** | 0.779 / 0.596 | 4.7e-4 | 465/132/1275362/315 |

- Best val-F1 epoch: **2**
- Max val-AUPRC epoch: **2**
- Seed-edge coverage ≈ 1.0 (epoch2: 1276274/1276276)
- `test_evaluated=false`; test untouched

## Gate note

Initial in-job gate failed on (1) 2 missing NeighborLoader seed edges and (2) NaN test placeholders in summary under `--skip_test_eval`. Gate tolerances fixed; **gate re-run offline only** (no second training job).
