# Small-HI + SAML-D mixed SSL Phase-2 smoke

> Twin: `results/diagnostics/smallhi_samld_mixed_ssl_phase2_smoke.json`  
> Unique run: `smallhi_samld_shared_core_mixed_tfmoe_phase2_smoke_seed2`

## Status

**Smoke PASSED** (`ok=true`, job `19515484`, ~14.8 min, exit 0). Stop for human review — no Phase-3 / extract / probe submitted.

## Resolved protocol

| Field | Value |
|---|---|
| Contract | `smallhi_samld_shared_core_v1` |
| Features | `[Timestamp, Amount Received, in_port, out_port, in_td, out_td]` (`edge_dim=6`) |
| Encoder | GIN shared, from scratch, seed 2 |
| Representation | R198 InfoNCE (`--direct_r198_infonce`), projection **omitted** (=false) |
| TF MoE | adaptive α/β on 3 causal targets; shared heads |
| Schedule | 50 steps, 1:1 alternating Small-HI / SAML-D (25 each) |
| BN | domain-specific running buffers; shared affine/encoder weights |
| Loss calib | per-domain `LossNormState`; first 5 steps/domain; α/β frozen until both done |
| Graph flags | reverse_mp, ego, ports, emlps, TDS, correct_reverse; `preserve_seed_edges` omitted (=false) |
| Norm | per-domain train-fit edge z-norm |
| SAML split | `samld_calendar_day_rezero_v1` (not integrity raw-calendar counts) |
| LR | encoder 2e-3, α/β 1e-3; linear warmup 10 + linear decay 40 |
| AMP | off |
| Test | `--skip_test_eval`; SAML-D `te_inds` emptied; no test metrics |

## Cache provenance

| Domain | Cache | Notes |
|---|---|---|
| Small-HI | `results/cache/temporal_flow_causal/Small-HI` | `temporal_flow_causal_v1`; MoE cols [0,2,3]; EdgeID=row index; train-only std via `load_tf_moe_context` |
| SAML-D | `results/cache/temporal_flow_causal_samld_shared_core_v1/SAML-D` | scaler SHA `8e6e5d1b…`; train/val EdgeID SHA match rezero card; **no test split file** |

Preflight refuses incompatible MoE order / EdgeID joins / wrong SAML hashes. No cache rebuild.

## Focused tests

```text
python -m pytest -q tests/test_mixed_ssl_phase2_smoke.py --tb=line
12 passed in 153.76s
```

**12 passed, 0 failed.**

## Smoke job

| Field | Value |
|---|---|
| Job ID | `19515484` **COMPLETED** (exit 0) |
| Resources | `mit_preemptable` / `mit_general` / `qos=normal`, 1×L40S, 16 CPU, 128G, wall ~14.8 min |
| Script | `slurm/run_mixed_ssl_smallhi_samld_phase2_smoke.sh` |
| Steps | Small-HI **25** / SAML-D **25** (50 total); all finite losses; nonzero encoder+MoE grads both domains |
| Calib | per-domain means frozen by step 8/9; α/β unfrozen after step 9 (`alpha_unfrozen_after_step=10`) |
| BN | both bundles changed vs init and differ (L1 HI↔SD ≈ 10.48); swap round-trip OK |
| Checkpoint reload | OK (`ckpt_reload_ok`) |
| Artifacts | `results/diagnostics/smallhi_samld_mixed_ssl_phase2_smoke/` |
| Checkpoint | `saved-models/smallhi_samld_shared_core_mixed_tfmoe_phase2_smoke_seed2/checkpoint_smoke.tar` |

## Source files

- `mixed_ssl_phase2/` (`__init__.py`, `bn.py`, `schedule.py`, `preflight.py`)
- `scripts/train_mixed_ssl_smallhi_samld_phase2_smoke.py`
- `tests/test_mixed_ssl_phase2_smoke.py`
- `slurm/run_mixed_ssl_smallhi_samld_phase2_smoke.sh`
- this note + diagnostics JSON

Historical single-domain trainers (`main.py` / `training.py` paths) were not modified for mixed logic.

## Explicit non-goals

No full run, multiseed, extraction, probe, category adapters, paired-domain updates, PaySim, or dependent DAG.

## Proposed Phase-3 seed-2 scout (not submitted)

After smoke gates pass: longer alternating scout (~2–5k steps or few epochs), still shared-core + domain BN + per-domain loss calib + adaptive TFMOE, validation logging only (no test), unique checkpoint name, then optional frozen linear probe on val — submit only after human review of Phase-2 smoke.
