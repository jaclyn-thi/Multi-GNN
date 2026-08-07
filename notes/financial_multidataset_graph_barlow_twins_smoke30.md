# Graph Barlow Twins 30-step smoke

**Verdict:** `PASS`
**Job ID:** `19597502`
**Generated:** 2026-08-04T01:00:57.457163+00:00

## Slurm flags

```
{
  "partition": "mit_normal_gpu",
  "account": "mit_amf_advanced_gpu",
  "qos": "mit_amf_advanced_gpu",
  "gres": "gpu:1",
  "cpus_per_task": "16",
  "mem": "128G",
  "time": "01:00:00",
  "loader_workers": 0
}
```

## Step / exposure counts

- optimizer steps: 30
- scheduler steps: 30
- per-domain: `{"Small-HI": 10, "SAML-D": 10, "Small-LI": 10}`
- first/last LR (full 3000-step schedule prefix): 0.0002 → 0.00028714524207011687

## Loss by domain (first vs last)

- **Small-HI**: L_total 37.7821 → 16.7711 (inv 0.486583→0.952153; red 37.2955→15.819)
- **SAML-D**: L_total 19.4445 → 12.2928 (inv 0.766666→1.03287; red 18.6778→11.2599)
- **Small-LI**: L_total 33.6085 → 16.629 (inv 0.313842→0.997512; red 33.2946→15.6315)

## Variance / effective rank

- effective_rank first→last: 1.5217599868774414 → 2.0524466037750244 (mean 5.443753484884898)
- view1 std median first→last: 0.7165226936340332 → 0.49898916482925415

## Gradients / parameter updates

- max encoder grad: 288.26143360938084
- max param update: 0.06455080758261096
- encoder changed: True
- mean Δθ by domain: `{"Small-HI": 0.03989280195010154, "SAML-D": 0.03738555734265068, "Small-LI": 0.03672166921139118}`

## Memory

- peak CUDA allocated: 17.099 GiB
- peak CUDA reserved: 20.045 GiB
- peak host RSS: 9.440 GiB
- mean sec/step: 1.0170582478089878

## Checkpoint reload

`{
  "ok": true,
  "path": "/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/checkpoints/financial_multidataset_graph_barlow_twins_smoke30_seed2/checkpoint_step_0030.pt",
  "sha256": "b4b73d1d768205b097b1b9d2c2821680a4cc2c57b47f43ae6fa873d121d393f0",
  "objective_id": "edge_aligned_graph_barlow_twins_r198",
  "global_step": 30,
  "scheduler_completed": 30,
  "has_bn_bundles": true,
  "forbidden": {
    "infonce": false,
    "tfmoe": false,
    "projection": false,
    "alpha_beta": false,
    "view2_detach": false
  }
}`

## Gates

```
{
  "exact_30_optimizer_steps": true,
  "exact_10_updates_per_domain": true,
  "scheduler_steps_match": true,
  "finite_losses_all_steps": true,
  "both_view_grads_nonzero_all_steps": true,
  "encoder_grads_nonzero_all_domains": true,
  "encoder_params_changed": true,
  "encoder_sha_changed_from_init": true,
  "each_domain_bn_changed": true,
  "domain_bn_bundles_distinct": true,
  "no_forbidden_objectives": true,
  "c_always_198x198": true,
  "checkpoint_reload_ok": true,
  "long_seed_hash_match": true,
  "no_test_metrics": true,
  "lr_schedule_not_rescaled": true,
  "init_sha_prefix_ok": true,
  "cuda_within_allocation": true,
  "host_rss_under_128g": true,
  "view_hashes_logged": true
}
```

**View-hash note:** View augmentation hashes logged every step. Historical MIXED_3DOMAIN_LONG did not log view hashes — no cross-arm view equality claimed.

## Proposed full run (NOT submitted)

```bash
sbatch --partition=mit_normal_gpu --account=mit_amf_advanced_gpu --qos=mit_amf_advanced_gpu --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=06:00:00 --job-name=gbt_r198_full3000 slurm/run_mixed_3domain_graph_barlow_twins_only_full.sh   # SCRIPT NOT CREATED / NOT SUBMITTED — full training remains blocked
```

Stop after 30-step smoke.
