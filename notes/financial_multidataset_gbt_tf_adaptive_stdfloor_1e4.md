# MIXED_3DOMAIN_GBT_TF_ADAPTIVE_STDFLOOR_1E4 full 3000-step training

**Job:** `19630663`
**Verdict:** `FAIL`
**Objective:** `edge_aligned_gbt_tf_adaptive_stdfloor_1e4`

## Locks
- Fresh Phase-3 shared init (not smoke / GBT recovery / Phase-4B InfoNCE)
- 3000 steps / 1000 per domain / LONG LR 600+2400
- std_floor=1e-4 GBT + TF MoE view1 + adaptive α/β
- Projection/AMP off; no test access; no extraction/probes

## Gates
```json
{
  "exact_3000_optimizer_steps": true,
  "exact_1000_updates_per_domain": true,
  "scheduler_steps_match": true,
  "finite_losses_all_steps": true,
  "both_view_grads_nonzero_all_steps": true,
  "moe_grads_nonzero_all_steps": true,
  "encoder_sha_changed_from_init": true,
  "moe_sha_changed_from_init": true,
  "each_domain_bn_changed": true,
  "domain_bn_bundles_distinct": true,
  "c_always_198x198": false,
  "milestone_checkpoints_reload_ok": true,
  "long_seed_hash_match_1000": true,
  "alpha_unfrozen_at_15": true,
  "all_domains_calibrated": true,
  "no_test_metrics": true,
  "no_extraction": true,
  "no_probe": true,
  "lr_schedule_long_matched": true,
  "init_sha_prefix_ok": true,
  "did_not_resume_smoke": true,
  "did_not_resume_gbt_recovery": true,
  "did_not_resume_phase4b_infonce": true,
  "fresh_phase3_shared_init": true,
  "objective_is_gbt_tf_adaptive_stdfloor": true,
  "projection_amp_off": true,
  "cuda_within_allocation": true,
  "host_rss_under_128g": true,
  "view_hashes_logged_without_historical_equality_claim": true,
  "api_contract_ok": true
}
```

## Checkpoints
```json
{
  "last": {
    "path": "/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/checkpoints/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4_seed2/checkpoint_last.pt",
    "global_step": 3000,
    "sha256": "a2c3c90d3f4382bc670e8176cdc1796689a1314f7efa7dcb16bf412dde32cca7"
  },
  "step_750": {
    "path": "/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/checkpoints/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4_seed2/checkpoint_step_0750.pt",
    "sha256": "82e457b3585da318a23a7630886b097debbe3e76b6c9bf763f379af2febf58f5",
    "reload_ok": true
  },
  "step_1500": {
    "path": "/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/checkpoints/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4_seed2/checkpoint_step_1500.pt",
    "sha256": "6616948c428e513bdc4b6e69062887de0803e5c968d37a78759644cc43936a9a",
    "reload_ok": true
  },
  "step_2250": {
    "path": "/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/checkpoints/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4_seed2/checkpoint_step_2250.pt",
    "sha256": "29cd8913e12f29411b19b894a2ec8f9241f631bf22d54618798f1f62ff9cac8d",
    "reload_ok": true
  },
  "step_3000": {
    "path": "/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/checkpoints/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4_seed2/checkpoint_step_3000.pt",
    "sha256": "24e10bea6c938840b7ff279effa39f0d0fe9d920749181f9289d6450a0efd02e",
    "reload_ok": true
  }
}
```

Training-integrity analysis only. No extraction/probes/test eval.

## Offline C_shape gate revalidation (no retrain)

- classification: `PASS_REVALIDATED_LOGGING_GATE`
- original failure reason: `FALSE_NEGATIVE_C_SHAPE_LOGGING_OMISSION`
- original aggregate SHA: `bc5cf2823806943c658800124ae9f2298818dca4d1b48ae56d764a482b42ada8` (unchanged)
- steps.jsonl SHA: `fd267f0d2458a6a2edd374e110448288815456b8ff92f77dbfa6770d9720d0ae` (unchanged)
- authorized_for_frozen_eval: `True`
- sidecar: `results/diagnostics/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4/training_integrity_revalidation.json`
- revalidated aggregate: `results/diagnostics/financial_multidataset_gbt_tf_adaptive_stdfloor_1e4/aggregate_revalidated.json`
