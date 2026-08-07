# Graph Barlow Twins stdfloor_1e4 full 3000-step training

**Arm:** `MIXED_3DOMAIN_GBT_STDFLOOR_1E4_FULL3000_SEED2`
**Objective:** `edge_aligned_graph_barlow_twins_r198_stdfloor_1e4`
**Verdict:** `PASS`
**Job ID:** `19605022`
**Generated:** 2026-08-04T04:03:31.817874+00:00

## Counts
- optimizer/scheduler: 3000/3000
- per-domain: `{"Small-HI": 1000, "SAML-D": 1000, "Small-LI": 1000}`
- LR: step1=0.0002, step600=0.002, step3000=0.00019999999999999996
- resume_exact_verified: false (fresh LONG stream match used instead)

## Gates
```
{
  "exact_3000_optimizer_steps": true,
  "exact_1000_updates_per_domain": true,
  "scheduler_steps_match": true,
  "finite_losses_all_steps": true,
  "both_view_grads_nonzero_all_steps": true,
  "encoder_params_changed": true,
  "encoder_sha_changed_from_init": true,
  "each_domain_bn_changed": true,
  "domain_bn_bundles_distinct": true,
  "c_always_198x198": true,
  "milestone_checkpoints_reload_ok": true,
  "long_seed_hash_match_1000": true,
  "no_test_metrics": true,
  "no_extraction": true,
  "no_probe": true,
  "lr_schedule_long_matched": true,
  "init_sha_prefix_ok": true,
  "did_not_resume_smoke": true,
  "did_not_resume_recovery": true,
  "did_not_resume_failed_official_full3000": true,
  "objective_is_stdfloor_1e4": true,
  "resume_exact_not_claimed": true,
  "cuda_within_allocation": true,
  "host_rss_under_128g": true,
  "view_hashes_logged": true
}
```

## Memory
- peak CUDA reserved: 20.631 GiB
- peak host RSS: 9.439 GiB
- mean sec/step: 0.8722607853690473

## Matching
- LONG seed-hash match (1000/domain): `{"Small-HI": {"ok": true, "n_compared": 1000}, "SAML-D": {"ok": true, "n_compared": 1000}, "Small-LI": {"ok": true, "n_compared": 1000}}`
- View hashes logged every step. LONG lacks historical view hashes — no cross-arm view equality claimed.

Training-integrity analysis only. No extraction/probes/test eval.
Does not resume recovery or failed official full3000.
