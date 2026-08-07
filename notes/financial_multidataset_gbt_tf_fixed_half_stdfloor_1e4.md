# MIXED_3DOMAIN_GBT_TF_FIXED_HALF_STDFLOOR_1E4 full 1500-step training

**Job:** `19643862`
**Verdict:** `PASS`
**Objective:** `edge_aligned_gbt_tf_fixed_half_stdfloor_1e4`

## Locks
- executed_stop_step=1500, schedule_horizon=3000
- fixed w_gbt=0.5, sum(w_tf)=0.5, learnable beta, alpha not optimized
- Fresh Phase-3 init; no smoke/adaptive resume; no test/extract/probe

## Gates
```json
{
  "exact_1500_optimizer_steps": true,
  "exact_500_updates_per_domain": true,
  "long_seed_hash_match_500": true,
  "w_gbt_always_0_5": true,
  "sum_w_tf_always_0_5": true,
  "beta_frozen_through_15": true,
  "beta_first_update_at_16_or_later": true,
  "alpha_unfrozen_at_15": true,
  "finite_losses_and_grads": true,
  "loss_reconstruction_ok": true,
  "c_always_198x198": true,
  "checkpoint_750_reload_ok": true,
  "checkpoint_1500_reload_ok": true,
  "checkpoint_last_reload_ok": true,
  "optimizer_excludes_alpha": true,
  "schedule_horizon_3000": true,
  "executed_stop_1500": true,
  "weight_mode_fixed_half": true,
  "learn_alpha_false": true,
  "orchestration_gates_ok": true,
  "no_test_metrics": true,
  "no_extraction": true,
  "no_probe": true,
  "fresh_phase3_init": true
}
```

## Checkpoints
```json
{
  "0750": {
    "path": "/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/checkpoints/financial_multidataset_gbt_tf_fixed_half_stdfloor_1e4_seed2/checkpoint_step_0750.pt",
    "exists": true,
    "sha256": "6a6ac1f33c4b6eee63565392ee3194f17868183766e755e0bf6b507169c96fae",
    "reload_ok": true,
    "objective_id": "edge_aligned_gbt_tf_fixed_half_stdfloor_1e4",
    "global_step": 750,
    "weight_mode": "fixed_half",
    "alpha_policy": "fixed_constant_0.5_not_learnable",
    "test_evaluated": false,
    "optimizer_n_groups": 2,
    "beta_only_optim_group": true,
    "alpha_constant": 0.5
  },
  "1500": {
    "path": "/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/checkpoints/financial_multidataset_gbt_tf_fixed_half_stdfloor_1e4_seed2/checkpoint_step_1500.pt",
    "exists": true,
    "sha256": "be0103a961132a24bac16e5693d8428cec95cb1767212ec9e2be895a1b5ebe37",
    "reload_ok": true,
    "objective_id": "edge_aligned_gbt_tf_fixed_half_stdfloor_1e4",
    "global_step": 1500,
    "weight_mode": "fixed_half",
    "alpha_policy": "fixed_constant_0.5_not_learnable",
    "test_evaluated": false,
    "optimizer_n_groups": 2,
    "beta_only_optim_group": true,
    "alpha_constant": 0.5
  },
  "last": {
    "path": "/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/checkpoints/financial_multidataset_gbt_tf_fixed_half_stdfloor_1e4_seed2/checkpoint_last.pt",
    "exists": true,
    "sha256": "2e28be94db62a0d5b2ee162adee45913c0e8edc17b60a712ea4f8f2b415da0cb",
    "reload_ok": true,
    "objective_id": "edge_aligned_gbt_tf_fixed_half_stdfloor_1e4",
    "global_step": 1500,
    "weight_mode": "fixed_half",
    "alpha_policy": "fixed_constant_0.5_not_learnable",
    "test_evaluated": false,
    "optimizer_n_groups": 2,
    "beta_only_optim_group": true,
    "alpha_constant": 0.5
  }
}
```

