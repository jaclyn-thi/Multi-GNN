# Phase-4A financial multi-dataset shared-core smoke

> Twin: `results/diagnostics/financial_multidataset_shared_core_phase4a_smoke.json`
> Job: **19531409** · unique: `financial_multidataset_shared_core_phase4a_smoke_seed2`

**Infrastructure/memory smoke only.** No 1500-step scout, extraction, probes, PaySim/Medium/AMLSim, adapters, or DAG.

## Verdict

All smoke gates **PASS**. Three-domain residency is easy under 128G (Slurm MaxRSS ≈ **10.27 GiB**). Full 1500-step scout is **recommended** at 128G / workers=0 after review (not submitted here).

## Contract

- Old `smallhi_samld_shared_core_v1`: **unchanged** (still Small-HI + SAML-D only).
- New `financial_multidataset_shared_core_v1`: Timestamp, Amount Received, in_port, out_port, in_td, out_td (`edge_dim=6`).
- Geometry equivalent to the two-domain vector; new protocol identity for broader registry.

## Small-LI provenance (locked)

- Cache: `results/cache/temporal_flow_causal/Small-LI` (`temporal_flow_causal_v1`)
- EdgeID == row index; MoE order [interarrival, past_7d_count, amount_vs_sender_past_mean]
- Train/val SHAs locked in preflight; train-only TF scaler at load; no test TF arrays loaded.

## Smoke

| Item | Value |
|---|---|
| Domains / schedule | Small-HI → SAML-D → Small-LI round-robin |
| Steps | 60 global · 20/domain |
| α/β | Frozen through completed step 15; first update at 16 |
| LossNorm | 5 obs/domain; all calibrated |
| Mean s/step | 1.895 |
| Time to first step | 1239s (dominated by graph build) |
| Process peak RSS | 9.44 GiB |
| Slurm MaxRSS | 10.27 GiB |
| CUDA peak alloc / reserved | 12.92 / 15.51 GiB |
| Checkpoint | `/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/checkpoints/financial_multidataset_shared_core_phase4a_smoke_seed2/checkpoint_step_0060.tar` reload OK |

## Phase-4B (proposal only — not submitted)

- 500 updates/domain · 1500 steps · same contract/objective
- Resources: `mit_preemptable` / `mit_general` / `qos=normal` · 128G · 16 CPU · 1 GPU · workers=0 · ≤08:00:00
- Validation only; no extraction/probe until training review

## Confirmations

- no full 1500-step run / extraction / probe / adapter / PaySim / Medium / AMLSim / DAG
