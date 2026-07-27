# Final corrected/no-preserve multiseed — post-smoke submission

- Submitted (UTC): `2026-07-27T01:26:01.936404+00:00`
- Smoke verified: `18951224` COMPLETED on node2804
- GPU jobs exclude: `node4104` (prior ECC failure)
- No smoke dependency (already passed); aggregate waits on all evals

## DAG

| Role | Job ID | Dependency |
|------|-------:|------------|
| integrity_smoke_seed2 (`SMOKE_PASSED`) | 18951224 | — |
| amlworld_seed1_confirmation (`AML_s1`) | 18952852 | — |
| amlworld_seed2_development (`AML_s2`) | 18952853 | — |
| amlworld_seed3_confirmation (`AML_s3`) | 18952854 | — |
| amlworld_seed4_confirmation (`AML_s4`) | 18952855 | — |
| paysim_seed1_P1P2P3 (`PS_s1`) | 18952856 | — |
| paysim_seed2_P1P2P3 (`PS_s2`) | 18952857 | — |
| paysim_seed3_P1P2P3 (`PS_s3`) | 18952858 | — |
| paysim_seed4_P1P2P3 (`PS_s4`) | 18952859 | — |
| matched_random_and_x_only_controls (`CONTROLS`) | 18952860 | — |
| cpu_aggregate_ensembles_registry (`AGG`) | 18952861 | afterok:18952852:18952853:18952854:18952855:18952856:18952857:18952858:18952859:18952860 |

## Expected artifacts

- **note**: `notes/final_corrected_no_preserve_multiseed.md`
- **json**: `results/diagnostics/final_corrected_no_preserve_multiseed.json`
- **cells**: `results/diagnostics/final_corrected_no_preserve_multiseed/cells/`
- **probabilities**: `results/diagnostics/final_corrected_no_preserve_multiseed/probabilities/`
- **embeddings**: `embeddings/final_corrected_no_preserve_multiseed/`

