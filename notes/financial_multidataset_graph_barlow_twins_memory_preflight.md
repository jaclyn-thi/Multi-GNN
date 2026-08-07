# Graph Barlow Twins dual-view GPU memory preflight

**Classification:** `PASS_WITH_HEADROOM`
**Job ID:** `19594777`
**Generated:** 2026-08-04T00:29:20.653170+00:00
**Mode:** `--memory-preflight-only` (no optimizer/scheduler updates)

## Slurm flags

```
{
  "partition": "mit_normal_gpu",
  "account": "mit_amf_advanced_gpu",
  "qos": "mit_amf_advanced_gpu",
  "gres": "gpu:1",
  "cpus_per_task": "16",
  "mem": "128G",
  "time": "02:00:00",
  "loader_workers": 0
}
```

## Formula verification

- Code pattern: `(z - mean) / (std_unbiased + 1e-15)`
- OK: `True`
- Note: Any chat/summary rendering as (Z-mean)/std + eps is a prose typo only.
- Implementation note clarifies typo: `False`

## Memory by domain

| Domain | B | peak reserved GiB | free@peak GiB | enc grad | v1/v2 grads | C KiB |
|--------|--:|------------------:|--------------:|---------:|------------|------:|
| Small-HI | 6619 | 8.271 | 35.581 | 269.1 | 0.1368/0.1414 | 153.1 |
| SAML-D | 5897 | 18.613 | 25.240 | 66.81 | 0.09291/0.09129 | 153.1 |
| Small-LI | 6673 | 8.625 | 35.228 | 314.5 | 0.1565/0.1483 | 153.1 |

**Max peak reserved:** 18.613 GiB
**Max peak allocated:** 16.663 GiB
**Min remaining headroom (total−peak_reserved):** 25.781 GiB
**Min CUDA free at peak stage:** 25.240 GiB
**Max host RSS:** 9.440 GiB
**GPU total:** 44.394 GiB
**Init encoder SHA unchanged:** True

## Gates

- integrity_ok: True
- oom: False
- fail_stage: None
- fail_error: None
- no_optimizer_step / no_scheduler_step: True
- no_test_split / no embeddings / no training ckpt: True

## Per-stage CSV

`/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/financial_multidataset_graph_barlow_twins_memory_preflight/per_stage_memory.csv`

## Proposed 30-step smoke (NOT submitted)

```bash
sbatch --partition=mit_normal_gpu --account=mit_amf_advanced_gpu --qos=mit_amf_advanced_gpu --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=01:00:00 slurm/run_mixed_3domain_graph_barlow_twins_only_smoke.sh
```

Stop after memory preflight. Do not run 30 steps, full training, extraction,
probes, or test evaluation from this job.
