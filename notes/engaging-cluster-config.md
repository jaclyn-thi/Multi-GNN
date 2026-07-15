# MIT Engaging cluster config (Slurm accounts, QoS, limits)

Single source of truth for which Slurm account/QoS/partition to request and the **verified**
effective walltime/resource limits. Update this file rather than scattering account details across
scripts. Individual `slurm/*.sh` scripts carry a short pointer back here.

## Advanced account — temporary, valid through **2026-08-01**

| Job type | `--partition` | `--account` | `--qos` |
|----------|---------------|-------------|---------|
| GPU | `mit_normal_gpu` | `mit_amf_advanced_gpu` | `mit_amf_advanced_gpu` |
| CPU | `mit_normal`     | `mit_amf_advanced_cpu` | `mit_amf_advanced_cpu` |

### Verified effective limits (checked 2026-07-07)

The **stricter of {partition MaxTime, QoS MaxWall}** is authoritative. Here the partition MaxTime is
the binding constraint for walltime on both queues.

| Resource | GPU (`mit_normal_gpu`) | CPU (`mit_normal`) |
|----------|------------------------|--------------------|
| **Effective max walltime** | **06:00:00** (partition MaxTime) | **12:00:00** (partition MaxTime) |
| Partition DefaultTime | 02:00:00 | none |
| QoS MaxWall (looser, not binding) | 2-00:00:00 | 4-00:00:00 |
| QoS MaxTRES per user | cpu=64, gres/gpu=4, mem=1T | cpu=512, mem=2T |

Commands used to verify (re-run if limits may have changed):

```bash
sacctmgr -n show qos mit_amf_advanced_gpu format=Name,MaxWall,MaxTRESPU,MaxJobsPU,MaxSubmitPU
sacctmgr -n show qos mit_amf_advanced_cpu format=Name,MaxWall,MaxTRESPU,MaxJobsPU,MaxSubmitPU
scontrol show partition mit_normal_gpu
scontrol show partition mit_normal
```

**Practical guidance:** even though the QoS allows multi-day sessions, the partition caps walltime at
6 h (GPU) / 12 h (CPU). Do **not** request more than the partition MaxTime, and keep using
historically-sufficient walltimes (extraction ≈ minutes → 04:00:00; CPU probes → 10:00:00) rather
than the maximum.

## Reverting after 2026-08-01

When Advanced access expires, drop the two extra directives from GPU/CPU scripts and fall back to
the base partition defaults:

```bash
# GPU: keep only
#SBATCH --partition=mit_normal_gpu
# CPU: keep only
#SBATCH --partition=mit_normal
# (remove the --account=mit_amf_advanced_* and --qos=mit_amf_advanced_* lines)
```

The base `mit_normal` / `mit_normal_gpu` partitions carry the same 12 h / 6 h walltime caps, so only
the account/QoS lines need removing.
