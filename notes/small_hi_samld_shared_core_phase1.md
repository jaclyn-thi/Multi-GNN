# Small-HI + SAML-D shared-core Phase 1

**Status:** Phase 1 complete — contract + tests + SAML-D TF cache integrity OK.  
**Stop for human review before mixed trainer / adapters / GPU work.**

## Contract

| Field | Value |
|---|---|
| Contract ID | `smallhi_samld_shared_core_v1` |
| Datasets | Small-HI, SAML-D |
| Final feature order | `[Timestamp, Amount Received, in_port, out_port, in_td, out_td]` |
| `edge_dim` | 6 (= base2 + ports2 + tds2) |
| Excluded from encoder inputs | Received Currency, Payment Format, Sent Currency, all labels, TF targets |
| TF role | prediction targets only (MoE: 3 causal cols) |
| Normalization | train-fit z-norm **per dataset**; validation reuses that dataset’s train scaler |
| Reverse MP | `correct_reverse_edge_features` port/TDS swap preserved |
| `preserve_seed_edges` | false |
| Projection | off (later trainer) |

**Not** the historical supervised Multi-GIN ports-only `edge_dim=6` (base4+ports2, TDS off). Dimension alone never identifies this contract.

## Source changes

- `shared_core_contract.py` — versioned contract + selection + train-fit scaler provenance
- `feature_contracts.py` — resolve/refuse PaySim neutralization path for shared-core
- `data_loading.py` — Small-HI/SAML-D shared-core load path (drop cats → ports+TDS → dim6 + auto train-fit z-norm)
- `util.py` — CLI help text
- `scripts/build_temporal_flow_causal_cache.py` — SAML-D unique root/version, train∪val only, MoE train scaler, refuse nonmatching overwrite
- `scripts/build_samld_shared_core_tf_cache_phase1.py` — Phase-1 wrapper + integrity + completion JSON
- `slurm/build_samld_shared_core_tf_cache_phase1.sh` — single CPU job (`mit_preemptable` / `mit_general` / `qos=normal`)
- `tests/test_small_hi_samld_shared_core_phase1.py` — focused unit tests

## Focused tests

```text
python -m pytest -q tests/test_small_hi_samld_shared_core_phase1.py --tb=line
10 passed in 102.73s (0:01:42)
```

**Pass/fail:** **10 passed, 0 failed** (re-run after SAML-D EdgeID≠row-index fix).

Coverage maps to the Phase-1 checklist (feature order, dim6 both datasets, cats/labels absent, per-dataset scalers, reverse swap, MoE shape/finite/causal/ties/label-free/EdgeID, test refusal, cache reload/metadata, historical contracts unchanged).

## Cache job

| Field | Value |
|---|---|
| First attempt | `19511713` — **failed**: SAML-D EdgeID unique but not equal to post-sort CSV row index (formatter assigns IDs then sorts by Timestamp). Builder updated to retain EdgeID for joins on SAML-D. |
| Job ID (current) | `19512052` (**completed**, integrity `ok=true`) |
| Resources | partition=`mit_preemptable`, account=`mit_general`, qos=`normal`, 128G, 16 CPU, 12h |
| Script | `slurm/build_samld_shared_core_tf_cache_phase1.sh` |
| Cache root | `results/cache/temporal_flow_causal_samld_shared_core_v1/SAML-D/` |
| Completion JSON | `results/diagnostics/small_hi_samld_shared_core_phase1_cache_completion.json` |

Writes **train/val only** (no test split file). MoE target standardization is **SAML-D train-only** (does not reuse Small-HI stats).

| Split | Rows | Positives (metadata only) |
|---|---:|---:|
| train | 5,715,293 | 5,764 |
| val | 1,900,105 | 1,984 |
| train∪val feature rows | 7,615,398 | — |
| MoE finite fraction | 1.0 | — |
| MoE scaler sha256 | `8e6e5d1b3d541dc76ecb9b4c7abddf149b3c923efb7b943e9a4971e26ce89d99` | — |

`edge_id_equals_row_index=false` (SAML-D formatter sort); EdgeIDs unique and used as join keys.

## Explicit non-goals (this phase)

- No test-split construction or evaluation
- No encoder training
- No embedding extraction / probing
- No category adapters
- No multi-dataset / mixed trainer
- No dependent GPU job

## Proposed next command (not submitted)

After cache integrity passes, a **20–50-step** mixed Small-HI+SAML-D smoke (trainer still to be implemented in Phase 2) would look like:

```bash
# PROPOSED ONLY — do not submit until Phase-2 trainer exists and cache completion is OK
sbatch --partition=mit_preemptable --account=mit_general --qos=normal \
  --gres=gpu:1 -c 8 --mem=64G -t 02:00:00 \
  --wrap '... python scripts/train_mixed_ssl_smallhi_samld_smoke.py \
    --feature_contract smallhi_samld_shared_core_v1 \
    --ports --tds --emlps --ego --reverse_mp \
    --correct_reverse_edge_features \
    --train_fit_edge_znorm \
    --preserve_seed_edges False \
    --projection off \
    --max_steps 50 \
    --skip_test_eval \
    --samld_tf_cache results/cache/temporal_flow_causal_samld_shared_core_v1/SAML-D \
    --seed 2'
```

Exact trainer entrypoint is Phase 2.
