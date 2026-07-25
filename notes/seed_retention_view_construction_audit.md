# Seed retention / view-construction audit

**Label:** seed-retention/view-construction ablation motivated by transaction-node vs transaction-edge mapping. **Not** a GCPAL reproduction.

Machine twin / Scout B artifacts use `preserve_seed` in run names.

## Coordination with TDS-off identity control (Scout A)

| Field | TDS-off control (Scout A) |
|-------|---------------------------|
| Job | **18491335** (`gin_40ep_s2_tdsoff`) |
| Script | `slurm/comparison_gin_emlps_ports_tds_off_asym_proj_8192neg_queue0_40ep_seed2.sh` |
| Run name | `gin_emlps_ports_tds_off_asym_proj_8192neg_queue0_40ep_seed2` |
| Started | 2026-07-21T22:58:35-04:00 (code already loaded before `--preserve_seed_edges` landed) |
| Git HEAD | `ed7a15c18ab75f3d8d2e4600113c32a7f25046c2` |
| Config | gin, `--reverse_mp --ego --ports --emlps`, **no `--tds`**, seed **2**, bs **8192**, accum **4**, negs **8192**, queue **0**, T **0.5**, projection 128→128, asymmetric, identity-only, 40 ep, `--testing`, extract+probe+feature ablation |
| First-batch log | requested_seed_edges=8192, shared_seed_edges=**6654** (≈81.2%) |

**Decision:** Scout A matches the required matched-pair recipe. **Do not submit another current-view control.** Submit only Scout B (`--preserve_seed_edges`).

## Part 1 — read-only audit findings

1. **Can a target/seed transaction be dropped from either view?** **Yes.** `generate_views` / `_hetero_random_edge_drop_view` apply a uniform Bernoulli keep mask over **all** forward edges, including seed/target `edge_id`s. Docstring previously stated seeds are not protected.

2. **Are contrastive embeddings only for seeds present in both views?** **Yes.** `select_shared_seed_edge_embeddings` keeps `seed ∩ view1 ∩ view2` and aligns by `edge_id`.

3. **Does 2048 → 1678 follow from drop rates?** **Yes.** With independent `p_drop=0.1` per view:  
   \(P(\text{both})=(0.9)^2=0.81\) → \(2048×0.81≈1659\). Observed **1678** is within sampling noise. Scout A first batch: \(8192×0.81≈6635\) vs observed **6654**.

4. **Independent across views?** **Yes.** Two independent `generate_views` / edge-drop draws (hetero: reverse synced to forward within a view).

5. **Fractions (expected at p=0.1):** both ≈81%; only one view ≈18%; neither ≈1%. Empirically consistent with GCPAL diagnostic and Scout A.

6. **Accum / loss normalization?** InfoNCE returns **mean over surviving anchors**; training divides by `accum_steps` and averages losses **per microbatch**, not per requested seed. Fewer survivors do **not** down-weight the step relative to a fuller batch; there is no compensation that restores “8192 seed units.”

7. **GCPAL transaction-as-node edge drop?** Documented edge dropping removes **adjacency relations between transaction nodes**, not the transaction instances themselves. In our edge-centric map, dropping a seed **edge** removes the transaction from readout **and** from MP — a translation mismatch.

**Distinction:** retaining a seed as a readout/query entity ≠ retaining its edge in the MP graph. Current default conflates them.

## Part 2 — smallest faithful opt-in

Implemented **`--preserve_seed_edges`** (default **off**):

- Independently drop non-seed/context edges as before.
- Force-keep every requested seed `edge_id` in both views (forward + synced reverse).
- Seed edge attributes still receive independent attr-masking.
- **Approximation documented:** seed relations remain in the message-passing graph (not a separated query-only readout). Separating query readout cleanly would be a larger refactor.

Files: `graph_augmentations.py`, `training.py` (`_contrastive_view_kwargs` + call sites), `util.py` CLI flag.  
Tests: `tests/test_preserve_seed_edges.py` (**6 passed**).

## Part 3 — Scout B only

| | Scout A (reuse) | Scout B (submit) |
|--|-----------------|------------------|
| View construction | current (seeds droppable) | `--preserve_seed_edges` |
| Unique name | `gin_emlps_ports_tds_off_asym_proj_8192neg_queue0_40ep_seed2` | `gin_emlps_ports_tds_off_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2` |
| Epochs | 40 | 40 (fits ≪ 5h15m; source ~3.5h wall) |
| Seed / recipe | identical otherwise | identical otherwise |

Independent GPUs: yes. Artifacts: non-colliding names.
