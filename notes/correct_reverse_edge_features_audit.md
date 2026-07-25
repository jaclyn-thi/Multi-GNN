# Correct reverse-edge feature semantics (opt-in)

**Provenance (implementation session):**
- `git rev-parse HEAD` at start: `ed7a15c18ab75f3d8d2e4600113c32a7f25046c2`
- Dirty tree preserved (unrelated user changes untouched)
- Flag: **`--correct_reverse_edge_features`** (default **off**)

This is a **thesis contrastive ablation** of temporal/port reverse-MP features.
It is **not** paper Multi-GIN+EU and **not** GCPAL reproduction.
Historical TDS-on runs remain labeled `inherited_legacy` / incorrect reverse semantics.

---

## Semantic audit

### Construction order (`data_loading.get_data` → homo `GraphData`)

1. Base edge features from dataset schema (AML: timestamp, amount, currency, payment_format) → **4-D**
2. If `--ports`: append **in_port**, **out_port** (destination incoming / source outgoing port indices)
3. If `--tds`: append **in_td**, **out_td** (time deltas on destination incoming / source outgoing adjacency)
4. `z_norm` over all edge_attr columns
5. If `--reverse_mp`: `create_hetero_obj` builds forward + reverse relations
6. Later: optional TF encoder inputs (after hetero; must not be mistaken for ports)
7. `add_arange_ids` **prepends** synthetic edge IDs (removed before the model via `exclude_edge_id` / first-column strip)

Resolved schema (ports+TDS): indices
`{in_port:4, out_port:5, in_td:6, out_td:7}` — **not** “last two = ports”.

### Directional roles under reverse MP

| Feature | Construction | Reverse MP action |
|---------|--------------|-------------------|
| base_* | edge-intrinsic | leave unchanged |
| in_port / out_port | dest-in / src-out roles | **swap once** |
| in_td / out_td | dest-in / src-out temporal roles | **swap once** (same directional logic as ports) |

### Inherited default (`correct_reverse_edge_features=False`)

Confirmed in `data_util.create_hetero_obj`:

1. Forward and reverse **alias** the same `edge_attr` storage
2. If `ports=True`, in-place swap of **trailing** columns `[:, [-1,-2]]`
3. With **ports+TDS**, trailing columns are **TDS**, so ports are not swapped and the swap mutates **both** relations
4. With **ports only**, trailing columns are ports (swap target is correct) but aliasing still mutates forward

Paper Multi-GIN+EU uses ports **without** TDS and **without** this flag → unchanged.

### Corrected opt-in (`--correct_reverse_edge_features`)

1. Reverse gets **independent** `edge_attr` storage (`clone`)
2. Forward left in original orientation
3. Swap named port columns and, when present, named TDS columns (schema-resolved)
4. No reliance on “last two columns”
5. Edge index flip, labels on forward, and later ID prepending unchanged

Metadata recorded in logs / contrastive checkpoints / supervised summaries:
`correct_reverse_edge_features`, `reverse_edge_feature_semantics` ∈ {`inherited_legacy`,`corrected`}, `edge_feature_schema`, `preserve_seed_edges`.

---

## Tests

- `tests/test_correct_reverse_edge_features.py`
- Updated `tests/test_multignn_supervised_parity.py` (legacy documentation; paper path without flag)
- Related: `tests/test_preserve_seed_edges.py`, `tests/test_rgcn_relation_type.py`

---

## Matched contrastive runs (seed 2, 40ep)

| | Run C | Run D |
|--|-------|-------|
| Unique name | `gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2` | `gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2` |
| TDS | on | on |
| `--correct_reverse_edge_features` | on | on |
| `--preserve_seed_edges` | **off** | **on** |
| Otherwise | match seed-2 40ep TDS-on recipe | same |

Comparators (existing): inherited TDS-on; TDS-off A; TDS-off preserve B.
