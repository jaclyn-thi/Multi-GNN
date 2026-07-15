# Positive-pair quality diagnostics

Inventory of prior contrastive positive / exclusion methods and design of higher-quality **self-supervised** soft-positive rules. **FNF modes are exclusions, not positives.**

Code hub: `contrastive_loss.py`, `morphology/contrast.py`, `knn_filter.py`, `knn_soft_positives.py`.

Diagnostic output (Task F): `results/diagnostics/positive_pair_candidate_diagnostics.json`, `notes/positive_pair_candidate_samples.md`.

---

## Task D — Prior methods inventory

| method | type | definition | coverage | positives/anchor | weight | cost | result | success/failure reason | fully SSL? | leakage |
|--------|------|------------|----------|------------------|--------|------|--------|------------------------|------------|---------|
| **Identity** | hard positive | Same `edge_id` in both augmented views | 100% aligned seeds | 1 | 1.0 | O(batch) | **baseline** | Required InfoNCE anchor | yes | none |
| **same_sender FNF** | exclusion | Drop negatives with same `from_id` | varies | 0 | — | O(B²) mask | mixed legacy | too broad; unstable F1 | yes | none |
| **same_receiver FNF** | exclusion | Drop negatives with same `to_id` | varies | 0 | — | O(B²) | weak (0.193 F1) | overly permissive negatives remain | yes | none |
| **same_endpoint FNF** | exclusion | Any endpoint overlap | high filter | 0 | — | O(B²) | seed1 spike 0.277 F1 | high-variance | yes | none |
| **same_pair FNF** | exclusion | Same ordered (src,dst) | moderate | 0 | — | O(B²) | embedding-only **↓** vs baseline; **full stack ↑** (0.319 F1) | helps ranking w/ morph probe, not representation alone | yes | none |
| **Multi-positive same_* ** | soft positive | Endpoint match in other view @ `multi_positive_weight` | batch-local | 0–many | 0.1 default | O(B²) | **closed negative** (~0.153–0.224 F1) | dilutes identity; wrong geometry as positives | yes | none |
| **KNN negative filter** | exclusion | Exclude cached feature neighbors | cache-dependent | 0 | — | precompute + O(k) | 0.209–0.176 F1 | feature neighbors ≠ semantic AML pairs | yes | none |
| **KNN soft positives** | soft positive | Cached cosine neighbors @ low weight | sparse | 0–m | 0.025 | heavy precompute | **0.067 F1** — failed | false semantic neighbors | yes | none |
| **Morphology-bin (M2)** | soft positive | Same quantile bin on morph features | batch | many (capped 256) | logsumexp | calib + O(B²) | **0.012–0.058 F1** | bins ≠ semantics; expert conflated | yes | calib train-only |
| **Motif / typology metadata** | — | **Not implemented** as contrastive positives | — | — | — | — | pattern typology diagnostics only | no label-free positive rule shipped | N/A | would leak if labels used |
| **degree_aware edge drop** | augmentation | Not a positive rule | — | — | — | — | negative vs emlps+tds | unrelated to positives | yes | none |

### Notes

- **FNF ≠ positive:** only removes false negatives from denominator.
- Best operational win from endpoint rules is **downstream** with full probe stack, not embedding-only SSL metrics.
- KNN and M2 tried to manufacture density of positives — both **harmed** ranking metrics.

---

## Task E — Candidate higher-quality positive rules (≤3)

### Rule 1 (recommended): `endpoint_role_temporal_v1`

**Definition:** Edge B is a soft positive for anchor A if:

1. ∃ shared node u appearing in both edges;
2. u has the **same role** on both (sender-on-u or receiver-on-u);
3. |t_B − t_A| ≤ W (W = 86400 s = 1 day default);
4. edge_id_B ≠ edge_id_A.

**Expected coverage:** ~61% anchors have ≥1 positive (Small-HI train sample, 400k rows); median 1, mean 73 (**hub-heavy**).

**Failure modes:** High-degree accounts dominate (max 4353 positives/anchor); may reinforce hub representations.

**Complexity:** O(E) indexing + O(avg_pos) per anchor; feasible offline index.

**High-degree dominance:** Yes — monitor top-10 endpoint share (diagnostic included).

**Labels / typology:** **None** — fully self-supervised.

**Why not “many positives” alone:** Rule is constrained by role + time; still needs **cap per anchor** (e.g. max 32) in implementation.

---

### Rule 2 (backup): `repeat_pair_forward_temporal_v1`

**Definition:** Same ordered (sender, receiver), 0 < t_B − t_A ≤ W (forward repeat only).

**Expected coverage:** ~24% anchors with ≥1 positive; mean 0.31; max 4 — **sparse, precise**.

**Failure modes:** Low recall of positives; may miss structurally related but different-counterparty patterns.

**Complexity:** O(E) pair index.

**Hub dominance:** Low.

**Labels:** None.

**Use when:** Rule 1 over-concentrates on hubs — blend at low weight or use as filter.

---

### Rule 3 (conditional): `shared_sender_temporal_amount_v1`

**Definition:** Same sender, |Δt| ≤ W, |log1p(a_B) − log1p(a_A)| ≤ 0.5.

**Expected coverage:** ~15% anchors with ≥1; 85% zero — **sparse**.

**Failure modes:** Misses cross-account structuring; amount threshold arbitrary.

**Complexity:** O(out_degree) per anchor.

**Labels:** None.

**Use when:** Amount coherence is hypothesized to matter — likely **secondary** to Rule 1.

---

## Task F — Diagnostic results (Small-HI train sample)

**Sampling:** train split, first 400k rows after split, 2000 anchors, seed=1, W=86400s. **No labels used.**

| rule | frac zero | frac one | mean pos | median | max | hub concern |
|------|-----------|----------|----------|--------|-----|-------------|
| endpoint_role_temporal_v1 | 39% | 23% | 73.4 | 1 | 4353 | **high** |
| shared_sender_temporal_amount_v1 | 85% | 9% | 7.3 | 0 | 690 | medium |
| repeat_pair_forward_temporal_v1 | 76% | 19% | 0.31 | 0 | 4 | low |

**Overlap (Jaccard mean):** endpoint vs repeat ≈ 0.13; endpoint vs sender_amount ≈ 0.05.

**Samples:** see `notes/positive_pair_candidate_samples.md` (~20 per rule).

**Full-dataset script:** `scripts/diagnose_positive_pair_candidates.py` — rerun with `--max-train-rows 0` on cluster CPU if needed (~45s for 400k train rows).

---

## Task G — Recommendations (contrastive positives)

| Item | Choice |
|------|--------|
| **Recommended rule** | `endpoint_role_temporal_v1` **with per-anchor cap** (e.g. 32) + hub down-weighting |
| **Backup rule** | `repeat_pair_forward_temporal_v1` at low soft weight |
| **Expected compute** | Index build O(E) once; +5–10% batch loss time if capped |
| **Implementation changes** | New mask in `contrastive_loss.py`; CLI `--soft_positive_mode endpoint_role_temporal`; incompatible with morph_contrast/KNN soft (existing guards) |
| **Main risks** | Hub dominance; semantic drift (same endpoint ≠ same behavior); tuning W and cap |
| **Single-scout config (do not launch)** | `same_pair_fnf` + `endpoint_role_temporal_v1` @ weight 0.05, W=86400, cap=32, gin emlps+tds 20ep s1, 20ep — compare embedding-only + full probe to FNF-only baseline |
| **Stopping rule** | Abort if embedding-only AUPRC **↓ >0.005** vs FNF-only **or** full-stack probe F1 does not improve vs FNF at ep20 |

**Confirm:** No cluster jobs submitted for this audit.
