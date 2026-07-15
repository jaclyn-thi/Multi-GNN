# PNA SSL fairness follow-up — consolidated interpretation

This note compares three PNA-related baselines against the GIN current-protocol reference. The width-aligned PNA scout is **one seed** and **not** a full architecture sweep; do not treat it as proof that PNA is universally better or worse than GIN. The main thesis result does not depend on PNA.

## Comparison table

| model | hidden | pre dim | post dim | params | representation | stack | AUROC | AUPRC | F1 | P@100 | lift@100 |
|-------|-------:|--------:|---------:|-------:|----------------|-------|------:|------:|---:|------:|---------:|
| original PNA | 20 | 60 | 128 | 42280 | pre | embedding_only | 0.947 | 0.119 | 0.214 | 0.32 | 171.43438857852266 |
| original PNA | 20 | 60 | 128 | 42280 | post | embedding_only | 0.946 | 0.112 | 0.208 | 0.33 | 176.7917132216015 |
| original PNA | 20 | 60 | 128 | 42280 | pre | embedding_plus_raw | 0.959 | 0.262 | 0.042 | 0.75 | 401.79934823091247 |
| original PNA | 20 | 60 | 128 | 42280 | post | embedding_plus_raw | 0.960 | 0.264 | 0.032 | 0.74 | 396.4420235878336 |
| width-aligned PNA | 65 | 195 | 128 | 321280 | pre | embedding_only | 0.960 | 0.230 | 0.298 | 0.83 | 444.6497020484171 |
| width-aligned PNA | 65 | 195 | 128 | 321280 | post | embedding_only | 0.954 | 0.147 | 0.216 | 0.56 | 300.0046182495345 |
| width-aligned PNA | 65 | 195 | 128 | 321280 | pre | embedding_plus_raw | 0.969 | 0.276 | 0.279 | 0.82 | 439.29247672253257 |
| width-aligned PNA | 65 | 195 | 128 | 321280 | post | embedding_plus_raw | 0.961 | 0.198 | 0.266 | 0.64 | 342.8624208566108 |
| GIN reference | 66 | 198 | 128 | None | post | embedding | 0.944 | 0.213 | 0.259 | — | — |

## Model geometry

| model | hidden | pre dim | post dim | params | notes |
|-------|-------:|--------:|---------:|-------:|-------|
| GIN reference | 66 | 198 | 128 | 182704 | architecture-sweep post-128 embedding-only |
| default PNA | 20 | 60 | 128 | 42280 | not capacity/hyperparameter matched |
| width-aligned PNA | 65 | 195 | 128 | 321280 | GIN-matched LR/dropout; seed 1 scout |

## Answers

1. **Upstream parity:** passed (see pytest log) (See tests/test_pna_upstream_parity.py results in Slurm log or pytest output.)
2. **Degree histogram:** audit pending — inherited minibatch behavior; see `notes/pna_degree_histogram_audit.md`.
3. **Pre-3h on original 60-d PNA:** expansion 60→128 (not GIN-style compression); see `notes/pre_embedding_3h_vs_post_embedding_pna_emlps_tds_seed1.md`.
4. **Width/hyperparameter alignment:** width65 scout uses GIN-matched LR/dropout; still one seed and not fully tuned PNA.
5. **Vs GIN:** GIN architecture-sweep post embedding AUPRC = 0.21330862216949034 (embedding-only).
6. **AUROC vs AUPRC:** compare width-aligned rows in per-run notes; high AUROC with low AUPRC remains possible under imbalance.
7. **Best downstream representation:** per-row winners in per-run notes (`pre` vs `post`, embedding-only vs +raw).
8. **Remaining gap:** conservative read — objective + hyperparameter confounds unless width-aligned scout closes most of the GIN gap without parity failures.

JSON: `/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/pna_ssl_fairness_followup.json`
Width-aligned probe JSON: `/orcd/home/002/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/pna_width_aligned_probe.json`