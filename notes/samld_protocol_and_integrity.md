# SAML-D protocol lock and integrity (P0)

> Twin: `results/diagnostics/samld_protocol_and_integrity.json`  
> Prior: `notes/multidataset_next_steps_preflight.md` (not repeated)  
> CPU job **19108637** (`mit_normal`). Failed precursor 19107936 (conda `nounset`).  
> **No GPU jobs. No follow-ups. No local runs. No training / GNN / embeddings / test eval.**

## Final verdict: **PASS_FOR_SUPERVISED_SMOKE**

Integrity is clean enough to run **one** protocol-A supervised smoke under the predeclared gate. Historical F1≈0.90 remains **untraceable** and must not be cited.

---

## 1. Three locked protocols

| ID | Role | edge_dim | ports | TDS | correct_reverse | preserve | Norm |
|----|------|---------:|:-----:|:---:|:---------------:|:--------:|------|
| `samld_supervised_multigin_eu_v1` | supervised baseline | **6** | on | **off** | **off** | off | **legacy per-graph z-norm** |
| `samld_frozen_aml_corrected_np_v1` | frozen AMLWorld transfer | **8** | on | **on** | **on** | off | **train-fit** |
| `samld_ssl_domain_v1` | SAML-only / future multidomain SSL | **8** | on | **on** | **on** | off | **train-fit**; labels excluded |

### A — `samld_supervised_multigin_eu_v1`

- gin + emlps + reverse_mp + ego; ports-only; TDS off; preserve off; corrected reverse **off** (paper-faithful Multi-GIN+EU).
- **Normalization:** `legacy_per_graph_edge_znorm` (`--train_fit_edge_znorm` **omitted**).
- **Why:** Matches PaySim/AML supervised Multi-GIN Candidate A parity. Attr z-norm is partially transductive; **not** for inductive transfer claims.
- **Hard refuse:** loading any edge_dim=8 / TDS / corrected-reverse AML SSL checkpoint.

### B — `samld_frozen_aml_corrected_np_v1`

- Must match locked encoder: `saved-models/checkpoint_gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2.tar`  
  SHA256 `18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6`
- ports+TDS, edge_dim=8, corrected reverse on, preserve off, **train-fit** z-norm.
- Assertions: ckpt sha256, ports/tds/correct_reverse/preserve, schema names `[base_0..3,in_port,out_port,in_td,out_td]`.
- **Hard refuse:** protocol-A or historical `checkpoint_multi-gin-eu-SAML-D-50epochs.tar` (edge_dim=6).

### C — `samld_ssl_domain_v1`

- Same feature contract as B (ports+TDS, edge_dim=8, corrected on, preserve off, train-fit).
- Labels **excluded** from SSL objective and selection (`uses_labels=false`).
- Multidomain extension: shared weights + **per-domain BN** (design only).
- **Hard refuse:** edge_dim=6 supervised baseline as SSL encoder without a separate conversion audit.

**Contract isolation:** A ⟂ B ⟂ C — no cross-loading of incompatible checkpoints.

---

## 2. Exact integrity results (job 19108637)

| Check | Result |
|-------|--------|
| Formatted SHA256 | `beb7f89ac19f648a860f3e10a33de818ccf763a588f741a2a0a73b6035619b8f` |
| Raw SHA256 | `5b71ce2ea7b47fe6f19da1aa151776b04ec74560a852c2c077df91d20b8b4ef9` |
| Columns | Exact `FORMATTED_TRANSACTION_COLUMNS` match |
| Rows | **9,504,852** (raw linecount matches) |
| Positives | **9,873** (π=0.103873%) |
| EdgeID unique | **yes** (n_unique = n_rows) |
| Duplicate rows | **0** |
| Feature-identical hashes spanning splits | **0** |
| Amount Sent == Amount Received | **all rows** (formatter policy) |
| Timestamp | nondecreasing; ties=1,750,972; days=**321** |
| Missing | **0** all columns |
| Unseen currency/payment vs train | **0** on val and test |
| Label in X / edge_attr / graph / norm / sampling / morph | **False** (code contract) |
| Perfect categorical label separators | **none** |

### Split (calendar_day, target 60/20/20)

| Split | Days | n | positives | π | index SHA256 |
|-------|------|--:|----------:|--:|--------------|
| Train | 0–191 (192d) | 5,707,315 | 5,751 | 0.100765% | `290713933cc655e9c70984bc3cb7f575ab26a03b8078a1337cda58892054935f` |
| Val | 192–255 (64d) | 1,899,523 | 1,986 | 0.104553% | `b08cdb815f82e6d37019e5e6ec9c5a6fd12c3f9d523f63b2768f6e4d0a99a38c` |
| Test | 256–320 (65d) | 1,898,014 | 2,136 | 0.112539% | `52d83d522af0783e9c1eb9984a47fe0c65bf95e5439afb6e63469219afa9d1aa` |

EdgeID cross-split overlap: **train∩val=0, train∩test=0, val∩test=0**.

### Entity overlap (allowed)

- Val accounts also in train: **78.4%**
- Test accounts also in train: **77.5%**  
→ **not** entity-inductive; report as allowed account overlap.

### Graph edge counts (loader policy)

| Graph | Edges | Seeds |
|-------|------:|------:|
| Train | 5,707,315 | 5,707,315 |
| Val (train∪val) | 7,606,838 | 1,899,523 |
| Test (all) | 9,504,852 | 1,898,014 |

### Message-passing temporal context (current loader)

| Kind | Verdict |
|------|---------|
| **Future split leakage** | **No** — train cannot see val/test edges; val cannot see test edges |
| **Within-split future context** | **Yes** — seeds can aggregate later edges inside the same split graph (no temporal neighbor filter) |
| **Legitimate earlier-history context** | **Yes** — val sees train; test sees train∪val (expanding history) |

Smoke must keep **`--testing` off** so test graph is never built/evaluated.

---

## 3. Historical F1 ≈ 0.90

**Classification: `untraceable`** (also **protocol-incomparable**). Not accepted as valid. Not proven contaminated.

| Evidence | Finding |
|----------|---------|
| Notes | `datasets.md` / `downstream-eval-plan.md` cite ~0.90 test F1 @ 1 ep |
| Surviving logs/JSON with the number | **None** |
| `saved-models/checkpoint_multi-gin-eu-SAML-D-50epochs.tar` | SHA `068a583d47b0645d83f38a2886baee8c0ab64cf987db89084f26c60a394d1cdb`; keys only `epoch=19` + states; **edge_dim=6**; no args/metrics |
| Legacy `run_saml_d_supervised_smoke.sh` | `--testing`, **no `--emlps`**, 1 ep — ≠ locked protocol A |
| Integrity | No label-in-features; no perfect categorical separators |

Do **not** assume leakage. Do **not** cite. Raw-feature signal vs prevalence remains untested (no new eval here).

---

## 4. Smoke gate (protocol A only) — do not submit yet

**Exact command (provided; not submitted):**

```bash
N_EPOCHS=2 RUN_NAME=samld_supervised_multigin_eu_v1_smoke_seed2 \
  sbatch slurm/run_samld_supervised_multigin_eu_v1_smoke.sh
```

Until that wrapper exists, equivalent flags (val-only; **omit** `--testing`, `--tds`, `--preserve_seed_edges`, `--correct_reverse_edge_features`, `--train_fit_edge_znorm`):

```bash
python main.py --data SAML-D --model gin --objective supervised \
  --unique_name samld_supervised_multigin_eu_v1_smoke_seed2 \
  --seed 2 --n_epochs 2 --batch_size 4096 \
  --num_neighs 100 100 --loader_num_workers 16 \
  --reverse_mp --ego --ports --emlps --tqdm
```

**Pass:** finite loss/grads; seed-edge CE; edge_dim=6; val coverage/class counts; report val AUROC/AUPRC/F1/P/R/PPR/confusion; compare only to prevalence + future X-only.  
**Fail:** nonfinite loss; label leakage; impossible coverage; unexplained leakage.  
**High F1 alone ≠ automatic failure.**

Artifact: `results/diagnostics/samld_supervised_smoke.json`.

---

## Unresolved risks

1. Within-split future neighbor context (declare; do not pretend causal MP).  
2. Protocol A legacy attr z-norm is transductive.  
3. Default test graph = all edges — keep locked for smoke.  
4. Frozen transfer currency/payment semantic mismatch.  
5. Historical F1 untraceable.  
6. Protocol-A smoke wrapper not yet on disk (command above is the contract).  
7. X-only control not yet run.

---

## Confirmation

- Training/model code **not** modified for this deliverable (audit script + CPU sbatch only).  
- **GPU jobs submitted: none.**  
- **Follow-up jobs: none.**  
- **Local runs: none** (integrity via Slurm CPU **19108637** only).  
