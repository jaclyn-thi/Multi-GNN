# Multi-GNN supervised parity audit (Small-HI Multi-GIN+EU)

Machine-readable twin: [`results/diagnostics/multignn_supervised_parity_audit.json`](../results/diagnostics/multignn_supervised_parity_audit.json)

Smoke job that produced the GPU audit + 1-epoch TDS-off path check: **18470252** (`hi_sup_parity`, 2026-07-21 17:45–18:09). All audit stages status=`ok`.

---

## 1. Paper target and current result

| Source | Minority-class F1 (argmax) |
|--------|---------------------------:|
| Egressy et al. Multi-GIN+EU (Small-HI) | **64.79 ± 1.22** |
| Formal thesis run `small_hi_legacy_supervised_gin_emlps_tds_100ep_seed1` (best-val ckpt) | **~0.529–0.539** ([train summary](supervised_Small-HI_small_hi_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.md), [eval note](eval_small_hi_legacy_supervised_gin_emlps_tds_100ep_seed1.md)) |
| Released upstream `checkpoint_multi-gin-eu-SmallHI-50epochs.tar` on current TDS-**off** split (read-only eval in this audit) | **0.6899** (coverage 863073 / 863900 seed edges) |

Gap vs paper for the formal run: about **11–12 F1 points**. Upstream checkpoint eval on the current TDS-off pipeline is **above** the paper mean, which is strong evidence that the data/model path (without TDS) can host Multi-GIN+EU weights.

---

## 2. Exact comparator

- Table comparator: **Multi-GIN+EU** = **64.79 ± 1.22** (not Multi-GIN).
- Requires `--emlps` (edge updates) plus the documented graph adaptations: `--reverse_mp --ego --ports`.
- Decision rule for reproduction: **argmax over two-class logits** (paper_argmax).

---

## 3. Configuration matrix

| Setting | Upstream paper/repo Multi-GIN+EU | Current `tds=False` | Formal run `tds=True` |
|---------|----------------------------------|---------------------|------------------------|
| `reverse_mp` | true | true | true |
| `ego` | true | true | true |
| `ports` | true | true | true |
| `emlps` | true | true | true |
| `tds` | **false** | false | **true** |
| `edge_dim` | **6** (4 raw + 2 ports) | 6 | **8** (+2 TDS) |
| Epochs (evidence) | released ckpt name `50epochs` | — | **100** |

Upstream README command (commit `252b025`):

```bash
python main.py --data Small_HI --model gin --emlps --reverse_mp --ego --ports
```

`--tds` exists as an optional CLI flag in code but is **absent** from that command and from the README adaptation list.

All **17** inventoried upstream-style checkpoints have `edge_emb` in_features = **6**.

---

## 4. First point of divergence

**Feature configuration:** the formal run enabled `--tds`, so edge attributes are **8-D** and the reverse-port swap (when `ports` is on) operates on the trailing pair — which are **TDS columns**, not ports.

Everything measured after that (model init, logits, weighted CE, grads, one Adam step) matches upstream GINe **when the same edge_dim is used**. Loader seed-edge selection and first-batch determinism also match under the audited settings.

Shared with upstream (not a current-only divergence for paper config): `create_hetero_obj` assigns the **same** `edge_attr` storage to forward and reverse, so the in-place port swap also mutates the forward relation.

---

## 5. Was the current formal run paper-compatible?

**No.**

Reasons, in order:

1. `--tds` was on → edge_dim 8 vs paper/released edge_dim 6.
2. That also activates the demonstrated ports+TDS wrong-column swap (see §6).
3. Training used **100** epochs; released Multi-GIN+EU Small-HI artifacts are named **50epochs** (best epoch in the inspected ckpt is 34).

The run summary’s `paper_comparable: true` / “configured to reproduce” flags are therefore **overstated** for Multi-GIN+EU table comparison.

---

## 6. Ranked root-cause findings

| Rank | Finding | Status |
|------|---------|--------|
| 1 | Formal Multi-GIN+EU comparator incorrectly included `--tds` (edge_dim 8 vs 6). | **Demonstrated** (README, paper adaptations, checkpoint inventory, config matrix) |
| 2 | With `ports`+`tds`, reverse swap uses `[:, [-1,-2]]` on TDS columns, not ports; aliasing mutates both relations. | **Demonstrated** (tiny semantics + xfails in `tests/test_multignn_supervised_parity.py`) |
| 3 | Epoch protocol may differ (100 vs ~50 / released `50epochs`). | **Likely** (filename + ckpt epoch 34); exact paper seed/mean protocol still partially **unresolved** |
| 4 | Legacy GINe vs upstream GINe (mapped classifier) init / logits / weighted CE / grads / Adam. | **Demonstrated match** when features match (`tds` false or true) |
| 5 | Upstream Multi-GIN+EU checkpoint load + paper_argmax on current TDS-off test split → F1 ≈ 0.69. | **Demonstrated** (read-only eval in audit JSON) |
| 6 | First LinkNeighborLoader batch determinism (seed 1, workers 0). | **Demonstrated** (exact hash equality) |
| 7 | Commit `252b025` reverse-MP fix. | Affects checkpoint `to_hetero` load order only; **not** feature construction |

Focused tests (local): **21 passed, 4 xfailed** (strict semantic xfails for ports+TDS alias / wrong-column swap — kept failing on purpose).

---

## 7. Recommended single full follow-up run

**Do not change** existing `slurm/train_small_hi_legacy_supervised_gin_{1_50,51_100}.sh`. Copy/adapt a **new** script.

Only flags that should differ from the formal TDS-on 100-ep run:

| Change | Formal | Recommended |
|--------|--------|-------------|
| `--tds` | present | **omit** |
| `--n_epochs` | 100 (split 50+50) | **50** (single job preferred) |
| `--unique_name` | `..._tds_100ep_seed1` | e.g. `small_hi_legacy_supervised_gin_emlps_ports_50ep_seed1` |

Keep unchanged: `--data Small-HI --model gin --objective supervised --supervised_head legacy --reverse_mp --ego --ports --emlps --batch_size 8192 --num_neighs 100 100 --seed 1 --save_model` (and the same Advanced GPU Slurm envelope). Prefer `--loader_num_workers 0` for loader parity with the audit, or keep 16 if matching the formal scripts’ I/O setup — document which.

Example command:

```bash
python main.py \
  --data Small-HI --model gin --objective supervised --supervised_head legacy \
  --unique_name small_hi_legacy_supervised_gin_emlps_ports_50ep_seed1 \
  --save_model --n_epochs 50 --batch_size 8192 --num_neighs 100 100 \
  --loader_num_workers 0 \
  --reverse_mp --ego --ports --emlps --seed 1 --tqdm
```

**Not submitted** in this audit (stop condition: wait for explicit approval).

A code fix for ports+TDS semantics is **not** a prerequisite for this paper-matched run (ports-only, no TDS). It **is** justified later for any TDS-on training (including contrastive scouts that keep `--tds`).

---

## 8. Estimated runtime

From formal Small-HI legacy parts: `hi_legacy_sup_p1_17718838` ≈ **3h10m** (17:31–20:42), `hi_legacy_sup_p2_17718839` ≈ **3h10m** (20:42–23:52). A single **50-epoch** TDS-off job should fit comfortably in one **06:00:00** Advanced GPU slot (same envelope as `slurm/train_small_hi_legacy_supervised_gin_1_50.sh`).

Parity smoke **18470252**: audit (~14 min) + 1-epoch TDS-off train (~10 min) ≈ **24 min** total.

---

## Linked artifacts

| Artifact | Path |
|----------|------|
| Audit JSON (all stages ok) | `results/diagnostics/multignn_supervised_parity_audit.json` |
| This note | `notes/multignn_supervised_parity_audit.md` |
| Smoke Slurm script | `slurm/smoke_small_hi_multignn_supervised_parity.sh` |
| Smoke stdout/stderr | `slurm-logs/hi_sup_parity_18470252.{out,err}` |
| Smoke train log | `logs/small_hi_multignn_supervised_parity_smoke_tds_off_seed1.log` |
| Smoke summaries | `notes/supervised_Small-HI_small_hi_multignn_supervised_parity_smoke_tds_off_seed1_summary.md`, `results/diagnostics/supervised_Small-HI_small_hi_multignn_supervised_parity_smoke_tds_off_seed1_{summary,epoch_history}.json` |
| Smoke checkpoints | `saved-models/small_hi_multignn_supervised_parity_smoke_tds_off_seed1/checkpoint_{last,best_val_f1}.tar` |
| Parity tests | `tests/test_multignn_supervised_parity.py` |
| Audit script | `scripts/audit_multignn_supervised_parity.py` |
| Formal (mismatched) run | `notes/supervised_Small-HI_small_hi_legacy_supervised_gin_emlps_tds_100ep_seed1_summary.md`, `notes/eval_small_hi_legacy_supervised_gin_emlps_tds_100ep_seed1.md` |
| Upstream ckpt evaluated | `saved-models/checkpoint_multi-gin-eu-SmallHI-50epochs.tar` (pool path resolved in JSON) |

Smoke metrics (1 epoch, `--testing`): train/val/test argmax F1 = **0.0** — expected for a path check only; not paper-comparable.
