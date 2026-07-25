# Evaluation protocols

Status: **canonical** · Date: 2026-07-22

Never place F1 from **argmax**, **fixed 0.5**, and **validation-selected threshold** in one unlabeled column. Never substitute **random-40** for the primary **temporal** result.

---

## 1. Multi-GIN supervised paper comparison

| Item | Rule |
|------|------|
| Dataset | Small-HI (paper Multi-GIN+EU) |
| Features | ports + ego + reverse_mp + emlps; **TDS off**; edge_dim=6 |
| Head | `--supervised_head legacy` |
| Checkpoint | Best **validation minority F1** (`checkpoint_best_val_f1.tar`) |
| Test decision | **Two-class argmax** over logits (`paper_argmax`) |
| Reported metric | Minority-class F1 (and companion AUROC/AUPRC if logged) |
| Paper target | 0.6479 ± 0.0122 |
| Formal aggregate | seeds 1–3 → 0.660 ± 0.060 (mean yes; variance no) |

Val-tuned threshold F1 on supervised logits is **diagnostic only** (`paper_comparable=false`).

---

## 2. Thesis temporal frozen evaluation (primary SSL)

| Item | Rule |
|------|------|
| Split | Temporal train / validation / test on transactions |
| Encoder | Frozen after contrastive (or other SSL) pretrain |
| Features | Probe stacks **X** (raw), **H** (embedding), **H‖X** (and morph variants when noted) |
| Representation source | `post_embedding` (128-d) vs `pre_embedding_3h` — always labeled |
| Ranking | AUROC, AUPRC |
| Classification F1 | Report **fixed 0.5** and **val-selected threshold** in **separate** fields/columns |
| Checkpoint for SSL | Not AML-F1 selected during pretrain (Papagei-style); use last or morph/loss policy as documented per run |

---

## 3. Random-40 diagnostic

| Item | Rule |
|------|------|
| Construction | 40% of edges randomly held for eval (exact construction in run JSON / script) |
| Status | **Transductive / diagnostic** |
| Use | Never enters the primary temporal thesis table; never replaces temporal claims |

---

## 4. GCPAL published comparison (labeling rules)

| Item | Rule |
|------|------|
| Published table | GCPAL paper Table 2 (training ratio / F1 as reported there) |
| This repo | **Inspired / reimplemented under assumptions** — paper code missing |
| Required label | `NOT AN EXACT GCPAL REPRODUCTION` on every txn-node result block |
| Comparability | Do not mark `paper_comparable=true` for GCPAL Table 2 |

Failed forensic audits that never produced complete metrics contribute **no** numeric comparison rows.
