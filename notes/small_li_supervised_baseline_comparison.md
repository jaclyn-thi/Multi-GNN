
# Small-LI Supervised Baseline Comparison

> **Architecture note (added Jul 2026).** This "supervised baseline" run used the **current
> embedding-head architecture** (`--supervised_head embedding`, the default), i.e. the
> `edge representation → 128-d embedding head → Linear(128,50) → Linear(50,2)` control. It is
> **NOT** the legacy IBM Multi-GNN / Egressy et al. reproduction (`--supervised_head legacy`,
> the `3·h → 50 → 25 → 2` head). Do not treat this row as the Egressy/Multi-GNN baseline.
>
> In the table below, the **`F1`** column is the **validation-tuned-threshold** F1 (diagnostic
> only, NOT paper-compatible), while **`F1@0.5`** corresponds to the paper-compatible
> **argmax over two-class logits** decision rule. These are different metrics and must not be
> compared or merged. New runs record the paper-compatible metric as `paper_argmax` and the tuned
> metric as `validation_tuned_threshold` (see `scripts/evaluate_supervised_gnn.py`), and select the
> checkpoint by validation minority F1 (`checkpoint_best_val_f1.tar`) rather than the final epoch.
>
> For the legacy (Egressy-head) counterpart, see
> [`small_li_legacy_supervised_scout.md`](small_li_legacy_supervised_scout.md).

Updated from `results/diagnostics/supervised_small_li_gin_emlps_tds_seed1.json` and the current Small-LI probe artifacts.

## Thesis-Relevant Takeaway

The supervised GINe final checkpoint is not competitive with the frozen SSL probes or engineered-feature probes on Small-LI ranking metrics. Its test AUROC is high (0.9313), but AUPRC is only 0.0060; practical alerting is especially weak, with P@100 0.0000, P@500 0.0020, and lift@500 2.9.

This supports a conservative reading: Small-LI is intrinsically difficult under very low prevalence, and thresholded F1 can be misleading when the selected threshold buys recall by accepting very low precision. The weak supervised final checkpoint does not prove SSL failure; the stronger frozen-probe rankings suggest prevalence and evaluation protocol are central.

## Supervised Final Checkpoint

| Split | AUROC | AUPRC | F1 | F1@0.5 | Precision | Recall | P@100 | P@500 | P@1000 | lift@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.9535 | 0.0091 | 0.0211 | 0.0000 | 0.0107 | 0.7247 | 0.0100 | 0.0140 | 0.0130 | 23.9 |
| train | 0.9533 | 0.0065 | 0.0150 | 0.0000 | 0.0076 | 0.7045 | 0.0200 | 0.0080 | 0.0090 | 17.8 |
| test | 0.9313 | 0.0060 | 0.0132 | 0.0000 | 0.0066 | 0.7893 | 0.0000 | 0.0020 | 0.0010 | 2.9 |

## Current References

| Reference | AUROC | AUPRC | F1 | F1@0.5 |
|---|---:|---:|---:|---:|
| Supervised final checkpoint | 0.9313 | 0.0060 | 0.0132 | 0.0000 |
| Plain SSL `embedding` (`cw=model`) | 0.8988 | 0.0166 | 0.0522 | 0.0524 |
| Plain SSL `embedding+raw` (`cw=model`) | 0.9093 | 0.0272 | 0.0757 | 0.0809 |
| Best generic Small-LI sweep AUPRC | 0.9236 | 0.0496 | 0.0742 | 0.0925 |

## Interpretation

- The supervised final checkpoint has high recall at the selected threshold (0.7893) but precision of only 0.0066, so it is not operationally useful.
- Frozen SSL plus features produces much stronger ranking. The best Small-LI sweep AUPRC is 0.0496, compared with 0.0060 for supervised final.
- Thesis-safe framing: Small-LI is a low-prevalence ranking task; alert-budget precision/lift should be emphasized over AUROC or broad-recall thresholded F1.

Artifacts: `results/diagnostics/supervised_small_li_gin_emlps_tds_seed1.json`, `results/diagnostics/probe_sweep_small_li_current_protocol.json`.
