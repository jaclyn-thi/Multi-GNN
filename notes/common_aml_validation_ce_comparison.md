# Common AMLWorld validation CE comparison

## 1. Artifact audit

```json
{
  "probe_predictions": {
    "per_example_logits_or_probs": false,
    "available": "aggregate final_probe_val_bce + epoch_history only",
    "cells_checked": [
      "/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/r198_only_lr_analysis/cells/direct_r198_infonce_40ep_seed2_linear_lr1e-3/epoch_03.json",
      "/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/r198_only_lr_analysis/cells/direct_r198_tfmoe_40ep_seed2_linear_lr2e-3/epoch_10.json",
      "/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/cells/direct_r198_infonce_40ep_seed2_linear_lr6p2e-3/epoch_10.json",
      "/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval/cells/direct_r198_tfmoe_40ep_seed2_linear_lr2e-3/epoch_10.json"
    ],
    "classifier_final_vs_selected": "final_probe_val_bce = PaperStyleMLP epoch 20; ranking metrics / best_probe_epoch = best val AUPRC epoch",
    "seed_only_excluded": true
  },
  "supervised_predictions": {
    "per_example_before_this_script": false,
    "checkpoints": {
      "final": "/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2/checkpoint_last.tar",
      "best_val": "/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/saved-models/small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2/checkpoint_best_val_f1.tar"
    },
    "native_ce_0_011657_source": "/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/supervised_Small-HI_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2_epoch_history.json epoch 50 train_loss (also ce_audit.json supervised_native.final_epoch.train_loss_weighted_ce)",
    "native_ce_0_011319_source": "/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/supervised_Small-HI_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2_epoch_history.json epoch 43 train_loss (best-validation epoch; TRAIN weighted CE, not validation)",
    "inference_necessary": false
  },
  "loss_definitions": {
    "probe": {
      "loss": "binary_cross_entropy_with_logits",
      "logits": "one_logit",
      "class_weights": null,
      "pos_weight": null,
      "reduction": "mean",
      "mlp_epochs": 20,
      "final_vs_selected": "final = last probe epoch BCE; selected = best_val_auprc epoch (ranking); both recorded separately"
    },
    "supervised": {
      "loss": "CrossEntropyLoss",
      "logits": "two_logit",
      "class_weights_source": "checkpoint config w_ce1/w_ce2",
      "reduction": "mean",
      "train_loss_aggregation": "example-mean of batch means (sum loss*n / N)",
      "validation_ce_logged": false
    }
  },
  "tfmoe_ablation_untouched": true,
  "test_accessed": false
}
```

## 2. Loss definitions

- **Probe:** one-logit `binary_cross_entropy_with_logits`, no class/pos weights, `reduction=mean`, 20 PaperStyleMLP epochs. `final_probe_val_bce` = epoch 20; ranking uses `best_probe_epoch` by val AUPRC.
- **Supervised:** two-logit `CrossEntropyLoss(weight=[w_ce1,w_ce2], reduction=mean)`. Weights read from checkpoint `config` (not hardcoded). Logged epoch loss is **train** example-mean weighted CE; validation CE was not logged.

## 3. Cohort

- Distinct EdgeID hashes: `['8e0a99e79e61f6324fe6d2461371b99c95b1e0d751f78bf3999e42d80f21208e', 'c89707af6a7458f5a5cfee613eb6832dfab8ce19d5456cdbf40ac6a27f5c6166']`
- Probe rows use full-subgraph embedding val EdgeIDs; supervised uses inferred val EdgeIDs. If hashes match, cohorts are identical.

## 4. Main table (final classifier / supervised checkpoints)

| Method | Feature protocol | Model checkpoint | Classifier meaning | Native final val loss | Common unweighted val CE | Common supervised-weighted val CE | n | positives | ID hash |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| DIRECT_R198 | R198_only | direct_r198_infonce_40ep_seed2_linear_lr1e-3 SSL ep3 | final_probe_epoch_20 | 0.004982 | 0.004982 | N/A | 965464 | 1036 | `c89707af6a7458f5` |
| DIRECT_H_TFMOE | R198_only | direct_r198_tfmoe_40ep_seed2_linear_lr2e-3 SSL ep10 | final_probe_epoch_20 | 0.004028 | 0.004028 | N/A | 965464 | 1036 | `c89707af6a7458f5` |
| DIRECT_R198 | R198_X_TF | direct_r198_infonce_40ep_seed2_linear_lr6p2e-3 SSL ep10 | final_probe_epoch_20 | 0.003700 | 0.003700 | N/A | 965464 | 1036 | `c89707af6a7458f5` |
| DIRECT_H_TFMOE | R198_X_TF | direct_r198_tfmoe_40ep_seed2_linear_lr2e-3 SSL ep10 | final_probe_epoch_20 | 0.003585 | 0.003585 | N/A | 965464 | 1036 | `c89707af6a7458f5` |
| DIRECT_R198 | R198_only | direct_r198_infonce_40ep_seed2_linear_lr1e-3 SSL ep40 | final_probe_epoch_20 | 0.006288 | 0.006288 | N/A | 965464 | 1036 | `c89707af6a7458f5` |
| DIRECT_H_TFMOE | R198_only | direct_r198_tfmoe_40ep_seed2_linear_lr2e-3 SSL ep40 | final_probe_epoch_20 | 0.004239 | 0.004239 | N/A | 965464 | 1036 | `c89707af6a7458f5` |
| DIRECT_R198 | R198_X_TF | direct_r198_infonce_40ep_seed2_linear_lr6p2e-3 SSL ep40 | final_probe_epoch_20 | 0.003969 | 0.003969 | N/A | 965464 | 1036 | `c89707af6a7458f5` |
| DIRECT_H_TFMOE | R198_X_TF | direct_r198_tfmoe_40ep_seed2_linear_lr2e-3 SSL ep40 | final_probe_epoch_20 | 0.003707 | 0.003707 | N/A | 965464 | 1036 | `c89707af6a7458f5` |
| supervised_MultiGIN | supervised_raw_edge_features | small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2 ep50 | final_supervised_epoch_50 |  | 0.006994 | 0.015480 | 965406 | 1035 | `8e0a99e79e61f632` |
| supervised_MultiGIN | supervised_raw_edge_features | small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2 ep43 | best_validation_supervised_checkpoint |  | 0.004958 | 0.012886 | 965406 | 1035 | `8e0a99e79e61f632` |

Selected-classifier probe rows (not final) are in the JSON under `rows` with `is_validation_selected_classifier=true`.

## 5. Direct answers

1. Existing predictions sufficient? **False** — Probe aggregates existed; supervised per-example val preds were missing (required one inference job).
2. Slurm job submitted? **True** — {"job_id": "19458946", "partition": "mit_preemptable", "account": "mit_general", "qos": "normal", "script": "slurm/run_common_aml_validation_ce.sh", "dependency_on_tfmoe_ablation": false, "advanced_account": false, "test_eval": false, "max_jobs_for_task": 1, "resubmitted_after_argparse_fix": true, "prior_failed_job": "19446400"}
3. Native losses directly comparable? **False** — Probe native loss = unweighted val BCE; supervised logged loss = TRAIN weighted CE. Different objective, split, and weighting.
4. Common unweighted CE values:
```json
{
  "DIRECT_R198|R198_only|final_probe_epoch_20|ssl3": 0.004981516394764185,
  "DIRECT_H_TFMOE|R198_only|final_probe_epoch_20|ssl10": 0.0040284316055476665,
  "DIRECT_R198|R198_X_TF|final_probe_epoch_20|ssl10": 0.003700042376294732,
  "DIRECT_H_TFMOE|R198_X_TF|final_probe_epoch_20|ssl10": 0.0035847709514200687,
  "DIRECT_R198|R198_only|final_probe_epoch_20|ssl40": 0.006287924479693174,
  "DIRECT_H_TFMOE|R198_only|final_probe_epoch_20|ssl40": 0.00423886813223362,
  "DIRECT_R198|R198_X_TF|final_probe_epoch_20|ssl40": 0.003968931268900633,
  "DIRECT_H_TFMOE|R198_X_TF|final_probe_epoch_20|ssl40": 0.0037070424295961857,
  "supervised_MultiGIN|supervised_raw_edge_features|final_supervised_epoch_50|sslNone": 0.006994187076211333,
  "supervised_MultiGIN|supervised_raw_edge_features|best_validation_supervised_checkpoint|sslNone": 0.004958493869670289
}
```
5. Common supervised-weighted CE values:
```json
{
  "DIRECT_R198|R198_only|final_probe_epoch_20|ssl3": null,
  "DIRECT_H_TFMOE|R198_only|final_probe_epoch_20|ssl10": null,
  "DIRECT_R198|R198_X_TF|final_probe_epoch_20|ssl10": null,
  "DIRECT_H_TFMOE|R198_X_TF|final_probe_epoch_20|ssl10": null,
  "DIRECT_R198|R198_only|final_probe_epoch_20|ssl40": null,
  "DIRECT_H_TFMOE|R198_only|final_probe_epoch_20|ssl40": null,
  "DIRECT_R198|R198_X_TF|final_probe_epoch_20|ssl40": null,
  "DIRECT_H_TFMOE|R198_X_TF|final_probe_epoch_20|ssl40": null,
  "supervised_MultiGIN|supervised_raw_edge_features|final_supervised_epoch_50|sslNone": 0.015480412957826796,
  "supervised_MultiGIN|supervised_raw_edge_features|best_validation_supervised_checkpoint|sslNone": 0.012886049444027669
}
```
6. Recomputed match logged? {"probe_unweighted_equals_native_final_bce": true, "supervised_weighted_val_vs_logged_train_ce": "Not expected to match: logged values are TRAIN CE; recomputed are VAL CE."}
7. Final vs validation-selected: see JSON `answers.7_final_vs_selected` (probe final = MLP ep20; supervised final = ep50; selected SSL/probe/supervised flagged separately).
8. Test data accessed? **no**
9. Active TFMOE jobs or their code paths modified? **no**

## 6. Proposed follow-up (NOT launched)

Deterministic PaperStyleMLP re-probe on existing full-subgraph embeddings for the cells listed in the audit, saving `edge_id`, `y`, and one-logit `logits` for final (ep20) and optionally selected probe epochs, to enable EdgeID-aligned common weighted CE for probes. Single dedicated job; do not touch TFMOE ablation DAG.

