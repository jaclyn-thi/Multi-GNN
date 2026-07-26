# PaySim frozen D+ transfer — final

**Protocol:** frozen AMLWorld Small-HI D+ encoders (seeds 1–3) → PaySim dual extract (pre-3h + post-128) → PaperStyleMLP; primary stack = **pre-3h H+X**; best epoch by val AUPRC; threshold by val F1; `--train_fit_edge_znorm`.

## Exact transfer claim

A self-supervised contrastive Multi-GIN encoder (D+) pretrained on AMLWorld Small-HI and evaluated frozen on PaySim with a supervised downstream MLP on pre-3h H+X (train-fit edge z-norm; ports+tds+emlps+corrected reverse) yields test AUPRC 0.1116 ± 0.0265 and F1@0.5 0.0704 ± 0.0478 over three encoder seeds. This is frozen encoder transfer with target-graph structural featurization, not pure feature-space zero-shot and not the historical ports-only logistic PaySim diagnostic.

## Limitations

Schema placeholders map PaySim transaction type into AML currency/payment slots without semantic alignment; ports/TDS are recomputed on PaySim; test message-passing uses the full timeline graph (Multi-GNN inherent scope); TF deferred; no protocol-compatible published PaySim numerical baseline; random and AML-supervised FT encoders are secondary and excluded from the primary mean.

## Primary three-seed pre-3h H+X

| Metric | Mean ± sample SD | Median |
|--------|-----------------:|-------:|
| test AUPRC | 0.1116 ± 0.0265 | 0.1090 |
| test AUROC | 0.6443 ± 0.0851 | 0.6698 |
| test F1@0.5 | 0.0704 ± 0.0478 | 0.0750 |
| test F1@val-thr | 0.0889 ± 0.0393 | 0.0986 |

### Thesis table (Markdown)

```markdown
| Seed | Test AUPRC | Test AUROC | F1@0.5 | F1@val-thr |
|-----:|-----------:|-----------:|-------:|-----------:|
| 1 | 0.1393 | 0.7138 | 0.0750 | 0.0986 |
| 2 | 0.0864 | 0.6698 | 0.0205 | 0.0456 |
| 3 | 0.1090 | 0.5494 | 0.1156 | 0.1224 |
| mean±sd | 0.1116 ± 0.0265 | 0.6443 ± 0.0851 | 0.0704 ± 0.0478 | 0.0889 ± 0.0393 |
```

### Thesis table (LaTeX)

```latex
\begin{tabular}{rcccc}
\toprule
Seed & Test AUPRC & Test AUROC & F1@0.5 & F1@val-thr \\
\midrule
1 & 0.1393 & 0.7138 & 0.0750 & 0.0986 \\
2 & 0.0864 & 0.6698 & 0.0205 & 0.0456 \\
3 & 0.1090 & 0.5494 & 0.1156 & 0.1224 \\
mean$\pm$sd & 0.1116 ± 0.0265 & 0.6443 ± 0.0851 & 0.0704 ± 0.0478 & 0.0889 ± 0.0393 \\
\bottomrule
\end{tabular}
```

## Controls and ablations

- X-only: `{
  "test_auprc": 0.09608876668509789,
  "test_auroc": 0.8517032242115923,
  "test_f1_0.5": 0.0,
  "test_f1_val_thr": 0.0481320192528077,
  "source_role": "seed2"
}`
- pre-3h H-only aggregate: `{
  "stack": "pre3h_H_only",
  "test_auprc": {
    "mean": 0.10678869388899442,
    "sample_std": 0.05606738242361996,
    "median": 0.1358553295366196,
    "n": 3,
    "values": [
      0.1358553295366196,
      0.14235427217011215,
      0.042156479960251514
    ]
  },
  "test_auroc": {
    "mean": 0.6961292027079297,
    "sample_std": 0.16817634666700018,
    "median": 0.7557878106055184,
    "n": 3,
    "values": [
      0.826343397396382,
      0.7557878106055184,
      0.5062564001218888
    ]
  },
  "test_f1_0.5": {
    "mean": 0.04841021204508754,
    "sample_std": 0.025080584569941746,
    "median": 0.039323273891175126,
    "n": 3,
    "values": [
      0.029139685476410736,
      0.039323273891175126,
      0.07676767676767676
    ]
  },
  "test_f1_val_thr": {
    "mean": 0.049934758982446634,
    "sample_std": 0.015146516398218226,
    "median": 0.05659950192438307,
    "n": 3,
    "values": [
      0.05659950192438307,
      0.032598714416896234,
      0.06060606060606061
    ]
  },
  "val_auprc": {
    "mean": 0.07912494194623668,
    "sample_std": 0.03054695542103715,
    "median": 0.0896990566826728,
    "n": 3,
    "values": [
      0.1029799085065371,
      0.044695860649500124,
      0.0896990566826728
    ]
  }
}`
- post-128 H+X aggregate: `{
  "stack": "post128_HxX",
  "test_auprc": {
    "mean": 0.10562287664849133,
    "sample_std": 0.017715278606971297,
    "median": 0.0967471653895174,
    "n": 3,
    "values": [
      0.1260215407900214,
      0.0940999237659352,
      0.0967471653895174
    ]
  },
  "test_auroc": {
    "mean": 0.7036845114719197,
    "sample_std": 0.1767689203998731,
    "median": 0.6323508075192226,
    "n": 3,
    "values": [
      0.9049740618613432,
      0.5737286650351932,
      0.6323508075192226
    ]
  },
  "test_f1_0.5": {
    "mean": 0.05869047753821578,
    "sample_std": 0.015981645531629136,
    "median": 0.06057845593258939,
    "n": 3,
    "values": [
      0.0418487008507703,
      0.06057845593258939,
      0.07364427583128766
    ]
  },
  "test_f1_val_thr": {
    "mean": 0.08503628257435242,
    "sample_std": 0.00650437658829308,
    "median": 0.08241387074983111,
    "n": 3,
    "values": [
      0.08241387074983111,
      0.08025247971145176,
      0.09244249726177436
    ]
  },
  "val_auprc": {
    "mean": 0.13674418262562482,
    "sample_std": 0.01408593581394525,
    "median": 0.13400851520233725,
    "n": 3,
    "values": [
      0.13400851520233725,
      0.15199728481023872,
      0.12422674786429849
    ]
  }
}`
- Ensembles: pre3h=True post128=True

## Final answers (1–15)

**1_primary_pre3h_HxX_mean_pm_sd:** `{"test_auprc": "0.1116 \u00b1 0.0265", "test_auroc": "0.6443 \u00b1 0.0851", "test_f1_0.5": "0.0704 \u00b1 0.0478", "test_f1_val_thr": "0.0889 \u00b1 0.0393", "median_test_auprc": 0.10896015372354298}`

**2_x_only:** `{"test_auprc": 0.09608876668509789, "test_auroc": 0.8517032242115923, "test_f1_0.5": 0.0, "test_f1_val_thr": 0.0481320192528077, "source_role": "seed2"}`

**3_pre3h_H_only_outperforms_X_only:** `true`

**4_pre3h_H_improves_HxX_over_X:** `true`

**5_post128_vs_pre3h:** `{"pre3h_HxX_auprc_mean": 0.11155380572540688, "post128_HxX_auprc_mean": 0.10562287664849133, "post128_better": false, "note": "Do not select representation using test; post-128 is sensitivity only."}`

**6_seed_variability:** `{"test_auprc_sample_std": 0.026511836567450264, "test_f1_0.5_sample_std": 0.047767037463813375, "per_seed_auprc": [0.13926714559833023, 0.08643411785434744, 0.10896015372354298]}`

**7_pretrained_H_vs_random_H:** `{"seed2_pre3h_H_auprc": 0.14235427217011215, "random_pre3h_H_auprc": 0.10758112970602782, "pretrained_better": true}`

**8_pre3h_ensemble:** `{"test_auprc": 0.13251344328067327, "test_f1_0.5": 0.07110507246376813, "improves_vs_mean": true}`

**9_post128_ensemble:** `{"test_auprc": 0.11868063019569138, "test_f1_0.5": 0.05703855806525211, "improves_vs_post128_mean": true}`

**10_ft_vs_frozen_seed2:** `{"frozen_seed2_pre3h_HxX_auprc": 0.08643411785434744, "ft_seed2_pre3h_HxX_auprc": 0.07511091845441752, "ft_helps": false, "included_in_primary_aggregate": false}`

**11_cross_dataset_transfer_supported:** `true`

**12_schema_preprocessing_caveats:** `"PaySim type\u2192currency/payment slots are schema placeholders, not AML-semantic equivalence; ports/TDS recomputed on PaySim; train-fit edge z-norm (inductive); test MP graph includes all edges (Multi-GNN inherent scope); no TF; AML scalers not transferred."`

**13_published_comparisons:** `"None numerically protocol-compatible (FAIL). Methodological PARTIAL only (Papagei-style frozen probe / GFM narrative). Do not cite historical ~0.866 ports-only logistic PaySim AUROC as D+ transfer."`

**14_no_paysim_labels_updated_encoder:** `true`

**15_no_automatic_followup_training_submitted:** `true`

## Registry rows (document for later ingest)

Worker jobs must not write the thesis registry. Suggested rows after this aggregate:

```json
[
  {
    "family": "paysim_frozen_dplus_transfer",
    "role": "primary_three_seed_pre3h_HxX",
    "metric_summary": "0.1116 \u00b1 0.0265",
    "source_json": "/home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN/results/diagnostics/paysim_dplus_transfer_final.json"
  }
]
```

Optional append was skipped by default (`--append_registry` not implied) to avoid concurrent registry writes.
