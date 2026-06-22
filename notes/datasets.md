# Datasets

How to format and load transaction graph data for Multi-GNN. Folder names under `aml-data/` must match the `--data` argument.

| Dataset | Role | Split mode | Label column |
| ------- | ---- | ------------ | ------------ |
| **IBM / AMLWorld** (Small-HI, Small-LI, …) | Pretrain + in-domain AML eval | `calendar_day` (~60/20/20) | `Is Laundering` |
| **PaySim** | External fraud transfer (downstream only) | `hourly_step` (~60/20/20) | `Is Laundering` (= `isFraud`) |
| **SAML-D** | In-domain AML (eval under review) | `calendar_day` (default spec) | `Is Laundering` |

---

## IBM synthetic AML (Kaggle)

Source: [IBM AML transaction datasets](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data) (Egressy et al., NeurIPS 2023).

1. Download CSV(s) from Kaggle (e.g. `HI-Small_Trans.csv` for the small, high-illicitness split).
2. Format each file:

```bash
python format_kaggle_files.py /path/to/HI-Small_Trans.csv
```

This writes `formatted_transactions.csv` next to the input.

3. Copy each formatted file into its own folder under `aml-data/`:

```
aml-data/
  Small-HI/
    formatted_transactions.csv
  Small-LI/
    formatted_transactions.csv
```

The Kaggle release has several sizes and illicitness levels — one folder per split you use.

---

## PaySim (downstream fraud transfer)

PaySim is an **external edge-level fraud benchmark**: AML-pretrained encoder → PaySim linear probe. It is **not** pretraining data.

1. Place raw CSV under `aml-data/PaySim/` (e.g. `PS_20174392719_1491204439457_log.csv`).
2. Format and validate:

```bash
python scripts/validate_paysim_data.py --format-raw
python scripts/validate_paysim_data.py              # quick stats
python scripts/validate_paysim_data.py --load-graph --testing   # full loader smoke (slow)
```

Or format directly:

```bash
python format_paysim.py aml-data/PaySim/PS_*.csv -o aml-data/PaySim/formatted_transactions.csv
```

3. Extract + probe (use `embeddings/paysim/` so AMLWorld embeddings are not overwritten):

```bash
python embedding_extraction.py \
  --data PaySim --model gin \
  --unique_name hi_contrastive_proj_sym_20ep_bestckpt \
  --embeddings_dir embeddings/paysim \
  --reverse_mp --ego --ports --testing

python linear_probe.py \
  --unique_name hi_contrastive_proj_sym_20ep_bestckpt \
  --embeddings_dir embeddings/paysim \
  --model gin --class_weight model --testing
```

Random-init baseline: add `--random_init` to extraction; use any `--unique_name` (e.g. `random_init_gin`).

**Slurm:**

```bash
sbatch slurm/run_paysim_load_smoke.sh
UNIQUE_NAME=hi_contrastive_proj_sym_20ep_bestckpt sbatch slurm/run_paysim_extract_probe.sh
UNIQUE_NAME=random_init_gin RANDOM_INIT=1 sbatch slurm/run_paysim_extract_probe.sh
bash slurm/submit_paysim_transfer.sh
ENCODERS="hi_contrastive_proj_sym_20ep_bestckpt" bash slurm/submit_paysim_probe_variants.sh
```

**Notes:** Balance fields and `isFlaggedFraud` are excluded at format time. Test fraud rate ~0.33% vs train ~0.08% (fraud concentrated in later steps). Report **AUROC** as the primary metric.

**Dev results (Jun 2026):** pretrained sym+proj test AUROC **0.866** vs random-init **0.730** (`class_weight=model`). Full tables: [`downstream-eval-plan.md` § PaySim](downstream-eval-plan.md#paysim--status-jun-2026).

**Code:** `format_paysim.py`, `dataset_specs.py`, `dataset_splits.py`, `scripts/validate_paysim_data.py`, `tests/test_format_paysim.py`.

---

## SAML-D (in-domain AML; eval under review)

Same edge-level AML objective as Small-HI (`--data SAML-D`, `calendar_day`, `Is Laundering`).

```bash
python format_saml_d_files.py /path/to/SAML-D.csv
# ensure aml-data/SAML-D/formatted_transactions.csv exists

python main.py --data SAML-D --model gin --objective supervised \
  --reverse_mp --ego --ports --testing
```

**Slurm smoke:** `sbatch slurm/run_saml_d_supervised_smoke.sh` · control: `sbatch slurm/run_small_hi_supervised_smoke.sh`.

**Caveat (Jun 2026):** 1-epoch supervised smoke reported test F1 ~0.90 on SAML-D vs ~0.00 on Small-HI under identical flags. Treat SAML-D supervised numbers as **suspect** until eval protocol is fixed. See [`downstream-eval-plan.md` § SAML-D](downstream-eval-plan.md#saml-d--status-jun-2026).

---

## Formatted schema and splits

**Columns:** `EdgeID`, `from_id`, `to_id`, `Timestamp`, `Amount Sent`, `Sent Currency`, `Amount Received`, `Received Currency`, `Payment Format`, `Is Laundering`.

**Splits:** IBM and SAML-D use **calendar day** (~60/20/20). PaySim uses **hourly step** (~60/20/20). Illicit edges are heavily underrepresented — **AUROC** from the linear probe is usually more informative than F1 on short runs.
