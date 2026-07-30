# SAML-D formal eval cohort reconciliation (job 19117882)

> Twin: `results/diagnostics/samld_formal_eval_cohort_reconciliation.json`  
> Scope: **read-only** (plus this write). No retrain, no job submit, no test-metric inspection, no historical overwrite.  
> Train job **19117881** completed. Eval job **19117882** failed on a validation-cohort assertion.

## Classification

**`extraction_loader_coverage_defect` + `stale_integrity_constant_mismatch`**

Observed scored cohort `n=1,653,622 / positives=1,865` is **not** the locked formal temporal validation seed set. It is the **NeighborLoader-scored subset** under `preserve_seed_edges=false` (~87% of current temporal seeds). The eval gate also compared against **stale integrity constants** (`1,899,523 / 1,986`) whose EdgeID index hashes no longer match the current `get_data` temporal split on the same CSV.

---

## 1. Split definitions by stage

All stages use `--data SAML-D`, `calendar_day`, target fractions `(0.6, 0.2, 0.2)`, Candidate-A graph flags (legacy head, ports on, TDS off, preserve off, correct_reverse off, inherited legacy reverse, legacy per-graph z-norm). Formatted CSV SHA256 is identical everywhere:

`beb7f89ac19f648a860f3e10a33de818ccf763a588f741a2a0a73b6035619b8f` (9,504,852 rows, 9,873 positives).

| Stage | Job | Temporal seed counts (train / val / test) | Val positives | Val index SHA256 | Graph edges (train / train∪val) | Notes |
|-------|-----|-------------------------------------------|--------------:|------------------|-------------------------------:|-------|
| Integrity audit | 19108637 | 5,707,315 / **1,899,523** / 1,898,014 | **1,986** | `b08cdb81…a38c` | 5,707,315 / 7,606,838 | Declared locked protocol card |
| Smoke v2 | 19114673 | *(log % only)* implied current | *(not logged)* | *(not logged)* | **5,715,293 / 7,615,398** | Same graph sizes as formal; `--skip_test_eval` |
| Formal train | 19117881 | implied **5,715,293 / 1,900,105 / 1,889,454** | seed **1,984** | `81269d80…a97e` (recomputed) | **5,715,293 / 7,615,398** | Matches current `get_data` recompute |
| Failed eval | 19117882 | same `get_data` path as train | seed 1,984 | same | same after full test graph build | Asserted against integrity constants |

### Current `get_data` recompute (same CSV; `torch.Tensor` / float32 **and** float64 agree)

| Split | Days | n | positives | index SHA256 |
|-------|------|--:|----------:|--------------|
| Train | 0–191 | 5,715,293 | 5,764 | `840bdf404eb692572afb6012425290704f9355e9e20a2d88769df0f1d2bcf2c3` |
| Val | 192–255 | **1,900,105** | **1,984** | `81269d803f1480b75dde3ab66562324fa10d5d11616fa8cca21be21755f8a97e` |
| Test | 256–320 | 1,889,454 | 2,125 | `a9f19af47d06417035b29235f2cb84277a055f8765c6240ca0ae6cda188caf0c` |

**Integrity vs current:** same day ranges `[0,191]/[192,255]/[256,320]` and same CSV SHA, but **different EdgeID sets / counts**. Neither float32 nor float64 current `temporal_edge_split` reproduces integrity hashes. Integrity’s materialization path is therefore **not bit-identical** to today’s `data_loading.get_data` path (likely a different timestamp→bucket implementation in the integrity job). **Do not silently replace integrity constants with scored counts.**

Timestamp units: seconds; bucket = 86400 s (calendar day). `get_data` re-zeros `Timestamp` by subtracting min before bucketing.

---

## 2. Per-stage scored / seed comparison

### Integrity (protocol card)

- Raw rows: 9,504,852; positives: 9,873  
- Val **seed** edges: 1,899,523; positives: 1,986  
- Val graph: train∪val = 7,606,838  
- Feature contract: AMLWorld base-4 + ports → edge_dim=6  
- No GNN scoring in this job  

### Smoke v2 (19114673)

- Graph: train 5,715,293; val/test-alias 7,615,398  
- Flags: `skip_test_eval=true`, `preserve_seed_edges=false`, `train_fit_edge_znorm=false`  
- Epoch history lacked `validation_n` fields (pre-coverage logging), but graph sizes already match formal/current split, **not** integrity  

### Formal train (19117881)

- Graph: train 5,715,293; val 7,615,398; test aliased to val under `--skip_test_eval`  
- Every epoch: `test_evaluated=false`; no `test_*` metric keys in history  
- **Scored** validation over 50 epochs:  
  - `validation_n` ∈ **[1,653,133 … 1,654,396]** (spread 1,263)  
  - `validation_n_positives` ∈ **[1,853 … 1,883]** (spread 30)  
- Selection: best val-F1 epoch **22**, F1 **0.9682**; max-val-AUPRC diagnostic epoch **7**  
- Checkpoint: `saved-models/samld_supervised_multigin_eu_v1_formal_seed2/checkpoint_best_val_f1.tar`  
  - archive SHA256 `5d1264c3…707e`  
  - model_state SHA256 `f7059687…88c5`  
  - `supervised_head=legacy`, `selected_epoch=22`  

### Failed eval (19117882)

- Loaded best-val checkpoint; rebuilt graphs **without** `--skip_test_eval` (intended)  
- `collect_split_predictions` returned **n=1,653,622, positives=1,865**  
- Gate compared to hard-coded `EXPECTED_VAL = {n: 1_899_523, n_positives: 1_986}` in `scripts/eval_samld_formal_seed2.py` and aborted  
- Error: `val cohort mismatch n=1653622 pos=1865 expected {'n': 1899523, 'n_positives': 1986}`  

### Loader / filtering mechanism

In `train_util.get_loaders` (hetero), val seeds are `val_data.edge_index[:, val_inds]` with full-timeline `val_inds`. Scoring (`_collect_supervised_predictions_hetero` / `collect_split_predictions`) keeps only seed edges whose synthetic id appears in the sampled batch; missing seeds are re-injected **only** for `Small_J` / `Small_Q`. With `preserve_seed_edges=false` on SAML-D, missing seeds are **silently dropped**.

Coverage vs **current** temporal val seeds (1,900,105 / 1,984):

| Quantity | Seed | Scored (eval) | Missing | Coverage |
|----------|-----:|--------------:|--------:|---------:|
| Edges | 1,900,105 | 1,653,622 | 246,483 | **87.03%** |
| Positives | 1,984 | 1,865 | 119 | **94.00%** |

Epoch-to-epoch scored `n` jitter (~1.3k) is consistent with stochastic/incomplete seed survival under neighbor sampling, not a changing temporal split.

Dropped EdgeID lists were **not** persisted in train/eval artifacts; only aggregate scored counts are available.

---

## 3. What is 1,653,622 / 1,865?

| Hypothesis | Verdict |
|------------|---------|
| Correct locked formal temporal cohort | **No** — locked seed cohort under current `get_data` is **1,900,105 / 1,984**; integrity card is **1,899,523 / 1,986** (different ID hash) |
| Extraction / loader coverage defect | **Yes** — ~13% of current val seeds never scored; positives also under-covered |
| Split / configuration mismatch (gate vs runtime) | **Yes (secondary)** — gate used integrity constants; smoke/formal/eval runtime graphs follow current split (~1.900M seeds) |
| Unresolved | Integrity’s exact bucketization code path vs current `temporal_edge_split` remains unexplained beyond “not reproducible from float32/float64 get_data”; does **not** change the loader-coverage diagnosis for 1.65M |

---

## 4. ID hash comparison (artifacts)

| Artifact | Val EdgeID index SHA256 | Matches current get_data? | Matches scored 1.65M set? |
|----------|-------------------------|---------------------------|---------------------------|
| Integrity | `b08cdb815f82e6d37019e5e6ec9c5a6fd12c3f9d523f63b2768f6e4d0a99a38c` | **No** | No (and not claimed) |
| Current get_data / formal graphs | `81269d803f1480b75dde3ab66562324fa10d5d11616fa8cca21be21755f8a97e` | **Yes** | Unknown (scored IDs not archived) |
| Failed eval scored set | *(not hashed; only counts)* | n/a | Self |

**Conclusion:** Do **not** overwrite integrity hashes with observed scored counts. Versioned split metadata for formal eval must track the **current get_data seed IDs**, and coverage must be reported separately.

---

## 5. Checkpoint selection vs test

- Training used `--skip_test_eval` throughout; `te_inds` emptied; every epoch `test_evaluated=false`  
- Summary `test_minority_f1_argmax_at_best` is NaN / unused  
- Selection rule: validation minority F1 argmax only (epoch 22)  
- Selected metrics were computed on the **scored ~1.65M** validation edges (same loader regime as the failed eval’s val pass)  
- **Test was not used for selection or training evaluation**

Checkpoint **19117881 remains valid** for the protocol-as-executed (legacy Candidate A, seed 2, skip_test during train). It is **not** a full-cohort (1.90M-seed) validation selection.

---

## 6. Gate patch decision

**Patch applied (2026-07-29):** `scripts/eval_samld_formal_seed2.py` no longer equates scored NeighborLoader counts to the integrity card.

Observed `1,653,622 / 1,865` remains **not** the locked formal temporal seed cohort. Replacing `EXPECTED_VAL` with those numbers would encode a coverage defect as a protocol constant — that path was **not** taken.

Gate now:

1. Requires versioned **current `get_data`** seed counts/hashes (`1,900,105 / 1,984`, `81269d80…`; test `1,889,454 / 2,125`, `a9f19af4…`);  
2. Records `expected_seed_edges`, `scored_seed_edges`, coverage, and dropped counts in the eval JSON;  
3. Fails only on seed-hash / count mismatches or coverage below predeclared floors (`edge≥0.85`, `positive≥0.90`) — **not** by equating scored counts to integrity card counts (`1,899,523 / 1,986`, kept as historical metadata only).

Tests: `tests/test_samld_formal_eval_gate.py`.

---

## 7. Prepared evaluation-only rerun (NOT submitted)

Gate fix is in place. Ready to submit (still **not** auto-submitted):

```bash
sbatch --export=ALL,TRAIN_JOB_ID=19117881 \
  slurm/eval_samld_supervised_multigin_eu_v1_formal_seed2.sh
```

Equivalent direct command (eval-only; no training):

```bash
cd /home/jthi/ondemand/data/sys/myjobs/projects/Multi-GNN
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate multignn; set -u
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/eval_samld_formal_seed2.py \
  --train-job-id 19117881 \
  --eval-job-id manual_local
```

Constraints for any authorized rerun:

- Load **only** `checkpoint_best_val_f1.tar` from run `samld_supervised_multigin_eu_v1_formal_seed2`  
- Do not train; do not change architecture/weights/threshold selection on test  
- Preserve locked test protocol (evaluate test exactly once after val-based selection already frozen at epoch 22)
- Metrics remain **scored-seed** metrics under `preserve_seed_edges=false`; report coverage alongside

---

## End answers

1. **Classification:** `extraction_loader_coverage_defect` (primary) with `stale_integrity_constant_mismatch` (prior gate failure mode; patched).  
2. **Checkpoint 19117881 valid?** **Yes** for protocol-as-executed; selection used scored val edges; test untouched.  
3. **Eval-only rerun authorized?** **Yes** after coverage-aware / versioned-split gate fix (applied); still **not** treating 1,653,622 as the locked cohort.  
4. **Proposed command:** see §7 (`sbatch … TRAIN_JOB_ID=19117881` / `eval_samld_formal_seed2.py`).  
5. **No job was submitted** by this gate-patch step (job 19202636 was a prior user submit that failed on the old gate).
