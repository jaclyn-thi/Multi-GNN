# SAML-D shared-core split reconciliation (pre–Phase 2)

> Twin: `results/diagnostics/samld_shared_core_split_reconciliation.json`  
> Scope: **read-only**. No Slurm jobs, no training, no embedding load, no TF `features.npy`, no cache rewrite, no code changes.

## Verdict

**Primary cause: different calendar-boundary convention (raw vs rezeroed timestamps).**

Same splitter (`dataset_splits.temporal_edge_split`), same mode (`calendar_day`), same fractions `(0.6, 0.2, 0.2)`, and the **same nominal day cuts** `[0–191] / [192–255] / [256–320]` — but integrity buckets on **raw** `Timestamp`, while `get_data`, supervised formal/smoke, and the Phase-1 TF cache bucket after **`Timestamp -= min(Timestamp)`**.

Because `ts_min = 38129` and `38129 % 86400 ≠ 0`, day walls shift by 38129 s. Edge membership at the train/val and val/test boundaries therefore differs even though reported day ranges look identical.

**Secondary label:** the integrity protocol card is **stale relative to the current loader** (not because the CSV changed).

---

## 1. Split sources (exact)

| Source | Function | Timestamp input | Dtype into splitter | Inclusion |
|--------|----------|-----------------|---------------------|-----------|
| **Integrity** (`scripts/validate_saml_d_integrity.py`, job 19108637) | `temporal_edge_split` | **raw** CSV `Timestamp` (no rezero) | `torch.long` | full CSV → train/val/**test** |
| **Supervised smoke/formal** (`data_loading.get_data`) | `temporal_edge_split` | **`Timestamp -= min`** then split | `torch.Tensor` → float32 | full CSV; smoke/formal used `--skip_test_eval` for graphs but split still defines all three partitions |
| **Frozen-transfer scout** | same `get_data` extract path | rezero (loader) | float32 | train/val seeds extracted; **coverage gates still cite integrity counts** (`EXPECTED_COUNTS` = 5,707,315 / 1,899,523) |
| **Phase-1 TF cache** (`build_temporal_flow_causal_cache` / phase1 wrapper) | `temporal_edge_split` | **`Timestamp -= min`** before split | `torch.Tensor` → float32 | **train∪val only** retained in cache; **no test file** |

Shared parameters everywhere: `split_mode=calendar_day`, `bucket_sec=86400`, `split_fractions=(0.6,0.2,0.2)`, dataset `SAML-D`.

CSV: `aml-data/SAML-D/formatted_transactions.csv`, n=9,504,852, `EdgeID` unique but **≠ row index** (formatter assigns IDs then sorts by time).

---

## 2. Observed count cards

| Protocol | train n / pos | val n / pos | test n / pos |
|----------|--------------:|------------:|-------------:|
| Locked integrity card | 5,707,315 / 5,751 | 1,899,523 / 1,986 | 1,898,014 / 2,136 |
| Current loader / Phase-1 cache | 5,715,293 / 5,764 | 1,900,105 / 1,984 | 1,889,454 / 2,125 |

Δ train = **+7,978**; Δ val = **+582** (= −7,978 from locked-val into train + **+8,560** from locked-test into val); Δ test = **−8,560**.

---

## 3. EdgeID-only comparison (no feature matrices)

Arrays used: Phase-1 `split_train_edge_id.npy`, `split_val_edge_id.npy`; locked/current partitions reconstructed from CSV columns `EdgeID`, `Timestamp`, `Is Laundering` only.

### Exact sizes + hashes

| Set | n | ordered SHA256 | set (sorted-unique) SHA256 |
|-----|--:|----------------|----------------------------|
| Cache train | 5,715,293 | `840bdf404eb692572afb6012425290704f9355e9e20a2d88769df0f1d2bcf2c3` | `ee765f7030c617fb9825f483a81e4458108c8ab0a95754a89c17f8f55e731a52` |
| Cache val | 1,900,105 | `81269d803f1480b75dde3ab66562324fa10d5d11616fa8cca21be21755f8a97e` | `81269d803f1480b75dde3ab66562324fa10d5d11616fa8cca21be21755f8a97e` |
| Locked train (EdgeIDs) | 5,707,315 | `b5c8e1c3129157f643b3af2afc3f3113a18dd32f82095abae9b810d7ca9f2ee3` | `290713933cc655e9c70984bc3cb7f575ab26a03b8078a1337cda58892054935f` |
| Locked val | 1,899,523 | `b08cdb815f82e6d37019e5e6ec9c5a6fd12c3f9d523f63b2768f6e4d0a99a38c` | `b08cdb815f82e6d37019e5e6ec9c5a6fd12c3f9d523f63b2768f6e4d0a99a38c` |
| Locked test | 1,898,014 | `52d83d522af0783e9c1eb9984a47fe0c65bf95e5439afb6e63469219afa9d1aa` | `52d83d522af0783e9c1eb9984a47fe0c65bf95e5439afb6e63469219afa9d1aa` |
| Current loader train EdgeIDs | 5,715,293 | = cache train | = cache train set |
| Current loader val EdgeIDs | 1,900,105 | = cache val | = cache val |
| Current loader test EdgeIDs | 1,889,454 | `a9f19af47d06417035b29235f2cb84277a055f8765c6240ca0ae6cda188caf0c` | same |

Integrity published `index_sha256` values hash **positional row indices**, not EdgeID order. They reproduce exactly under `integrity_raw_long`. Phase-1 cache stores **EdgeID** arrays; they match the **current rezero loader** bit-for-bit (ordered and as sets).

### Cache vs locked overlaps

| Query | train | val | test |
|-------|------:|----:|-----:|
| cache-train ∩ locked-* | **5,707,315** | **7,978** | **0** |
| cache-val ∩ locked-* | **0** | **1,891,545** | **8,560** |

IDs unique to each protocol (boundary swap):

- only in current/cache train (not locked train): **7,978** — all from **locked val**
- only in locked val (not current val): **7,978** — all move to **current train**
- only in current/cache val (not locked val): **8,560** — all from **locked test**
- only in locked test (not current test): **8,560** — all move to **current val**
- current test ⊂ locked test (no current-only test IDs)

---

## 4. Did locked-test EdgeIDs enter the Phase-1 cache?

**Yes — 8,560 locked-test EdgeIDs are in Phase-1 cache val.**

They are **not** in the current loader’s test partition (`current_test ∩ cache_train∪val = 0`). Under the rezero convention they are validation edges; under the integrity/raw convention they were test.

No Phase-1 `split_test_edge_id.npy` was written; leakage is vs the **integrity-card** test set, not vs the current loader test set.

---

## 5. Cause classification

| Hypothesis | Verdict |
|------------|---------|
| Stale integrity card | **Secondary yes** — card does not match current `get_data` |
| Changed split implementation | **No** — same `temporal_edge_split` |
| **Different calendar boundary convention** | **Primary yes** — raw vs `Timestamp-=min`; `ts_min%86400=38129` |
| Split recomputed after filtering | **No** — both split on full CSV first; cache only drops test **after** split |
| Row-index / EdgeID confusion | **Not the count delta** — explains hash *identity* differences (integrity hashes positions; cache stores EdgeIDs), not the ±7978/8560 membership shift |
| Other | float32 vs long on **raw** timestamps can move 1 edge at boundaries (`raw_float32` val n=1,899,522); negligible vs rezero effect |

Reproduction:

- `integrity_raw_long` → locked counts + locked positional hashes  
- `get_data_rezero_float32` / `rezero_long` → Phase-1 / formal counts; cache EdgeIDs identical  

---

## 6. Canonical split recommendation (for mixed training)

**Use the current loader / Phase-1 cache convention:**

`Timestamp -= min(Timestamp)` → `temporal_edge_split(..., calendar_day, (0.6,0.2,0.2))`

Suggested ID: `samld_calendar_day_rezero_v1`

| Split | n | positives | EdgeID ordered SHA256 |
|-------|--:|----------:|------------------------|
| train | 5,715,293 | 5,764 | `840bdf40…f2c3` |
| val | 1,900,105 | 1,984 | `81269d80…a97e` |
| test | 1,889,454 | 2,125 | `a9f19af4…af0c` |

**Why**

1. Matches `data_loading.get_data`, supervised SAML-D smoke/formal graphs, and the existing Phase-1 TF cache (no rebuild needed for split alignment).  
2. Integrity/raw card is already known-stale vs formal training (`notes/samld_formal_eval_cohort_reconciliation.md`).  
3. Frozen-transfer should stop gating coverage against integrity 5.707M/1.900M and use these hashes/counts instead.

**Do not** mix integrity-card EdgeID expectations with rezero caches without an explicit remap.

---

## 7. Confirmations

- No Slurm jobs submitted  
- No encoder training / embedding load / probing  
- No TF feature matrix load (`features.npy` unused)  
- No cache rewrite / code change  
- No test-metric evaluation  

Stop for human review before Phase 2.
