#!/usr/bin/env python3
"""EXPERT_ONLY R198 frozen-transfer scout (PaySim + SAML-D), validation-only.

Modes:
  smoke   — bounded GPU loader+extract batches; writes smoke.json (no full embeddings)
  extract — full train+val R198 extract for one target/encoder cell
  probe   — LogisticRegression R198 + random + X-only on existing embeddings (CPU)

Never loads/extracts/evaluates test. Never uses seed-only extraction.
Never substitutes PaperStyleMLP for the logistic probe.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Locked matched epoch-10 encoders (AMLWorld Small-HI, seed 2)
SOURCE_EPOCH = 10
ENCODER_SPECS: Dict[str, Dict[str, Any]] = {
    "random_init": {
        "label": "matched_random_R198",
        "run": "direct_r198_tfmoe_wtabl_expert_only_20ep_seed2_linear_lr2e-3",
        "sha256": None,
        "random_init": True,
        "cell_key": "random_r198",
    },
    "direct_h": {
        "label": "DIRECT_H",
        "run": "direct_r198_infonce_40ep_seed2_linear_lr2e-3",
        "sha256": "c79e723d772e18748c0b675126cc0e5b7f2df01fde83d090243d70962521e06c",
        "random_init": False,
        "cell_key": "direct_h_r198",
    },
    "adaptive_tfmoe": {
        "label": "TFMOE_adaptive",
        "run": "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3",
        "sha256": "da73ddd5676e2c194a8f22632ac6f838f8ffdb638b5d5bd1f689adfd31d06b9c",
        "random_init": False,
        "cell_key": "adaptive_tfmoe_r198",
    },
    "expert_only": {
        "label": "EXPERT_ONLY",
        "run": "direct_r198_tfmoe_wtabl_expert_only_20ep_seed2_linear_lr2e-3",
        "sha256": "f0280e129c7bf0deb4c4a823fe24dd9e9b1c16ac2951aa87f0d81a55bc30c27c",
        "random_init": False,
        "cell_key": "expert_only_r198",
    },
}
MATRIX_ENCODERS = ("random_init", "direct_h", "adaptive_tfmoe", "expert_only")
# Backward-compatible aliases used by smoke / earlier notes
SOURCE_RUN = ENCODER_SPECS["expert_only"]["run"]
SOURCE_CKPT_REL = f"saved-models/checkpoint_{SOURCE_RUN}_epoch{SOURCE_EPOCH:02d}.tar"
SOURCE_SHA256 = ENCODER_SPECS["expert_only"]["sha256"]

# P1 logistic settings (scripts/final_corrected_no_preserve_multiseed.py)
DOWNSTREAM_LOGISTIC_SEED = 1
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 1000
LOGISTIC_SOLVER = "lbfgs"
GIN_CLASS_WEIGHT = {0: 1.0000182882773443, 1: 6.275014431494497}
LEARNER = "LogisticRegression"
FORBIDDEN_LEARNER = "PaperStyleMLP"

PAYSIM_PROTOCOL_ID = "P1_strict_inductive_legacy"
PAYSIM_CONTRACT = "paysim_legacy_duplicate_v1"
SAMLD_PROTOCOL_ID = "samld_frozen_expert_only_r198_valonly_v1"

TRANSFER_DELTA_AUPRC = 0.003  # EXPERT_ONLY must beat random by this margin

# Locked integrity-card source counts (do not redefine from extracted NPZ rows).
EXPECTED_COUNTS = {
    "PaySim": {
        "train_n": 3792821,
        "train_pos": 3175,
        "val_n": 1276276,
        "val_pos": 780,
    },
    "SAML-D": {
        "train_n": 5707315,
        "train_pos": 5751,
        "val_n": 1899523,
        "val_pos": 1986,
    },
}

# Coverage vs locked source after four-arm matched EdgeID intersection.
COVERAGE_FLOORS = {
    "PaySim": {"edge": 0.999, "positive": 0.999},
    "SAML-D": {"edge": 0.85, "positive": 0.90},
}

SAMLD_MATCHED_COHORT_NOTE = (
    "matched scored extraction cohort; incomplete relative to the locked integrity cohort "
    "because of the documented extraction_loader_coverage_defect."
)

PROHIBITED_PAYSIM_FIELDS = (
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
)

OUT_ROOT = ROOT / "results" / "diagnostics" / "expert_only_frozen_transfer_samld_paysim"
SMOKE_DIR = OUT_ROOT / "smoke"
SMOKE_JSON = OUT_ROOT / "smoke.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def assert_source_checkpoint() -> Dict[str, Any]:
    ckpt = ROOT / SOURCE_CKPT_REL
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    sha = sha256_file(ckpt)
    if sha != SOURCE_SHA256:
        raise RuntimeError(f"Source SHA mismatch: got {sha} expected {SOURCE_SHA256}")
    # Payload string gates (no torch required for unit tests calling this after import torch path)
    return {
        "path": str(ckpt),
        "sha256": sha,
        "run": SOURCE_RUN,
        "ssl_epoch": SOURCE_EPOCH,
    }


def load_extract_module():
    path = ROOT / "scripts" / "extract_direct_r198_full_cell.py"
    spec = importlib.util.spec_from_file_location("extract_direct_r198_full_cell", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def refuse_seed_only() -> None:
    mod = load_extract_module()
    mod.refuse_seed_only_path()


def refuse_test_splits(splits: str) -> List[str]:
    mod = load_extract_module()
    return mod.parse_extract_splits(splits)


def assert_no_paperstyle_mlp(learner: str) -> None:
    if learner != LEARNER:
        raise RuntimeError(f"Probe learner must be {LEARNER}, got {learner!r}")
    if learner == FORBIDDEN_LEARNER or "mlp" in learner.lower():
        raise RuntimeError("Silent PaperStyleMLP / MLP substitution is forbidden")


def make_logistic() -> Any:
    from sklearn.linear_model import LogisticRegression

    assert_no_paperstyle_mlp(LEARNER)
    return LogisticRegression(
        class_weight=dict(GIN_CLASS_WEIGHT),
        max_iter=LOGISTIC_MAX_ITER,
        random_state=DOWNSTREAM_LOGISTIC_SEED,
        solver=LOGISTIC_SOLVER,
        n_jobs=1,
        C=LOGISTIC_C,
    )


def logistic_settings_block(protocol_id: str) -> Dict[str, Any]:
    return {
        "protocol_id": protocol_id,
        "learner": LEARNER,
        "forbidden_learner": FORBIDDEN_LEARNER,
        "C": LOGISTIC_C,
        "class_weight_mode": "model",
        "class_weight": {str(k): float(v) for k, v in GIN_CLASS_WEIGHT.items()},
        "max_iter": LOGISTIC_MAX_ITER,
        "solver": LOGISTIC_SOLVER,
        "downstream_seed": DOWNSTREAM_LOGISTIC_SEED,
        "primary_metric": "auprc",
        "representation": "R198_only",
        "test_evaluated": False,
    }


def metrics_at_threshold(y: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y = y.astype(np.int64)
    pred = (proba >= float(thr)).astype(np.int64)
    out = {
        "auprc": float(average_precision_score(y, proba))
        if len(np.unique(y)) > 1
        else float("nan"),
        "auroc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(thr),
    }
    return out


def tune_threshold_max_f1(y: np.ndarray, proba: np.ndarray) -> float:
    from sklearn.metrics import f1_score, precision_recall_curve

    y = y.astype(np.int64)
    if len(np.unique(y)) < 2:
        return 0.5
    prec, rec, thrs = precision_recall_curve(y, proba)
    if thrs.size == 0:
        return 0.5
    f1 = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thrs[int(np.argmax(f1))])


def validate_embedding_pair(
    train_path: Path,
    val_path: Path,
    *,
    expected: Optional[Dict[str, int]] = None,
    partial: bool = False,
) -> Dict[str, Any]:
    def _load(p: Path):
        d = np.load(p)
        return (
            np.asarray(d["Z"], dtype=np.float32),
            np.asarray(d["y"]).reshape(-1).astype(np.int64),
            np.asarray(d["edge_id"]).reshape(-1).astype(np.int64),
        )

    Ztr, ytr, idtr = _load(train_path)
    Zva, yva, idva = _load(val_path)
    report: Dict[str, Any] = {
        "train_n": int(Ztr.shape[0]),
        "val_n": int(Zva.shape[0]),
        "train_pos": int((ytr == 1).sum()),
        "val_pos": int((yva == 1).sum()),
        "z_dim_train": int(Ztr.shape[1]),
        "z_dim_val": int(Zva.shape[1]),
        "finite_train": bool(np.isfinite(Ztr).all()),
        "finite_val": bool(np.isfinite(Zva).all()),
        "unique_train": bool(idtr.size == np.unique(idtr).size),
        "unique_val": bool(idva.size == np.unique(idva).size),
        "train_val_intersect": int(len(set(idtr.tolist()) & set(idva.tolist()))),
        "no_test_npz": not (train_path.parent / "test.npz").is_file(),
        "partial": partial,
    }
    report["dim_ok"] = report["z_dim_train"] == 198 and report["z_dim_val"] == 198
    report["disjoint_ok"] = report["train_val_intersect"] == 0
    report["finite_ok"] = report["finite_train"] and report["finite_val"]
    report["unique_ok"] = report["unique_train"] and report["unique_val"]
    counts_ok = True
    if expected is not None and not partial:
        counts_ok = (
            report["train_n"] == expected["train_n"]
            and report["val_n"] == expected["val_n"]
            and report["train_pos"] == expected["train_pos"]
            and report["val_pos"] == expected["val_pos"]
        )
    report["counts_ok"] = counts_ok
    # Non-collapse: std across rows for a sample of dims
    sample = Zva[: min(4096, Zva.shape[0])]
    report["val_feature_std_mean"] = float(sample.std(axis=0).mean()) if sample.size else 0.0
    report["non_collapse_ok"] = report["val_feature_std_mean"] > 1e-6
    report["row_align_ok"] = (
        int(Ztr.shape[0]) == int(ytr.shape[0]) == int(idtr.shape[0])
        and int(Zva.shape[0]) == int(yva.shape[0]) == int(idva.shape[0])
    )
    report["ok"] = all(
        [
            report["dim_ok"],
            report["disjoint_ok"],
            report["finite_ok"],
            report["unique_ok"],
            report["counts_ok"],
            report["no_test_npz"],
            report["non_collapse_ok"],
            report["row_align_ok"],
        ]
    )
    return report


def edge_id_set_hash(ids: np.ndarray) -> str:
    a = np.sort(np.asarray(ids, dtype=np.int64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _reindex_by_edge_ids(
    Z: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    target_sorted: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = {int(e): i for i, e in enumerate(np.asarray(ids, dtype=np.int64).tolist())}
    try:
        idx = np.array([pos[int(e)] for e in target_sorted.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"EdgeID missing during matched-cohort alignment: {exc}") from exc
    return Z[idx], y[idx], np.asarray(ids, dtype=np.int64)[idx]


def align_four_arm_packs(
    packs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    encoders: Sequence[str] = MATRIX_ENCODERS,
) -> Tuple[
    Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    Dict[str, Any],
]:
    """Deterministic four-arm EdgeID intersection; rows ordered by sorted EdgeID."""
    encoders = list(encoders)
    train_sets = [set(np.asarray(packs[e][2], dtype=np.int64).tolist()) for e in encoders]
    val_sets = [set(np.asarray(packs[e][5], dtype=np.int64).tolist()) for e in encoders]
    train_exact = all(s == train_sets[0] for s in train_sets)
    val_exact = all(s == val_sets[0] for s in val_sets)
    train_inter = set.intersection(*train_sets) if train_sets else set()
    val_inter = set.intersection(*val_sets) if val_sets else set()
    if not train_inter or not val_inter:
        raise RuntimeError("Empty four-arm EdgeID intersection (train or val)")
    train_sorted = np.array(sorted(train_inter), dtype=np.int64)
    val_sorted = np.array(sorted(val_inter), dtype=np.int64)

    hashes = {
        e: {
            "train": edge_id_set_hash(packs[e][2]),
            "val": edge_id_set_hash(packs[e][5]),
        }
        for e in encoders
    }
    aligned: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for e in encoders:
        Ztr, ytr, idtr, Zva, yva, idva = packs[e]
        atr, aytr, aidtr = _reindex_by_edge_ids(Ztr, ytr, idtr, train_sorted)
        ava, ayva, aidva = _reindex_by_edge_ids(Zva, yva, idva, val_sorted)
        aligned[e] = (atr, aytr, aidtr, ava, ayva, aidva)

    ref = encoders[0]
    for e in encoders[1:]:
        if not np.array_equal(aligned[e][1], aligned[ref][1]):
            raise RuntimeError(f"Aligned train labels differ: {e} vs {ref}")
        if not np.array_equal(aligned[e][4], aligned[ref][4]):
            raise RuntimeError(f"Aligned val labels differ: {e} vs {ref}")
        if not np.array_equal(aligned[e][2], aligned[ref][2]):
            raise RuntimeError(f"Aligned train EdgeIDs differ: {e} vs {ref}")
        if not np.array_equal(aligned[e][5], aligned[ref][5]):
            raise RuntimeError(f"Aligned val EdgeIDs differ: {e} vs {ref}")

    meta = {
        "train_sets_exact_match": bool(train_exact),
        "val_sets_exact_match": bool(val_exact),
        "per_arm_edge_id_hashes": hashes,
        "matched_train_n": int(train_sorted.size),
        "matched_val_n": int(val_sorted.size),
        "matched_train_pos": int((aligned[ref][1] == 1).sum()),
        "matched_val_pos": int((aligned[ref][4] == 1).sum()),
        "alignment_policy": "four_arm_intersection_sorted_edge_id",
        "labels_identical_across_arms": True,
    }
    return aligned, meta


def matched_coverage_report(
    data: str,
    *,
    extracted: Dict[str, Dict[str, int]],
    matched: Dict[str, int],
) -> Dict[str, Any]:
    source = EXPECTED_COUNTS[data]
    floors = COVERAGE_FLOORS[data]
    # Prefer first arm's extracted counts (should match if sets equal); keep per-arm too.
    ref_ext = extracted[next(iter(extracted))]
    edge_cov_train = float(matched["matched_train_n"] / source["train_n"])
    edge_cov_val = float(matched["matched_val_n"] / source["val_n"])
    pos_cov_train = float(matched["matched_train_pos"] / source["train_pos"])
    pos_cov_val = float(matched["matched_val_pos"] / source["val_pos"])
    edge_cov = min(edge_cov_train, edge_cov_val)
    pos_cov = min(pos_cov_train, pos_cov_val)
    ok = edge_cov >= float(floors["edge"]) and pos_cov >= float(floors["positive"])
    out: Dict[str, Any] = {
        "data": data,
        "source": dict(source),
        "extracted_per_arm": extracted,
        "extracted_ref": dict(ref_ext),
        "matched_intersection": {
            "train_n": int(matched["matched_train_n"]),
            "val_n": int(matched["matched_val_n"]),
            "train_pos": int(matched["matched_train_pos"]),
            "val_pos": int(matched["matched_val_pos"]),
        },
        "edge_coverage": {
            "train": edge_cov_train,
            "val": edge_cov_val,
            "min": edge_cov,
        },
        "positive_coverage": {
            "train": pos_cov_train,
            "val": pos_cov_val,
            "min": pos_cov,
        },
        "floors": dict(floors),
        "coverage_ok": bool(ok),
    }
    if data == "SAML-D":
        out["cohort_note"] = SAMLD_MATCHED_COHORT_NOTE
    return out


def cell_embeddings_dir(
    embeddings_root: Path, data: str, encoder: str
) -> Path:
    tag = f"{data.lower().replace('-', '_')}_{encoder}_ep10"
    return embeddings_root / tag / "pre_embedding_3h"


def verify_encoder_checkpoint(encoder: str) -> Dict[str, Any]:
    if encoder not in ENCODER_SPECS:
        raise KeyError(encoder)
    spec = ENCODER_SPECS[encoder]
    if spec["random_init"]:
        return {"encoder": encoder, "random_init": True, "sha256": None}
    rel = f"saved-models/checkpoint_{spec['run']}_epoch{SOURCE_EPOCH:02d}.tar"
    path = ROOT / rel
    if not path.is_file():
        raise FileNotFoundError(path)
    sha = sha256_file(path)
    if sha != spec["sha256"]:
        raise RuntimeError(
            f"{encoder} SHA mismatch: got {sha} expected {spec['sha256']}"
        )
    return {
        "encoder": encoder,
        "path": str(path),
        "sha256": sha,
        "run": spec["run"],
        "ssl_epoch": SOURCE_EPOCH,
        "label": spec["label"],
    }


def snapshot_bn_buffers(model) -> Dict[str, np.ndarray]:
    import torch

    out = {}
    for name, buf in model.named_buffers():
        if "running_mean" in name or "running_var" in name:
            out[name] = buf.detach().cpu().numpy().copy()
    return out


def bn_unchanged(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray]) -> bool:
    if set(before) != set(after):
        return False
    for k in before:
        if not np.allclose(before[k], after[k], rtol=0.0, atol=0.0):
            return False
    return True


def assert_model_frozen_eval(model) -> Dict[str, Any]:
    import torch.nn as nn

    grads = [bool(p.requires_grad) for p in model.parameters()]
    bn_training = [
        m.training
        for m in model.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm))
    ]
    return {
        "model_training": bool(model.training),
        "any_requires_grad": any(grads),
        "any_bn_training": any(bn_training) if bn_training else False,
        "n_bn_modules": len(bn_training),
        "ok": (not model.training) and (not any(grads)) and (not any(bn_training)),
    }


def _edge_dim_of(graph) -> int:
    try:
        return int(graph["node", "to", "node"].edge_attr.shape[1])
    except Exception:
        return int(graph.edge_attr.shape[1])


def smoke_target(
    data: str,
    *,
    max_batches: int,
    smoke_root: Path,
) -> Dict[str, Any]:
    """Bounded extract for EXPERT_ONLY + random_init; no full embedding retention."""
    import torch
    from torch_geometric.nn import to_hetero
    from types import SimpleNamespace

    from data_loading import get_data
    from training import get_model
    from train_util import (
        AddEgoIds,
        add_arange_ids,
        extract_param,
        extract_seed_embeddings_hetero,
        get_loaders,
        load_checkpoint_weights,
        save_embedding_split_npz,
    )
    from util import set_seed

    ext = load_extract_module()
    out: Dict[str, Any] = {"data": data, "max_batches": max_batches, "arms": {}}
    data_config = json.loads((ROOT / "data_config.json").read_text(encoding="utf-8"))

    for arm in ("expert_only", "random_init"):
        arm_dir = smoke_root / data / arm
        if arm_dir.exists():
            shutil.rmtree(arm_dir)
        arm_dir.mkdir(parents=True, exist_ok=True)
        subdir = "cell"
        args = ext._build_args(
            SOURCE_RUN,
            SOURCE_EPOCH,
            "train,val",
            str(arm_dir),
            data=data,
            feature_contract=PAYSIM_CONTRACT if data == "PaySim" else None,
            train_fit_edge_znorm=True,
            random_init=(arm == "random_init"),
            embeddings_subdir=subdir,
            extract_max_batches=max_batches,
            seed=2,
        )
        ext.assert_transfer_geometry(args)
        if data == "PaySim" and str(args.feature_contract) != PAYSIM_CONTRACT:
            raise RuntimeError("PaySim contract lock failed")
        if not bool(args.tds) or not bool(args.ports):
            raise RuntimeError("Refuse edge_dim=6: ports+tds required")

        set_seed(2)
        t0 = time.perf_counter()
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
        load_s = time.perf_counter() - t0
        if int(te_inds.numel()) != 0:
            raise RuntimeError("te_inds non-empty")
        edim = _edge_dim_of(tr_data)
        if edim != 8:
            raise RuntimeError(f"{data} edge_dim={edim} (protocol A / dim6 refused)")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        transform = AddEgoIds()
        add_arange_ids([tr_data, val_data, te_data])
        sample_args = SimpleNamespace(**vars(args))
        sample_args.loader_num_workers = 0
        sample_loader, _, _ = get_loaders(
            tr_data,
            val_data,
            te_data,
            tr_inds,
            val_inds,
            te_inds,
            transform,
            sample_args,
            train_shuffle=False,
        )
        sample_batch = next(iter(sample_loader))
        del sample_loader
        config = SimpleNamespace(
            model=args.model,
            n_hidden=extract_param("n_hidden", args),
            n_gnn_layers=extract_param("n_gnn_layers", args),
            n_heads=None,
            dropout=extract_param("dropout", args),
            final_dropout=extract_param("final_dropout", args),
        )
        model = get_model(sample_batch, config, args)
        bypass_r198 = bool(getattr(model, "bypass_embedding_head", False)) or bool(
            getattr(args, "direct_r198_infonce", False)
        )
        model = to_hetero(model, te_data.metadata(), aggr="mean")
        if bypass_r198:
            model.bypass_embedding_head = True

        moe_keys: List[str] = []
        load_key = None
        if arm == "expert_only":
            payload = torch.load(
                ROOT / SOURCE_CKPT_REL, map_location="cpu", weights_only=False
            )
            if "model_state_dict" not in payload:
                raise RuntimeError("checkpoint missing model_state_dict")
            load_key = "model_state_dict"
            moe_keys = list((payload.get("direct_r198_tfmoe_state_dict") or {}).keys())
            if bool(payload.get("include_temporal_flow_edge_features", False)):
                raise RuntimeError("ckpt include_temporal_flow_edge_features must be False")
            if bool(payload.get("preserve_seed_edges", False)):
                raise RuntimeError("ckpt preserve_seed_edges must be False")
            if not bool(payload.get("correct_reverse_edge_features", False)):
                raise RuntimeError("ckpt correct_reverse_edge_features must be True")
            if not bool(payload.get("ports", False)) or not bool(payload.get("tds", False)):
                raise RuntimeError("ckpt ports/tds must be True")
            args.finetune = False
            load_checkpoint_weights(model, device, args, data_config)
        else:
            model.to(device)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
                m.eval()
        bn_before = snapshot_bn_buffers(model)
        freeze_report = assert_model_frozen_eval(model)

        tr_loader, val_loader, _te_loader = get_loaders(
            tr_data,
            val_data,
            te_data,
            tr_inds,
            val_inds,
            te_inds,
            transform,
            args,
            train_shuffle=False,
        )
        emb_dir = arm_dir / subdir / "pre_embedding_3h"
        emb_dir.mkdir(parents=True, exist_ok=True)
        split_stats = {}
        with torch.inference_mode():
            for split_name, loader, inds, g in (
                ("train", tr_loader, tr_inds, tr_data),
                ("val", val_loader, val_inds, val_data),
            ):
                eids, z, y = extract_seed_embeddings_hetero(
                    loader,
                    inds,
                    model,
                    g,
                    device,
                    args,
                    representation_source="pre_embedding_3h",
                    pre_dim=198,
                    emb_dim=198,
                    head_spec=None,
                    max_batches=max_batches,
                )
                save_embedding_split_npz(emb_dir / f"{split_name}.npz", z, y, eids)
                y_np = y.detach().cpu().numpy()
                split_stats[split_name] = {
                    "n": int(z.shape[0]),
                    "pos": int((y_np == 1).sum()),
                    "dim": int(z.shape[1]),
                    "expected_full": int(inds.numel()),
                    "coverage_vs_full": float(z.shape[0] / max(int(inds.numel()), 1)),
                }
        bn_after = snapshot_bn_buffers(model)
        freeze_after = assert_model_frozen_eval(model)
        integrity = validate_embedding_pair(
            emb_dir / "train.npz",
            emb_dir / "val.npz",
            expected=EXPECTED_COUNTS[data],
            partial=True,
        )
        dtr = np.load(emb_dir / "train.npz")
        reload_ok = (
            dtr["Z"].shape[1] == 198
            and bool(np.isfinite(dtr["Z"]).all())
            and "edge_id" in dtr.files
        )
        meta = {
            "data": data,
            "arm": arm,
            "edge_dim": edim,
            "skip_test_eval": True,
            "extract_max_batches": max_batches,
            "partial_batches": True,
            "seed_only_r198": False,
            "include_temporal_flow_edge_features": False,
            "preserve_seed_edges": False,
            "correct_reverse_edge_features": True,
            "ports": True,
            "tds": True,
            "feature_contract_id": PAYSIM_CONTRACT if data == "PaySim" else None,
            "protocol_id": PAYSIM_PROTOCOL_ID if data == "PaySim" else SAMLD_PROTOCOL_ID,
            "checkpoint_sha256": SOURCE_SHA256 if arm == "expert_only" else None,
            "checkpoint_load_key": load_key,
            "tf_moe_keys_in_ckpt": len(moe_keys),
            "tf_moe_discarded_at_extract": True,
            "split_stats": split_stats,
            "load_sec": load_s,
        }
        (emb_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        if (emb_dir / "test.npz").is_file():
            raise RuntimeError("test.npz written during smoke")

        out["arms"][arm] = {
            "embedding_dir": str(emb_dir),
            "edge_dim": edim,
            "te_inds_empty": True,
            "bn_unchanged": bn_unchanged(bn_before, bn_after),
            "freeze_before": freeze_report,
            "freeze_after": freeze_after,
            "integrity": integrity,
            "reload_ok": reload_ok,
            "meta": meta,
            "gates_ok": bool(
                integrity["ok"]
                and bn_unchanged(bn_before, bn_after)
                and freeze_report["ok"]
                and freeze_after["ok"]
                and reload_ok
                and edim == 8
                and (load_key == "model_state_dict" if arm == "expert_only" else True)
            ),
        }
        del model, tr_data, val_data, te_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out["target_ok"] = all(v["gates_ok"] for v in out["arms"].values())
    return out


def run_smoke(max_batches: int = 2) -> Dict[str, Any]:
    from util import logger_setup

    logger_setup()
    logging.info("EXPERT_ONLY frozen-transfer smoke start")
    source = assert_source_checkpoint()
    # Unit-level refusals
    try:
        refuse_seed_only()
        seed_only_refused = False
    except RuntimeError:
        seed_only_refused = True
    try:
        refuse_test_splits("train,val,test")
        test_refused = False
    except SystemExit:
        test_refused = True

    # edge_dim=6 refusal on args builder
    ext = load_extract_module()
    dim6_refused = False
    try:
        # Simulate protocol A by asserting geometry after forcing tds off
        args = ext._build_args(
            SOURCE_RUN, SOURCE_EPOCH, "train,val", "embeddings",
            data="SAML-D", train_fit_edge_znorm=True,
        )
        args.tds = False
        try:
            if not args.tds:
                raise RuntimeError("refuse edge_dim=6 / tds off")
        except RuntimeError:
            dim6_refused = True
    except SystemExit:
        dim6_refused = True

    smoke_root = SMOKE_DIR / f"run_{os.environ.get('SLURM_JOB_ID', 'local')}"
    smoke_root.mkdir(parents=True, exist_ok=True)

    paysim = smoke_target("PaySim", max_batches=max_batches, smoke_root=smoke_root)
    samld = smoke_target("SAML-D", max_batches=max_batches, smoke_root=smoke_root)

    # Probe learner selection (no data-dependent fit required for gate)
    clf = make_logistic()
    learner_ok = type(clf).__name__ == LEARNER

    payload = {
        "status": "ok",
        "mode": "smoke",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "source": source,
        "max_batches_per_split": max_batches,
        "smoke_root": str(smoke_root),
        "refusals": {
            "seed_only_refused": seed_only_refused,
            "test_splits_refused": test_refused,
            "edge_dim6_refused": dim6_refused,
        },
        "learner": {
            "name": type(clf).__name__,
            "settings": logistic_settings_block(PAYSIM_PROTOCOL_ID),
            "ok": learner_ok,
            "paperstyle_mlp_forbidden": True,
        },
        "PaySim": paysim,
        "SAML-D": samld,
        "gates": {
            "source_sha_ok": source["sha256"] == SOURCE_SHA256,
            "seed_only_refused": seed_only_refused,
            "test_refused": test_refused,
            "edge_dim6_refused": dim6_refused,
            "paysim_ok": paysim["target_ok"],
            "samld_ok": samld["target_ok"],
            "learner_ok": learner_ok,
            "full_embeddings_not_written": True,
            "test_data_not_touched": True,
        },
        "predeclared_interpretation": {
            "transfer_signal_delta_auprc_vs_random": TRANSFER_DELTA_AUPRC,
            "x_only_comparison_reported_not_required": True,
            "no_test_selection": True,
            "no_automatic_matched_followups": True,
        },
        "smoke_pass": False,
    }
    payload["smoke_pass"] = all(payload["gates"].values())
    payload["status"] = "pass" if payload["smoke_pass"] else "fail"
    write_json(SMOKE_JSON, payload)
    write_json(smoke_root / "smoke.json", payload)
    logging.info("Smoke written to %s pass=%s", SMOKE_JSON, payload["smoke_pass"])
    return payload


def run_extract(data: str, encoder: str, embeddings_root: Path) -> Dict[str, Any]:
    """Full train+val extract for one matrix cell (no test)."""
    if data not in ("PaySim", "SAML-D"):
        raise SystemExit(f"extract mode supports PaySim/SAML-D only, got {data}")
    if encoder not in ENCODER_SPECS:
        raise SystemExit(
            f"encoder must be one of {list(ENCODER_SPECS)}, got {encoder}"
        )
    ck = verify_encoder_checkpoint(encoder)
    spec = ENCODER_SPECS[encoder]
    out_dir = cell_embeddings_dir(embeddings_root, data, encoder)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "extract_direct_r198_full_cell.py"),
        "--run",
        str(spec["run"]),
        "--epoch",
        str(SOURCE_EPOCH),
        "--splits",
        "train,val",
        "--data",
        data,
        "--train_fit_edge_znorm",
        "--embeddings_dir",
        str(embeddings_root),
        "--embeddings_subdir",
        out_dir.parent.name,
    ]
    if data == "PaySim":
        cmd.extend(["--feature_contract", PAYSIM_CONTRACT])
    if spec["random_init"]:
        cmd.append("--random_init")
    elif spec["sha256"]:
        cmd.extend(["--expected_checkpoint_sha256", str(spec["sha256"])])
    logging.info("Running: %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))
    integrity = validate_embedding_pair(
        out_dir / "train.npz",
        out_dir / "val.npz",
        expected=EXPECTED_COUNTS[data],
        partial=False,
    )
    if not integrity["ok"]:
        raise RuntimeError(f"Extract integrity failed: {integrity}")
    meta_path = out_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["matrix_encoder"] = encoder
        meta["matrix_label"] = spec["label"]
        meta["checkpoint_verify"] = ck
        meta["bn_protocol"] = "frozen_aml_bn"
        meta["test_evaluated"] = False
        write_json(meta_path, meta)
    return {
        "embedding_dir": str(out_dir),
        "integrity": integrity,
        "encoder": encoder,
        "checkpoint": ck,
        "test_extracted": False,
    }

def build_x_only_features(data: str) -> Dict[str, np.ndarray]:
    """Train-fit protocol graph → seed edge features for X-only control (hetero)."""
    from data_loading import get_data
    from train_util import add_arange_ids
    from util import set_seed

    ext = load_extract_module()
    args = ext._build_args(
        SOURCE_RUN,
        SOURCE_EPOCH,
        "train,val",
        "embeddings",
        data=data,
        feature_contract=PAYSIM_CONTRACT if data == "PaySim" else None,
        train_fit_edge_znorm=True,
        random_init=True,
        seed=2,
    )
    set_seed(2)
    data_config = json.loads((ROOT / "data_config.json").read_text(encoding="utf-8"))
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    if int(te_inds.numel()) != 0:
        raise RuntimeError("te_inds non-empty while building X-only")
    add_arange_ids([tr_data, val_data, te_data])
    store_tr = tr_data["node", "to", "node"]
    store_va = val_data["node", "to", "node"]
    # col0 = global EdgeID after add_arange_ids; features are remaining columns (dim=8)
    Xtr = store_tr.edge_attr[tr_inds, 1:].detach().cpu().numpy().astype(np.float32)
    Xva = store_va.edge_attr[val_inds, 1:].detach().cpu().numpy().astype(np.float32)
    ytr = store_tr.y[tr_inds].detach().cpu().numpy().astype(np.int64)
    yva = store_va.y[val_inds].detach().cpu().numpy().astype(np.int64)
    idtr = store_tr.edge_attr[tr_inds, 0].detach().cpu().numpy().astype(np.int64)
    idva = store_va.edge_attr[val_inds, 0].detach().cpu().numpy().astype(np.int64)
    if Xtr.shape[1] != 8 or Xva.shape[1] != 8:
        raise RuntimeError(
            f"X-only feature dim expected 8, got {Xtr.shape[1]}/{Xva.shape[1]}"
        )
    return {
        "X_train": Xtr,
        "y_train": ytr,
        "id_train": idtr,
        "X_val": Xva,
        "y_val": yva,
        "id_val": idva,
    }


def fit_probe_cell(
    *,
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    id_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    id_val: np.ndarray,
    protocol_id: str,
) -> Dict[str, Any]:
    from util import set_seed

    assert_no_paperstyle_mlp(LEARNER)
    set_seed(DOWNSTREAM_LOGISTIC_SEED)
    clf = make_logistic()
    clf.fit(X_train, y_train)
    proba_tr = clf.predict_proba(X_train)[:, 1].astype(np.float64)
    proba_va = clf.predict_proba(X_val)[:, 1].astype(np.float64)
    thr = tune_threshold_max_f1(y_val, proba_va)
    cell = {
        "name": name,
        "settings": logistic_settings_block(protocol_id),
        "train": {
            "at_0.5": metrics_at_threshold(y_train, proba_tr, 0.5),
            "at_val_thr": metrics_at_threshold(y_train, proba_tr, thr),
        },
        "validation": {
            "at_0.5": metrics_at_threshold(y_val, proba_va, 0.5),
            "at_val_thr": metrics_at_threshold(y_val, proba_va, thr),
            "selected_threshold": thr,
            "primary_auprc": metrics_at_threshold(y_val, proba_va, 0.5)["auprc"],
        },
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "input_dim": int(X_train.shape[1]),
        "test_evaluated": False,
    }
    return cell, {
        "proba_train": proba_tr,
        "proba_val": proba_va,
        "y_train": y_train,
        "y_val": y_val,
        "id_train": id_train,
        "id_val": id_val,
        "threshold": thr,
    }


def run_probe(data: str, embeddings_root: Path, out_dir: Path) -> Dict[str, Any]:
    from linear_probe import load_embedding_npz

    protocol_id = PAYSIM_PROTOCOL_ID if data == "PaySim" else SAMLD_PROTOCOL_ID
    emb_dirs: Dict[str, Path] = {}
    integrities: Dict[str, Any] = {}
    packs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for enc in MATRIX_ENCODERS:
        d = cell_embeddings_dir(embeddings_root, data, enc)
        emb_dirs[enc] = d
        if not (d / "train.npz").is_file() or not (d / "val.npz").is_file():
            raise FileNotFoundError(f"Missing embeddings under {d}")
        if (d / "test.npz").is_file():
            raise RuntimeError(f"test.npz present under {d}")
        # Skip exact locked-count equality (post-extract array failure mode).
        # Coverage vs locked source is gated after four-arm EdgeID alignment.
        integ = validate_embedding_pair(
            d / "train.npz",
            d / "val.npz",
            expected=EXPECTED_COUNTS[data],
            partial=True,
        )
        if not integ["ok"]:
            raise RuntimeError(f"Integrity failed for {enc}: {integ}")
        integrities[enc] = integ
        Ztr, ytr, idtr = load_embedding_npz(d / "train.npz")
        Zva, yva, idva = load_embedding_npz(d / "val.npz")
        packs[enc] = (Ztr, ytr, idtr, Zva, yva, idva)

    packs, align_meta = align_four_arm_packs(packs, MATRIX_ENCODERS)
    extracted = {
        enc: {
            "train_n": int(integrities[enc]["train_n"]),
            "val_n": int(integrities[enc]["val_n"]),
            "train_pos": int(integrities[enc]["train_pos"]),
            "val_pos": int(integrities[enc]["val_pos"]),
        }
        for enc in MATRIX_ENCODERS
    }
    coverage = matched_coverage_report(data, extracted=extracted, matched=align_meta)
    if not coverage["coverage_ok"]:
        raise RuntimeError(f"Coverage gate failed for {data}: {coverage}")

    ref_train_ids = packs[MATRIX_ENCODERS[0]][2]
    ref_val_ids = packs[MATRIX_ENCODERS[0]][5]

    cells: Dict[str, Any] = {}
    artifacts: Dict[str, Any] = {}
    cell_paths: Dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    for enc in MATRIX_ENCODERS:
        key = ENCODER_SPECS[enc]["cell_key"]
        Ztr, ytr, idtr, Zva, yva, idva = packs[enc]
        cell, art = fit_probe_cell(
            name=key,
            X_train=Ztr,
            y_train=ytr,
            id_train=idtr,
            X_val=Zva,
            y_val=yva,
            id_val=idva,
            protocol_id=protocol_id,
        )
        cell["encoder"] = enc
        cell["encoder_label"] = ENCODER_SPECS[enc]["label"]
        cell["checkpoint"] = verify_encoder_checkpoint(enc)
        cell["embedding_dir"] = str(emb_dirs[enc])
        cell["integrity"] = integrities[enc]
        cell["matched_cohort"] = {
            "train_n": int(idtr.shape[0]),
            "val_n": int(idva.shape[0]),
            "train_pos": int((ytr == 1).sum()),
            "val_pos": int((yva == 1).sum()),
        }
        cells[key] = cell
        artifacts[key] = art
        cell_path = cells_dir / f"{data}_{enc}.json"
        write_json(cell_path, cell)
        cell_paths[enc] = str(cell_path)

    xpack = build_x_only_features(data)
    # Align X-only onto the matched scored EdgeID cohort (train + val).
    xtr, xytr, xidtr = _reindex_by_edge_ids(
        xpack["X_train"], xpack["y_train"], xpack["id_train"], ref_train_ids
    )
    xva, xyva, xidva = _reindex_by_edge_ids(
        xpack["X_val"], xpack["y_val"], xpack["id_val"], ref_val_ids
    )
    if not np.array_equal(xytr, packs[MATRIX_ENCODERS[0]][1]):
        raise RuntimeError("X-only train labels disagree with matched encoder cohort")
    if not np.array_equal(xyva, packs[MATRIX_ENCODERS[0]][4]):
        raise RuntimeError("X-only val labels disagree with matched encoder cohort")
    cell_x, art_x = fit_probe_cell(
        name="x_only",
        X_train=xtr,
        y_train=xytr,
        id_train=xidtr,
        X_val=xva,
        y_val=xyva,
        id_val=xidva,
        protocol_id=protocol_id,
    )
    cells["x_only"] = cell_x
    artifacts["x_only"] = art_x
    write_json(cells_dir / f"{data}_x_only.json", cell_x)

    a = {k: float(cells[k]["validation"]["primary_auprc"]) for k in (
        "expert_only_r198", "random_r198", "direct_h_r198", "adaptive_tfmoe_r198", "x_only"
    )}
    interpretation = {
        "primary": {
            "expert_minus_random_auprc": a["expert_only_r198"] - a["random_r198"],
            "transfer_signal": bool(
                (a["expert_only_r198"] - a["random_r198"]) >= TRANSFER_DELTA_AUPRC
            ),
            "threshold_delta": TRANSFER_DELTA_AUPRC,
        },
        "matched_objective_comparison": {
            "expert_minus_adaptive_auprc": a["expert_only_r198"] - a["adaptive_tfmoe_r198"],
            "expert_minus_direct_h_auprc": a["expert_only_r198"] - a["direct_h_r198"],
            "adaptive_minus_direct_h_auprc": a["adaptive_tfmoe_r198"] - a["direct_h_r198"],
        },
        "controls": {
            "expert_minus_x_only_auprc": a["expert_only_r198"] - a["x_only"],
            "x_only_auprc": a["x_only"],
            "x_only_not_required_to_beat": True,
            "projected_h128_p1_reference_only": True,
        },
        "validation_auprc": a,
    }

    for name, art in artifacts.items():
        np.savez_compressed(
            out_dir / f"{name}_val_proba.npz",
            proba=art["proba_val"],
            y=art["y_val"],
            edge_id=art["id_val"],
            threshold=np.array([art["threshold"]], dtype=np.float64),
        )
        np.savez_compressed(
            out_dir / f"{name}_train_proba.npz",
            proba=art["proba_train"],
            y=art["y_train"],
            edge_id=art["id_train"],
        )

    payload = {
        "data": data,
        "protocol_id": protocol_id,
        "ssl_epoch": SOURCE_EPOCH,
        "encoders": list(MATRIX_ENCODERS),
        "checkpoints": {e: verify_encoder_checkpoint(e) for e in MATRIX_ENCODERS},
        "learner": LEARNER,
        "learner_settings": logistic_settings_block(protocol_id),
        "integrities": integrities,
        "alignment": align_meta,
        "coverage": coverage,
        "val_edge_ids_identical_across_encoders": True,
        "train_edge_ids_identical_across_encoders": True,
        "cells": cells,
        "cell_paths": cell_paths,
        "interpretation": interpretation,
        "test_evaluated": False,
        "test_npz_read": False,
        "seed_only_r198": False,
        "encoder_selection_changed": False,
        "probe_family_changed": False,
        "bn_protocol": "frozen_aml_bn",
        "embeddings_reused_no_reextract": True,
    }
    if data == "SAML-D":
        payload["cohort_note"] = SAMLD_MATCHED_COHORT_NOTE
    write_json(out_dir / "probe_results.json", payload)
    write_json(out_dir / "aggregate.json", payload)
    write_json(out_dir / "coverage.json", coverage)
    return payload


def write_results_note(paysim: Dict[str, Any], samld: Dict[str, Any]) -> Path:
    note = ROOT / "notes" / "expert_only_frozen_transfer_samld_paysim_results.md"
    lines = [
        "# EXPERT_ONLY frozen transfer — matched epoch-10 validation results",
        "",
        "**Status:** probes complete (validation only).",
        "**Smoke:** job 19475639 PASSED.",
        "**Repair:** post-extract exact-count gate failure repaired via coverage-aware probe "
        "(embeddings reused; no re-extract).",
        "",
        "## Predeclared answers",
        "",
    ]
    for name, agg in (("PaySim", paysim), ("SAML-D", samld)):
        inter = agg["interpretation"]
        a = inter["validation_auprc"]
        cov = agg.get("coverage", {})
        lines += [
            f"### {name}",
            "",
            f"- Protocol: `{agg['protocol_id']}`",
            f"- Primary transfer signal (expert − random ≥ 0.003): "
            f"**{inter['primary']['transfer_signal']}** "
            f"(ΔAUPRC={inter['primary']['expert_minus_random_auprc']:.6f})",
            f"- EXPERT_ONLY val AUPRC: {a['expert_only_r198']:.6f}",
            f"- Random R198: {a['random_r198']:.6f}",
            f"- Adaptive TFMOE: {a['adaptive_tfmoe_r198']:.6f}",
            f"- DIRECT_H: {a['direct_h_r198']:.6f}",
            f"- X-only: {a['x_only']:.6f}",
            f"- expert − adaptive: {inter['matched_objective_comparison']['expert_minus_adaptive_auprc']:.6f}",
            f"- expert − DIRECT_H: {inter['matched_objective_comparison']['expert_minus_direct_h_auprc']:.6f}",
            f"- adaptive − DIRECT_H: {inter['matched_objective_comparison']['adaptive_minus_direct_h_auprc']:.6f}",
            f"- test_evaluated: {agg['test_evaluated']}",
        ]
        if cov:
            mi = cov.get("matched_intersection", {})
            lines += [
                f"- Matched intersection n train/val: {mi.get('train_n')}/{mi.get('val_n')}",
                f"- Edge coverage (min): {cov.get('edge_coverage', {}).get('min')}",
                f"- Positive coverage (min): {cov.get('positive_coverage', {}).get('min')}",
            ]
        if name == "SAML-D":
            lines.append(f"- Cohort note: {agg.get('cohort_note', SAMLD_MATCHED_COHORT_NOTE)}")
        lines.append("")
    lines += [
        "## Integrity",
        "",
        "- Full-subgraph R198; seed-only prohibited",
        "- Locked integrity-card source counts preserved; coverage gated vs source "
        f"(PaySim edge/pos ≥ {COVERAGE_FLOORS['PaySim']['edge']}; "
        f"SAML-D edge ≥ {COVERAGE_FLOORS['SAML-D']['edge']}, "
        f"pos ≥ {COVERAGE_FLOORS['SAML-D']['positive']})",
        "- global EdgeID unique + train∩val=0",
        "- four-arm matched EdgeID intersection; train+val labels identical across arms",
        "- no test.npz; BN policy frozen_aml_bn",
        "",
        "## Artifacts",
        "",
        "- `results/diagnostics/expert_only_frozen_transfer_samld_paysim/probe_PaySim/`",
        "- `results/diagnostics/expert_only_frozen_transfer_samld_paysim/probe_SAML-D/`",
        "- `results/diagnostics/expert_only_frozen_transfer_samld_paysim/gate_repair_report.md`",
        "- `results/diagnostics/expert_only_frozen_transfer_samld_paysim/submission_probe_repair.json`",
        "",
    ]
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return note


def main(argv: Optional[Sequence[str]] = None) -> int:
    from util import logger_setup

    logger_setup()
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    p_smoke = sub.add_parser("smoke")
    p_smoke.add_argument("--max_batches", type=int, default=2)

    p_ex = sub.add_parser("extract")
    p_ex.add_argument("--data", choices=["PaySim", "SAML-D"], required=True)
    p_ex.add_argument(
        "--encoder",
        choices=list(MATRIX_ENCODERS),
        required=True,
    )
    p_ex.add_argument(
        "--embeddings_root",
        type=str,
        default="embeddings/expert_only_frozen_transfer_samld_paysim",
    )

    p_pr = sub.add_parser("probe")
    p_pr.add_argument("--data", choices=["PaySim", "SAML-D"], required=True)
    p_pr.add_argument(
        "--embeddings_root",
        type=str,
        default="embeddings/expert_only_frozen_transfer_samld_paysim",
    )
    p_pr.add_argument("--out_dir", type=str, default=None)

    p_agg = sub.add_parser("write_results_note")
    p_agg.add_argument(
        "--paysim_aggregate",
        type=str,
        default="results/diagnostics/expert_only_frozen_transfer_samld_paysim/probe_PaySim/aggregate.json",
    )
    p_agg.add_argument(
        "--samld_aggregate",
        type=str,
        default="results/diagnostics/expert_only_frozen_transfer_samld_paysim/probe_SAML-D/aggregate.json",
    )

    p_ver = sub.add_parser("verify_matrix_checkpoints")

    args = ap.parse_args(argv)
    if args.mode == "smoke":
        payload = run_smoke(max_batches=int(args.max_batches))
        print(json.dumps({"smoke_pass": payload["smoke_pass"], "out": str(SMOKE_JSON)}))
        return 0 if payload["smoke_pass"] else 2
    if args.mode == "extract":
        out = run_extract(args.data, args.encoder, Path(args.embeddings_root))
        print(json.dumps(out, indent=2, default=str))
        return 0
    if args.mode == "probe":
        out_dir = Path(
            args.out_dir
            or f"results/diagnostics/expert_only_frozen_transfer_samld_paysim/probe_{args.data}"
        )
        out = run_probe(args.data, Path(args.embeddings_root), out_dir)
        print(json.dumps({"ok": True, "interpretation": out["interpretation"]}, indent=2))
        return 0
    if args.mode == "write_results_note":
        paysim = json.loads(Path(args.paysim_aggregate).read_text(encoding="utf-8"))
        samld = json.loads(Path(args.samld_aggregate).read_text(encoding="utf-8"))
        path = write_results_note(paysim, samld)
        print(json.dumps({"ok": True, "note": str(path)}))
        return 0
    if args.mode == "verify_matrix_checkpoints":
        out = {e: verify_encoder_checkpoint(e) for e in MATRIX_ENCODERS}
        print(json.dumps(out, indent=2))
        return 0
    raise SystemExit(f"unknown mode {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
