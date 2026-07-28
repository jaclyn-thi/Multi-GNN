#!/usr/bin/env python3
"""Sequential AMLWorld→PaySim self-supervised adaptation scout (exploratory/post-hoc).

Subcommands: smoke | train_arm | eval | aggregate

Arms:
  A aml_init_paysim_ssl — weight continuation from locked AML seed-2 on PaySim train SSL
  B random_init_paysim_ssl — matched PaySim-only SSL from scratch
Controls (eval): frozen_aml, bn_only, x_only; AMLWorld retention with original BN restored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import to_hetero

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loading import get_data  # noqa: E402
from feature_contracts import CONTRACT_LEGACY  # noqa: E402
from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba  # noqa: E402
from linear_probe import load_embedding_npz  # noqa: E402
from ranking_metrics import alert_budget_metrics  # noqa: E402
from train_util import (  # noqa: E402
    AddEgoIds,
    FORWARD_EDGE_TYPE,
    REVERSE_EDGE_TYPE,
    add_arange_ids,
    extract_param,
    get_loaders,
    load_checkpoint_auxiliary_modules,
    load_checkpoint_weights,
    load_model,
)
from training import get_model, train_gnn  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

TAG = "sequential_aml_to_paysim_ssl"
MODEL_ROOT = ROOT / "saved-models" / TAG
RESULT_ROOT = ROOT / "results" / "diagnostics" / TAG
EMBED_ROOT = ROOT / "embeddings" / TAG
CELLS = RESULT_ROOT / "cells"
BN_DIR = RESULT_ROOT / "bn"
LOG_DIR = RESULT_ROOT / "logs"
SMOKE_JSON = RESULT_ROOT / "smoke.json"
SCOUT_JSON = ROOT / "results" / "diagnostics" / f"{TAG}_scout.json"
NOTES_MD = ROOT / "notes" / f"{TAG}_scout.md"
SUBMISSION_JSON = RESULT_ROOT / "submission.json"

SOURCE_UNIQUE = "gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2"
SOURCE_CKPT = ROOT / f"saved-models/checkpoint_{SOURCE_UNIQUE}.tar"
SOURCE_SHA256 = "18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6"

CONTRACT = CONTRACT_LEGACY  # paysim_legacy_duplicate_v1
ENCODER_SEED = 2
DOWNSTREAM_LOGISTIC_SEED = 1
MLP_SEED = 2
MLP_EPOCHS = 15
MLP_LR = 1e-3
MLP_BS = 8192
FULL_STEPS = 500
FULL_EPOCHS = 5
SMOKE_STEPS = 2
ACCUM_FULL = 4
ACCUM_SMOKE = 1

UNIQUE_AML_INIT = "seq_aml2ps_aml_init_seed2"
UNIQUE_RANDOM = "seq_aml2ps_rand_seed2"
UNIQUE_FROZEN = "seq_aml2ps_frozen_ref_seed2"
UNIQUE_BN_ONLY = "seq_aml2ps_bn_only_seed2"

X_ONLY_CELL = (
    ROOT
    / "results/diagnostics/final_corrected_no_preserve_multiseed/cells"
    / "control_X_only_paysim_legacy_duplicate_v1.json"
)
TF_CACHE = ROOT / "results/cache/temporal_flow_causal/Small-HI/features.npy"
CONTINUATION_LABEL = "checkpoint_weight_continuation_with_optimizer_reset"

GATE_AUPRC_MARGIN = 0.003
GATE_AML_REGRESS_MAX = 0.02


def common_tags() -> Dict[str, Any]:
    return {
        "exploratory_posthoc": True,
        "table_eligible": False,
        "sequential_domain_adaptive_ssl": True,
        "joint_multidomain_pretraining": False,
        "supervised_encoder_updates": False,
        "test_evaluated": False,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, dict):
        obj = {**obj, **common_tags()}
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def code_provenance() -> Dict[str, Any]:
    def _run(cmd: List[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=str(ROOT), stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""

    files = [
        "scripts/sequential_aml_to_paysim_ssl_scout.py",
        "training.py",
        "train_util.py",
        "data_loading.py",
        "util.py",
    ]
    return {
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "source_file_sha256": {f: sha256_file(ROOT / f) for f in files if (ROOT / f).is_file()},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def hash_state_dict(sd: Dict[str, torch.Tensor], *, include: str = "all") -> str:
    h = hashlib.sha256()
    for name in sorted(sd.keys()):
        is_bn = name.endswith(("running_mean", "running_var", "num_batches_tracked"))
        if include == "bn_stats" and not is_bn:
            continue
        if include == "learned" and is_bn:
            continue
        h.update(name.encode())
        h.update(sd[name].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def extract_bn_state(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu().clone()
        for k, v in sd.items()
        if k.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }


def apply_bn_to_state_dict(
    model_sd: Dict[str, torch.Tensor], bn_sd: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    out = {k: v.clone() for k, v in model_sd.items()}
    for k, v in bn_sd.items():
        if k not in out:
            raise KeyError(f"BN key missing from model state: {k}")
        out[k] = v.clone()
    return out


def predeclared_gate() -> Dict[str, Any]:
    return {
        "written_before_full_arm_val_scores": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "promote_aml_init_only_if_all": [
            f"PaySim val AUPRC(aml_init) - AUPRC(frozen_aml) >= {GATE_AUPRC_MARGIN}",
            f"PaySim val AUPRC(aml_init) - AUPRC(bn_only) >= {GATE_AUPRC_MARGIN}",
            f"PaySim val AUPRC(aml_init) - AUPRC(random_init) >= {GATE_AUPRC_MARGIN}",
            "PaySim val AUPRC(aml_init) > AUPRC(x_only)",
            f"AMLWorld pre-3h H+X+TF val AUPRC with restored original AML BN "
            f"regresses by <= {GATE_AML_REGRESS_MAX} vs original checkpoint",
            "coverage, finite-gradient, non-collapse, label-exclusion, checkpoint, provenance pass",
        ],
        "thresholds": {
            "paysim_auprc_margin_abs": GATE_AUPRC_MARGIN,
            "amlworld_hxxtf_val_auprc_max_regression_abs": GATE_AML_REGRESS_MAX,
        },
        "eval_splits": "validation_only",
        "test_forbidden": True,
        **common_tags(),
    }


def label_exclusion_report() -> Dict[str, Any]:
    return {
        "labels_may_be_present_on_loader_edge_label": True,
        "used_in_seed_sampling": False,
        "used_in_augmentations": False,
        "used_in_infonce": False,
        "used_in_optimizer_updates": False,
        "used_in_checkpoint_selection": False,
        "used_in_stopping": False,
        "used_in_bn_recalibration": False,
        "selection_stopping": "fixed_horizon_or_ssl_train_loss_only",
        "assertion": "PaySim fraud labels must not enter SSL objective/selection/BN recal",
    }


def gin_model_class_weight() -> Dict[int, float]:
    args = create_parser().parse_args(["--data", "PaySim", "--model", "gin", "--testing"])
    return {0: float(extract_param("w_ce1", args)), 1: float(extract_param("w_ce2", args))}


def tune_thr_max_f1(y: np.ndarray, proba: np.ndarray) -> float:
    y = y.astype(np.int64)
    if len(np.unique(y)) < 2:
        return 0.5
    prec, rec, thrs = precision_recall_curve(y, proba)
    if thrs.size == 0:
        return 0.5
    f1 = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thrs[int(np.argmax(f1))])


def metrics_block(y: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    y = y.astype(np.int64)
    pred = (proba >= float(thr)).astype(np.int64)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    out = {
        "auroc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(thr),
        "positive_prediction_rate": float(pred.mean()) if pred.size else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": float(y.shape[0]),
        "n_positives": float(int(y.sum())),
        "positive_rate": float(y.mean()) if y.size else 0.0,
    }
    out.update(alert_budget_metrics(y, proba))
    return out


def ids_hash(ids: np.ndarray) -> Dict[str, Any]:
    ids = np.asarray(ids, dtype=np.int64)
    return {
        "n": int(ids.shape[0]),
        "sha256": hashlib.sha256(ids.tobytes()).hexdigest() if ids.size else None,
    }


def verify_and_assert_source_checkpoint() -> Dict[str, Any]:
    if not SOURCE_CKPT.is_file():
        raise SystemExit(f"missing source checkpoint {SOURCE_CKPT}")
    digest = sha256_file(SOURCE_CKPT)
    if digest != SOURCE_SHA256:
        raise SystemExit(f"source sha256 mismatch got={digest} expected={SOURCE_SHA256}")
    ckpt = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
    schema = ckpt.get("edge_feature_schema") or {}
    edge_dim = int(schema.get("edge_dim") or 0)
    if edge_dim != 8:
        # fallback: inspect edge_emb weight
        sd = ckpt["model_state_dict"]
        ew = [v for k, v in sd.items() if "edge_emb" in k and k.endswith("weight")]
        if not ew or int(ew[0].shape[-1]) != 8:
            raise SystemExit(f"edge_dim expected 8, schema={schema}")
        edge_dim = 8
    checks = {
        "sha256_verified": True,
        "sha256": digest,
        "ports": bool(ckpt.get("ports", True)),
        "tds": bool(ckpt.get("tds", True)),
        "correct_reverse_edge_features": bool(ckpt.get("correct_reverse_edge_features", False)),
        "preserve_seed_edges": bool(ckpt.get("preserve_seed_edges", False)),
        "include_temporal_flow_edge_features": bool(
            ckpt.get("include_temporal_flow_edge_features", False)
        ),
        "embedding_dim": int(ckpt.get("embedding_dim", 128)),
        "edge_dim": edge_dim,
        "has_contrast_projection": "contrast_projection_state_dict" in ckpt,
        "has_model_state_dict": "model_state_dict" in ckpt,
        "model": "gin",
        "ego": True,
        "emlps": True,
        "reverse_mp": True,
    }
    required_true = [
        "ports",
        "tds",
        "correct_reverse_edge_features",
        "has_contrast_projection",
        "has_model_state_dict",
    ]
    for k in required_true:
        if not checks[k]:
            raise SystemExit(f"source checkpoint assert failed: {k}={checks[k]}")
    if checks["preserve_seed_edges"]:
        raise SystemExit("preserve_seed_edges must be false on source")
    if checks["include_temporal_flow_edge_features"]:
        raise SystemExit("TF edge features must be false on source")
    if checks["embedding_dim"] != 128:
        raise SystemExit(f"embedding_dim={checks['embedding_dim']} != 128")
    if checks["edge_dim"] != 8:
        raise SystemExit(f"edge_dim={checks['edge_dim']} != 8")
    return {
        "path": str(SOURCE_CKPT),
        "expected_sha256": SOURCE_SHA256,
        **checks,
        "epoch": ckpt.get("epoch"),
    }


def stage_checkpoint(unique: str) -> Path:
    dest = ROOT / "saved-models" / f"checkpoint_{unique}.tar"
    mirror = MODEL_ROOT / f"checkpoint_{unique}.tar"
    if dest.resolve() == SOURCE_CKPT.resolve() or mirror.resolve() == SOURCE_CKPT.resolve():
        raise SystemExit("refusing to stage over locked source checkpoint")
    if dest.is_file() or mirror.is_file():
        raise SystemExit(f"refusing overwrite of existing staged ckpt for {unique}")
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    blob = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
    blob.pop("optimizer_state_dict", None)
    torch.save(blob, dest)
    shutil.copy2(dest, mirror)
    src_sd = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    for k, v in blob["model_state_dict"].items():
        if k not in src_sd or not torch.equal(v.cpu(), src_sd[k].cpu()):
            dest.unlink(missing_ok=True)
            mirror.unlink(missing_ok=True)
            raise SystemExit(f"staged weights mismatch on key {k}")
    return dest


def mirror_ckpt(unique: str, *, finetuned: bool) -> Path:
    suffix = "_finetuned" if finetuned else ""
    src = ROOT / "saved-models" / f"checkpoint_{unique}{suffix}.tar"
    if not src.is_file():
        raise SystemExit(f"missing checkpoint to mirror: {src}")
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    dst = MODEL_ROOT / src.name
    shutil.copy2(src, dst)
    return dst


def locked_paysim_train_argv(
    unique: str,
    *,
    finetune: bool,
    max_optimizer_steps: int,
    n_epochs: int,
    accum_steps: int,
) -> List[str]:
    argv = [
        "--data", "PaySim",
        "--model", "gin",
        "--testing",
        "--tqdm",
        "--objective", "contrastive",
        "--unique_name", unique,
        "--seed", str(ENCODER_SEED),
        "--batch_size", "8192",
        "--num_neighs", "100", "100",
        "--loader_num_workers", "0",
        "--n_epochs", str(int(n_epochs)),
        "--max_optimizer_steps", str(int(max_optimizer_steps)),
        "--save_model",
        "--checkpoint_policy", "last",
        "--reverse_mp", "--ego", "--ports", "--emlps", "--tds",
        "--correct_reverse_edge_features",
        "--train_fit_edge_znorm",
        "--feature_contract", CONTRACT,
        "--contrast_projection_head",
        "--contrast_projection_hidden", "128",
        "--contrast_projection_dim", "128",
        "--contrastive_asymmetric",
        "--contrastive_num_neg_samples", "8192",
        "--contrastive_memory_bank_size", "0",
        "--contrastive_accum_steps", str(int(accum_steps)),
        "--contrastive_temperature", "0.5",
    ]
    if finetune:
        argv.append("--finetune")
    return argv


def assert_train_argv_contract(argv: List[str]) -> Dict[str, Any]:
    ns = create_parser().parse_args(argv)
    if str(ns.data) != "PaySim":
        raise SystemExit("data must be PaySim")
    if str(getattr(ns, "feature_contract", None)) != CONTRACT:
        raise SystemExit(f"feature_contract must be {CONTRACT}")
    if not bool(ns.train_fit_edge_znorm):
        raise SystemExit("train_fit_edge_znorm required")
    if bool(getattr(ns, "preserve_seed_edges", False)):
        raise SystemExit("preserve must be off")
    if not bool(ns.correct_reverse_edge_features):
        raise SystemExit("correct_reverse required")
    if not (ns.ports and ns.tds and ns.emlps and ns.ego and ns.reverse_mp):
        raise SystemExit("architecture flags incomplete")
    return {
        "feature_contract": str(ns.feature_contract),
        "train_fit_edge_znorm": True,
        "correct_reverse_edge_features": True,
        "preserve_seed_edges": False,
        "seed": int(ns.seed),
        "batch_size": int(ns.batch_size),
        "contrastive_accum_steps": int(ns.contrastive_accum_steps),
        "max_optimizer_steps": int(ns.max_optimizer_steps),
        "finetune": bool(ns.finetune),
    }


def parse_train_log(text: str) -> Dict[str, Any]:
    hashes = re.findall(
        r"scout_batch_log epoch=(\d+) step=(\d+) seed_ids_sha256=([0-9a-f]+) n_seeds=(\d+)",
        text,
    )
    opt_finished = re.findall(
        r"Contrastive training finished: total_optimizer_steps=(\d+) max_optimizer_steps=(\S+)",
        text,
    )
    train_loss = re.findall(r"Train Loss:\s*([0-9.eE+-]+)", text)
    return {
        "batch_hashes": [
            {"epoch": int(e), "step": int(s), "seed_ids_sha256": h, "n_seeds": int(n)}
            for e, s, h, n in hashes
        ],
        "total_optimizer_steps": int(opt_finished[-1][0]) if opt_finished else None,
        "max_optimizer_steps_logged": opt_finished[-1][1] if opt_finished else None,
        "optimizer_reset_provenance_logged": CONTINUATION_LABEL in text,
        "rng_barrier_logged": "Training-stream RNG barrier" in text,
        "last_train_loss": float(train_loss[-1]) if train_loss else None,
    }


def recalibrate_bn(model, tr_loader, device) -> Dict[str, Any]:
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    learned_b = hash_state_dict(before, include="learned")
    bn_b = hash_state_dict(before, include="bn_stats")
    for p in model.parameters():
        p.requires_grad = False
    model.train()
    n = 0
    with torch.no_grad():
        for batch in tr_loader:
            batch[FORWARD_EDGE_TYPE].edge_attr = batch[FORWARD_EDGE_TYPE].edge_attr[:, 1:]
            batch[REVERSE_EDGE_TYPE].edge_attr = batch[REVERSE_EDGE_TYPE].edge_attr[:, 1:]
            batch.to(device)
            model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            n += 1
    model.eval()
    after = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    learned_a = hash_state_dict(after, include="learned")
    bn_a = hash_state_dict(after, include="bn_stats")
    if learned_b != learned_a:
        raise RuntimeError("BN recalibration changed learned parameters")
    for k in before:
        if k.endswith(("running_mean", "running_var", "num_batches_tracked")):
            continue
        if not torch.equal(before[k], after[k]):
            raise RuntimeError(f"learned tensor changed during BN recal: {k}")
    return {
        "n_batches": n,
        "learned_hash_before": learned_b,
        "learned_hash_after": learned_a,
        "bn_stats_hash_before": bn_b,
        "bn_stats_hash_after": bn_a,
        "learned_unchanged": True,
        "bn_stats_changed": bn_b != bn_a,
        "labels_used": False,
    }


def build_hetero_model(ns, te_data, tr_data, val_data, tr_inds, val_inds, te_inds, device):
    from types import SimpleNamespace

    config = SimpleNamespace(
        model=ns.model,
        n_hidden=extract_param("n_hidden", ns),
        n_gnn_layers=extract_param("n_gnn_layers", ns),
        n_heads=None,
        dropout=extract_param("dropout", ns),
        final_dropout=extract_param("final_dropout", ns),
    )
    transform = AddEgoIds() if ns.ego else None
    add_arange_ids([tr_data, val_data, te_data])
    sample_args = argparse.Namespace(**vars(ns))
    sample_args.loader_num_workers = 0
    tr_loader, _, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, sample_args, train_shuffle=False
    )
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, ns)
    emb_dim = int(getattr(model, "embedding_dim", 128))
    model = to_hetero(model, te_data.metadata(), aggr="mean").to(device)
    return model, emb_dim, transform, tr_loader


def save_bn_snapshot(path: Path, sd: Dict[str, torch.Tensor], meta: Dict[str, Any]) -> Dict[str, Any]:
    bn = extract_bn_state(sd)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"bn_state_dict": bn, "meta": meta}, path)
    return {
        "path": str(path),
        "bn_stats_hash": hash_state_dict(bn, include="bn_stats"),
        "n_tensors": len(bn),
        **meta,
    }


def _run_main_train(argv: List[str], log_path: Path) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, str(ROOT / "main.py"), *argv]
    with log_path.open("w") as lf:
        lf.write("CMD " + " ".join(cmd) + "\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT, check=False)
    text = log_path.read_text(errors="replace")
    if proc.returncode != 0:
        raise SystemExit(f"train failed rc={proc.returncode}; see {log_path}")
    out = parse_train_log(text)
    out["returncode"] = proc.returncode
    out["log_path"] = str(log_path)
    return out


def _run_train_gnn_inprocess(argv: List[str], log_path: Path, data_bundle, data_config) -> Dict[str, Any]:
    ns = create_parser().parse_args(argv)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = data_bundle
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_logger.addHandler(fh)
    try:
        logging.info("INPROCESS_TRAIN argv=%s", " ".join(argv))
        set_seed(int(ns.seed))
        train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, ns, data_config)
    finally:
        root_logger.removeHandler(fh)
        fh.close()
    out = parse_train_log(log_path.read_text(errors="replace"))
    out["returncode"] = 0
    out["log_path"] = str(log_path)
    out["inprocess"] = True
    return out


def require_smoke_passed() -> Dict[str, Any]:
    if not SMOKE_JSON.is_file():
        raise SystemExit(f"missing smoke artifact {SMOKE_JSON}")
    smoke = json.loads(SMOKE_JSON.read_text())
    if smoke.get("passed") is not True:
        raise SystemExit("smoke did not pass; refusing full arms")
    return smoke


def evaluate_gate(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Pure gate on PaySim/AML val AUPRC floats."""
    a = metrics["aml_init"]
    checks = {
        "vs_frozen": (a - metrics["frozen_aml"]) >= GATE_AUPRC_MARGIN,
        "vs_bn_only": (a - metrics["bn_only"]) >= GATE_AUPRC_MARGIN,
        "vs_random": (a - metrics["random_init"]) >= GATE_AUPRC_MARGIN,
        "vs_x_only": a > metrics["x_only"],
        "aml_retention": (metrics["aml_original"] - metrics["aml_continued_orig_bn"])
        <= GATE_AML_REGRESS_MAX,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "deltas": {
            "aml_init_minus_frozen": a - metrics["frozen_aml"],
            "aml_init_minus_bn_only": a - metrics["bn_only"],
            "aml_init_minus_random": a - metrics["random_init"],
            "aml_init_minus_x_only": a - metrics["x_only"],
            "aml_original_minus_continued": metrics["aml_original"] - metrics["aml_continued_orig_bn"],
        },
        "thresholds": {
            "paysim_auprc_margin_abs": GATE_AUPRC_MARGIN,
            "aml_regress_max": GATE_AML_REGRESS_MAX,
        },
    }


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def cmd_smoke(_args: argparse.Namespace) -> int:
    logger_setup()
    for p in (RESULT_ROOT, EMBED_ROOT, CELLS, BN_DIR, LOG_DIR, MODEL_ROOT):
        p.mkdir(parents=True, exist_ok=True)
    job_tag = os.environ.get("SLURM_JOB_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        return _cmd_smoke_impl(str(job_tag))
    except BaseException as err:
        payload = {"passed": False, "error": str(err), "job_tag": job_tag, **common_tags()}
        write_json(SMOKE_JSON, payload)
        logging.exception("smoke failed")
        return 1


def _cmd_smoke_impl(job_tag: str) -> int:
    t0 = time.perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source_meta = verify_and_assert_source_checkpoint()
    gate = predeclared_gate()
    write_json(RESULT_ROOT / "predeclared_gate.json", gate)

    smoke_unique = f"seq_aml2ps_smoke_aml_init_{job_tag}"
    staged = stage_checkpoint(smoke_unique)
    argv = locked_paysim_train_argv(
        smoke_unique,
        finetune=True,
        max_optimizer_steps=SMOKE_STEPS,
        n_epochs=1,
        accum_steps=ACCUM_SMOKE,
    )
    flags = assert_train_argv_contract(argv)

    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)
    ns = create_parser().parse_args(argv)
    set_seed(ENCODER_SEED)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns, data_config)
    # Contract / norm integrity
    summary = getattr(ns, "feature_contract_summary", None) or getattr(tr_data, "feature_contract", None)
    if summary is not None and isinstance(summary, dict):
        cid = summary.get("feature_contract_id") or summary.get("contract_id")
        if cid and cid != CONTRACT:
            raise SystemExit(f"loaded contract {cid} != {CONTRACT}")
    if not bool(ns.train_fit_edge_znorm):
        raise SystemExit("train_fit not set after parse")

    # Original BN snapshot from source
    src_blob = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
    bn_orig_meta = save_bn_snapshot(
        BN_DIR / "original_aml_bn.pt",
        src_blob["model_state_dict"],
        {"role": "original_aml", "source": SOURCE_UNIQUE},
    )

    # BN-only control on PaySim train
    model_bn, _, transform, tr_loader = build_hetero_model(
        ns, te_data, tr_data, val_data, tr_inds, val_inds, te_inds, device
    )
    ns_bn = argparse.Namespace(**vars(ns))
    ns_bn.unique_name = smoke_unique
    ns_bn.finetune = False
    load_checkpoint_weights(model_bn, device, ns_bn, data_config)
    bn_only_report = recalibrate_bn(model_bn, tr_loader, device)
    if not bn_only_report["bn_stats_changed"]:
        raise SystemExit("BN-only recal did not change BN stats")
    if not bn_only_report["learned_unchanged"]:
        raise SystemExit("BN-only changed learned weights")
    bn_only_meta = save_bn_snapshot(
        BN_DIR / f"bn_only_smoke_{job_tag}.pt",
        model_bn.state_dict(),
        {"role": "bn_only", "labels_used": False},
    )
    del model_bn

    # SSL two-step continuation (in-process, reuse loaded PaySim graph)
    log_path = LOG_DIR / f"smoke_train_{job_tag}.log"
    train_meta = _run_train_gnn_inprocess(
        argv, log_path, (tr_data, val_data, te_data, tr_inds, val_inds, te_inds), data_config
    )
    if train_meta["total_optimizer_steps"] != SMOKE_STEPS:
        raise SystemExit(f"expected {SMOKE_STEPS} steps, got {train_meta['total_optimizer_steps']}")
    if not train_meta["optimizer_reset_provenance_logged"]:
        raise SystemExit("missing optimizer reset provenance")
    if not np.isfinite(train_meta["last_train_loss"] or np.nan):
        raise SystemExit("non-finite train loss")

    ft = ROOT / f"saved-models/checkpoint_{smoke_unique}_finetuned.tar"
    if not ft.is_file():
        raise SystemExit(f"missing finetuned smoke ckpt {ft}")
    mirror_ckpt(smoke_unique, finetuned=True)
    ft_blob = torch.load(ft, map_location="cpu", weights_only=False)
    learned_delta = hash_state_dict(ft_blob["model_state_dict"], include="learned") != hash_state_dict(
        src_blob["model_state_dict"], include="learned"
    )
    bn_delta = hash_state_dict(ft_blob["model_state_dict"], include="bn_stats") != hash_state_dict(
        src_blob["model_state_dict"], include="bn_stats"
    )
    if not learned_delta:
        raise SystemExit("SSL smoke did not change learned weights")
    if not bn_delta:
        raise SystemExit("SSL smoke did not change BN buffers")
    adapted_bn = save_bn_snapshot(
        BN_DIR / f"aml_init_smoke_adapted_bn_{job_tag}.pt",
        ft_blob["model_state_dict"],
        {"role": "aml_init_smoke_adapted"},
    )

    # Reload integrity
    model_r, _, _, _ = build_hetero_model(
        ns, te_data, tr_data, val_data, tr_inds, val_inds, te_inds, device
    )
    ns_r = argparse.Namespace(**vars(ns))
    ns_r.unique_name = smoke_unique
    ns_r.finetune = True
    load_checkpoint_weights(model_r, device, ns_r, data_config)
    ok_reload = hash_state_dict(model_r.state_dict(), include="learned") == hash_state_dict(
        ft_blob["model_state_dict"], include="learned"
    )
    if not ok_reload:
        raise SystemExit("reload learned-hash mismatch")

    report = {
        "passed": True,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_tag": job_tag,
        "device": str(device),
        "source_checkpoint": source_meta,
        "predeclared_gate": gate,
        "flags": flags,
        "contract": CONTRACT,
        "normalization": "paysim_train_fit_edge_znorm",
        "smoke_unique": smoke_unique,
        "staged_init_ckpt": str(staged),
        "finetuned_ckpt": str(ft),
        "train": train_meta,
        "bn_original": bn_orig_meta,
        "bn_only": {**bn_only_report, **bn_only_meta},
        "bn_adapted_ssl": adapted_bn,
        "learned_weights_changed_ssl": learned_delta,
        "bn_stats_changed_ssl": bn_delta,
        "learned_weights_unchanged_bn_only": True,
        "label_exclusion": label_exclusion_report(),
        "artifact_roots": {
            "models": str(MODEL_ROOT),
            "embeddings": str(EMBED_ROOT),
            "results": str(RESULT_ROOT),
        },
        "smoke_wall_sec": time.perf_counter() - t0,
        "code_provenance": code_provenance(),
        "design_class": "matched_configuration_one_seed_exploratory_ablation",
        "exact_batch_pairing": False,
    }
    write_json(SMOKE_JSON, report)
    if json.loads(SMOKE_JSON.read_text()).get("passed") is not True:
        raise SystemExit("smoke.json not passed after write")
    logging.info("SMOKE PASSED → %s", SMOKE_JSON)
    return 0


# ---------------------------------------------------------------------------
# train_arm
# ---------------------------------------------------------------------------


def cmd_train_arm(args: argparse.Namespace) -> int:
    logger_setup()
    require_smoke_passed()
    for p in (RESULT_ROOT, MODEL_ROOT, LOG_DIR, CELLS, BN_DIR):
        p.mkdir(parents=True, exist_ok=True)
    arm = args.arm
    if arm not in ("aml_init", "random_init"):
        raise SystemExit("arm must be aml_init or random_init")
    unique = UNIQUE_AML_INIT if arm == "aml_init" else UNIQUE_RANDOM
    job_tag = os.environ.get("SLURM_JOB_ID") or "manual"
    source_meta = verify_and_assert_source_checkpoint()
    write_json(RESULT_ROOT / "predeclared_gate.json", predeclared_gate())

    if arm == "aml_init":
        stage_checkpoint(unique)
        finetune = True
    else:
        # Ensure we do not accidentally finetune from a leftover staged file.
        for p in (
            ROOT / f"saved-models/checkpoint_{unique}.tar",
            ROOT / f"saved-models/checkpoint_{unique}_finetuned.tar",
            MODEL_ROOT / f"checkpoint_{unique}.tar",
            MODEL_ROOT / f"checkpoint_{unique}_finetuned.tar",
        ):
            if p.is_file():
                raise SystemExit(f"refusing random_init with existing ckpt {p}")
        finetune = False

    argv = locked_paysim_train_argv(
        unique,
        finetune=finetune,
        max_optimizer_steps=FULL_STEPS,
        n_epochs=FULL_EPOCHS,
        accum_steps=ACCUM_FULL,
    )
    flags = assert_train_argv_contract(argv)
    log_path = LOG_DIR / f"train_{arm}_{job_tag}.log"
    train_meta = _run_main_train(argv, log_path)
    if train_meta["total_optimizer_steps"] != FULL_STEPS:
        raise SystemExit(
            f"{arm} expected {FULL_STEPS} steps got {train_meta['total_optimizer_steps']}"
        )
    if arm == "aml_init" and not train_meta["optimizer_reset_provenance_logged"]:
        raise SystemExit("aml_init missing optimizer reset provenance")
    if not train_meta.get("rng_barrier_logged"):
        logging.warning("RNG barrier log not found (non-fatal if older binary)")

    if arm == "aml_init":
        ckpt_path = ROOT / f"saved-models/checkpoint_{unique}_finetuned.tar"
        mirror_ckpt(unique, finetuned=True)
    else:
        ckpt_path = ROOT / f"saved-models/checkpoint_{unique}.tar"
        if not ckpt_path.is_file():
            raise SystemExit(f"missing trained random ckpt {ckpt_path}")
        mirror_ckpt(unique, finetuned=False)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    bn_meta = save_bn_snapshot(
        BN_DIR / f"{arm}_adapted_bn.pt",
        blob["model_state_dict"],
        {"role": arm, "unique": unique},
    )
    # Also persist original BN once
    if not (BN_DIR / "original_aml_bn.pt").is_file():
        src = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
        save_bn_snapshot(
            BN_DIR / "original_aml_bn.pt",
            src["model_state_dict"],
            {"role": "original_aml"},
        )

    cell = {
        "arm": arm,
        "unique_name": unique,
        "finetune": finetune,
        "continuation_label": CONTINUATION_LABEL if finetune else "from_scratch_random_init",
        "source_checkpoint": source_meta if finetune else None,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "train": {
            **train_meta,
            "batch_hashes_first_32": train_meta.get("batch_hashes", [])[:32],
        },
        "flags": flags,
        "bn_adapted": bn_meta,
        "label_exclusion": label_exclusion_report(),
        "exact_batch_pairing": False,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "code_provenance": code_provenance(),
    }
    write_json(CELLS / f"{arm}_train.json", cell)
    logging.info("train_arm %s done → %s", arm, CELLS / f"{arm}_train.json")
    return 0


# ---------------------------------------------------------------------------
# eval helpers
# ---------------------------------------------------------------------------


def _extract(
    *,
    data: str,
    unique: str,
    emb_subdir: str,
    representation_source: str,
    train_fit: bool,
    feature_contract: Optional[str],
    finetune: bool,
    batch_size: int,
    extract_splits: str = "train,val",
) -> Path:
    import embedding_extraction as ee

    if "test" in extract_splits.split(","):
        raise SystemExit("test split extraction forbidden in this harness")
    argv = [
        "--data", data,
        "--model", "gin",
        "--testing",
        "--tqdm",
        "--unique_name", unique,
        "--embeddings_dir", str(EMBED_ROOT),
        "--embeddings_subdir", emb_subdir,
        "--batch_size", str(batch_size),
        "--loader_num_workers", "0",
        "--num_neighs", "100", "100",
        "--representation_source", representation_source,
        "--extract_splits", extract_splits,
        "--reverse_mp", "--ego", "--ports", "--tds", "--emlps",
        "--correct_reverse_edge_features",
        "--seed", str(ENCODER_SEED),
    ]
    if train_fit:
        argv.append("--train_fit_edge_znorm")
    if feature_contract:
        argv.extend(["--feature_contract", feature_contract])
    if finetune:
        argv.append("--finetune")
    p = create_parser()
    p.add_argument("--embeddings_dir", type=str, default="embeddings")
    p.add_argument("--random_init", action="store_true")
    p.add_argument("--checkpoint_suffix", type=str, default="")
    p.add_argument("--embeddings_subdir", type=str, default=None)
    p.add_argument("--representation_source", type=str, default="post_embedding")
    p.add_argument("--extract_splits", type=str, default="train,val,test")
    ns = p.parse_args(argv)
    set_seed(ns.seed)
    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns, data_config)
    out = Path(
        ee.run_embedding_extraction(
            tr_data, val_data, te_data, tr_inds, val_inds, te_inds, ns, data_config
        )
    )
    if (out / "test.npz").is_file():
        raise SystemExit(f"test.npz written under {out} — forbidden")
    return out


def _logistic_val_only(emb_dir: Path, *, bn_protocol: str) -> Dict[str, Any]:
    if (emb_dir / "test.npz").is_file():
        raise SystemExit(f"test.npz present under {emb_dir}")
    z_tr, y_tr, ids_tr = load_embedding_npz(emb_dir / "train.npz")
    z_va, y_va, ids_va = load_embedding_npz(emb_dir / "val.npz")
    cw = gin_model_class_weight()
    set_seed(DOWNSTREAM_LOGISTIC_SEED)
    clf = LogisticRegression(
        class_weight=cw,
        max_iter=1000,
        random_state=DOWNSTREAM_LOGISTIC_SEED,
        solver="lbfgs",
        n_jobs=1,
        C=1.0,
    )
    clf.fit(z_tr, y_tr)
    proba = clf.predict_proba(z_va)[:, 1].astype(np.float64)
    thr = tune_thr_max_f1(y_va, proba)
    return {
        "validation_metrics_at_0.5": metrics_block(y_va, proba, 0.5),
        "validation_metrics_at_val_optimal_f1": metrics_block(y_va, proba, thr),
        "ids": {"train": ids_hash(ids_tr), "val": ids_hash(ids_va)},
        "test_evaluated": False,
        "learner": "LogisticRegression",
        "class_weight_mode": "model",
        "C": 1.0,
        "downstream_seed": DOWNSTREAM_LOGISTIC_SEED,
        "feature_contract": CONTRACT,
        "bn_protocol": bn_protocol,
        "normalization": "paysim_train_fit_edge_znorm",
        "feature_stack": "H_only_post128",
        "embeddings_dir": str(emb_dir),
    }


def _run_amlworld_val_only(emb_pre: Path, emb_post: Path, device: torch.device) -> Dict[str, Any]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "probe_feature_ablation", ROOT / "scripts" / "probe_feature_ablation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["probe_feature_ablation"] = mod
    spec.loader.exec_module(mod)
    df, df_train, tr_ids, va_ids, te_ids, dspec = mod.load_dataset_frames(
        "Small-HI", str(ROOT / "data_config.json")
    )
    del te_ids
    y_all = df[dspec.label_col].to_numpy().astype(np.int64)
    x_raw, _, _, _ = mod.build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    if not TF_CACHE.is_file():
        raise SystemExit(f"missing TF cache {TF_CACHE}")
    tf_feat = np.load(TF_CACHE).astype(np.float32)
    stacks: Dict[str, Any] = {}
    for stack_name, emb_dir in (("pre3h_HxXTF", emb_pre), ("post128_H_only", emb_post)):
        feats: Dict[str, Dict[str, np.ndarray]] = {}
        for sp, expected_ids in (("train", tr_ids), ("val", va_ids)):
            del expected_ids
            z, y, ids = load_embedding_npz(emb_dir / f"{sp}.npz")
            if not np.array_equal(y, y_all[ids]):
                raise SystemExit(f"AML label alignment failure {sp} {stack_name}")
            if stack_name == "post128_H_only":
                mat = z.astype(np.float32)
            else:
                mat = np.concatenate([z, x_raw[ids], tf_feat[ids]], axis=1).astype(np.float32)
            feats[sp] = {"X": mat, "y": y, "ids": ids}
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(feats["train"]["X"]).astype(np.float32)
        x_va = scaler.transform(feats["val"]["X"]).astype(np.float32)
        torch.manual_seed(MLP_SEED)
        np.random.seed(MLP_SEED)
        model = PaperStyleMLP(int(x_tr.shape[1])).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
        x_t = torch.from_numpy(x_tr)
        y_t = torch.from_numpy(feats["train"]["y"].astype(np.float32))
        n = x_tr.shape[0]
        best_auprc, best_state, best_ep = -1.0, None, -1
        for ep in range(MLP_EPOCHS):
            model.train()
            perm = np.random.RandomState(MLP_SEED * 1009 + ep).permutation(n)
            for start in range(0, n, MLP_BS):
                idx = perm[start : start + MLP_BS]
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    model(x_t[idx].to(device)), y_t[idx].to(device)
                )
                loss.backward()
                opt.step()
            pva = _predict_proba(model, x_va, batch_size=MLP_BS, device=device)
            auprc = float(average_precision_score(feats["val"]["y"], pva))
            if auprc > best_auprc + 1e-12:
                best_auprc = auprc
                best_ep = ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        assert best_state is not None
        model.load_state_dict(best_state)
        pva = _predict_proba(model, x_va, batch_size=MLP_BS, device=device)
        thr = tune_thr_max_f1(feats["val"]["y"], pva)
        stacks[stack_name] = {
            "best_epoch_by_val_auprc": best_ep,
            "best_val_auprc": best_auprc,
            "learner": "PaperStyleMLP",
            "learner_seed": MLP_SEED,
            "is_primary": stack_name == "pre3h_HxXTF",
            "validation_metrics_at_0.5": metrics_block(feats["val"]["y"], pva, 0.5),
            "validation_metrics_at_val_optimal_f1": metrics_block(feats["val"]["y"], pva, thr),
            "ids": {sp: ids_hash(feats[sp]["ids"]) for sp in feats},
            "test_evaluated": False,
        }
    return stacks


def _write_ckpt_with_bn(unique: str, model_sd: Dict[str, torch.Tensor], base_ckpt: Path) -> Path:
    blob = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    blob["model_state_dict"] = model_sd
    blob.pop("optimizer_state_dict", None)
    dest = ROOT / "saved-models" / f"checkpoint_{unique}.tar"
    if dest.resolve() == SOURCE_CKPT.resolve():
        raise SystemExit("refusing overwrite source")
    torch.save(blob, dest)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dest, MODEL_ROOT / dest.name)
    return dest


def cmd_eval(_args: argparse.Namespace) -> int:
    logger_setup()
    require_smoke_passed()
    for p in (RESULT_ROOT, EMBED_ROOT, CELLS, BN_DIR, MODEL_ROOT):
        p.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source_meta = verify_and_assert_source_checkpoint()
    write_json(RESULT_ROOT / "predeclared_gate.json", predeclared_gate())

    # Ensure frozen ref staged
    if not (ROOT / f"saved-models/checkpoint_{UNIQUE_FROZEN}.tar").is_file():
        stage_checkpoint(UNIQUE_FROZEN)

    # --- BN-only checkpoint (learned=source, BN=PaySim-train recal) ---
    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)
    argv_ps = locked_paysim_train_argv(
        UNIQUE_BN_ONLY, finetune=False, max_optimizer_steps=0, n_epochs=1, accum_steps=1
    )
    # remove save/max steps noise for data load
    ns = create_parser().parse_args(
        [a for a in argv_ps if a not in ("--save_model",)] + []
    )
    ns.unique_name = UNIQUE_FROZEN
    ns.max_optimizer_steps = 0
    ns.finetune = False
    set_seed(ENCODER_SEED)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns, data_config)
    model_bn, _, _, tr_loader = build_hetero_model(
        ns, te_data, tr_data, val_data, tr_inds, val_inds, te_inds, device
    )
    load_checkpoint_weights(model_bn, device, ns, data_config)
    bn_report = recalibrate_bn(model_bn, tr_loader, device)
    if not bn_report["learned_unchanged"] or not bn_report["bn_stats_changed"]:
        raise SystemExit(f"bn_only integrity failed: {bn_report}")
    # Build bn_only unique from frozen blob + adapted BN
    frozen_blob = torch.load(
        ROOT / f"saved-models/checkpoint_{UNIQUE_FROZEN}.tar", map_location="cpu", weights_only=False
    )
    bn_sd = extract_bn_state(model_bn.state_dict())
    merged = apply_bn_to_state_dict(frozen_blob["model_state_dict"], bn_sd)
    _write_ckpt_with_bn(UNIQUE_BN_ONLY, merged, ROOT / f"saved-models/checkpoint_{UNIQUE_FROZEN}.tar")
    save_bn_snapshot(BN_DIR / "bn_only_adapted_bn.pt", merged, {"role": "bn_only", **bn_report})
    del model_bn

    paysim: Dict[str, Any] = {}

    # frozen_aml
    emb = _extract(
        data="PaySim",
        unique=UNIQUE_FROZEN,
        emb_subdir="frozen_aml/paysim_post128",
        representation_source="post_embedding",
        train_fit=True,
        feature_contract=CONTRACT,
        finetune=False,
        batch_size=4096,
    )
    paysim["frozen_aml"] = _logistic_val_only(emb, bn_protocol="frozen_aml_bn")

    # bn_only
    emb = _extract(
        data="PaySim",
        unique=UNIQUE_BN_ONLY,
        emb_subdir="bn_only/paysim_post128",
        representation_source="post_embedding",
        train_fit=True,
        feature_contract=CONTRACT,
        finetune=False,
        batch_size=4096,
    )
    paysim["bn_only"] = _logistic_val_only(emb, bn_protocol="paysim_train_bn_recal")
    paysim["bn_only"]["bn_recalibration"] = bn_report

    # aml_init
    emb = _extract(
        data="PaySim",
        unique=UNIQUE_AML_INIT,
        emb_subdir="aml_init/paysim_post128",
        representation_source="post_embedding",
        train_fit=True,
        feature_contract=CONTRACT,
        finetune=True,
        batch_size=4096,
    )
    paysim["aml_init"] = _logistic_val_only(emb, bn_protocol="paysim_adapted_bn_from_ssl")

    # random_init
    emb = _extract(
        data="PaySim",
        unique=UNIQUE_RANDOM,
        emb_subdir="random_init/paysim_post128",
        representation_source="post_embedding",
        train_fit=True,
        feature_contract=CONTRACT,
        finetune=False,
        batch_size=4096,
    )
    paysim["random_init"] = _logistic_val_only(emb, bn_protocol="paysim_adapted_bn_from_ssl")

    # x_only from existing val cell
    if not X_ONLY_CELL.is_file():
        raise SystemExit(f"missing x_only cell {X_ONLY_CELL}")
    xcell = json.loads(X_ONLY_CELL.read_text())
    x_val = xcell.get("validation", {})
    paysim["x_only"] = {
        "source_cell": str(X_ONLY_CELL),
        "feature_contract_id": xcell.get("feature_contract_id"),
        "validation": x_val,
        "validation_auprc_at_0.5": (x_val.get("threshold_0.5") or {}).get("auprc"),
        "test_evaluated": False,
        "test_used_for_gate": False,
        "note": "Located existing multiseed X-only control; val metrics only used for gate",
    }

    # AML retention: original + continued with original BN restored
    aml: Dict[str, Any] = {}
    emb_pre = _extract(
        data="Small-HI",
        unique=SOURCE_UNIQUE,
        emb_subdir="aml_retention/original/pre3h",
        representation_source="pre_embedding_3h",
        train_fit=False,
        feature_contract=None,
        finetune=False,
        batch_size=8192,
    )
    emb_post = _extract(
        data="Small-HI",
        unique=SOURCE_UNIQUE,
        emb_subdir="aml_retention/original/post128",
        representation_source="post_embedding",
        train_fit=False,
        feature_contract=None,
        finetune=False,
        batch_size=8192,
    )
    aml["original"] = _run_amlworld_val_only(emb_pre, emb_post, device)
    aml["original"]["bn_protocol"] = "original_aml_bn"

    # Continued weights + restore original BN into a temp unique for extract
    cont_unique = "seq_aml2ps_aml_init_origbn_seed2"
    ft = ROOT / f"saved-models/checkpoint_{UNIQUE_AML_INIT}_finetuned.tar"
    if not ft.is_file():
        raise SystemExit(f"missing aml_init finetuned {ft}")
    orig_bn_path = BN_DIR / "original_aml_bn.pt"
    if not orig_bn_path.is_file():
        raise SystemExit("missing original_aml_bn.pt")
    orig_bn = torch.load(orig_bn_path, map_location="cpu", weights_only=False)["bn_state_dict"]
    cont_blob = torch.load(ft, map_location="cpu", weights_only=False)
    restored_sd = apply_bn_to_state_dict(cont_blob["model_state_dict"], orig_bn)
    _write_ckpt_with_bn(cont_unique, restored_sd, ft)

    emb_pre_c = _extract(
        data="Small-HI",
        unique=cont_unique,
        emb_subdir="aml_retention/aml_init_origbn/pre3h",
        representation_source="pre_embedding_3h",
        train_fit=False,
        feature_contract=None,
        finetune=False,
        batch_size=8192,
    )
    emb_post_c = _extract(
        data="Small-HI",
        unique=cont_unique,
        emb_subdir="aml_retention/aml_init_origbn/post128",
        representation_source="post_embedding",
        train_fit=False,
        feature_contract=None,
        finetune=False,
        batch_size=8192,
    )
    aml["aml_init_original_bn"] = _run_amlworld_val_only(emb_pre_c, emb_post_c, device)
    aml["aml_init_original_bn"]["bn_protocol"] = "restored_original_aml_bn"
    aml["aml_init_original_bn"]["retention_primary"] = True

    summary = {
        "source_checkpoint": source_meta,
        "paysim": paysim,
        "amlworld": aml,
        "label_exclusion": label_exclusion_report(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "code_provenance": code_provenance(),
    }
    write_json(RESULT_ROOT / "eval_summary.json", summary)
    write_json(CELLS / "eval_summary.json", summary)
    logging.info("eval done → %s", RESULT_ROOT / "eval_summary.json")
    return 0


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------


def cmd_aggregate(_args: argparse.Namespace) -> int:
    logger_setup()
    eval_path = RESULT_ROOT / "eval_summary.json"
    if not eval_path.is_file():
        raise SystemExit(f"missing {eval_path}")
    ev = json.loads(eval_path.read_text())
    paysim = ev["paysim"]
    aml = ev["amlworld"]

    def _auprc(block: Dict[str, Any]) -> float:
        if "validation_metrics_at_0.5" in block:
            return float(block["validation_metrics_at_0.5"]["auprc"])
        if "validation_auprc_at_0.5" in block:
            return float(block["validation_auprc_at_0.5"])
        raise KeyError("no val auprc")

    metrics = {
        "aml_init": _auprc(paysim["aml_init"]),
        "frozen_aml": _auprc(paysim["frozen_aml"]),
        "bn_only": _auprc(paysim["bn_only"]),
        "random_init": _auprc(paysim["random_init"]),
        "x_only": float(paysim["x_only"]["validation_auprc_at_0.5"]),
        "aml_original": float(aml["original"]["pre3h_HxXTF"]["validation_metrics_at_0.5"]["auprc"]),
        "aml_continued_orig_bn": float(
            aml["aml_init_original_bn"]["pre3h_HxXTF"]["validation_metrics_at_0.5"]["auprc"]
        ),
    }
    gate = evaluate_gate(metrics)
    out = {
        "title": "sequential_aml_to_paysim_ssl_scout",
        "scope": "sequential_domain_adaptive_ssl_validation_gate",
        "predeclared_gate": predeclared_gate(),
        "metrics_paysim_val_auprc": {
            k: metrics[k] for k in ("aml_init", "frozen_aml", "bn_only", "random_init", "x_only")
        },
        "metrics_amlworld_val_auprc_pre3h_HxXTF": {
            "original": metrics["aml_original"],
            "aml_init_with_restored_original_bn": metrics["aml_continued_orig_bn"],
        },
        "gate": gate,
        "promote_aml_init": gate["passed"],
        "eval_summary": str(eval_path),
        "smoke": str(SMOKE_JSON),
        "code_provenance": code_provenance(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "follow_up_jobs_submitted": False,
    }
    write_json(RESULT_ROOT / "aggregate.json", out)
    write_json(SCOUT_JSON, out)

    md = f"""# Sequential AMLWorld→PaySim SSL scout

> Exploratory / post-hoc. `table_eligible=false`. Validation only.
> `sequential_domain_adaptive_ssl=true`. Not joint multidomain pretraining.

- Promote AML→PaySim (`aml_init`): **{'PASS' if gate['passed'] else 'FAIL'}**

## PaySim val AUPRC (post-128 H, logistic)

| Arm | Val AUPRC |
|-----|----------:|
| frozen_aml | {metrics['frozen_aml']:.6f} |
| bn_only | {metrics['bn_only']:.6f} |
| random_init_paysim_ssl | {metrics['random_init']:.6f} |
| aml_init_paysim_ssl | {metrics['aml_init']:.6f} |
| x_only | {metrics['x_only']:.6f} |

## AMLWorld retention (pre-3h H+X+TF, restored original BN for continued)

| Encoder | Val AUPRC |
|---------|----------:|
| original AML | {metrics['aml_original']:.6f} |
| aml_init + original BN | {metrics['aml_continued_orig_bn']:.6f} |

## Gate deltas

```json
{json.dumps(gate, indent=2)}
```

Artifacts: `{SCOUT_JSON}`, `{RESULT_ROOT / 'aggregate.json'}`, cells under `{CELLS}`.
"""
    NOTES_MD.parent.mkdir(parents=True, exist_ok=True)
    NOTES_MD.write_text(md)
    logging.info("aggregate done promote=%s", gate["passed"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    t = sub.add_parser("train_arm")
    t.add_argument("--arm", required=True, choices=["aml_init", "random_init"])
    sub.add_parser("eval")
    sub.add_parser("aggregate")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "smoke":
        return cmd_smoke(args)
    if args.cmd == "train_arm":
        return cmd_train_arm(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    if args.cmd == "aggregate":
        return cmd_aggregate(args)
    raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
