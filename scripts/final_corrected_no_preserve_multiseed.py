#!/usr/bin/env python3
"""Final frozen corrected/no-preserve multiseed evaluation (seeds 1–4).

Final-results evaluation — not architecture search.
Encoder weights remain frozen. No GNN training or fine-tuning.
Test metrics never select configuration, checkpoint, learner, threshold,
normalization, or feature contract.

Subcommands (compute nodes via sbatch only):
  smoke | eval_amlworld | eval_paysim | eval_controls | aggregate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
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
from feature_contracts import CONTRACT_LEGACY, CONTRACT_TYPE_ONLY  # noqa: E402
from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba  # noqa: E402
from linear_probe import load_embedding_npz  # noqa: E402
from ranking_metrics import alert_budget_metrics  # noqa: E402
from train_util import (  # noqa: E402
    AddEgoIds,
    add_arange_ids,
    expected_seed_edge_ids,
    extract_param,
    extract_seed_embeddings_hetero,
    get_loaders,
    load_checkpoint_weights,
    log_seed_coverage,
    save_embedding_split_npz,
)
from training import get_model  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

TAG = "final_corrected_no_preserve_multiseed"
EMBED_ROOT = ROOT / "embeddings" / TAG
RESULT_ROOT = ROOT / "results" / "diagnostics" / TAG
PROBA_ROOT = RESULT_ROOT / "probabilities"
CELLS = RESULT_ROOT / "cells"
FINAL_JSON = ROOT / "results" / "diagnostics" / f"{TAG}.json"
FINAL_MD = ROOT / "notes" / f"{TAG}.md"

UNIQUE_TMPL = "gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed{seed}"
SEED2_SHA = "18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6"
EXPECTED_SHA = {
    1: "5e59b5f2147003745f9f0c33933cf45a5dc0a4193f291fc8315e99ebec507e3c",
    2: SEED2_SHA,
    3: "4ea55c74a55e657797fffbd532d1f29f04ee5942e63ddd1b1caddda6d27283f8",
    4: "31aae0f9b3e8040e815916bf042e32a22647d622cdb839d374a53c10313adaeb",
}
EXPECTED_EDGE_DIM = 8
FORWARD_EDGE = ("node", "to", "node")
REV_EDGE = ("node", "rev_to", "node")
DOWNSTREAM_LOGISTIC_SEED = 1
RANDOM_INIT_SEED = 2
MLP_EPOCHS = 15
MLP_LR = 1e-3
MLP_BS = 8192
MLP_SEED = 2
TF_CACHE = ROOT / "results/cache/temporal_flow_causal/Small-HI"

SEEDS_ALL = (1, 2, 3, 4)
SEEDS_CONFIRMATION = (1, 3, 4)
DEV_SEED = 2

# Predeclared PaySim protocols (locked before any test inspection).
PAYSIM_PROTOCOLS = (
    {
        "id": "P1_strict_inductive_legacy",
        "label": "P1 strict inductive primary",
        "feature_contract": CONTRACT_LEGACY,
        "train_fit": True,
        "bn_recal": False,
        "bn_protocol": "frozen_aml_bn",
        "role": "primary",
        "contract_note": "legacy compatibility contract (type duplicated into currency+payment); not claimed semantically ideal",
    },
    {
        "id": "P2_label_free_target_bn_legacy",
        "label": "P2 label-free target adaptation",
        "feature_contract": CONTRACT_LEGACY,
        "train_fit": True,
        "bn_recal": True,
        "bn_protocol": "target_train_only_running_stats",
        "role": "adaptation",
        "contract_note": "same legacy contract; BN stats adapted on PaySim train only; no labels",
    },
    {
        "id": "P3_type_only_sensitivity",
        "label": "P3 type-only sensitivity",
        "feature_contract": CONTRACT_TYPE_ONLY,
        "train_fit": True,
        "bn_recal": False,
        "bn_protocol": "frozen_aml_bn",
        "role": "sensitivity",
        "contract_note": "semantically cleaner sensitivity; must not replace P1 based on test",
    },
)

SOURCE_SNAPSHOT_FILES = (
    "graph_augmentations.py",
    "training.py",
    "util.py",
    "data_loading.py",
    "embedding_extraction.py",
    "train_util.py",
    "feature_contracts.py",
    "ranking_metrics.py",
    "linear_probe.py",
    "scripts/final_corrected_no_preserve_multiseed.py",
    "scripts/probe_feature_ablation.py",
    "gcpal_txn_node/eval_mlp.py",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_provenance() -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = None
    try:
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
        dirty_manifest = [ln[3:] for ln in dirty]
    except Exception:
        dirty_manifest = []
    src = {}
    for rel in SOURCE_SNAPSHOT_FILES:
        p = ROOT / rel
        if p.is_file():
            src[rel] = sha256_file(p)
    return {
        "git_commit": commit,
        "dirty_file_count": len(dirty_manifest),
        "dirty_tree_manifest": dirty_manifest[:100],
        "source_file_sha256": src,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "files_must_remain_unchanged_until_dag_finishes": list(SOURCE_SNAPSHOT_FILES),
    }


def ids_hash(ids: np.ndarray) -> Dict[str, Any]:
    ids = np.asarray(ids, dtype=np.int64)
    return {
        "n": int(ids.shape[0]),
        "n_unique": int(np.unique(ids).shape[0]),
        "edge_id_sum": int(ids.sum()) if ids.size else 0,
        "sha256_of_ids_bytes": hashlib.sha256(ids.tobytes()).hexdigest() if ids.size else None,
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def hetero_edge_dim(data) -> int:
    return int(data[FORWARD_EDGE].edge_attr.shape[1])


def unique_name(seed: int) -> str:
    return UNIQUE_TMPL.format(seed=int(seed))


def ckpt_path(seed: int) -> Path:
    return ROOT / "saved-models" / f"checkpoint_{unique_name(seed)}.tar"


def seed_role(seed: int) -> str:
    return "development" if int(seed) == DEV_SEED else "confirmation"


def verify_checkpoint(seed: int) -> Tuple[Path, str, Dict[str, Any]]:
    ckpt = ckpt_path(seed)
    if not ckpt.is_file():
        raise SystemExit(f"missing checkpoint {ckpt}")
    sha = sha256_file(ckpt)
    expected = EXPECTED_SHA.get(int(seed))
    if expected and sha != expected:
        raise SystemExit(f"seed{seed} sha mismatch: {sha} != {expected}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    if bool(payload.get("preserve_seed_edges")):
        raise SystemExit(f"seed{seed}: preserve_seed_edges enabled — abort")
    if not bool(payload.get("correct_reverse_edge_features")):
        raise SystemExit(f"seed{seed}: corrected reverse not set — abort")
    if payload.get("reverse_edge_feature_semantics") != "corrected":
        raise SystemExit(f"seed{seed}: reverse semantics not corrected — abort")
    meta = {
        "path": str(ckpt),
        "sha256": sha,
        "epoch": int(payload.get("epoch", -1)),
        "embedding_dim": int(payload.get("embedding_dim", -1)),
        "preserve_seed_edges": False,
        "correct_reverse_edge_features": True,
        "reverse_edge_feature_semantics": "corrected",
        "ports": bool(payload.get("ports")),
        "tds": bool(payload.get("tds")),
        "encoder_frozen": True,
    }
    return ckpt, sha, meta


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
    out = {
        "auroc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(thr),
        "positive_prediction_rate": float(pred.mean()) if y.size else 0.0,
        "tp": float(((pred == 1) & (y == 1)).sum()),
        "fp": float(((pred == 1) & (y == 0)).sum()),
        "tn": float(((pred == 0) & (y == 0)).sum()),
        "fn": float(((pred == 0) & (y == 1)).sum()),
        "n": float(y.shape[0]),
        "n_positives": float(int(y.sum())),
        "positive_rate": float(y.mean()) if y.size else 0.0,
    }
    out.update(alert_budget_metrics(y, proba))
    return out


def hash_state_dict(sd: Dict[str, torch.Tensor], *, include: str) -> str:
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


def base_argv(
    *,
    data: str,
    unique: str,
    emb_dir: str,
    emb_subdir: str,
    seed: int,
    train_fit: bool,
    representation_source: str = "post_embedding",
    batch_size: int = 4096,
    feature_contract: Optional[str] = None,
    random_init: bool = False,
    extract_splits: str = "train,val,test",
) -> List[str]:
    argv = [
        "--data", data, "--model", "gin", "--testing", "--tqdm",
        "--unique_name", unique,
        "--embeddings_dir", emb_dir,
        "--embeddings_subdir", emb_subdir,
        "--batch_size", str(batch_size),
        "--loader_num_workers", "0",
        "--num_neighs", "100", "100",
        "--representation_source", representation_source,
        "--extract_splits", extract_splits,
        "--reverse_mp", "--ego", "--ports", "--tds", "--emlps",
        "--correct_reverse_edge_features",
        "--seed", str(seed),
    ]
    if train_fit:
        argv.append("--train_fit_edge_znorm")
    if feature_contract:
        argv.extend(["--feature_contract", feature_contract])
    if random_init:
        argv.append("--random_init")
    return argv


def parse_extract_args(argv: List[str]):
    p = create_parser()
    p.add_argument("--embeddings_dir", type=str, default="embeddings")
    p.add_argument("--random_init", action="store_true")
    p.add_argument("--checkpoint_suffix", type=str, default="")
    p.add_argument("--embeddings_subdir", type=str, default=None)
    p.add_argument("--representation_source", type=str, default="post_embedding")
    p.add_argument("--extract_splits", type=str, default="train,val,test")
    return p.parse_args(argv)


def build_hetero_model(ns, te_data, tr_data, val_data, tr_inds, val_inds, te_inds, device):
    from types import SimpleNamespace

    config = SimpleNamespace(
        model=ns.model,
        n_hidden=extract_param("n_hidden", ns),
        n_gnn_layers=extract_param("n_gnn_layers", ns),
        n_heads=extract_param("n_heads", ns) if ns.model == "gat" else None,
        dropout=extract_param("dropout", ns),
        final_dropout=extract_param("final_dropout", ns),
    )
    transform = AddEgoIds() if ns.ego else None
    add_arange_ids([tr_data, val_data, te_data])
    sample_args = SimpleNamespace(**vars(ns))
    sample_args.loader_num_workers = 0
    sample_loader, _, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, sample_args, train_shuffle=False
    )
    sample_batch = next(iter(sample_loader))
    del sample_loader
    model = get_model(sample_batch, config, ns)
    model = to_hetero(model, te_data.metadata(), aggr="mean")
    model.to(device)
    return model, transform


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
            batch[FORWARD_EDGE].edge_attr = batch[FORWARD_EDGE].edge_attr[:, 1:]
            batch[REV_EDGE].edge_attr = batch[REV_EDGE].edge_attr[:, 1:]
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
            raise RuntimeError(f"learned tensor changed: {k}")
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


def save_proba_npz(
    path: Path,
    transaction_ids: np.ndarray,
    labels: np.ndarray,
    proba: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tids = np.asarray(transaction_ids, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(proba, dtype=np.float64)
    np.savez_compressed(
        path,
        transaction_id=tids,
        edge_id=tids,
        label=y,
        y=y,
        predicted_probability=p,
        proba=p,
    )


def _load_proba(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    if "transaction_id" in d.files:
        ids = d["transaction_id"].astype(np.int64)
    else:
        ids = d["edge_id"].astype(np.int64)
    if "label" in d.files:
        y = d["label"].astype(np.int64)
    else:
        y = d["y"].astype(np.int64)
    if "predicted_probability" in d.files:
        proba = d["predicted_probability"].astype(np.float64)
    else:
        proba = d["proba"].astype(np.float64)
    return ids, y, proba


def fit_mlp_stack(
    feats: Dict[str, Dict[str, np.ndarray]],
    device: torch.device,
) -> Dict[str, Any]:
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(feats["train"]["X"]).astype(np.float32)
    x_va = scaler.transform(feats["val"]["X"]).astype(np.float32)
    x_te = scaler.transform(feats["test"]["X"]).astype(np.float32)

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
    model.load_state_dict(best_state)
    model.to(device)
    ptr = _predict_proba(model, x_tr, batch_size=MLP_BS, device=device)
    pva = _predict_proba(model, x_va, batch_size=MLP_BS, device=device)
    pte = _predict_proba(model, x_te, batch_size=MLP_BS, device=device)
    thr = tune_thr_max_f1(feats["val"]["y"], pva)
    return {
        "best_epoch_by_val_auprc": best_ep,
        "best_val_auprc": best_auprc,
        "validation_selected_threshold": thr,
        "feature_dim": int(x_tr.shape[1]),
        "learner": "PaperStyleMLP",
        "learner_seed": MLP_SEED,
        "mlp_epochs": MLP_EPOCHS,
        "mlp_lr": MLP_LR,
        "mlp_batch_size": MLP_BS,
        "proba": {"train": ptr, "val": pva, "test": pte},
        "threshold_provenance": "max_f1_on_validation_only",
        "test_used_for_selection": False,
    }


def extract_representation(
    *,
    data: str,
    seed: int,
    unique: str,
    subdir: str,
    train_fit: bool,
    representation_source: str,
    batch_size: int,
    device: torch.device,
    feature_contract: Optional[str] = None,
    bn_recal: bool = False,
    random_init: bool = False,
    extract_seed_override: Optional[int] = None,
) -> Tuple[Path, Optional[Dict[str, Any]]]:
    import embedding_extraction as ee

    use_seed = int(extract_seed_override if extract_seed_override is not None else seed)
    argv = base_argv(
        data=data,
        unique=unique,
        emb_dir=str(EMBED_ROOT),
        emb_subdir=subdir,
        seed=use_seed,
        train_fit=train_fit,
        representation_source=representation_source,
        batch_size=batch_size,
        feature_contract=feature_contract,
        random_init=random_init,
    )
    ns = parse_extract_args(argv)
    set_seed(ns.seed)
    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)

    if not bn_recal:
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns, data_config)
        if hetero_edge_dim(tr_data) != EXPECTED_EDGE_DIM:
            raise SystemExit(f"edge_dim={hetero_edge_dim(tr_data)} for {subdir}")
        out = ee.run_embedding_extraction(
            tr_data, val_data, te_data, tr_inds, val_inds, te_inds, ns, data_config
        )
        meta_path = out / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        meta.update({
            "train_fit_edge_znorm": train_fit,
            "bn_protocol": "n/a_random" if random_init else "frozen_aml_bn",
            "feature_contract_id": feature_contract,
            "protocol_tag": subdir,
            "encoder_frozen": True,
            "random_init": random_init,
        })
        write_json(meta_path, meta)
        return out, None

    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns, data_config)
    if hetero_edge_dim(tr_data) != EXPECTED_EDGE_DIM:
        raise SystemExit(f"edge_dim={hetero_edge_dim(tr_data)} for {subdir}")
    model, transform = build_hetero_model(
        ns, te_data, tr_data, val_data, tr_inds, val_inds, te_inds, device
    )
    load_checkpoint_weights(model, device, ns, data_config)
    for p in model.parameters():
        p.requires_grad = False
    tr_loader, val_loader, te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, ns, train_shuffle=False
    )
    bn_info = recalibrate_bn(model, tr_loader, device)
    model.eval()
    out_dir = EMBED_ROOT / subdir
    # run_embedding_extraction nests by representation_source; mirror for BN path
    if representation_source == "post_embedding":
        save_dir = out_dir
    else:
        save_dir = out_dir / representation_source
    save_dir.mkdir(parents=True, exist_ok=True)
    for split_name, loader, inds, graph in (
        ("train", tr_loader, tr_inds, tr_data),
        ("val", val_loader, val_inds, val_data),
        ("test", te_loader, te_inds, te_data),
    ):
        expected = expected_seed_edge_ids(loader.data, inds, hetero=True)
        edge_ids, z, y = extract_seed_embeddings_hetero(
            loader, inds, model, graph, device, ns,
            representation_source=representation_source, pre_dim=None, emb_dim=128, head_spec=None,
        )
        log_seed_coverage(edge_ids, expected, split_name)
        save_embedding_split_npz(save_dir / f"{split_name}.npz", z, y, edge_ids)
    write_json(save_dir / "meta.json", {
        "bn_recalibration": bn_info,
        "train_fit_edge_znorm": train_fit,
        "bn_protocol": "target_train_only_running_stats",
        "feature_contract_id": feature_contract,
        "source_unique_name": unique,
        "encoder_frozen": True,
        "labels_used_in_bn_recal": False,
    })
    return save_dir, bn_info


def resolve_emb_dir(out: Path, subdir: str, representation_source: str) -> Path:
    if (out / "train.npz").is_file():
        return out
    nested = EMBED_ROOT / subdir / representation_source
    if (nested / "train.npz").is_file():
        return nested
    alt = out / representation_source
    if (alt / "train.npz").is_file():
        return alt
    raise SystemExit(f"missing embeddings under {out} / {nested}")


def run_logistic_protocol(
    emb_dir: Path,
    *,
    seed: Optional[int],
    protocol_id: str,
    feature_contract: str,
    bn_protocol: str,
    ckpt_meta: Optional[Dict[str, Any]],
    encoder_role: str,
) -> Dict[str, Any]:
    splits = {}
    for sp in ("train", "val", "test"):
        z, y, ids = load_embedding_npz(emb_dir / f"{sp}.npz")
        splits[sp] = {"Z": z, "y": y, "ids": ids}

    cw = gin_model_class_weight()
    set_seed(DOWNSTREAM_LOGISTIC_SEED)
    clf = LogisticRegression(
        class_weight=cw, max_iter=1000, random_state=DOWNSTREAM_LOGISTIC_SEED,
        solver="lbfgs", n_jobs=1, C=1.0,
    )
    clf.fit(splits["train"]["Z"], splits["train"]["y"])
    proba = {sp: clf.predict_proba(splits[sp]["Z"])[:, 1].astype(np.float64) for sp in splits}
    thr = tune_thr_max_f1(splits["val"]["y"], proba["val"])

    proba_paths = {}
    for sp in ("train", "val", "test"):
        tag = f"seed{seed}" if seed is not None else encoder_role
        path = PROBA_ROOT / f"{tag}_{protocol_id}_{sp}.npz"
        save_proba_npz(path, splits[sp]["ids"], splits[sp]["y"], proba[sp])
        proba_paths[sp] = str(path)

    rep = {
        "seed": seed,
        "encoder_role": encoder_role if seed is None else seed_role(int(seed)),
        "protocol_id": protocol_id,
        "feature_contract_id": feature_contract,
        "normalization_protocol": "paysim_train_fit_edge_znorm",
        "bn_protocol": bn_protocol,
        "class_weight_mode": "model",
        "C": 1.0,
        "learner": "LogisticRegression",
        "downstream_seed": DOWNSTREAM_LOGISTIC_SEED,
        "feature_stack": "H_only_post128",
        "representation_layer": "post_embedding",
        "representation_dim": 128,
        "checkpoint": ckpt_meta,
        "embeddings_dir": str(emb_dir),
        "proba_npz": proba_paths,
        "ids": {sp: ids_hash(splits[sp]["ids"]) for sp in splits},
        "coverage": {
            sp: {
                "n": int(splits[sp]["y"].shape[0]),
                "n_positives": int(splits[sp]["y"].sum()),
                "positive_rate": float(splits[sp]["y"].mean()),
            }
            for sp in splits
        },
        "validation_selected_threshold": thr,
        "threshold_provenance": "max_f1_on_paysim_validation_only",
        "test_used_for_selection": False,
        "encoder_frozen": True,
        "validation": {
            "threshold_0.5": metrics_block(splits["val"]["y"], proba["val"], 0.5),
            "threshold_val_selected": metrics_block(splits["val"]["y"], proba["val"], thr),
        },
        "test": {
            "threshold_0.5": metrics_block(splits["test"]["y"], proba["test"], 0.5),
            "threshold_val_selected": metrics_block(splits["test"]["y"], proba["test"], thr),
        },
        "code_provenance": code_provenance(),
    }
    out_name = f"seed{seed}_{protocol_id}.json" if seed is not None else f"{encoder_role}_{protocol_id}.json"
    write_json(CELLS / out_name, rep)
    return rep


def run_amlworld(seed: int, batch_size: int, device: torch.device) -> Dict[str, Any]:
    import importlib.util

    unique = unique_name(seed)
    ckpt, ckpt_sha, ckpt_meta = verify_checkpoint(seed)

    # --- primary: pre-3h ---
    sub_pre = f"seed{seed}_amlworld_pre3h"
    out_pre, _ = extract_representation(
        data="Small-HI", seed=seed, unique=unique, subdir=sub_pre,
        train_fit=False, representation_source="pre_embedding_3h",
        batch_size=max(batch_size, 8192), device=device,
    )
    emb_pre = resolve_emb_dir(out_pre, sub_pre, "pre_embedding_3h")

    # --- diagnostic: post-128 ---
    sub_post = f"seed{seed}_amlworld_post128"
    out_post, _ = extract_representation(
        data="Small-HI", seed=seed, unique=unique, subdir=sub_post,
        train_fit=False, representation_source="post_embedding",
        batch_size=max(batch_size, 8192), device=device,
    )
    emb_post = resolve_emb_dir(out_post, sub_post, "post_embedding")

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
    y_all = df[dspec.label_col].to_numpy().astype(np.int64)
    x_raw, _, _, _ = mod.build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    tf_path = TF_CACHE / "features.npy"
    if not tf_path.is_file():
        raise SystemExit(f"missing TF cache {tf_path}")
    tf_feat = np.load(tf_path).astype(np.float32)

    def build_feats(emb_dir: Path, stack_name: str) -> Dict[str, Dict[str, np.ndarray]]:
        feats = {}
        for sp, expected_ids in (("train", tr_ids), ("val", va_ids), ("test", te_ids)):
            z, y, ids = load_embedding_npz(emb_dir / f"{sp}.npz")
            if not np.array_equal(y, y_all[ids]):
                raise SystemExit(f"AML label mismatch seed={seed} {sp} {stack_name}")
            if stack_name.endswith("H_only"):
                mat = z.astype(np.float32)
            else:
                mat = np.concatenate([z, x_raw[ids], tf_feat[ids]], axis=1).astype(np.float32)
            feats[sp] = {"X": mat, "y": y, "ids": ids}
        return feats

    stacks_spec = (
        ("pre3h_H_only", emb_pre, "pre_embedding_3h", False),
        ("pre3h_HxXTF", emb_pre, "pre_embedding_3h", True),
        ("post128_H_only", emb_post, "post_embedding", False),
    )
    stacks: Dict[str, Any] = {}
    for stack_name, emb_dir, rep_layer, _primary in stacks_spec:
        feats = build_feats(emb_dir, stack_name)
        fit = fit_mlp_stack(feats, device)
        proba_paths = {}
        for sp in ("train", "val", "test"):
            path = PROBA_ROOT / f"seed{seed}_amlworld_{stack_name}_{sp}.npz"
            save_proba_npz(path, feats[sp]["ids"], feats[sp]["y"], fit["proba"][sp])
            proba_paths[sp] = str(path)
        thr = fit["validation_selected_threshold"]
        stacks[stack_name] = {
            "best_epoch_by_val_auprc": fit["best_epoch_by_val_auprc"],
            "best_val_auprc": fit["best_val_auprc"],
            "validation_selected_threshold": thr,
            "feature_dim": fit["feature_dim"],
            "learner": fit["learner"],
            "learner_seed": fit["learner_seed"],
            "representation_layer": rep_layer,
            "is_primary": stack_name == "pre3h_HxXTF",
            "is_diagnostic": stack_name == "post128_H_only",
            "threshold_provenance": fit["threshold_provenance"],
            "test_used_for_selection": False,
            "encoder_frozen": True,
            "proba_npz": proba_paths,
            "ids": {sp: ids_hash(feats[sp]["ids"]) for sp in feats},
            "coverage": {
                sp: {
                    "n": int(feats[sp]["y"].shape[0]),
                    "n_positives": int(feats[sp]["y"].sum()),
                    "positive_rate": float(feats[sp]["y"].mean()),
                }
                for sp in feats
            },
            "validation": {
                "threshold_0.5": metrics_block(feats["val"]["y"], fit["proba"]["val"], 0.5),
                "threshold_val_selected": metrics_block(feats["val"]["y"], fit["proba"]["val"], thr),
            },
            "test": {
                "threshold_0.5": metrics_block(feats["test"]["y"], fit["proba"]["test"], 0.5),
                "threshold_val_selected": metrics_block(feats["test"]["y"], fit["proba"]["test"], thr),
            },
        }

    report = {
        "seed": seed,
        "encoder_role": seed_role(seed),
        "unique_name": unique,
        "checkpoint": ckpt_meta,
        "dataset": "Small-HI",
        "split": "temporal",
        "preserve_seed_edges": False,
        "correct_reverse_edge_features": True,
        "encoder_frozen": True,
        "test_used_for_selection": False,
        "stacks": stacks,
        "embeddings": {"pre3h": str(emb_pre), "post128": str(emb_post)},
        "code_provenance": code_provenance(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(CELLS / f"seed{seed}_amlworld.json", report)
    return report


def run_paysim(seed: int, batch_size: int, device: torch.device) -> Dict[str, Any]:
    unique = unique_name(seed)
    ckpt, ckpt_sha, ckpt_meta = verify_checkpoint(seed)
    protocols_out: Dict[str, Any] = {}
    for proto in PAYSIM_PROTOCOLS:
        pid = proto["id"]
        logging.info("=== PaySim %s seed=%s ===", pid, seed)
        subdir = f"seed{seed}_{pid}"
        emb, bn_info = extract_representation(
            data="PaySim",
            seed=seed,
            unique=unique,
            subdir=subdir,
            train_fit=bool(proto["train_fit"]),
            representation_source="post_embedding",
            batch_size=batch_size,
            device=device,
            feature_contract=str(proto["feature_contract"]),
            bn_recal=bool(proto["bn_recal"]),
        )
        emb_dir = resolve_emb_dir(emb, subdir, "post_embedding")
        rep = run_logistic_protocol(
            emb_dir,
            seed=seed,
            protocol_id=pid,
            feature_contract=str(proto["feature_contract"]),
            bn_protocol=str(proto["bn_protocol"]),
            ckpt_meta=ckpt_meta,
            encoder_role=seed_role(seed),
        )
        if bn_info is not None:
            rep["bn_recalibration"] = bn_info
            write_json(CELLS / f"seed{seed}_{pid}.json", rep)
        protocols_out[pid] = {
            "label": proto["label"],
            "role": proto["role"],
            "contract_note": proto["contract_note"],
            "cell": str(CELLS / f"seed{seed}_{pid}.json"),
            "test_auroc": rep["test"]["threshold_0.5"]["auroc"],
            "test_auprc": rep["test"]["threshold_0.5"]["auprc"],
        }

    summary = {
        "seed": seed,
        "encoder_role": seed_role(seed),
        "unique_name": unique,
        "checkpoint": ckpt_meta,
        "protocols": protocols_out,
        "encoder_frozen": True,
        "test_used_for_selection": False,
        "code_provenance": code_provenance(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(CELLS / f"seed{seed}_paysim_summary.json", summary)
    return summary


def _edge_attr_matrix_for_split(data, inds) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_edge_attr_no_id, y, transaction_ids) for seed edges.

    ``inds`` are positional indices into the hetero forward edge store (same as
    ``expected_seed_edge_ids``). Requires ``add_arange_ids`` so col0 is the
    transaction id.
    """
    edge = data[FORWARD_EDGE]
    attr = edge.edge_attr
    y_all = edge.y
    inds_t = torch.as_tensor(inds, dtype=torch.long)
    if attr.shape[1] < EXPECTED_EDGE_DIM + 1:
        raise SystemExit(
            f"X-only expected edge_attr width>={EXPECTED_EDGE_DIM + 1} after arange ids; got {attr.shape[1]}"
        )
    ids = attr[inds_t, 0].detach().cpu().numpy().astype(np.int64)
    x = attr[inds_t, 1:].detach().cpu().numpy().astype(np.float32)
    y = y_all[inds_t].detach().cpu().numpy().astype(np.int64)
    return x, y, ids


def run_x_only_control(feature_contract: str, device: torch.device) -> Dict[str, Any]:
    """Logistic on PaySim edge_attr (post-contract, train-fit z-norm already in get_data)."""
    argv = base_argv(
        data="PaySim",
        unique="x_only_control",
        emb_dir=str(EMBED_ROOT),
        emb_subdir=f"controls_x_only_{feature_contract}",
        seed=DOWNSTREAM_LOGISTIC_SEED,
        train_fit=True,
        feature_contract=feature_contract,
    )
    ns = parse_extract_args(argv)
    set_seed(ns.seed)
    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns, data_config)
    if hetero_edge_dim(tr_data) != EXPECTED_EDGE_DIM:
        raise SystemExit(f"X-only pre-id edge_dim={hetero_edge_dim(tr_data)}")
    add_arange_ids([tr_data, val_data, te_data])
    x_tr, y_tr, id_tr = _edge_attr_matrix_for_split(tr_data, tr_inds)
    x_va, y_va, id_va = _edge_attr_matrix_for_split(val_data, val_inds)
    x_te, y_te, id_te = _edge_attr_matrix_for_split(te_data, te_inds)
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr).astype(np.float32)
    x_va_s = scaler.transform(x_va).astype(np.float32)
    x_te_s = scaler.transform(x_te).astype(np.float32)
    cw = gin_model_class_weight()
    set_seed(DOWNSTREAM_LOGISTIC_SEED)
    clf = LogisticRegression(
        class_weight=cw, max_iter=1000, random_state=DOWNSTREAM_LOGISTIC_SEED,
        solver="lbfgs", n_jobs=1, C=1.0,
    )
    clf.fit(x_tr_s, y_tr)
    proba = {
        "train": clf.predict_proba(x_tr_s)[:, 1].astype(np.float64),
        "val": clf.predict_proba(x_va_s)[:, 1].astype(np.float64),
        "test": clf.predict_proba(x_te_s)[:, 1].astype(np.float64),
    }
    thr = tune_thr_max_f1(y_va, proba["val"])
    ids = {"train": id_tr, "val": id_va, "test": id_te}
    ys = {"train": y_tr, "val": y_va, "test": y_te}
    proba_paths = {}
    for sp in ("train", "val", "test"):
        path = PROBA_ROOT / f"control_x_only_{feature_contract}_{sp}.npz"
        save_proba_npz(path, ids[sp], ys[sp], proba[sp])
        proba_paths[sp] = str(path)
    pid = f"X_only_{feature_contract}"
    rep = {
        "control": "X_only",
        "feature_contract_id": feature_contract,
        "normalization_protocol": "paysim_train_fit_StandardScaler_on_edge_attr",
        "bn_protocol": "n/a",
        "learner": "LogisticRegression",
        "downstream_seed": DOWNSTREAM_LOGISTIC_SEED,
        "class_weight_mode": "model",
        "C": 1.0,
        "feature_dim": int(x_tr_s.shape[1]),
        "validation_selected_threshold": thr,
        "threshold_provenance": "max_f1_on_paysim_validation_only",
        "test_used_for_selection": False,
        "encoder_frozen": True,
        "no_encoder": True,
        "proba_npz": proba_paths,
        "ids": {sp: ids_hash(ids[sp]) for sp in ids},
        "validation": {
            "threshold_0.5": metrics_block(y_va, proba["val"], 0.5),
            "threshold_val_selected": metrics_block(y_va, proba["val"], thr),
        },
        "test": {
            "threshold_0.5": metrics_block(y_te, proba["test"], 0.5),
            "threshold_val_selected": metrics_block(y_te, proba["test"], thr),
        },
        "code_provenance": code_provenance(),
    }
    write_json(CELLS / f"control_{pid}.json", rep)
    return rep


def cmd_smoke(args: argparse.Namespace) -> int:
    """Short integrity smoke on seed 2 (GPU)."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt, sha, meta = verify_checkpoint(DEV_SEED)
    unique = unique_name(DEV_SEED)

    # AMLWorld: build model from sample WITH id column still present
    argv = base_argv(
        data="Small-HI", unique=unique, emb_dir=str(EMBED_ROOT),
        emb_subdir="smoke_aml", seed=DEV_SEED, train_fit=False,
        representation_source="post_embedding", batch_size=int(args.batch_size),
        extract_splits="train,val,test",
    )
    ns = parse_extract_args(argv)
    set_seed(ns.seed)
    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns, data_config)
    model, transform = build_hetero_model(
        ns, te_data, tr_data, val_data, tr_inds, val_inds, te_inds, device
    )
    load_checkpoint_weights(model, device, ns, data_config)
    for p in model.parameters():
        p.requires_grad = False
    learned_hash = hash_state_dict(model.state_dict(), include="learned")
    tr_loader, _, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, ns, train_shuffle=False
    )
    batch = next(iter(tr_loader))
    # strip id for forward
    batch[FORWARD_EDGE].edge_attr = batch[FORWARD_EDGE].edge_attr[:, 1:]
    batch[REV_EDGE].edge_attr = batch[REV_EDGE].edge_attr[:, 1:]
    assert batch[FORWARD_EDGE].edge_attr.shape[1] == EXPECTED_EDGE_DIM
    batch.to(device)
    model.eval()
    with torch.no_grad():
        out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
    # PaySim legacy contract load check
    argv_ps = base_argv(
        data="PaySim", unique=unique, emb_dir=str(EMBED_ROOT),
        emb_subdir="smoke_paysim", seed=DEV_SEED, train_fit=True,
        feature_contract=CONTRACT_LEGACY, batch_size=int(args.batch_size),
    )
    ns_ps = parse_extract_args(argv_ps)
    tr_ps, val_ps, te_ps, tr_i, val_i, te_i = get_data(ns_ps, data_config)
    if hetero_edge_dim(tr_ps) != EXPECTED_EDGE_DIM:
        raise SystemExit(f"PaySim smoke edge_dim={hetero_edge_dim(tr_ps)}")

    report = {
        "passed": True,
        "seed": DEV_SEED,
        "checkpoint": meta,
        "aml_forward_ok": True,
        "paysim_legacy_edge_dim": EXPECTED_EDGE_DIM,
        "encoder_grad_disabled": True,
        "learned_hash": learned_hash,
        "out_node_keys": list(out.keys()) if isinstance(out, dict) else str(type(out)),
        "code_provenance": code_provenance(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(RESULT_ROOT / "smoke.json", report)
    logging.info("Smoke passed: %s", RESULT_ROOT / "smoke.json")
    return 0


def cmd_eval_amlworld(args: argparse.Namespace) -> int:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    CELLS.mkdir(parents=True, exist_ok=True)
    PROBA_ROOT.mkdir(parents=True, exist_ok=True)
    EMBED_ROOT.mkdir(parents=True, exist_ok=True)
    run_amlworld(int(args.seed), int(args.batch_size), device)
    return 0


def cmd_eval_paysim(args: argparse.Namespace) -> int:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    CELLS.mkdir(parents=True, exist_ok=True)
    PROBA_ROOT.mkdir(parents=True, exist_ok=True)
    EMBED_ROOT.mkdir(parents=True, exist_ok=True)
    run_paysim(int(args.seed), int(args.batch_size), device)
    return 0


def cmd_eval_controls(args: argparse.Namespace) -> int:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    CELLS.mkdir(parents=True, exist_ok=True)
    PROBA_ROOT.mkdir(parents=True, exist_ok=True)
    EMBED_ROOT.mkdir(parents=True, exist_ok=True)

    controls: Dict[str, Any] = {"random": {}, "x_only": {}}
    # Matched random encoders once per relevant contract (not once per pretrained seed)
    for contract in (CONTRACT_LEGACY, CONTRACT_TYPE_ONLY):
        subdir = f"controls_random_{contract}"
        emb, _ = extract_representation(
            data="PaySim",
            seed=RANDOM_INIT_SEED,
            unique=unique_name(DEV_SEED),  # architecture match; weights random
            subdir=subdir,
            train_fit=True,
            representation_source="post_embedding",
            batch_size=int(args.batch_size),
            device=device,
            feature_contract=contract,
            bn_recal=False,
            random_init=True,
            extract_seed_override=RANDOM_INIT_SEED,
        )
        emb_dir = resolve_emb_dir(emb, subdir, "post_embedding")
        # Map to primary-like protocol ids for reporting
        pid = f"random_{contract}"
        rep = run_logistic_protocol(
            emb_dir,
            seed=None,
            protocol_id=pid,
            feature_contract=contract,
            bn_protocol="frozen_aml_bn",
            ckpt_meta=None,
            encoder_role="random_init",
        )
        # rewrite cell name for clarity
        write_json(CELLS / f"control_{pid}.json", rep)
        controls["random"][contract] = {
            "cell": str(CELLS / f"control_{pid}.json"),
            "test_auroc": rep["test"]["threshold_0.5"]["auroc"],
            "test_auprc": rep["test"]["threshold_0.5"]["auprc"],
        }

    for contract in (CONTRACT_LEGACY, CONTRACT_TYPE_ONLY):
        xo = run_x_only_control(contract, device)
        controls["x_only"][contract] = {
            "cell": str(CELLS / f"control_X_only_{contract}.json"),
            "test_auroc": xo["test"]["threshold_0.5"]["auroc"],
            "test_auprc": xo["test"]["threshold_0.5"]["auprc"],
        }

    summary = {
        "controls": controls,
        "note": "Controls computed once (not per pretrained seed). Random uses --random_init seed=2.",
        "test_used_for_selection": False,
        "code_provenance": code_provenance(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(CELLS / "controls_summary.json", summary)
    return 0


def _mean_sd_med(vals: Sequence[float]) -> Dict[str, Any]:
    a = np.asarray(list(vals), dtype=np.float64)
    return {
        "mean": float(a.mean()) if a.size else float("nan"),
        "sample_std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)) if a.size else float("nan"),
        "n": int(a.size),
        "values": [float(x) for x in a],
    }


def _agg_metric_block(cells: List[Dict[str, Any]], split: str, thr_key: str, metric: str) -> Dict[str, Any]:
    return _mean_sd_med([c[split][thr_key][metric] for c in cells])


def _ensemble_for_protocol(protocol_id: str, seeds: Sequence[int]) -> Dict[str, Any]:
    val_maps, test_maps = [], []
    for s in seeds:
        vid, vy, vp = _load_proba(PROBA_ROOT / f"seed{s}_{protocol_id}_val.npz")
        tid, ty, tp = _load_proba(PROBA_ROOT / f"seed{s}_{protocol_id}_test.npz")
        val_maps.append({int(e): (int(yy), float(pp)) for e, yy, pp in zip(vid, vy, vp)})
        test_maps.append({int(e): (int(yy), float(pp)) for e, yy, pp in zip(tid, ty, tp)})

    def _intersect(maps):
        common = set(maps[0].keys())
        for m in maps[1:]:
            common &= set(m.keys())
        common = sorted(common)
        y = np.asarray([maps[0][e][0] for e in common], dtype=np.int64)
        for m in maps[1:]:
            for e, yy in zip(common, y):
                if m[e][0] != int(yy):
                    raise SystemExit(f"label mismatch edge={e} protocol={protocol_id}")
        proba = np.asarray([float(np.mean([m[e][1] for m in maps])) for e in common], dtype=np.float64)
        return np.asarray(common, dtype=np.int64), y, proba

    val_ids, val_y, val_p = _intersect(val_maps)
    test_ids, test_y, test_p = _intersect(test_maps)
    thr = tune_thr_max_f1(val_y, val_p)
    for sp, ids, y, p in (
        ("val", val_ids, val_y, val_p),
        ("test", test_ids, test_y, test_p),
    ):
        save_proba_npz(PROBA_ROOT / f"ensemble_{protocol_id}_{sp}.npz", ids, y, p)
    return {
        "protocol_id": protocol_id,
        "seeds": list(seeds),
        "weights": "equal",
        "learned_weights": False,
        "n_val_intersection": int(val_ids.shape[0]),
        "n_test_intersection": int(test_ids.shape[0]),
        "n_val_positives": int(val_y.sum()),
        "n_test_positives": int(test_y.sum()),
        "validation_selected_threshold": thr,
        "threshold_provenance": "max_f1_on_ensemble_validation_proba",
        "test_used_for_selection": False,
        "validation": {
            "threshold_0.5": metrics_block(val_y, val_p, 0.5),
            "threshold_val_selected": metrics_block(val_y, val_p, thr),
        },
        "test": {
            "threshold_0.5": metrics_block(test_y, test_p, 0.5),
            "threshold_val_selected": metrics_block(test_y, test_p, thr),
        },
    }


def _aml_ensemble(stack_name: str, seeds: Sequence[int]) -> Dict[str, Any]:
    val_maps, test_maps = [], []
    for s in seeds:
        vid, vy, vp = _load_proba(PROBA_ROOT / f"seed{s}_amlworld_{stack_name}_val.npz")
        tid, ty, tp = _load_proba(PROBA_ROOT / f"seed{s}_amlworld_{stack_name}_test.npz")
        val_maps.append({int(e): (int(yy), float(pp)) for e, yy, pp in zip(vid, vy, vp)})
        test_maps.append({int(e): (int(yy), float(pp)) for e, yy, pp in zip(tid, ty, tp)})

    def _intersect(maps):
        common = set(maps[0].keys())
        for m in maps[1:]:
            common &= set(m.keys())
        common = sorted(common)
        y = np.asarray([maps[0][e][0] for e in common], dtype=np.int64)
        for m in maps[1:]:
            for e, yy in zip(common, y):
                if m[e][0] != int(yy):
                    raise SystemExit(f"AML label mismatch edge={e} stack={stack_name}")
        proba = np.asarray([float(np.mean([m[e][1] for m in maps])) for e in common], dtype=np.float64)
        return np.asarray(common, dtype=np.int64), y, proba

    val_ids, val_y, val_p = _intersect(val_maps)
    test_ids, test_y, test_p = _intersect(test_maps)
    thr = tune_thr_max_f1(val_y, val_p)
    for sp, ids, y, p in (
        ("val", val_ids, val_y, val_p),
        ("test", test_ids, test_y, test_p),
    ):
        save_proba_npz(PROBA_ROOT / f"ensemble_amlworld_{stack_name}_{sp}.npz", ids, y, p)
    return {
        "stack": stack_name,
        "seeds": list(seeds),
        "weights": "equal",
        "learned_weights": False,
        "n_val_intersection": int(val_ids.shape[0]),
        "n_test_intersection": int(test_ids.shape[0]),
        "n_val_positives": int(val_y.sum()),
        "n_test_positives": int(test_y.sum()),
        "validation_selected_threshold": thr,
        "threshold_provenance": "max_f1_on_ensemble_validation_proba",
        "test_used_for_selection": False,
        "validation": {
            "threshold_0.5": metrics_block(val_y, val_p, 0.5),
            "threshold_val_selected": metrics_block(val_y, val_p, thr),
        },
        "test": {
            "threshold_0.5": metrics_block(test_y, test_p, 0.5),
            "threshold_val_selected": metrics_block(test_y, test_p, thr),
        },
    }


def _robustness_block(seed_list: Sequence[int], aml_cells, paysim_cells) -> Dict[str, Any]:
    out: Dict[str, Any] = {"seeds": list(seed_list), "amlworld": {}, "paysim": {}}
    for stack in ("pre3h_HxXTF", "pre3h_H_only", "post128_H_only"):
        cells = [aml_cells[s]["stacks"][stack] for s in seed_list]
        out["amlworld"][stack] = {
            "per_seed": {
                str(s): {
                    "test_threshold_0.5": aml_cells[s]["stacks"][stack]["test"]["threshold_0.5"],
                    "test_threshold_val_selected": aml_cells[s]["stacks"][stack]["test"]["threshold_val_selected"],
                    "validation_threshold_0.5": aml_cells[s]["stacks"][stack]["validation"]["threshold_0.5"],
                    "coverage_test": aml_cells[s]["stacks"][stack]["coverage"]["test"],
                }
                for s in seed_list
            },
            "aggregate_test_threshold_0.5": {
                m: _agg_metric_block(cells, "test", "threshold_0.5", m)
                for m in ("auroc", "auprc", "f1", "precision", "recall")
            },
            "aggregate_test_threshold_val_selected": {
                m: _agg_metric_block(cells, "test", "threshold_val_selected", m)
                for m in ("auroc", "auprc", "f1", "precision", "recall")
            },
        }
    for proto in PAYSIM_PROTOCOLS:
        pid = proto["id"]
        cells = [paysim_cells[s][pid] for s in seed_list]
        out["paysim"][pid] = {
            "per_seed": {
                str(s): {
                    "test_threshold_0.5": paysim_cells[s][pid]["test"]["threshold_0.5"],
                    "test_threshold_val_selected": paysim_cells[s][pid]["test"]["threshold_val_selected"],
                    "coverage_test": paysim_cells[s][pid]["coverage"]["test"],
                }
                for s in seed_list
            },
            "aggregate_test_threshold_0.5": {
                m: _agg_metric_block(cells, "test", "threshold_0.5", m)
                for m in ("auroc", "auprc", "f1", "precision", "recall")
            },
            "aggregate_test_threshold_val_selected": {
                m: _agg_metric_block(cells, "test", "threshold_val_selected", m)
                for m in ("auroc", "auprc", "f1", "precision", "recall")
            },
        }
    return out


def cmd_aggregate(_: argparse.Namespace) -> int:
    missing = []
    aml_cells = {}
    paysim_cells = {}
    for s in SEEDS_ALL:
        ap = CELLS / f"seed{s}_amlworld.json"
        if not ap.is_file():
            missing.append(str(ap))
        else:
            aml_cells[s] = json.loads(ap.read_text())
        for proto in PAYSIM_PROTOCOLS:
            pp = CELLS / f"seed{s}_{proto['id']}.json"
            if not pp.is_file():
                missing.append(str(pp))
            else:
                paysim_cells.setdefault(s, {})[proto["id"]] = json.loads(pp.read_text())
    ctrl_path = CELLS / "controls_summary.json"
    if not ctrl_path.is_file():
        missing.append(str(ctrl_path))
    if missing:
        raise SystemExit(f"refuse aggregate; missing: {missing}")

    controls = json.loads(ctrl_path.read_text())
    confirmation = _robustness_block(SEEDS_CONFIRMATION, aml_cells, paysim_cells)
    descriptive = _robustness_block(SEEDS_ALL, aml_cells, paysim_cells)
    confirmation["note"] = "Confirmation aggregate excludes development seed 2."
    descriptive["note"] = "Descriptive aggregate includes development seed 2; not an independent confirmation."

    ensembles = {
        "amlworld": {
            stack: _aml_ensemble(stack, SEEDS_ALL)
            for stack in ("pre3h_HxXTF", "pre3h_H_only", "post128_H_only")
        },
        "paysim": {proto["id"]: _ensemble_for_protocol(proto["id"], SEEDS_ALL) for proto in PAYSIM_PROTOCOLS},
        "secondary_only": True,
        "not_included_in_robustness_mean": True,
    }

    # Cautious external comparators (recorded numbers; no claim of superiority from n=4)
    comparators = {
        "published_multi_gin_eu_fixed_decision_f1": {
            "value": 64.79,
            "units": "percent_f1_mean",
            "source": "notes/multignn_supervised_parity_audit.md (Egressy et al.)",
            "note": "Supervised Multi-GIN+EU paper figure; not directly comparable to frozen SSL probe F1.",
        },
        "reproduced_supervised_multi_gin_eu": {
            "source": "notes/multignn_supervised_parity_audit.md / thesis registry",
            "note": "Compare cautiously; different objective and decision rule (paper_argmax).",
        },
        "previous_preserve_seed_frozen_dplus": {
            "source": "results/diagnostics/paysim_dplus_transfer_final.json",
            "note": "Preserve-seed D+ frozen transfer; different encoder recipe (preserve ON).",
        },
    }

    payload = {
        "title": TAG,
        "scope": "final_frozen_corrected_no_preserve_multiseed",
        "not_architecture_search": True,
        "encoder_frozen": True,
        "test_used_for_selection": False,
        "seeds": {
            "development": DEV_SEED,
            "confirmation": list(SEEDS_CONFIRMATION),
            "descriptive": list(SEEDS_ALL),
            "seed5_included": False,
        },
        "recipe": {
            "unique_template": UNIQUE_TMPL,
            "preserve_seed_edges": False,
            "correct_reverse_edge_features": True,
            "contrastive_asymmetric": True,
            "projection_dim": 128,
            "embedding_dim": 128,
            "negatives": 8192,
            "queue": 0,
            "accum": 4,
            "temperature": 0.5,
            "epochs": 40,
        },
        "checkpoints": {str(s): aml_cells[s]["checkpoint"] for s in SEEDS_ALL},
        "paysim_protocols_predeclared": PAYSIM_PROTOCOLS,
        "confirmation_aggregate": confirmation,
        "descriptive_aggregate": descriptive,
        "ensembles": ensembles,
        "controls": controls,
        "comparators": comparators,
        "claim_language": (
            "Frozen corrected/no-preserve Multi-GIN contrastive encoders (seeds 1–4) evaluated "
            "with locked AMLWorld PaperStyleMLP (pre-3h H+X+TF primary; post-128 H diagnostic) "
            "and locked PaySim logistic probes (P1 legacy zero-shot primary; P2 label-free BN "
            "adaptation; P3 type-only sensitivity). n=4 does not support claims of statistical "
            "superiority. Do not equate P2 with pure zero-shot. Do not make an unqualified GCPAL "
            "comparison."
        ),
        "recommendation_placeholder": (
            "Filled for thesis judgment after human review of aggregates; aggregate job does not "
            "auto-declare this the final primary encoder without the claim language above."
        ),
        "artifacts": {
            "final_json": str(FINAL_JSON),
            "final_md": str(FINAL_MD),
            "cells": str(CELLS),
            "probabilities": str(PROBA_ROOT),
            "embeddings": str(EMBED_ROOT),
        },
        "code_provenance": code_provenance(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(FINAL_JSON, payload)

    def fmt(a: Dict[str, Any]) -> str:
        return f"{a['mean']:.4f} ± {a['sample_std']:.4f} (med {a['median']:.4f}; n={a['n']})"

    def per_seed_row(block, metric="auprc"):
        parts = []
        for s, v in block["per_seed"].items():
            parts.append(f"s{s}={v['test_threshold_0.5'][metric]:.4f}")
        return ", ".join(parts)

    lines = [
        "# Final corrected/no-preserve multiseed evaluation",
        "",
        "> Final-results evaluation of frozen encoders. Not architecture search. Test never selected protocols.",
        "",
        f"- Development seed: **{DEV_SEED}**",
        f"- Confirmation seeds: **{list(SEEDS_CONFIRMATION)}**",
        f"- Descriptive aggregate: seeds **{list(SEEDS_ALL)}** (includes development seed)",
        "- Seed 5: not required / not included",
        "- Encoder frozen everywhere; no GNN train/finetune",
        "",
        "## 1. AMLWorld primary (pre-3h H+X+TF, PaperStyleMLP)",
        "",
        "### Confirmation (seeds 1,3,4)",
        f"- Test AUPRC@0.5: {fmt(confirmation['amlworld']['pre3h_HxXTF']['aggregate_test_threshold_0.5']['auprc'])}",
        f"- Test AUROC@0.5: {fmt(confirmation['amlworld']['pre3h_HxXTF']['aggregate_test_threshold_0.5']['auroc'])}",
        f"- Test F1@0.5: {fmt(confirmation['amlworld']['pre3h_HxXTF']['aggregate_test_threshold_0.5']['f1'])}",
        f"- Test F1@val-thr: {fmt(confirmation['amlworld']['pre3h_HxXTF']['aggregate_test_threshold_val_selected']['f1'])}",
        f"- Per-seed AUPRC: {per_seed_row(confirmation['amlworld']['pre3h_HxXTF'])}",
        "",
        "### Descriptive (seeds 1–4; includes development seed 2)",
        f"- Test AUPRC@0.5: {fmt(descriptive['amlworld']['pre3h_HxXTF']['aggregate_test_threshold_0.5']['auprc'])}",
        f"- Test AUROC@0.5: {fmt(descriptive['amlworld']['pre3h_HxXTF']['aggregate_test_threshold_0.5']['auroc'])}",
        f"- Test F1@0.5: {fmt(descriptive['amlworld']['pre3h_HxXTF']['aggregate_test_threshold_0.5']['f1'])}",
        f"- Test F1@val-thr: {fmt(descriptive['amlworld']['pre3h_HxXTF']['aggregate_test_threshold_val_selected']['f1'])}",
        f"- Per-seed AUPRC: {per_seed_row(descriptive['amlworld']['pre3h_HxXTF'])}",
        "",
        "## 2. AMLWorld post-128 H diagnostic (PaperStyleMLP; not a replacement for primary)",
        "",
        f"- Confirmation AUPRC@0.5: {fmt(confirmation['amlworld']['post128_H_only']['aggregate_test_threshold_0.5']['auprc'])}",
        f"- Descriptive AUPRC@0.5: {fmt(descriptive['amlworld']['post128_H_only']['aggregate_test_threshold_0.5']['auprc'])}",
        "",
        "## 3. PaySim P1 — strict inductive primary (`paysim_legacy_duplicate_v1`)",
        "",
        "> Legacy compatibility contract (type duplicated). Not claimed semantically ideal.",
        "",
        f"- Confirmation AUPRC@0.5: {fmt(confirmation['paysim']['P1_strict_inductive_legacy']['aggregate_test_threshold_0.5']['auprc'])}",
        f"- Descriptive AUPRC@0.5: {fmt(descriptive['paysim']['P1_strict_inductive_legacy']['aggregate_test_threshold_0.5']['auprc'])}",
        f"- Per-seed (desc) AUPRC: {per_seed_row(descriptive['paysim']['P1_strict_inductive_legacy'])}",
        "",
        "## 4. PaySim P2 — label-free target BN adaptation (legacy)",
        "",
        "> Not pure zero-shot; BN running stats adapted on PaySim train without labels.",
        "",
        f"- Confirmation AUPRC@0.5: {fmt(confirmation['paysim']['P2_label_free_target_bn_legacy']['aggregate_test_threshold_0.5']['auprc'])}",
        f"- Descriptive AUPRC@0.5: {fmt(descriptive['paysim']['P2_label_free_target_bn_legacy']['aggregate_test_threshold_0.5']['auprc'])}",
        "",
        "## 5. PaySim P3 — type-only sensitivity (`paysim_type_only_v1`)",
        "",
        "> Sensitivity only; must not replace P1 based on test metrics.",
        "",
        f"- Confirmation AUPRC@0.5: {fmt(confirmation['paysim']['P3_type_only_sensitivity']['aggregate_test_threshold_0.5']['auprc'])}",
        f"- Descriptive AUPRC@0.5: {fmt(descriptive['paysim']['P3_type_only_sensitivity']['aggregate_test_threshold_0.5']['auprc'])}",
        "",
        "## 6. Matched controls",
        "",
        f"- Random + legacy test AUPRC: {controls['controls']['random'][CONTRACT_LEGACY]['test_auprc']:.4f}",
        f"- Random + type_only test AUPRC: {controls['controls']['random'][CONTRACT_TYPE_ONLY]['test_auprc']:.4f}",
        f"- X-only + legacy test AUPRC: {controls['controls']['x_only'][CONTRACT_LEGACY]['test_auprc']:.4f}",
        f"- X-only + type_only test AUPRC: {controls['controls']['x_only'][CONTRACT_TYPE_ONLY]['test_auprc']:.4f}",
        "",
        "## 7. Confirmation vs descriptive aggregates",
        "",
        "See JSON sections `confirmation_aggregate` and `descriptive_aggregate`.",
        "",
        "## 8. Equal-weight ensembles (secondary; not in robustness mean)",
        "",
        f"- AML primary HxXTF test AUPRC@0.5: {ensembles['amlworld']['pre3h_HxXTF']['test']['threshold_0.5']['auprc']:.4f}",
        f"- AML primary HxXTF test F1@val-thr: {ensembles['amlworld']['pre3h_HxXTF']['test']['threshold_val_selected']['f1']:.4f}",
        f"- PaySim P1 test AUPRC@0.5: {ensembles['paysim']['P1_strict_inductive_legacy']['test']['threshold_0.5']['auprc']:.4f}",
        f"- PaySim P1 test F1@val-thr: {ensembles['paysim']['P1_strict_inductive_legacy']['test']['threshold_val_selected']['f1']:.4f}",
        "",
        "## 9. Limitations and thesis-safe claim language",
        "",
        payload["claim_language"],
        "",
        "- n=4 does not support statistical superiority claims.",
        "- Keep P1 (zero-shot / frozen BN) and P2 (label-free BN adaptation) clearly separated.",
        "- Do not make an unqualified GCPAL comparison.",
        "- Supervised Multi-GIN+EU and preserve-seed D+ are cautious external references only.",
        "",
        "## 10. Recommendation (final primary encoder?)",
        "",
        "Pending interactive judgment after human review of these aggregates. "
        "This automated note does not auto-promote the recipe beyond the locked corrected/no-preserve family.",
        "",
        "## Artifacts",
        "",
        f"- `{FINAL_MD}`",
        f"- `{FINAL_JSON}`",
        f"- cells: `{CELLS}/`",
        f"- probabilities: `{PROBA_ROOT}/`",
        f"- embeddings: `{EMBED_ROOT}/`",
        "",
    ]
    FINAL_MD.write_text("\n".join(lines) + "\n")

    # Append registry rows (do not rewrite historical rows)
    reg_path = ROOT / "results/diagnostics/thesis_experiment_registry.json"
    if reg_path.is_file():
        reg = json.loads(reg_path.read_text())
        rows = reg.get("rows", [])
        existing = {r.get("run_id") for r in rows if isinstance(r, dict)}
        new_rows = [
            {
                "run_id": f"{TAG}|amlworld|pre3h_HxXTF|descriptive_n4",
                "dataset": "Small-HI",
                "objective": "contrastive_frozen_eval",
                "encoder": "gin_emlps_ports_tds_corrected_nopreserve",
                "seed": "1-4",
                "thesis_role": "thesis_supporting",
                "test_auprc": descriptive["amlworld"]["pre3h_HxXTF"]["aggregate_test_threshold_0.5"]["auprc"]["mean"],
                "test_auroc": descriptive["amlworld"]["pre3h_HxXTF"]["aggregate_test_threshold_0.5"]["auroc"]["mean"],
                "source": str(FINAL_JSON),
                "preserve_seed_edges": False,
                "correct_reverse_edge_features": True,
            },
            {
                "run_id": f"{TAG}|paysim|P1_legacy_frozen_bn|descriptive_n4",
                "dataset": "PaySim",
                "objective": "contrastive_frozen_transfer",
                "encoder": "gin_emlps_ports_tds_corrected_nopreserve",
                "seed": "1-4",
                "thesis_role": "thesis_supporting",
                "feature_contract_id": CONTRACT_LEGACY,
                "test_auprc": descriptive["paysim"]["P1_strict_inductive_legacy"]["aggregate_test_threshold_0.5"]["auprc"]["mean"],
                "test_auroc": descriptive["paysim"]["P1_strict_inductive_legacy"]["aggregate_test_threshold_0.5"]["auroc"]["mean"],
                "source": str(FINAL_JSON),
                "preserve_seed_edges": False,
            },
        ]
        added = 0
        for nr in new_rows:
            if nr["run_id"] not in existing:
                rows.append(nr)
                added += 1
        reg["rows"] = rows
        reg["row_count"] = len(rows)
        write_json(reg_path, reg)
        notes_reg = ROOT / "notes/thesis_experiment_registry.md"
        if notes_reg.is_file() and added:
            with notes_reg.open("a") as f:
                f.write(
                    f"\n\n## {TAG}\n\n"
                    f"- See `{FINAL_MD}` / `{FINAL_JSON}`\n"
                    f"- Appended {added} registry row(s); historical rows unchanged.\n"
                )

    logging.info("Wrote %s and %s", FINAL_JSON, FINAL_MD)
    return 0


def main() -> None:
    logger_setup()
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("smoke")
    ps.add_argument("--device", default="cuda:0")
    ps.add_argument("--batch_size", type=int, default=4096)
    ps.set_defaults(func=cmd_smoke)

    for name, fn in (
        ("eval_amlworld", cmd_eval_amlworld),
        ("eval_paysim", cmd_eval_paysim),
    ):
        pe = sub.add_parser(name)
        pe.add_argument("--seed", type=int, required=True, choices=SEEDS_ALL)
        pe.add_argument("--device", default="cuda:0")
        pe.add_argument("--batch_size", type=int, default=4096)
        pe.set_defaults(func=fn)

    pc = sub.add_parser("eval_controls")
    pc.add_argument("--device", default="cuda:0")
    pc.add_argument("--batch_size", type=int, default=4096)
    pc.set_defaults(func=cmd_eval_controls)

    pa = sub.add_parser("aggregate")
    pa.set_defaults(func=cmd_aggregate)

    args = p.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
