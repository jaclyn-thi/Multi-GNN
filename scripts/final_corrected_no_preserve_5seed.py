#!/usr/bin/env python3
"""Final five-seed corrected-TDS / no-preserve replication (eval + aggregate).

Training is handled by slurm/train_corrected_nopreserve_40ep_seed.sh.
This module runs on compute nodes only for eval_seed / aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

TAG = "final_corrected_no_preserve_5seed"
EMBED_ROOT = ROOT / "embeddings" / TAG
RESULT_ROOT = ROOT / "results" / "diagnostics" / TAG
PROBA_ROOT = RESULT_ROOT / "probas"
CELLS = RESULT_ROOT / "cells"
FINAL_JSON = ROOT / "results" / "diagnostics" / f"{TAG}.json"
FINAL_MD = ROOT / "notes" / f"{TAG}.md"

UNIQUE_TMPL = "gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed{seed}"
SEED2_SHA = "18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6"
EXPECTED_EDGE_DIM = 8
FORWARD_EDGE = ("node", "to", "node")
REV_EDGE = ("node", "rev_to", "node")
DOWNSTREAM_LOGISTIC_SEED = 1
MLP_EPOCHS = 15
MLP_LR = 1e-3
MLP_BS = 8192
MLP_SEED = 2
TF_CACHE = ROOT / "results/cache/temporal_flow_causal/Small-HI"

PAYSIM_PROTOCOLS = (
    "inductive_trainfit_frozen_bn",
    "inductive_trainfit_target_train_bn",
    "transductive_pergraph_frozen_bn",
)

RANDOM_TRAINFIT_EMB = (
    ROOT / "embeddings/paysim_preserve_normalization_ablation/random_edge8_trainfit_post128"
)
RANDOM_PERGRAPH_EMB = (
    ROOT / "embeddings/paysim_preserve_normalization_ablation/random_edge8_pergraph_post128"
)
RANDOM_TRAINFIT_CELL = (
    ROOT
    / "results/diagnostics/paysim_preserve_normalization_ablation/cells"
    / "random_edge8_trainfit_post128_logistic_cw_model.json"
)
RANDOM_PERGRAPH_CELL = (
    ROOT
    / "results/diagnostics/paysim_preserve_normalization_ablation/cells"
    / "random_edge8_pergraph_post128_logistic_cw_model.json"
)
DPLUS_FINAL = ROOT / "results/diagnostics/paysim_dplus_transfer_final.json"
DPLUS_ROLE2 = ROOT / "results/diagnostics/paysim_dplus_transfer_final/role_seed2.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def base_argv(*, data: str, unique: str, emb_dir: str, emb_subdir: str, seed: int,
              train_fit: bool, representation_source: str = "post_embedding",
              batch_size: int = 4096) -> List[str]:
    argv = [
        "--data", data, "--model", "gin", "--testing", "--tqdm",
        "--unique_name", unique,
        "--embeddings_dir", emb_dir,
        "--embeddings_subdir", emb_subdir,
        "--batch_size", str(batch_size),
        "--loader_num_workers", "0",
        "--num_neighs", "100", "100",
        "--representation_source", representation_source,
        "--reverse_mp", "--ego", "--ports", "--tds", "--emlps",
        "--correct_reverse_edge_features",
        "--seed", str(seed),
    ]
    if train_fit:
        argv.append("--train_fit_edge_znorm")
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
    }


def save_proba_npz(path: Path, edge_ids: np.ndarray, y: np.ndarray, proba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        edge_id=np.asarray(edge_ids, dtype=np.int64),
        y=np.asarray(y, dtype=np.int64),
        proba=np.asarray(proba, dtype=np.float64),
    )


def run_logistic_and_save(
    emb_dir: Path,
    *,
    seed: int,
    protocol: str,
    ckpt: Path,
    ckpt_sha: str,
) -> Dict[str, Any]:
    splits = {}
    for sp in ("train", "val", "test"):
        z, y, ids = load_embedding_npz(emb_dir / f"{sp}.npz")
        splits[sp] = {"Z": z, "y": y, "ids": ids}

    out_cw: Dict[str, Any] = {}
    for cw_mode in ("model", "none"):
        cw: Any = gin_model_class_weight() if cw_mode == "model" else None
        set_seed(DOWNSTREAM_LOGISTIC_SEED)
        clf = LogisticRegression(
            class_weight=cw, max_iter=1000, random_state=DOWNSTREAM_LOGISTIC_SEED,
            solver="lbfgs", n_jobs=1, C=1.0,
        )
        clf.fit(splits["train"]["Z"], splits["train"]["y"])
        proba = {sp: clf.predict_proba(splits[sp]["Z"])[:, 1].astype(np.float64) for sp in splits}
        thr = tune_thr_max_f1(splits["val"]["y"], proba["val"])
        # save test proba aligned by edge_id
        proba_path = PROBA_ROOT / f"seed{seed}_{protocol}_cw_{cw_mode}_test.npz"
        save_proba_npz(proba_path, splits["test"]["ids"], splits["test"]["y"], proba["test"])
        rep = {
            "seed": seed,
            "protocol": protocol,
            "class_weight_mode": cw_mode,
            "C": 1.0,
            "learner": "LogisticRegression",
            "feature_stack": "H_only_post128",
            "checkpoint_path": str(ckpt),
            "checkpoint_sha256": ckpt_sha,
            "embeddings_dir": str(emb_dir),
            "proba_test_npz": str(proba_path),
            "ids": {sp: ids_hash(splits[sp]["ids"]) for sp in splits},
            "validation_selected_threshold": thr,
            "threshold_0.5": metrics_block(splits["test"]["y"], proba["test"], 0.5),
            "threshold_val_selected": metrics_block(splits["test"]["y"], proba["test"], thr),
            "val_ranking": {
                "auroc": float(roc_auc_score(splits["val"]["y"], proba["val"])),
                "auprc": float(average_precision_score(splits["val"]["y"], proba["val"])),
            },
        }
        out_path = CELLS / f"seed{seed}_{protocol}_logistic_cw_{cw_mode}.json"
        write_json(out_path, rep)
        out_cw[cw_mode] = {"path": str(out_path), "threshold_0.5": rep["threshold_0.5"], "proba": str(proba_path)}
        logging.info("Wrote %s auroc=%.4f", out_path, rep["threshold_0.5"]["auroc"])
    return out_cw


def extract_post128(
    *,
    seed: int,
    unique: str,
    subdir: str,
    train_fit: bool,
    bn_recal: bool,
    batch_size: int,
    device: torch.device,
) -> Tuple[Path, Optional[Dict[str, Any]]]:
    import embedding_extraction as ee

    argv = base_argv(
        data="PaySim", unique=unique, emb_dir=str(EMBED_ROOT), emb_subdir=subdir,
        seed=seed, train_fit=train_fit, batch_size=batch_size,
    )
    ns = parse_extract_args(argv)
    set_seed(ns.seed)
    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)

    if not bn_recal:
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns, data_config)
        if hetero_edge_dim(tr_data) != EXPECTED_EDGE_DIM:
            raise SystemExit(f"edge_dim={hetero_edge_dim(tr_data)}")
        out = ee.run_embedding_extraction(
            tr_data, val_data, te_data, tr_inds, val_inds, te_inds, ns, data_config
        )
        meta_path = out / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        meta.update({
            "train_fit_edge_znorm": train_fit,
            "bn_protocol": "frozen_aml_bn",
            "protocol_tag": subdir,
        })
        write_json(meta_path, meta)
        return out, None

    # BN recal path: load, recalibrate on train, extract all splits
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns, data_config)
    if hetero_edge_dim(tr_data) != EXPECTED_EDGE_DIM:
        raise SystemExit(f"edge_dim={hetero_edge_dim(tr_data)}")
    model, transform = build_hetero_model(ns, te_data, tr_data, val_data, tr_inds, val_inds, te_inds, device)
    load_checkpoint_weights(model, device, ns, data_config)
    tr_loader, val_loader, te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, ns, train_shuffle=False
    )
    bn_info = recalibrate_bn(model, tr_loader, device)
    model.eval()
    out_dir = EMBED_ROOT / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, loader, inds, graph in (
        ("train", tr_loader, tr_inds, tr_data),
        ("val", val_loader, val_inds, val_data),
        ("test", te_loader, te_inds, te_data),
    ):
        expected = expected_seed_edge_ids(loader.data, inds, hetero=True)
        edge_ids, z, y = extract_seed_embeddings_hetero(
            loader, inds, model, graph, device, ns,
            representation_source="post_embedding", pre_dim=None, emb_dim=128, head_spec=None,
        )
        log_seed_coverage(edge_ids, expected, split_name)
        save_embedding_split_npz(out_dir / f"{split_name}.npz", z, y, edge_ids)
    write_json(out_dir / "meta.json", {
        "bn_recalibration": bn_info,
        "train_fit_edge_znorm": train_fit,
        "bn_protocol": "target_train_only_running_stats",
        "source_unique_name": unique,
    })
    return out_dir, bn_info


def run_amlworld(seed: int, unique: str, ckpt: Path, ckpt_sha: str, batch_size: int, device: torch.device) -> Dict[str, Any]:
    import embedding_extraction as ee
    import importlib.util

    argv = base_argv(
        data="Small-HI", unique=unique, emb_dir=str(EMBED_ROOT),
        emb_subdir=f"seed{seed}_amlworld_pre3h", seed=seed, train_fit=False,
        representation_source="pre_embedding_3h", batch_size=max(batch_size, 8192),
    )
    ns = parse_extract_args(argv)
    set_seed(ns.seed)
    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns, data_config)
    out = ee.run_embedding_extraction(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, ns, data_config
    )
    emb_dir = out if (out / "train.npz").is_file() else EMBED_ROOT / f"seed{seed}_amlworld_pre3h" / "pre_embedding_3h"
    if not (emb_dir / "train.npz").is_file():
        raise SystemExit(f"missing AML embeds at {emb_dir}")

    spec = importlib.util.spec_from_file_location(
        "probe_feature_ablation", ROOT / "scripts" / "probe_feature_ablation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["probe_feature_ablation"] = mod
    spec.loader.exec_module(mod)
    df, df_train, tr_ids, va_ids, te_ids, dspec = mod.load_dataset_frames("Small-HI", str(ROOT / "data_config.json"))
    y_all = df[dspec.label_col].to_numpy().astype(np.int64)
    x_raw, _, _, _ = mod.build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    tf_feat = np.load(TF_CACHE / "features.npy").astype(np.float32)

    stacks = {}
    for stack_name in ("pre3h_H_only", "pre3h_HxXTF"):
        feats = {}
        for sp, expected_ids in (("train", tr_ids), ("val", va_ids), ("test", te_ids)):
            z, y, ids = load_embedding_npz(emb_dir / f"{sp}.npz")
            if not np.array_equal(y, y_all[ids]):
                raise SystemExit(f"AML label mismatch {sp}")
            if stack_name == "pre3h_H_only":
                mat = z.astype(np.float32)
            else:
                mat = np.concatenate([z, x_raw[ids], tf_feat[ids]], axis=1).astype(np.float32)
            feats[sp] = {"X": mat, "y": y, "ids": ids}
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(feats["train"]["X"]).astype(np.float32)
        x_va = scaler.transform(feats["val"]["X"]).astype(np.float32)
        x_te = scaler.transform(feats["test"]["X"]).astype(np.float32)

        torch.manual_seed(MLP_SEED)
        np.random.seed(MLP_SEED)
        model = PaperStyleMLP(int(x_tr.shape[1])).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
        x_t = torch.from_numpy(x_tr.astype(np.float32))
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
        pva = _predict_proba(model, x_va, batch_size=MLP_BS, device=device)
        pte = _predict_proba(model, x_te, batch_size=MLP_BS, device=device)
        thr = tune_thr_max_f1(feats["val"]["y"], pva)
        proba_path = PROBA_ROOT / f"seed{seed}_amlworld_{stack_name}_test.npz"
        save_proba_npz(proba_path, feats["test"]["ids"], feats["test"]["y"], pte)
        stacks[stack_name] = {
            "best_epoch_by_val_auprc": best_ep,
            "best_val_auprc": best_auprc,
            "validation_selected_threshold": thr,
            "threshold_0.5": metrics_block(feats["test"]["y"], pte, 0.5),
            "threshold_val_selected": metrics_block(feats["test"]["y"], pte, thr),
            "val_ranking": {
                "auroc": float(roc_auc_score(feats["val"]["y"], pva)),
                "auprc": float(average_precision_score(feats["val"]["y"], pva)),
            },
            "proba_test_npz": str(proba_path),
            "feature_dim": int(x_tr.shape[1]),
            "ids": {sp: ids_hash(feats[sp]["ids"]) for sp in feats},
            "learner": "PaperStyleMLP",
        }

    report = {
        "seed": seed,
        "unique_name": unique,
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": ckpt_sha,
        "dataset": "Small-HI",
        "stacks": stacks,
        "preserve_seed_edges": False,
        "correct_reverse_edge_features": True,
    }
    write_json(CELLS / f"seed{seed}_amlworld_mlp.json", report)
    return report


def cmd_eval_seed(args: argparse.Namespace) -> int:
    seed = int(args.seed)
    unique = unique_name(seed)
    ckpt = ckpt_path(seed)
    if not ckpt.is_file():
        raise SystemExit(f"missing checkpoint {ckpt}")
    ckpt_sha = sha256_file(ckpt)
    if seed == 2 and ckpt_sha != SEED2_SHA:
        raise SystemExit(f"seed2 hash mismatch: {ckpt_sha}")
    payload = torch.load(ckpt, map_location="cpu")
    epoch = payload.get("epoch")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    CELLS.mkdir(parents=True, exist_ok=True)
    PROBA_ROOT.mkdir(parents=True, exist_ok=True)
    EMBED_ROOT.mkdir(parents=True, exist_ok=True)

    logging.info("=== AMLWorld eval seed=%s ===", seed)
    aml = run_amlworld(seed, unique, ckpt, ckpt_sha, int(args.batch_size), device)

    paysim = {}
    # 1) inductive trainfit frozen BN
    logging.info("=== PaySim inductive_trainfit_frozen_bn seed=%s ===", seed)
    emb1, _ = extract_post128(
        seed=seed, unique=unique, subdir=f"seed{seed}_inductive_trainfit_frozen_bn",
        train_fit=True, bn_recal=False, batch_size=int(args.batch_size), device=device,
    )
    paysim["inductive_trainfit_frozen_bn"] = run_logistic_and_save(
        emb1, seed=seed, protocol="inductive_trainfit_frozen_bn", ckpt=ckpt, ckpt_sha=ckpt_sha
    )

    # 2) inductive trainfit + target BN recal
    logging.info("=== PaySim inductive_trainfit_target_train_bn seed=%s ===", seed)
    emb2, bn_info = extract_post128(
        seed=seed, unique=unique, subdir=f"seed{seed}_inductive_trainfit_target_train_bn",
        train_fit=True, bn_recal=True, batch_size=int(args.batch_size), device=device,
    )
    paysim["inductive_trainfit_target_train_bn"] = run_logistic_and_save(
        emb2, seed=seed, protocol="inductive_trainfit_target_train_bn", ckpt=ckpt, ckpt_sha=ckpt_sha
    )
    paysim["inductive_trainfit_target_train_bn"]["bn_recalibration"] = bn_info

    # 3) transductive per-graph frozen BN
    logging.info("=== PaySim transductive_pergraph_frozen_bn seed=%s ===", seed)
    emb3, _ = extract_post128(
        seed=seed, unique=unique, subdir=f"seed{seed}_transductive_pergraph_frozen_bn",
        train_fit=False, bn_recal=False, batch_size=int(args.batch_size), device=device,
    )
    paysim["transductive_pergraph_frozen_bn"] = run_logistic_and_save(
        emb3, seed=seed, protocol="transductive_pergraph_frozen_bn", ckpt=ckpt, ckpt_sha=ckpt_sha
    )

    summary = {
        "seed": seed,
        "unique_name": unique,
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_epoch": epoch,
        "preserve_seed_edges": False,
        "correct_reverse_edge_features": True,
        "amlworld": {
            "pre3h_H_only": aml["stacks"]["pre3h_H_only"]["threshold_0.5"],
            "pre3h_HxXTF": aml["stacks"]["pre3h_HxXTF"]["threshold_0.5"],
        },
        "paysim_primary_cw_model_auroc": {
            p: paysim[p]["model"]["threshold_0.5"]["auroc"] for p in PAYSIM_PROTOCOLS
        },
        "paysim": paysim,
        "job_id": __import__("os").environ.get("SLURM_JOB_ID"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "no_test_selection": True,
    }
    write_json(CELLS / f"seed{seed}_summary.json", summary)
    logging.info("Wrote seed %s summary", seed)
    return 0


def _mean_sd_med(vals: List[float]) -> Dict[str, float]:
    a = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "sample_std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "n": int(a.size),
        "values": [float(x) for x in a],
    }


def _load_proba(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    return d["edge_id"].astype(np.int64), d["y"].astype(np.int64), d["proba"].astype(np.float64)


def cmd_aggregate(_: argparse.Namespace) -> int:
    seeds = [1, 2, 3, 4, 5]
    summaries = {}
    missing = []
    for s in seeds:
        p = CELLS / f"seed{s}_summary.json"
        if not p.is_file():
            missing.append(str(p))
        else:
            summaries[s] = json.loads(p.read_text())
    if missing:
        raise SystemExit(f"refuse aggregate; missing: {missing}")

    # Verify random controls
    random_reuse = {}
    for label, emb, cellp in (
        ("trainfit", RANDOM_TRAINFIT_EMB, RANDOM_TRAINFIT_CELL),
        ("pergraph", RANDOM_PERGRAPH_EMB, RANDOM_PERGRAPH_CELL),
    ):
        if not emb.is_dir() or not cellp.is_file():
            raise SystemExit(f"missing random control {label}")
        cell = json.loads(cellp.read_text())
        random_reuse[label] = {
            "embeddings": str(emb),
            "cell": str(cellp),
            "auroc": cell["threshold_0.5"]["auroc"],
            "auprc": cell["threshold_0.5"]["auprc"],
            "verified": True,
        }

    per_seed = {}
    for s, summ in summaries.items():
        per_seed[str(s)] = {
            "checkpoint_sha256": summ["checkpoint_sha256"],
            "checkpoint_epoch": summ.get("checkpoint_epoch"),
            "amlworld": summ["amlworld"],
            "paysim_cw_model": {
                p: json.loads((CELLS / f"seed{s}_{p}_logistic_cw_model.json").read_text())["threshold_0.5"]
                for p in PAYSIM_PROTOCOLS
            },
        }

    aggregates = {}
    for protocol in PAYSIM_PROTOCOLS:
        aurocs, auprcs = [], []
        for s in seeds:
            m = per_seed[str(s)]["paysim_cw_model"][protocol]
            aurocs.append(m["auroc"])
            auprcs.append(m["auprc"])
        aggregates[protocol] = {
            "auroc": _mean_sd_med(aurocs),
            "auprc": _mean_sd_med(auprcs),
        }

    aml_h_auroc = _mean_sd_med([per_seed[str(s)]["amlworld"]["pre3h_H_only"]["auroc"] for s in seeds])
    aml_hx_auroc = _mean_sd_med([per_seed[str(s)]["amlworld"]["pre3h_HxXTF"]["auroc"] for s in seeds])
    aml_h_auprc = _mean_sd_med([per_seed[str(s)]["amlworld"]["pre3h_H_only"]["auprc"] for s in seeds])
    aml_hx_auprc = _mean_sd_med([per_seed[str(s)]["amlworld"]["pre3h_HxXTF"]["auprc"] for s in seeds])

    # Equal-weight ensemble by edge_id intersection for primary inductive_trainfit_frozen_bn
    ensemble = {}
    for protocol in PAYSIM_PROTOCOLS:
        maps = []
        for s in seeds:
            path = PROBA_ROOT / f"seed{s}_{protocol}_cw_model_test.npz"
            eids, y, proba = _load_proba(path)
            maps.append({int(e): (int(yy), float(pp)) for e, yy, pp in zip(eids, y, proba)})
        common = set(maps[0].keys())
        for m in maps[1:]:
            common &= set(m.keys())
        common = sorted(common)
        if not common:
            raise SystemExit(f"empty intersection for {protocol}")
        y_common = np.asarray([maps[0][e][0] for e in common], dtype=np.int64)
        # verify labels agree
        for m in maps[1:]:
            for e, yy in zip(common, y_common):
                if m[e][0] != int(yy):
                    raise SystemExit(f"label mismatch on edge {e} protocol {protocol}")
        proba_ens = np.asarray(
            [float(np.mean([m[e][1] for m in maps])) for e in common], dtype=np.float64
        )
        # threshold: need val — use mean of per-seed val thresholds from cells as diagnostic only;
        # for ensemble report thr0.5 and a val-free ranking; also tune thr on... we don't have val proba ensemble.
        # Report thr0.5 + ranking metrics; for val-selected, skip or use 0.5 only with note.
        thr05 = metrics_block(y_common, proba_ens, 0.5)
        ens_path = PROBA_ROOT / f"ensemble_{protocol}_cw_model_test.npz"
        save_proba_npz(ens_path, np.asarray(common, dtype=np.int64), y_common, proba_ens)
        ensemble[protocol] = {
            "n_intersection": len(common),
            "n_positives_retained": int(y_common.sum()),
            "positive_rate": float(y_common.mean()),
            "threshold_0.5": thr05,
            "proba_npz": str(ens_path),
            "note": "equal_weight mean proba; threshold_val_selected not applied (no shared val fit)",
        }

    dplus_cmp = {}
    if DPLUS_FINAL.is_file():
        dplus = json.loads(DPLUS_FINAL.read_text())
        dplus_cmp["primary_pre3h_HxX"] = dplus.get("primary_pre3h_HxX")
    if DPLUS_ROLE2.is_file():
        r2 = json.loads(DPLUS_ROLE2.read_text())
        dplus_cmp["seed2_post128_H_only"] = r2.get("stacks", {}).get("post128_H_only", {}).get("threshold_0.5")

    payload = {
        "title": TAG,
        "primary_result": "five individual frozen corrected/no-preserve encoders",
        "secondary_result": "fixed equal-weight five-encoder ensemble",
        "recipe": {
            "unique_template": UNIQUE_TMPL,
            "preserve_seed_edges": False,
            "correct_reverse_edge_features": True,
            "contrastive_asymmetric": True,
            "negatives": 8192,
            "queue": 0,
            "accum": 4,
            "temperature": 0.5,
            "epochs": 40,
            "matched_seed2": UNIQUE_TMPL.format(seed=2),
            "seed2_sha256": SEED2_SHA,
        },
        "per_seed": per_seed,
        "aggregates_paysim_cw_model": aggregates,
        "aggregates_amlworld": {
            "pre3h_H_only_auroc": aml_h_auroc,
            "pre3h_H_only_auprc": aml_h_auprc,
            "pre3h_HxXTF_auroc": aml_hx_auroc,
            "pre3h_HxXTF_auprc": aml_hx_auprc,
        },
        "ensemble": ensemble,
        "random_controls_reused": random_reuse,
        "dplus_preserve_comparison": dplus_cmp,
        "tradeoff_table_seed2": {
            "aml_HxXTF_auroc": per_seed["2"]["amlworld"]["pre3h_HxXTF"]["auroc"],
            "paysim_inductive_trainfit_frozen_bn_auroc": per_seed["2"]["paysim_cw_model"]["inductive_trainfit_frozen_bn"]["auroc"],
            "paysim_transductive_pergraph_auroc": per_seed["2"]["paysim_cw_model"]["transductive_pergraph_frozen_bn"]["auroc"],
        },
        "no_encoder_training_in_eval": True,
        "test_used_for_selection": False,
        "historical_dplus_overwritten": False,
    }
    write_json(FINAL_JSON, payload)

    def fmt_agg(a):
        return f"{a['mean']:.4f} ± {a['sample_std']:.4f} (med {a['median']:.4f})"

    lines = [
        "# Final corrected-TDS / no-preserve five-seed replication",
        "",
        "Primary: five individual frozen encoders (seeds 1–5). "
        "Secondary: fixed equal-weight probability ensemble.",
        "",
        "## Recipe",
        "- Exact match to `gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2`",
        "- preserve_seed_edges **OFF**; corrected reverse; asym proj; 8192 neg; queue 0; accum 4; T=0.5; 40ep",
        "",
        "## PaySim primary (post-128 H-only logistic, cw=model) — inductive_trainfit_frozen_bn",
        f"- AUROC {fmt_agg(aggregates['inductive_trainfit_frozen_bn']['auroc'])}",
        f"- AUPRC {fmt_agg(aggregates['inductive_trainfit_frozen_bn']['auprc'])}",
        "",
        "## PaySim protocols (AUROC mean±sd)",
    ]
    for p in PAYSIM_PROTOCOLS:
        lines.append(f"- **{p}**: {fmt_agg(aggregates[p]['auroc'])}")
    lines += [
        "",
        "## Ensemble (equal-weight, edge-ID intersection)",
        f"- inductive_trainfit_frozen_bn: n={ensemble['inductive_trainfit_frozen_bn']['n_intersection']} "
        f"pos={ensemble['inductive_trainfit_frozen_bn']['n_positives_retained']} "
        f"AUROC={ensemble['inductive_trainfit_frozen_bn']['threshold_0.5']['auroc']:.4f}",
        "",
        "## AMLWorld",
        f"- pre3h H AUROC {fmt_agg(aml_h_auroc)} AUPRC {fmt_agg(aml_h_auprc)}",
        f"- pre3h HxXTF AUROC {fmt_agg(aml_hx_auroc)} AUPRC {fmt_agg(aml_hx_auprc)}",
        "",
        "## Random controls (reused)",
        f"- trainfit AUROC {random_reuse['trainfit']['auroc']:.4f}",
        f"- pergraph AUROC {random_reuse['pergraph']['auroc']:.4f}",
        "",
        "## Artifacts",
        f"- `{FINAL_MD}`",
        f"- `{FINAL_JSON}`",
        f"- cells: `{CELLS}/`",
        f"- probas: `{PROBA_ROOT}/`",
        "",
    ]
    for s in seeds:
        ps = per_seed[str(s)]["paysim_cw_model"]["inductive_trainfit_frozen_bn"]
        lines.append(
            f"- seed{s} inductive_trainfit_frozen_bn: AUROC {ps['auroc']:.4f} AUPRC {ps['auprc']:.4f}"
        )
    FINAL_MD.write_text("\n".join(lines) + "\n")
    logging.info("Wrote %s and %s", FINAL_JSON, FINAL_MD)
    return 0


def main() -> None:
    logger_setup()
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("eval_seed")
    pe.add_argument("--seed", type=int, required=True, choices=(1, 2, 3, 4, 5))
    pe.add_argument("--device", default="cuda:0")
    pe.add_argument("--batch_size", type=int, default=4096)
    pe.set_defaults(func=cmd_eval_seed)
    pa = sub.add_parser("aggregate")
    pa.set_defaults(func=cmd_aggregate)
    args = p.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
