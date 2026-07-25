"""AMLWorld→PaySim frozen D+ transfer (locked protocol).

Never updates the encoder with PaySim labels. Never writes under legacy
``embeddings/paysim/hi_contrastive_*`` paths. No TF evaluation.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import logging
import math
import os
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import to_hetero

from data_loading import get_data
from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba, _select_threshold_f1
from ranking_metrics import alert_budget_metrics
from train_util import (
    AddEgoIds,
    add_arange_ids,
    expected_seed_edge_ids,
    extract_param,
    extract_seed_embeddings_hetero_dual,
    get_loaders,
    log_seed_coverage,
    resolve_embedding_head_linear,
    save_embedding_split_npz,
)
from training import get_model
from util import create_parser, set_seed

ROOT = Path(__file__).resolve().parent

_pfa_spec = importlib.util.spec_from_file_location(
    "probe_feature_ablation", ROOT / "scripts" / "probe_feature_ablation.py"
)
_pfa = importlib.util.module_from_spec(_pfa_spec)
assert _pfa_spec.loader is not None
sys.modules["probe_feature_ablation"] = _pfa
_pfa_spec.loader.exec_module(_pfa)
build_full_feature_matrix = _pfa.build_full_feature_matrix
load_dataset_frames = _pfa.load_dataset_frames

# ---------------------------------------------------------------------------
# Locked checkpoints
# ---------------------------------------------------------------------------

LOCKED_CHECKPOINTS: Dict[str, Dict[str, Any]] = {
    "seed1": {
        "role": "primary",
        "encoder_seed": 1,
        "train_job": 18801429,
        "epoch": 34,
        "path": ROOT
        / "saved-models"
        / "checkpoint_edge_dplus_corrected_preserve_40ep_seed1_final.tar",
        "load_key": "model_state_dict",
        "run_tag": "dplus_seed1",
    },
    "seed2": {
        "role": "primary",
        "encoder_seed": 2,
        "train_job": 18514684,
        "epoch": 40,
        "path": ROOT
        / "saved-models"
        / "checkpoint_gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2.tar",
        "load_key": "model_state_dict",
        "run_tag": "dplus_seed2",
    },
    "seed3": {
        "role": "primary",
        "encoder_seed": 3,
        "train_job": 18802579,
        "epoch": 29,
        "path": ROOT
        / "saved-models"
        / "checkpoint_edge_dplus_corrected_preserve_40ep_seed3_final.tar",
        "load_key": "model_state_dict",
        "run_tag": "dplus_seed3",
    },
    "ft_seed2": {
        "role": "secondary_ft",
        "encoder_seed": 2,
        "train_job": 18801435,
        "epoch": 18,
        "path": ROOT
        / "saved-models"
        / "dplus_partial_finetune_hxxtf_seed2"
        / "checkpoint_best_val_auprc.tar",
        "load_key": "encoder_state_dict",
        "run_tag": "ft_encoder_seed2",
        "refuse_keys": ("classifier_state_dict",),
    },
    "random_init": {
        "role": "control_random",
        "encoder_seed": 2,
        "train_job": None,
        "epoch": None,
        "path": None,
        "load_key": None,
        "run_tag": "random_init_dplus",
        "random_seed": 2,
    },
}

EMBEDDINGS_ROOT = ROOT / "embeddings" / "paysim_dplus_transfer_final"
RESULTS_DIR = ROOT / "results" / "diagnostics" / "paysim_dplus_transfer_final"
LEGACY_PAYSIM_EMBED_DIR = ROOT / "embeddings" / "paysim"

PRE3H_DIM = 198
POST128_DIM = 128
EXPECTED_EDGE_DIM = 8
MLP_EPOCHS = 15
MLP_LR = 1e-3
MLP_BS = 8192
DOWNSTREAM_SEED = 2
DEFAULT_EXTRACT_BS = 4096
DEFAULT_SMOKE_MAX_BATCHES = 3

STACKS = (
    "X_only",
    "pre3h_H_only",
    "pre3h_HxX",
    "post128_H_only",
    "post128_HxX",
)

# Full-job wall budget (Advanced GPU partition MaxTime).
GPU_WALL_LIMIT_SEC = 6 * 3600
# Conservative margin: max(20 minutes, 25% of projected core runtime).
RUNTIME_MARGIN_FLOOR_SEC = 20 * 60
RUNTIME_MARGIN_FRAC = 0.25


def project_full_runtime(
    *,
    one_time_setup_sec: float,
    extract_sec_per_batch: float,
    smoke_n_batches: int,
    full_n_batches: int,
    downstream_probe_sec: float,
    projected_downstream_sec: Optional[float] = None,
    wall_limit_sec: float = GPU_WALL_LIMIT_SEC,
    margin_floor_sec: float = RUNTIME_MARGIN_FLOOR_SEC,
    margin_frac: float = RUNTIME_MARGIN_FRAC,
) -> Dict[str, Any]:
    """Project full-job runtime with one-time setup amortized once.

    Dual capture runs pre-3h and post-128 in a single forward, so the extract
    cost is charged once as ``projected_dual_extract_sec``. For the requested
    formula breakdown:

    - ``projected_pre3h_extract`` = dual extract cost (captures both)
    - ``projected_post128_extract`` = 0 (same forward; not a second pass)

    Formula::

        total = one_time_setup
              + projected_pre3h_extract
              + projected_post128_extract
              + projected_downstream_eval
              + margin

        margin = max(margin_floor_sec, margin_frac * core)

    where ``core`` is the sum of the four non-margin terms.
    """
    if smoke_n_batches <= 0:
        raise ValueError("smoke_n_batches must be positive")
    if extract_sec_per_batch < 0:
        raise ValueError("extract_sec_per_batch must be non-negative")

    projected_dual = float(extract_sec_per_batch) * float(full_n_batches)
    projected_pre3h = projected_dual
    projected_post128 = 0.0  # dual forward; documented explicitly
    if projected_downstream_sec is None:
        # Scale measured probe by epoch ratio if only 1 smoke epoch ran (caller sets).
        projected_downstream = float(downstream_probe_sec)
    else:
        projected_downstream = float(projected_downstream_sec)

    one_time = float(one_time_setup_sec)
    core = one_time + projected_pre3h + projected_post128 + projected_downstream
    margin = max(float(margin_floor_sec), float(margin_frac) * core)
    total = core + margin
    return {
        "formula": (
            "total = one_time_setup + projected_pre3h_extract "
            "+ projected_post128_extract + projected_downstream_eval + margin; "
            "margin = max(margin_floor_sec, margin_frac * core); "
            "dual_capture: post128 extract charged as 0 (same forward as pre3h)"
        ),
        "inputs": {
            "one_time_setup_sec": one_time,
            "extract_sec_per_batch": float(extract_sec_per_batch),
            "smoke_n_batches": int(smoke_n_batches),
            "full_n_batches": int(full_n_batches),
            "downstream_probe_sec_measured": float(downstream_probe_sec),
            "projected_downstream_sec": projected_downstream,
            "wall_limit_sec": float(wall_limit_sec),
            "margin_floor_sec": float(margin_floor_sec),
            "margin_frac": float(margin_frac),
        },
        "one_time_setup_sec": one_time,
        "projected_pre3h_extract_sec": projected_pre3h,
        "projected_post128_extract_sec": projected_post128,
        "projected_dual_extract_sec": projected_dual,
        "projected_downstream_eval_sec": projected_downstream,
        "core_sec": core,
        "margin_sec": margin,
        "est_total_sec": total,
        "est_total_hours": total / 3600.0,
        "fits_6h_gpu": bool(total < float(wall_limit_sec) * 0.95),
        "note": (
            "One-time setup (data load, ports, TDS, model construct, checkpoint load) "
            "is amortized once and never multiplied by batch count. "
            "Extract projection uses measured sec/batch from dual forward only."
        ),
    }



def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ids_hash(ids: np.ndarray) -> Dict[str, Any]:
    ids = np.asarray(ids).astype(np.int64)
    return {
        "n": int(ids.shape[0]),
        "n_unique": int(np.unique(ids).shape[0]),
        "n_duplicate_rows": int(ids.shape[0] - np.unique(ids).shape[0]),
        "edge_id_sum": int(ids.sum()) if ids.size else 0,
        "edge_id_first": int(ids[0]) if ids.size else None,
        "edge_id_last": int(ids[-1]) if ids.size else None,
        "sha256_of_ids_bytes": hashlib.sha256(ids.tobytes()).hexdigest() if ids.size else None,
    }


def assert_not_legacy_paysim_path(path: Path) -> None:
    """Refuse writes under historical June 2026 PaySim embedding trees."""
    resolved = path.resolve()
    legacy = LEGACY_PAYSIM_EMBED_DIR.resolve()
    try:
        resolved.relative_to(legacy)
    except ValueError:
        return
    raise RuntimeError(
        f"Refusing to write under legacy PaySim embeddings path {resolved} "
        f"(historical hi_contrastive_* / random_init_gin artifacts must be preserved). "
        f"Use {EMBEDDINGS_ROOT}/ instead."
    )


def build_paysim_dplus_args(
    *,
    batch_size: int = DEFAULT_EXTRACT_BS,
    loader_num_workers: int = 0,
    seed: int = 2,
    unique_name: str = "paysim_dplus_transfer",
    tqdm: bool = True,
) -> Any:
    """Locked PaySim extract args matching D+ edge_dim=8 geometry."""
    argv = [
        "--data",
        "PaySim",
        "--model",
        "gin",
        "--batch_size",
        str(int(batch_size)),
        "--num_neighs",
        "100",
        "100",
        "--loader_num_workers",
        str(int(loader_num_workers)),
        "--seed",
        str(int(seed)),
        "--reverse_mp",
        "--ego",
        "--ports",
        "--emlps",
        "--tds",
        "--correct_reverse_edge_features",
        "--train_fit_edge_znorm",
        "--testing",
        "--unique_name",
        unique_name,
    ]
    if tqdm:
        argv.append("--tqdm")
    args = create_parser().parse_args(argv)
    args.checkpoint_suffix = ""
    args.finetune = False
    args.include_temporal_flow_edge_features = False
    args.embedding_dim = 128
    args.preserve_seed_edges = False  # extract-time: no contrastive seed preservation
    # Explicitly disable augmentation levers if present on namespace
    if hasattr(args, "edge_drop_target_rate"):
        args.edge_drop_target_rate = 0.0
    if hasattr(args, "edge_attr_mask_rate"):
        args.edge_attr_mask_rate = 0.0
    return args


def hash_model_state(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(arr.tobytes())
    return h.hexdigest()


def freeze_encoder(model: nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            m.track_running_stats = True


def assert_bn_eval(model: nn.Module) -> None:
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            if m.training:
                raise RuntimeError(f"BatchNorm still in train mode: {name}")


def load_encoder_state_dict_strict(
    model: nn.Module,
    ckpt_path: Path,
    *,
    load_key: str,
    refuse_keys: Sequence[str] = (),
) -> Dict[str, Any]:
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    payload = torch.load(ckpt_path, map_location="cpu")
    for bad in refuse_keys:
        if bad in payload:
            logging.info("Ignoring transfer-forbidden key %s from %s", bad, ckpt_path.name)
    if load_key not in payload:
        raise KeyError(f"{ckpt_path} missing required key {load_key!r}; keys={list(payload.keys())}")
    if load_key == "encoder_state_dict" and "classifier_state_dict" in payload:
        # Explicit policy: never touch classifier weights
        logging.info(
            "FT checkpoint contains classifier_state_dict (%d tensors) — not loaded",
            len(payload["classifier_state_dict"]),
        )
    sd = payload[load_key]
    # Shape gate for edge_emb
    edge_keys = [k for k in sd if "edge_emb" in k and k.endswith("weight")]
    for k in edge_keys:
        shape = tuple(sd[k].shape)
        if shape[-1] != EXPECTED_EDGE_DIM:
            raise ValueError(
                f"edge_dim mismatch on {k}: got in_features={shape[-1]}, expected {EXPECTED_EDGE_DIM}"
            )
    incompatible = model.load_state_dict(sd, strict=True)
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise RuntimeError(f"strict load failed: missing={missing} unexpected={unexpected}")
    return {
        "path": str(ckpt_path),
        "sha256": file_sha256(ckpt_path),
        "load_key": load_key,
        "epoch": payload.get("epoch") or payload.get("global_epoch") or payload.get("early_stop", {}).get("best_epoch"),
        "n_tensors": len(sd),
        "edge_emb_keys": {k: list(sd[k].shape) for k in edge_keys},
        "classifier_loaded": False,
    }


def _build_model_config(args) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        n_hidden=extract_param("n_hidden", args),
        n_gnn_layers=extract_param("n_gnn_layers", args),
        n_heads=None,
        dropout=extract_param("dropout", args),
        final_dropout=extract_param("final_dropout", args),
        lr=extract_param("lr", args),
    )


def construct_paysim_encoder(
    tr_data,
    val_data,
    te_data,
    tr_inds,
    val_inds,
    te_inds,
    args,
    device: torch.device,
) -> Tuple[nn.Module, Any, int, int]:
    transform = AddEgoIds() if args.ego else None
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
    config = _build_model_config(args)
    model = get_model(sample_batch, config, args)
    emb_dim = int(getattr(model, "embedding_dim", 128))
    head_spec = resolve_embedding_head_linear(model, emb_dim)
    pre_dim = int(head_spec.in_features)
    model = to_hetero(model, te_data.metadata(), aggr="mean")
    model.to(device)
    # Verify constructed edge_dim
    for name, p in model.named_parameters():
        if "edge_emb" in name and name.endswith("weight"):
            if int(p.shape[-1]) != EXPECTED_EDGE_DIM:
                raise RuntimeError(
                    f"Constructed model {name} has in_features={p.shape[-1]}, "
                    f"expected {EXPECTED_EDGE_DIM} (ports+tds)"
                )
    return model, head_spec, pre_dim, emb_dim


def dual_extract_splits(
    model,
    head_spec,
    pre_dim: int,
    emb_dim: int,
    tr_data,
    val_data,
    te_data,
    tr_inds,
    val_inds,
    te_inds,
    args,
    device: torch.device,
    out_root: Path,
    *,
    max_batches: Optional[int] = None,
    splits: Sequence[str] = ("train", "val", "test"),
) -> Dict[str, Any]:
    assert_not_legacy_paysim_path(out_root)
    pre_dir = out_root / "pre_embedding_3h"
    post_dir = out_root / "post_embedding_128"
    pre_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)

    transform = AddEgoIds() if args.ego else None
    tr_loader, val_loader, te_loader = get_loaders(
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
    split_map = {
        "train": (tr_loader, tr_inds, tr_data),
        "val": (val_loader, val_inds, val_data),
        "test": (te_loader, te_inds, te_data),
    }
    report: Dict[str, Any] = {"splits": {}, "pre_dir": str(pre_dir), "post_dir": str(post_dir)}
    freeze_encoder(model)
    assert_bn_eval(model)
    hash_before = hash_model_state(model)

    total_batches = 0
    t_extract_all = time.perf_counter()
    for split_name in splits:
        loader, split_inds, graph_data = split_map[split_name]
        expected = expected_seed_edge_ids(loader.data, split_inds, hetero=True)
        t_split = time.perf_counter()
        with torch.inference_mode():
            edge_ids, z_pre, z_post, y = extract_seed_embeddings_hetero_dual(
                loader,
                split_inds,
                model,
                graph_data,
                device,
                args,
                pre_dim=pre_dim,
                emb_dim=emb_dim,
                head_spec=head_spec,
                max_batches=max_batches,
            )
        split_sec = float(time.perf_counter() - t_split)
        # Infer batch count from rows / batch_size when max_batches caps the pass.
        n_rows = int(edge_ids.shape[0])
        bs = max(1, int(getattr(args, "batch_size", DEFAULT_EXTRACT_BS)))
        if max_batches is not None:
            n_batches_split = int(max_batches)
        else:
            n_batches_split = max(1, int(math.ceil(n_rows / bs)))
        total_batches += n_batches_split
        if z_pre.shape[1] != PRE3H_DIM:
            raise RuntimeError(f"{split_name} pre3h dim {z_pre.shape[1]} != {PRE3H_DIM}")
        if z_post.shape[1] != POST128_DIM:
            raise RuntimeError(f"{split_name} post128 dim {z_post.shape[1]} != {POST128_DIM}")
        if edge_ids.shape[0] != z_pre.shape[0] or edge_ids.shape[0] != z_post.shape[0]:
            raise RuntimeError(f"{split_name} row-count mismatch pre/post/ids")
        if not torch.equal(edge_ids, edge_ids):
            raise RuntimeError("edge_id self-inequality")
        # identical row order already guaranteed by dual API; sanity on finiteness
        if not torch.isfinite(z_pre).all() or not torch.isfinite(z_post).all():
            raise RuntimeError(f"{split_name} non-finite embeddings")
        log_seed_coverage(edge_ids, expected, split_name)
        if max_batches is None:
            save_embedding_split_npz(pre_dir / f"{split_name}.npz", z_pre, y, edge_ids)
            save_embedding_split_npz(post_dir / f"{split_name}.npz", z_post, y, edge_ids)
        eid_np = edge_ids.detach().cpu().numpy().astype(np.int64)
        y_np = y.detach().cpu().numpy().astype(np.int64)
        found = set(eid_np.tolist())
        exp = set(expected.detach().cpu().tolist()) if max_batches is None else found
        report["splits"][split_name] = {
            "n": int(eid_np.shape[0]),
            "n_batches": n_batches_split,
            "extract_sec": split_sec,
            "pre_dim": int(z_pre.shape[1]),
            "post_dim": int(z_post.shape[1]),
            "ids": ids_hash(eid_np),
            "n_positives": int(y_np.sum()),
            "coverage": float(len(found & exp) / max(len(exp), 1)),
            "n_missing": int(len(exp - found)),
            "n_extra": int(len(found - exp)),
            "pre_post_id_aligned": True,
            "finite": True,
            "dual_capture": True,
        }
        # Keep tensors for smoke / optional return
        report["splits"][split_name]["_tensors"] = {
            "edge_ids": eid_np,
            "z_pre": z_pre.detach().cpu().numpy().astype(np.float32),
            "z_post": z_post.detach().cpu().numpy().astype(np.float32),
            "y": y_np,
        }

    extract_sec = float(time.perf_counter() - t_extract_all)
    hash_after = hash_model_state(model)
    if hash_before != hash_after:
        raise RuntimeError("Encoder parameters/buffers changed during extraction")
    report["encoder_hash_before"] = hash_before
    report["encoder_hash_after"] = hash_after
    report["encoder_hash_unchanged"] = True
    report["timing"] = {
        "dual_extract_sec": extract_sec,
        "n_batches_total": int(total_batches),
        "sec_per_batch": (extract_sec / total_batches) if total_batches else float("nan"),
        "pre3h_extract_sec": extract_sec,  # dual: same forward
        "post128_extract_sec": 0.0,  # dual: not a second pass
        "capture_mode": "dual_one_forward",
    }
    return report


def load_paysim_x_matrix(
    data_config_path: str = "data_config.json",
) -> Dict[str, Any]:
    df, df_train, tr_ids, va_ids, te_ids, spec = load_dataset_frames(
        "PaySim", str(ROOT / data_config_path)
    )
    y_all = df[spec.label_col].to_numpy().astype(np.int64)
    x_raw, feat_names, _, meta = build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    return {
        "df": df,
        "df_train": df_train,
        "tr_ids": np.asarray(tr_ids).astype(np.int64),
        "va_ids": np.asarray(va_ids).astype(np.int64),
        "te_ids": np.asarray(te_ids).astype(np.int64),
        "spec": spec,
        "y_all": y_all,
        "x_raw": x_raw.astype(np.float32),
        "feat_names": feat_names,
        "x_dim": int(x_raw.shape[1]),
        "meta": meta,
    }


def _metrics_block(y: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score

    y = y.astype(np.int64)
    pred = (proba >= float(thr)).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = int(y.shape[0])
    out = {
        "auroc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(thr),
        "positive_prediction_rate": float(pred.mean()) if n else 0.0,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "n": float(n),
    }
    out.update(alert_budget_metrics(y, proba))
    return out


def train_mlp_select_val_auprc(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    device: torch.device,
    seed: int = DOWNSTREAM_SEED,
    epochs: int = MLP_EPOCHS,
    lr: float = MLP_LR,
    batch_size: int = MLP_BS,
    max_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """Train PaperStyleMLP; select best epoch by validation AUPRC; thr on val F1."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_ep = int(max_epochs) if max_epochs is not None else int(epochs)
    model = PaperStyleMLP(int(x_tr.shape[1])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_t = torch.from_numpy(x_tr.astype(np.float32))
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    n = x_tr.shape[0]
    best_auprc = -1.0
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, float]] = []

    for ep in range(n_ep):
        model.train()
        perm = np.random.RandomState(seed * 1009 + ep).permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = x_t[idx].to(device)
            yb = y_t[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            opt.step()
        proba_va = _predict_proba(model, x_va, batch_size=batch_size, device=device)
        auprc = float(average_precision_score(y_va, proba_va)) if len(np.unique(y_va)) > 1 else float("nan")
        auroc = float(roc_auc_score(y_va, proba_va)) if len(np.unique(y_va)) > 1 else float("nan")
        history.append({"epoch": float(ep + 1), "val_auprc": auprc, "val_auroc": auroc})
        if auprc > best_auprc + 1e-12:
            best_auprc = auprc
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = n_ep
        best_auprc = float("nan")

    model.load_state_dict(best_state)
    model.to(device)
    proba_va = _predict_proba(model, x_va, batch_size=batch_size, device=device)
    proba_te = _predict_proba(model, x_te, batch_size=batch_size, device=device)
    thr = _select_threshold_f1(y_va, proba_va)
    return {
        "best_epoch": int(best_epoch),
        "best_val_auprc": float(best_auprc),
        "history": history,
        "val_ranking": {
            "auprc": float(average_precision_score(y_va, proba_va)) if len(np.unique(y_va)) > 1 else float("nan"),
            "auroc": float(roc_auc_score(y_va, proba_va)) if len(np.unique(y_va)) > 1 else float("nan"),
            "n": float(y_va.shape[0]),
        },
        "val_at_selected_threshold": _metrics_block(y_va, proba_va, thr),
        "threshold_0.5": _metrics_block(y_te, proba_te, 0.5),
        "threshold_val_selected": {
            **_metrics_block(y_te, proba_te, thr),
            "validation_selected_threshold": float(thr),
        },
        "validation_selected_threshold": float(thr),
        "proba_val": proba_va.astype(np.float32),
        "proba_test": proba_te.astype(np.float32),
        "learner": "PaperStyleMLP",
        "mlp_epochs": n_ep,
        "mlp_lr": lr,
        "downstream_seed": seed,
        "selection": "best_epoch_by_validation_auprc",
        "threshold_rule": "validation_max_f1_grid",
    }


def assemble_stack(
    name: str,
    z_pre: Optional[np.ndarray],
    z_post: Optional[np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    if name == "X_only":
        return x.astype(np.float32)
    if name == "pre3h_H_only":
        assert z_pre is not None
        return z_pre.astype(np.float32)
    if name == "pre3h_HxX":
        assert z_pre is not None
        return np.concatenate([z_pre, x], axis=1).astype(np.float32)
    if name == "post128_H_only":
        assert z_post is not None
        return z_post.astype(np.float32)
    if name == "post128_HxX":
        assert z_post is not None
        return np.concatenate([z_post, x], axis=1).astype(np.float32)
    raise ValueError(name)


def evaluate_stacks(
    *,
    splits_emb: Dict[str, Dict[str, np.ndarray]],
    x_by_id: np.ndarray,
    y_all: np.ndarray,
    device: torch.device,
    stacks: Sequence[str] = STACKS,
    max_mlp_epochs: Optional[int] = None,
    save_proba_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run locked MLP stacks. ``splits_emb[sp]`` has keys edge_ids, z_pre, z_post, y."""
    out: Dict[str, Any] = {}
    for stack in stacks:
        feats = {}
        for sp in ("train", "val", "test"):
            ids = splits_emb[sp]["edge_ids"]
            z_pre = splits_emb[sp].get("z_pre")
            z_post = splits_emb[sp].get("z_post")
            x = x_by_id[ids]
            y = splits_emb[sp]["y"]
            if not np.array_equal(y, y_all[ids]):
                raise RuntimeError(f"label mismatch on {sp} for stack {stack}")
            feats[sp] = {
                "X": assemble_stack(stack, z_pre, z_post, x),
                "y": y.astype(np.int64),
                "ids": ids.astype(np.int64),
            }
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(feats["train"]["X"]).astype(np.float32)
        x_va = scaler.transform(feats["val"]["X"]).astype(np.float32)
        x_te = scaler.transform(feats["test"]["X"]).astype(np.float32)
        metrics = train_mlp_select_val_auprc(
            x_tr,
            feats["train"]["y"],
            x_va,
            feats["val"]["y"],
            x_te,
            feats["test"]["y"],
            device=device,
            seed=DOWNSTREAM_SEED,
            max_epochs=max_mlp_epochs,
        )
        if save_proba_dir is not None:
            save_proba_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                save_proba_dir / f"{stack}_proba.npz",
                val_ids=feats["val"]["ids"],
                test_ids=feats["test"]["ids"],
                val_proba=metrics.pop("proba_val"),
                test_proba=metrics.pop("proba_test"),
                val_y=feats["val"]["y"],
                test_y=feats["test"]["y"],
            )
        else:
            metrics.pop("proba_val", None)
            metrics.pop("proba_test", None)
        metrics["feature_dim"] = int(x_tr.shape[1])
        metrics["ids_val"] = ids_hash(feats["val"]["ids"])
        metrics["ids_test"] = ids_hash(feats["test"]["ids"])
        out[stack] = metrics
    return out


def load_embedding_pair(run_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    from linear_probe import load_embedding_npz

    pre_dir = run_dir / "pre_embedding_3h"
    post_dir = run_dir / "post_embedding_128"
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for sp in ("train", "val", "test"):
        z_pre, y_pre, ids_pre = load_embedding_npz(pre_dir / f"{sp}.npz")
        z_post, y_post, ids_post = load_embedding_npz(post_dir / f"{sp}.npz")
        if not np.array_equal(ids_pre, ids_post):
            raise RuntimeError(f"{run_dir} {sp}: pre/post edge_id mismatch")
        if not np.array_equal(y_pre, y_post):
            raise RuntimeError(f"{run_dir} {sp}: pre/post y mismatch")
        if z_pre.shape[1] != PRE3H_DIM or z_post.shape[1] != POST128_DIM:
            raise RuntimeError(f"{run_dir} {sp}: unexpected dims")
        out[sp] = {
            "edge_ids": ids_pre.astype(np.int64),
            "z_pre": z_pre.astype(np.float32),
            "z_post": z_post.astype(np.float32),
            "y": y_pre.astype(np.int64),
        }
    return out


def rss_gb() -> float:
    # Linux: ru_maxrss is kilobytes
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def gpu_mem_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated()) / (1024.0 ** 3)


def run_role(
    role: str,
    *,
    data_config: str = "data_config.json",
    batch_size: int = DEFAULT_EXTRACT_BS,
    device_str: str = "cuda:0",
    skip_extract: bool = False,
    max_batches: Optional[int] = None,
    max_mlp_epochs: Optional[int] = None,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Full extract+eval for one locked role."""
    if role not in LOCKED_CHECKPOINTS:
        raise ValueError(f"Unknown role {role!r}; expected one of {sorted(LOCKED_CHECKPOINTS)}")
    meta = LOCKED_CHECKPOINTS[role]
    run_tag = str(meta["run_tag"])
    out_root = EMBEDDINGS_ROOT / run_tag
    assert_not_legacy_paysim_path(out_root)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    set_seed(int(meta.get("encoder_seed") or meta.get("random_seed") or 2))

    args = build_paysim_dplus_args(
        batch_size=batch_size,
        loader_num_workers=0,
        seed=int(meta.get("encoder_seed") or 2),
        unique_name=f"paysim_dplus_{run_tag}",
        tqdm=not smoke,
    )
    args.record_stage_timings = True
    args._stage_timings = {}
    with open(ROOT / data_config) as f:
        data_config_obj = json.load(f)

    stage_timings: Dict[str, float] = {}
    t0 = time.time()
    t_get = time.perf_counter()
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config_obj)
    stage_timings["get_data_wall_sec"] = float(time.perf_counter() - t_get)
    stage_timings.update({k: float(v) for k, v in (getattr(args, "_stage_timings", {}) or {}).items()})
    # Residual graph/model prep not attributed to ports/tds/csv
    accounted = (
        float(stage_timings.get("data_loading_csv_sec", 0.0))
        + float(stage_timings.get("ports_construction_sec", 0.0))
        + float(stage_timings.get("tds_construction_sec", 0.0))
    )
    stage_timings["other_graph_prep_sec"] = max(
        0.0, float(stage_timings.get("get_data_total_sec", stage_timings["get_data_wall_sec"])) - accounted
    )
    load_info: Dict[str, Any] = {"mode": "random_init" if meta["path"] is None else "checkpoint"}

    extract_report: Dict[str, Any] = {}

    if not skip_extract:
        if meta["path"] is None:
            set_seed(int(meta.get("random_seed") or 2))
        t_model = time.perf_counter()
        model, head_spec, pre_dim, emb_dim = construct_paysim_encoder(
            tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, device
        )
        stage_timings["model_construct_sec"] = float(time.perf_counter() - t_model)
        if meta["path"] is not None:
            t_ckpt = time.perf_counter()
            load_info = load_encoder_state_dict_strict(
                model,
                Path(meta["path"]),
                load_key=str(meta["load_key"]),
                refuse_keys=tuple(meta.get("refuse_keys") or ()),
            )
            model.to(device)
            stage_timings["checkpoint_load_sec"] = float(time.perf_counter() - t_ckpt)
        else:
            load_info = {
                "mode": "random_init",
                "random_seed": int(meta.get("random_seed") or 2),
                "classifier_loaded": False,
            }
            stage_timings["checkpoint_load_sec"] = 0.0
        freeze_encoder(model)
        hash0 = hash_model_state(model)
        extract_report = dual_extract_splits(
            model,
            head_spec,
            pre_dim,
            emb_dim,
            tr_data,
            val_data,
            te_data,
            tr_inds,
            val_inds,
            te_inds,
            args,
            device,
            out_root,
            max_batches=max_batches,
            splits=("train", "val", "test"),
        )
        hash1 = hash_model_state(model)
        if hash0 != hash1:
            raise RuntimeError("Encoder hash changed after freeze+extract")
        load_info["encoder_hash"] = hash0
    else:
        extract_report = {"skipped": True, "pre_dir": str(out_root / "pre_embedding_3h")}
        stage_timings["model_construct_sec"] = 0.0
        stage_timings["checkpoint_load_sec"] = 0.0

    # Build PaySim X (labels unused until MLP)
    t_x = time.perf_counter()
    x_pack = load_paysim_x_matrix(data_config)
    stage_timings["paysim_x_build_sec"] = float(time.perf_counter() - t_x)
    # Integrity: labels not in encoder inputs — checked via feature groups (edge_native only)
    if "Is Laundering" in (x_pack.get("feat_names") or []):
        raise RuntimeError("Label column leaked into PaySim X features")

    stacks = list(STACKS)
    if role in ("seed1", "seed3"):
        # X-only only needed once (seed2); still allow H stacks
        stacks = [s for s in STACKS if s != "X_only"]
    if role == "random_init":
        stacks = ["pre3h_H_only", "pre3h_HxX"]
    if role == "ft_seed2":
        stacks = ["pre3h_H_only", "pre3h_HxX"]
    if role == "seed2":
        stacks = list(STACKS)

    if smoke:
        # Use in-memory tensors from extract; one MLP step
        splits_emb = {}
        for sp, rec in extract_report["splits"].items():
            t = rec["_tensors"]
            splits_emb[sp] = {
                "edge_ids": t["edge_ids"],
                "z_pre": t["z_pre"],
                "z_post": t["z_post"],
                "y": t["y"],
            }
        # Align X by edge_id (ids are CSV EdgeIDs)
        t_down = time.perf_counter()
        stack_metrics = evaluate_stacks(
            splits_emb=splits_emb,
            x_by_id=x_pack["x_raw"],
            y_all=x_pack["y_all"],
            device=device,
            stacks=["pre3h_HxX"],
            max_mlp_epochs=1,
            save_proba_dir=None,
        )
        stage_timings["downstream_probe_sec"] = float(time.perf_counter() - t_down)
    else:
        splits_emb = load_embedding_pair(out_root)
        # Drop private tensors if any
        proba_dir = RESULTS_DIR / "proba" / run_tag
        t_down = time.perf_counter()
        stack_metrics = evaluate_stacks(
            splits_emb=splits_emb,
            x_by_id=x_pack["x_raw"],
            y_all=x_pack["y_all"],
            device=device,
            stacks=stacks,
            max_mlp_epochs=max_mlp_epochs,
            save_proba_dir=proba_dir,
        )
        stage_timings["downstream_probe_sec"] = float(time.perf_counter() - t_down)

    # Strip private tensors from extract report for JSON
    clean_extract = copy.deepcopy(extract_report)
    for sp in clean_extract.get("splits", {}):
        clean_extract["splits"][sp].pop("_tensors", None)

    one_time_setup = (
        float(stage_timings.get("data_loading_csv_sec", 0.0))
        + float(stage_timings.get("ports_construction_sec", 0.0))
        + float(stage_timings.get("tds_construction_sec", 0.0))
        + float(stage_timings.get("other_graph_prep_sec", 0.0))
        + float(stage_timings.get("model_construct_sec", 0.0))
        + float(stage_timings.get("checkpoint_load_sec", 0.0))
        + float(stage_timings.get("paysim_x_build_sec", 0.0))
    )
    stage_timings["one_time_setup_sec"] = one_time_setup

    # Full extract batch counts from split sizes (seed edges).
    n_tr = int(tr_inds.numel()) if hasattr(tr_inds, "numel") else int(len(tr_inds))
    n_va = int(val_inds.numel()) if hasattr(val_inds, "numel") else int(len(val_inds))
    n_te = int(te_inds.numel()) if hasattr(te_inds, "numel") else int(len(te_inds))
    bs = max(1, int(batch_size))
    full_batches = (
        int(math.ceil(n_tr / bs)) + int(math.ceil(n_va / bs)) + int(math.ceil(n_te / bs))
    )
    extract_timing = (extract_report or {}).get("timing") or {}
    smoke_n_batches = int(extract_timing.get("n_batches_total") or 0)
    sec_per_batch = float(extract_timing.get("sec_per_batch") or float("nan"))

    elapsed = time.time() - t0
    report = {
        "role": role,
        "run_tag": run_tag,
        "meta": {k: (str(v) if isinstance(v, Path) else v) for k, v in meta.items()},
        "load": load_info,
        "protocol": {
            "data": "PaySim",
            "flags": [
                "gin",
                "reverse_mp",
                "ego",
                "ports",
                "tds",
                "emlps",
                "correct_reverse_edge_features",
                "train_fit_edge_znorm",
            ],
            "loader_num_workers": 0,
            "persistent_workers": False,
            "batch_size": batch_size,
            "num_neighs": [100, 100],
            "tf": False,
            "augmentation": False,
            "primary_stack": "pre3h_HxX",
            "downstream": {
                "mlp": "PaperStyleMLP",
                "epochs": MLP_EPOCHS,
                "lr": MLP_LR,
                "batch_size": MLP_BS,
                "seed": DOWNSTREAM_SEED,
                "selection": "best_epoch_by_validation_auprc",
                "threshold": "validation_max_f1",
            },
            "x_features": "edge_native one_hot train-fit",
            "x_dim": x_pack["x_dim"],
            "pre3h_dim": PRE3H_DIM,
            "post128_dim": POST128_DIM,
            "expected_edge_dim": EXPECTED_EDGE_DIM,
            "labels_update_encoder": False,
            "legacy_paysim_artifacts_untouched": True,
        },
        "extract": clean_extract,
        "stacks": stack_metrics,
        "runtime": {
            "elapsed_sec": elapsed,
            "rss_gb": rss_gb(),
            "gpu_mem_gb": gpu_mem_gb(),
            "smoke": smoke,
            "max_batches": max_batches,
            "stage_timings": stage_timings,
            "split_sizes": {"train": n_tr, "val": n_va, "test": n_te},
            "full_n_batches": full_batches,
            "smoke_n_batches": smoke_n_batches,
            "extract_sec_per_batch": sec_per_batch,
        },
        "integrity": {
            "encoder_frozen": True,
            "bn_eval": True,
            "no_classifier_transfer": True,
            "train_fit_edge_znorm": True,
            "no_tf": True,
        },
    }
    return report


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=2, default=_default) + "\n")
