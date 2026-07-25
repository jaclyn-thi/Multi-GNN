"""D+ partial fine-tuning: MLP warmup then final-block unfreeze.

Distinct from ``--finetune``, contrastive pretrain, legacy supervised,
txn-node GCPAL, and neighbor-poscomplete paths.

Locked protocol (notes/final_dplus_experiment_preflight.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import to_hetero

import importlib.util
import sys as _sys

from data_loading import get_data
from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba, _select_threshold_f1
from linear_probe import load_embedding_npz
from train_util import (
    FORWARD_EDGE_TYPE,
    REVERSE_EDGE_TYPE,
    AddEgoIds,
    PreEmbeddingCapture,
    add_arange_ids,
    extract_param,
    extract_seed_embeddings_hetero,
    get_loaders,
    load_checkpoint_weights,
    resolve_embedding_head_linear,
)
from training import get_model
from util import create_parser, set_seed

ROOT = Path(__file__).resolve().parent

_pfa_spec = importlib.util.spec_from_file_location(
    "probe_feature_ablation",
    ROOT / "scripts" / "probe_feature_ablation.py",
)
_pfa = importlib.util.module_from_spec(_pfa_spec)
assert _pfa_spec.loader is not None
_sys.modules["probe_feature_ablation"] = _pfa
_pfa_spec.loader.exec_module(_pfa)
build_full_feature_matrix = _pfa.build_full_feature_matrix
load_dataset_frames = _pfa.load_dataset_frames

DPLUS_UNIQUE = "gin_emlps_ports_tds_corrected_preserve_seed_asym_proj_8192neg_queue0_40ep_seed2"
DPLUS_CKPT = ROOT / "saved-models" / f"checkpoint_{DPLUS_UNIQUE}.tar"
DPLUS_CKPT_SHA256 = "a320920141f585c5825cbd63ce760a845fb434a9b162d4c87270dc72b0442b87"
DPLUS_PRE3H_DIR = ROOT / "embeddings" / DPLUS_UNIQUE / "pre_embedding_3h"
TF_CACHE = ROOT / "results/cache/temporal_flow_causal/Small-HI"

CLF_LR = 1e-3
ENC_LR = 1e-4
WARMUP_EPOCHS_DEFAULT = 5
MAX_TOTAL_EPOCHS = 20
EARLY_STOP_PATIENCE = 5
MLP_HIDDEN = 128
MLP_DROPOUT = 0.1
MLP_BATCH_SIZE = 8192
STACK_DIM = 227

STAGE2_TRAINABLE_PREFIXES = ("convs.1.", "emlps.1.", "batch_norms.1.")
STAGE2_REQUIRED_SUBSTRINGS = (
    "convs.1.node__to__node.",
    "convs.1.node__rev_to__node.",
    "emlps.1.0.node__to__node.",
    "emlps.1.0.node__rev_to__node.",
    "emlps.1.2.node__to__node.",
    "emlps.1.2.node__rev_to__node.",
)


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def assert_dplus_checkpoint(path: Path = DPLUS_CKPT) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = file_sha256(path)
    if digest != DPLUS_CKPT_SHA256:
        raise ValueError(f"sha256 mismatch: {digest} != {DPLUS_CKPT_SHA256}")
    ckpt = torch.load(path, map_location="cpu")
    if int(ckpt.get("epoch", -1)) != 40:
        raise ValueError(f"Expected epoch 40, got {ckpt.get('epoch')}")
    if not bool(ckpt.get("correct_reverse_edge_features")):
        raise ValueError("correct_reverse_edge_features must be True")
    if not bool(ckpt.get("preserve_seed_edges")):
        raise ValueError("preserve_seed_edges must be True")
    if str(ckpt.get("reverse_edge_feature_semantics")) != "corrected":
        raise ValueError("reverse_edge_feature_semantics must be corrected")
    return {
        "path": str(path),
        "sha256": digest,
        "epoch": int(ckpt["epoch"]),
        "n_model_tensors": len(ckpt["model_state_dict"]),
    }


def _matches_prefix(name: str, prefixes: Sequence[str]) -> bool:
    return any(name.startswith(p) for p in prefixes)


def apply_trainability(model: nn.Module, stage: str) -> Dict[str, List[str]]:
    trainable: List[str] = []
    frozen: List[str] = []
    for name, p in model.named_parameters():
        if stage == "warmup":
            p.requires_grad = False
            frozen.append(name)
        elif stage == "partial":
            want = _matches_prefix(name, STAGE2_TRAINABLE_PREFIXES)
            p.requires_grad = bool(want)
            (trainable if want else frozen).append(name)
        else:
            raise ValueError(stage)
    if stage == "partial":
        blob = " ".join(trainable)
        for needle in STAGE2_REQUIRED_SUBSTRINGS:
            if needle not in blob:
                raise RuntimeError(f"Missing trainable params for {needle}")
        if not any(n.startswith("batch_norms.1.") for n in trainable):
            raise RuntimeError("Missing trainable batch_norms.1 params")
        if any(n.startswith("embedding_head.") for n in trainable):
            raise RuntimeError("embedding_head must remain frozen")
    return {"trainable": trainable, "frozen": frozen}


def set_encoder_modes(model: nn.Module, stage: str) -> None:
    model.eval()
    if stage != "partial":
        return
    for name, module in model.named_modules():
        if name.startswith("convs.1") or name.startswith("emlps.1") or name.startswith("batch_norms.1"):
            module.train()


def snapshot_params(model: nn.Module, names: Optional[Iterable[str]] = None) -> Dict[str, torch.Tensor]:
    wanted = set(names) if names is not None else None
    out: Dict[str, torch.Tensor] = {}
    for n, p in model.named_parameters():
        if wanted is None or n in wanted:
            out[n] = p.detach().cpu().clone()
    return out


def param_deltas(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor]) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for n in before:
        if n not in after:
            continue
        d = (after[n].float() - before[n].float()).abs().max().item()
        if d > 0:
            deltas[n] = float(d)
    return deltas


def build_graph_args(seed: int = 2, loader_num_workers: int = 0) -> argparse.Namespace:
    args = create_parser().parse_args(
        [
            "--data", "Small-HI", "--model", "gin", "--batch_size", "8192",
            "--num_neighs", "100", "100", "--loader_num_workers", str(loader_num_workers),
            "--seed", str(seed), "--reverse_mp", "--ego", "--ports", "--emlps", "--tds",
            "--correct_reverse_edge_features", "--preserve_seed_edges", "--testing",
            "--unique_name", DPLUS_UNIQUE, "--tqdm",
        ]
    )
    args.checkpoint_suffix = ""
    args.finetune = False
    args.include_temporal_flow_edge_features = False
    args.embedding_dim = 128
    return args


def load_dplus_hetero_encoder(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config, device):
    transform = AddEgoIds() if args.ego else None
    add_arange_ids([tr_data, val_data, te_data])
    sample_args = SimpleNamespace(**vars(args))
    sample_args.loader_num_workers = 0
    sample_loader, _, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, sample_args, train_shuffle=False
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
        lr=extract_param("lr", args),
    )
    model = get_model(sample_batch, config, args)
    head_spec = resolve_embedding_head_linear(model, int(getattr(model, "embedding_dim", 128)))
    pre_dim = int(head_spec.in_features)
    model = to_hetero(model, te_data.metadata(), aggr="mean")
    epoch = load_checkpoint_weights(model, device, args, data_config)
    model.to(device)
    return model, head_spec, pre_dim, int(epoch), config


def load_hxxtf_matrices(data_config_path: str = "data_config.json") -> Dict[str, Any]:
    df, df_train, tr_ids, va_ids, te_ids, spec = load_dataset_frames("Small-HI", data_config_path)
    y_all = df[spec.label_col].to_numpy().astype(np.int64)
    x_raw, _, _, _ = build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    tf_feat = np.load(TF_CACHE / "features.npy").astype(np.float32)
    if x_raw.shape[1] != 24:
        raise ValueError(f"expected X dim 24, got {x_raw.shape[1]}")
    if tf_feat.shape[1] != 5:
        raise ValueError(f"expected TF dim 5, got {tf_feat.shape[1]}")

    splits: Dict[str, Dict[str, np.ndarray]] = {}
    for sp in ("train", "val", "test"):
        z, y, ids = load_embedding_npz(DPLUS_PRE3H_DIR / f"{sp}.npz")
        if z.shape[1] != 198:
            raise ValueError(f"expected H dim 198, got {z.shape[1]}")
        if not np.array_equal(y, y_all[ids]):
            raise ValueError(f"label mismatch on {sp}")
        hxxtf = np.concatenate([z, x_raw[ids], tf_feat[ids]], axis=1).astype(np.float32)
        if hxxtf.shape[1] != STACK_DIM:
            raise ValueError(f"expected stack dim {STACK_DIM}, got {hxxtf.shape[1]}")
        splits[sp] = {"Z": z, "y": y, "ids": ids, "X": hxxtf}

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(splits["train"]["X"]).astype(np.float32)
    x_va = scaler.transform(splits["val"]["X"]).astype(np.float32)
    x_te = scaler.transform(splits["test"]["X"]).astype(np.float32)
    return {
        "x_raw": x_raw,
        "tf_feat": tf_feat,
        "y_all": y_all,
        "scaler": scaler,
        "splits": splits,
        "x_tr": x_tr,
        "y_tr": splits["train"]["y"],
        "ids_tr": splits["train"]["ids"],
        "x_va": x_va,
        "y_va": splits["val"]["y"],
        "ids_va": splits["val"]["ids"],
        "x_te": x_te,
        "y_te": splits["test"]["y"],
        "ids_te": splits["test"]["ids"],
        "winning_mlp_weights_found": False,
        "mlp_init_path": "from_scratch_identical_recipe",
        "winning_mlp_note": (
            "Job 18678029 did not persist PaperStyleMLP weights; classifier is "
            "initialized from scratch with seed=2 / lr=1e-3 / BCE recipe."
        ),
    }


def tf_leakage_audit() -> Dict[str, Any]:
    meta = json.loads((TF_CACHE / "meta.json").read_text())
    causal = meta.get("causal_history_policy") or {}
    ok = bool(causal.get("past_only")) and not bool(causal.get("uses_labels"))
    return {
        "past_only": bool(causal.get("past_only")),
        "uses_labels": bool(causal.get("uses_labels")),
        "val_sees_train_history": bool(causal.get("val_sees_train_history")),
        "test_sees_train_and_val_history": bool(causal.get("test_sees_train_and_val_history")),
        "ok": ok,
    }


@torch.no_grad()
def equivalence_check_pre3h(
    model, head_spec, pre_dim, tr_data, tr_inds, args, device,
    cached_z=None, cached_ids=None, *, max_batches: int = 2, atol: float = 1e-4,
    ckpt_path: Path = DPLUS_CKPT,
) -> Dict[str, Any]:
    """Verify FT encoder matches the locked D+ checkpoint before any update.

    Neighbor sampling makes live-vs-offline-cache comparison non-deterministic.
    Pass gate: (1) state_dict vs checkpoint tensors, (2) FT vs fresh deepcopy of
    the same weights on identical materialized batches.
    """
    import copy

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_sd = ckpt["model_state_dict"]
    model_sd = model.state_dict()
    weight_max_abs = 0.0
    weight_mismatched = 0
    for k, v in ckpt_sd.items():
        if k not in model_sd:
            weight_mismatched += 1
            continue
        d = float((model_sd[k].detach().cpu().float() - v.float()).abs().max().item())
        weight_max_abs = max(weight_max_abs, d)
        if d > atol:
            weight_mismatched += 1

    transform = AddEgoIds() if args.ego else None
    tr_loader, _, _ = get_loaders(
        tr_data, tr_data, tr_data, tr_inds, tr_inds[:1], tr_inds[:1],
        transform, args, train_shuffle=False,
    )
    batches = []
    for i, b in enumerate(tr_loader):
        if i >= max_batches:
            break
        batches.append(b)

    ref_model = copy.deepcopy(model)
    ref_model.load_state_dict(model_sd)
    ref_model.eval()
    model.eval()

    matched = 0
    max_abs = 0.0
    live_rows = 0
    cache_matched = 0
    cache_max = 0.0
    cache_map = None
    if cached_z is not None and cached_ids is not None:
        cache_map = {int(i): cached_z[j] for j, i in enumerate(cached_ids.tolist())}

    capture_ft = GradPreEmbeddingCapture(model, pre_dim=pre_dim, emb_dim=128, head_spec=head_spec)
    capture_ref = GradPreEmbeddingCapture(ref_model, pre_dim=pre_dim, emb_dim=128, head_spec=head_spec)
    try:
        for batch in batches:
            b_ft = copy.deepcopy(batch)
            b_ref = copy.deepcopy(batch)
            h_ft, _y, eids = forward_pre3h_from_loader_batch(
                model, b_ft, tr_inds, tr_loader.data, capture_ft, device
            )
            h_ref, _y2, eids2 = forward_pre3h_from_loader_batch(
                ref_model, b_ref, tr_inds, tr_loader.data, capture_ref, device
            )
            live_rows += int(h_ft.shape[0])
            ft_map = {
                int(e): h_ft[j].detach().cpu().numpy()
                for j, e in enumerate(eids.detach().cpu().tolist())
            }
            for j, e in enumerate(eids2.detach().cpu().tolist()):
                e = int(e)
                if e not in ft_map:
                    continue
                diff = float(np.abs(ft_map[e] - h_ref[j].detach().cpu().numpy()).max())
                max_abs = max(max_abs, diff)
                matched += 1
                if cache_map is not None and e in cache_map:
                    cdiff = float(np.abs(ft_map[e] - cache_map[e]).max())
                    cache_max = max(cache_max, cdiff)
                    cache_matched += 1
    finally:
        capture_ft.remove()
        capture_ref.remove()

    weights_ok = weight_mismatched == 0 and weight_max_abs <= atol
    live_vs_ref_ok = matched > 0 and max_abs <= atol
    return {
        "weight_max_abs_diff": weight_max_abs,
        "weight_mismatched_tensors": weight_mismatched,
        "weights_ok": weights_ok,
        "matched_ids": matched,
        "max_abs_diff": max_abs,
        "live_vs_ref_ok": live_vs_ref_ok,
        "ok": weights_ok and live_vs_ref_ok,
        "atol": atol,
        "live_rows": live_rows,
        "cache_diagnostic": {
            "matched_ids": cache_matched,
            "max_abs_diff": cache_max,
            "note": "Neighbor-sampling variance vs offline cache; not used as pass gate.",
        }
        if cache_map is not None
        else None,
    }


class GradPreEmbeddingCapture(PreEmbeddingCapture):
    """Like PreEmbeddingCapture but keeps autograd for encoder fine-tuning."""

    def _hook(self, module, inputs, output):
        self.captured[id(output)] = inputs[0]


def ranking_metrics(y: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    y = y.astype(np.int64)
    return {
        "auprc": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "auroc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "n": float(y.shape[0]),
    }


def f1_at_threshold(y: np.ndarray, proba: np.ndarray, thr: float) -> float:
    pred = (proba >= float(thr)).astype(np.int64)
    return float(f1_score(y.astype(np.int64), pred, zero_division=0))


def train_mlp_epoch(clf, x, y, *, optimizer, device, seed, epoch, batch_size=MLP_BATCH_SIZE) -> float:
    clf.train()
    x_t = torch.from_numpy(x.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))
    n = x.shape[0]
    perm = np.random.RandomState(seed * 1009 + epoch).permutation(n)
    total = 0.0
    steps = 0
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        xb = x_t[idx].to(device)
        yb = y_t[idx].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(clf(xb), yb)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu())
        steps += 1
    return total / max(steps, 1)


def eval_clf(clf, x, y, *, device, split_name="val", allow_test=False) -> Dict[str, float]:
    if split_name == "test" and not allow_test:
        raise RuntimeError("Test evaluation blocked during selection")
    proba = _predict_proba(clf, x, batch_size=MLP_BATCH_SIZE, device=device)
    thr = _select_threshold_f1(y, proba)
    out = ranking_metrics(y, proba)
    out["f1_at_selected"] = f1_at_threshold(y, proba, thr)
    out["threshold"] = float(thr)
    return out


@dataclass
class EarlyStopState:
    best_auprc: float = -1.0
    best_f1: float = -1.0
    best_epoch: int = -1
    patience: int = EARLY_STOP_PATIENCE
    patience_left: int = EARLY_STOP_PATIENCE
    stopped: bool = False

    def step(self, epoch: int, val_auprc: float, val_f1: float) -> bool:
        improved = False
        if val_auprc > self.best_auprc + 1e-12 or (
            abs(val_auprc - self.best_auprc) <= 1e-12 and val_f1 > self.best_f1
        ):
            self.best_auprc = float(val_auprc)
            self.best_f1 = float(val_f1)
            self.best_epoch = int(epoch)
            self.patience_left = int(self.patience)
            improved = True
        else:
            self.patience_left -= 1
            if self.patience_left <= 0:
                self.stopped = True
        return improved


@dataclass
class FinetuneState:
    stage: str = "warmup"
    global_epoch: int = 0
    warmup_epochs_done: int = 0
    partial_epochs_done: int = 0
    early: EarlyStopState = field(default_factory=EarlyStopState)
    history: List[Dict[str, Any]] = field(default_factory=list)


def save_finetune_checkpoint(path: Path, *, encoder, clf, optimizer, state: FinetuneState, scaler, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder_state_dict": encoder.state_dict(),
        "classifier_state_dict": clf.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "stage": state.stage,
        "global_epoch": state.global_epoch,
        "warmup_epochs_done": state.warmup_epochs_done,
        "partial_epochs_done": state.partial_epochs_done,
        "early_stop": asdict(state.early),
        "history": state.history,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "meta": meta,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },
        "protocol": {
            "clf_lr": CLF_LR,
            "enc_lr": ENC_LR,
            "max_total_epochs": MAX_TOTAL_EPOCHS,
            "stage2_trainable_prefixes": list(STAGE2_TRAINABLE_PREFIXES),
            "embedding_head_frozen": True,
            "source_checkpoint_sha256": DPLUS_CKPT_SHA256,
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_finetune_checkpoint(path: Path, encoder, clf, optimizer=None) -> FinetuneState:
    payload = torch.load(path, map_location="cpu")
    encoder.load_state_dict(payload["encoder_state_dict"])
    clf.load_state_dict(payload["classifier_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    early_d = payload.get("early_stop") or {}
    early = EarlyStopState(**{k: early_d[k] for k in EarlyStopState.__dataclass_fields__ if k in early_d})
    state = FinetuneState(
        stage=str(payload.get("stage", "warmup")),
        global_epoch=int(payload.get("global_epoch", 0)),
        warmup_epochs_done=int(payload.get("warmup_epochs_done", 0)),
        partial_epochs_done=int(payload.get("partial_epochs_done", 0)),
        early=early,
        history=list(payload.get("history") or []),
    )
    rng = payload.get("rng") or {}
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return state


def build_optimizer(stage: str, encoder: nn.Module, clf: PaperStyleMLP) -> torch.optim.Optimizer:
    groups = [{"params": [p for p in clf.parameters() if p.requires_grad], "lr": CLF_LR, "name": "classifier"}]
    if stage == "partial":
        enc_params = [p for p in encoder.parameters() if p.requires_grad]
        if not enc_params:
            raise RuntimeError("No encoder params trainable in partial stage")
        # Confirm reverse relations present
        names = [n for n, p in encoder.named_parameters() if p.requires_grad]
        if not any("rev_to" in n for n in names):
            raise RuntimeError("Reverse-relation params missing from encoder optimizer group")
        groups.append({"params": enc_params, "lr": ENC_LR, "name": "encoder_final_block"})
    return torch.optim.Adam(groups)


def optimizer_group_report(opt: torch.optim.Optimizer) -> List[Dict[str, Any]]:
    out = []
    for i, g in enumerate(opt.param_groups):
        out.append({
            "index": i,
            "name": g.get("name"),
            "lr": float(g["lr"]),
            "n_params": int(sum(p.numel() for p in g["params"])),
        })
    return out


def forward_pre3h_from_loader_batch(model, batch, split_inds, loader_data, capture, device):
    store = FORWARD_EDGE_TYPE
    fwd = batch[store]
    split_inds_cpu = split_inds.detach().cpu()
    batch_edge_inds = split_inds_cpu[fwd.input_id.detach().cpu()]
    batch_edge_ids = loader_data[store].edge_attr.detach().cpu()[batch_edge_inds, 0]
    mask = torch.isin(fwd.edge_attr[:, 0].detach().cpu(), batch_edge_ids)
    edge_ids_all = fwd.edge_attr[:, 0].long().clone()
    fwd.edge_attr = fwd.edge_attr[:, 1:]
    batch[REVERSE_EDGE_TYPE].edge_attr = batch[REVERSE_EDGE_TYPE].edge_attr[:, 1:]
    batch = batch.to(device)
    capture.clear()
    out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
    z = out[store]
    h = capture.get(z)
    mask_dev = mask.to(device, non_blocking=True)
    return h[mask_dev], fwd.y[mask_dev].float(), edge_ids_all[mask].to(device)


def pack_online_features(h, edge_ids, x_raw, tf_feat, scaler, device):
    ids = edge_ids.detach().cpu().numpy().astype(np.int64)
    x = torch.from_numpy(x_raw[ids].astype(np.float32)).to(device)
    tf = torch.from_numpy(tf_feat[ids].astype(np.float32)).to(device)
    feats = torch.cat([h, x, tf], dim=1)
    mean = torch.from_numpy(scaler.mean_.astype(np.float32)).to(device)
    scale = scaler.scale_.astype(np.float32).copy()
    scale[scale == 0] = 1.0
    scale_t = torch.from_numpy(scale).to(device)
    return (feats - mean) / scale_t


def run_partial_epoch_online(
    encoder, clf, optimizer, loader, split_inds, capture, x_raw, tf_feat, scaler, device,
    *, max_batches=None,
):
    set_encoder_modes(encoder, "partial")
    clf.train()
    total_loss = 0.0
    steps = 0
    grad_norm_sum = 0.0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        h, yb, eids = forward_pre3h_from_loader_batch(
            encoder, batch, split_inds, loader.data, capture, device
        )
        if not h.requires_grad:
            raise RuntimeError(
                "pre-3h tensor has requires_grad=False; use GradPreEmbeddingCapture for FT"
            )
        feats = pack_online_features(h, eids, x_raw, tf_feat, scaler, device)
        loss = nn.functional.binary_cross_entropy_with_logits(clf(feats), yb.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {i}")
        loss.backward()
        gn = 0.0
        enc_gn = 0.0
        for n, p in list(clf.named_parameters()) + [
            (n, p) for n, p in encoder.named_parameters() if p.requires_grad
        ]:
            if p.grad is not None:
                g = float(p.grad.detach().norm().cpu())
                if not np.isfinite(g):
                    raise RuntimeError(f"non-finite grad at step {i} ({n})")
                gn += g
                if not n.startswith("net."):  # clf uses .net; crude — count encoder below
                    pass
        for p in encoder.parameters():
            if p.requires_grad and p.grad is not None:
                enc_gn += float(p.grad.detach().norm().cpu())
        if enc_gn <= 0:
            raise RuntimeError(f"encoder gradients are zero at step {i}")
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        grad_norm_sum += gn
        steps += 1
    return {
        "loss": total_loss / max(steps, 1),
        "grad_norm": grad_norm_sum / max(steps, 1),
        "steps": float(steps),
    }


def count_named_params(model: nn.Module, names: Sequence[str]) -> int:
    wanted = set(names)
    return int(sum(p.numel() for n, p in model.named_parameters() if n in wanted))
