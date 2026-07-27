#!/usr/bin/env python3
"""Schema-level categorical masking scout (seed-2, validation-only gate).

Subcommands: smoke | train | eval_arm | aggregate

Canonical representation: post-128 H for AMLWorld and PaySim.
Does not evaluate test. Does not overwrite corrected/no-preserve control ckpt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
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
from feature_contracts import CONTRACT_LEGACY, CONTRACT_TYPE_ONLY  # noqa: E402
from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba  # noqa: E402
from graph_augmentations import generate_views  # noqa: E402
from linear_probe import load_embedding_npz  # noqa: E402
from ranking_metrics import alert_budget_metrics  # noqa: E402
from train_util import (  # noqa: E402
    AddEgoIds,
    FORWARD_EDGE_TYPE,
    REVERSE_EDGE_TYPE,
    add_arange_ids,
    attach_edge_id_from_batch,
    extract_param,
    get_hetero_seed_edge_ids,
    get_loaders,
    load_checkpoint_weights,
)
from training import _contrastive_view_kwargs  # noqa: E402
from training import get_model  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

TAG = "schema_mask_scout"
RESULT_ROOT = ROOT / "results" / "diagnostics" / TAG
EMBED_ROOT = ROOT / "embeddings" / TAG
CELLS = RESULT_ROOT / "cells"
NOTES_MD = ROOT / "notes" / f"{TAG}.md"
FINAL_JSON = ROOT / "results" / "diagnostics" / f"{TAG}.json"
PREDECLARED = RESULT_ROOT / "predeclared_gate.json"
TF_CACHE = ROOT / "results/cache/temporal_flow_causal/Small-HI"

CONTROL_UNIQUE = "gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2"
CONTROL_CKPT = ROOT / f"saved-models/checkpoint_{CONTROL_UNIQUE}.tar"
CONTROL_SHA = "18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6"
ENCODER_SEED = 2
RANDOM_INIT_SEED = 2
DOWNSTREAM_LOGISTIC_SEED = 1
MLP_SEED = 2
MLP_EPOCHS = 15
MLP_LR = 1e-3
MLP_BS = 8192

# Predeclared gate thresholds (written before any downstream result is read).
GATE_PAYSIM_AUPRC_DELTA = 0.003
GATE_PAYSIM_F1_DELTA = 0.01
GATE_AML_AUPRC_REGRESSION_MAX = 0.02

ARMS = {
    "control": {
        "unique": CONTROL_UNIQUE,
        "ckpt": CONTROL_CKPT,
        "train": False,
        "mask_prob": None,
    },
    "schema_mask_p025": {
        "unique": "schema_mask_scout_p025_seed2",
        "ckpt": ROOT / "saved-models/checkpoint_schema_mask_scout_p025_seed2.tar",
        "train": True,
        "mask_prob": 0.25,
    },
    "schema_mask_p050": {
        "unique": "schema_mask_scout_p050_seed2",
        "ckpt": ROOT / "saved-models/checkpoint_schema_mask_scout_p050_seed2.tar",
        "train": True,
        "mask_prob": 0.50,
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def ids_hash(ids: np.ndarray) -> Dict[str, Any]:
    ids = np.asarray(ids, dtype=np.int64)
    return {
        "n": int(ids.shape[0]),
        "n_unique": int(np.unique(ids).shape[0]),
        "edge_id_sum": int(ids.sum()) if ids.size else 0,
        "sha256_of_ids_bytes": hashlib.sha256(ids.tobytes()).hexdigest() if ids.size else None,
    }


def code_provenance() -> Dict[str, Any]:
    def _run(cmd: List[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=str(ROOT), stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""

    porcelain = _run(["git", "status", "--porcelain"])
    dirty = [ln[3:] for ln in porcelain.splitlines() if ln.strip()]
    src = [
        "graph_augmentations.py",
        "training.py",
        "util.py",
        "scripts/schema_mask_scout.py",
    ]
    return {
        "git_commit": _run(["git", "rev-parse", "HEAD"]) or None,
        "dirty_file_count": len(dirty),
        "dirty_tree_manifest": dirty[:200],
        "source_file_sha256": {r: sha256_file(ROOT / r) for r in src if (ROOT / r).is_file()},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_predeclared_gate() -> Dict[str, Any]:
    gate = {
        "written_before_any_downstream_result": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_representation": "post_128_H",
        "paysim_primary_contract": CONTRACT_TYPE_ONLY,
        "paysim_sensitivity_contract": CONTRACT_LEGACY,
        "thresholds": {
            "paysim_type_only_val_auprc_improve_abs": GATE_PAYSIM_AUPRC_DELTA,
            "paysim_type_only_val_f1_improve_abs": GATE_PAYSIM_F1_DELTA,
            "must_beat_matched_random_val_auprc": True,
            "amlworld_hxxtf_val_auprc_max_regression_abs": GATE_AML_AUPRC_REGRESSION_MAX,
        },
        "selection_rule": (
            "Masked arm passes iff (PaySim type_only val AUPRC +>=0.003 OR F1 +>=0.01) "
            "AND beats matched random AUPRC AND AMLWorld H+X+TF val AUPRC regresses by <=0.02. "
            "If both pass, pick higher PaySim type_only AUPRC; AMLWorld AUPRC tiebreak."
        ),
        "control_amlworld_baseline": (
            "Resolved by eval_arm --arm control under this script (post-128); "
            "prior pre-3h metrics are NOT reused."
        ),
        "code_provenance": code_provenance(),
    }
    write_json(PREDECLARED, gate)
    return gate


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


def metrics_block(y: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, Any]:
    y = y.astype(np.int64)
    pred = (proba >= float(thr)).astype(np.int64)
    out: Dict[str, Any] = {
        "auroc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "threshold": float(thr),
        "positive_prediction_rate": float(pred.mean()) if y.size else 0.0,
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
        "n": int(y.shape[0]),
        "n_positives": int(y.sum()),
    }
    out.update(alert_budget_metrics(y, proba))
    return out


def locked_train_argv(unique: str, mask_prob: float, *, n_epochs: int = 40) -> List[str]:
    """Exact corrected/no-preserve seed-2 recipe + semantic group mask."""
    return [
        "--data", "Small-HI",
        "--model", "gin",
        "--tqdm",
        "--batch_size", "8192",
        "--num_neighs", "100", "100",
        "--loader_num_workers", "16",
        "--save_model",
        "--seed", str(ENCODER_SEED),
        "--unique_name", unique,
        "--n_epochs", str(n_epochs),
        "--objective", "contrastive",
        "--reverse_mp", "--ego", "--ports", "--emlps", "--tds",
        "--correct_reverse_edge_features",
        "--contrast_projection_head",
        "--contrast_projection_hidden", "128",
        "--contrast_projection_dim", "128",
        "--checkpoint_policy", "best",
        "--contrastive_asymmetric",
        "--contrastive_num_neg_samples", "8192",
        "--contrastive_memory_bank_size", "0",
        "--contrastive_accum_steps", "4",
        "--contrastive_temperature", "0.5",
        "--edge_drop_target_rate", "0.1",
        "--edge_attr_mask_rate", "0.1",
        "--semantic_group_mask",
        "--categorical_group_mask_prob", str(mask_prob),
        "--testing",
    ]


def parse_ns(argv: List[str]):
    return create_parser().parse_args(argv)


def cmd_smoke(args: argparse.Namespace) -> None:
    logger_setup()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    write_predeclared_gate()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()

    argv = locked_train_argv("schema_mask_scout_smoke", 0.5, n_epochs=1)
    ns = parse_ns(argv)
    ns.loader_num_workers = 0
    set_seed(ns.seed)
    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns, data_config)
    transform = AddEgoIds() if ns.ego else None
    add_arange_ids([tr_data, val_data, te_data])
    from types import SimpleNamespace
    from contrastive_projection import ContrastiveProjectionHead

    config = SimpleNamespace(
        model=ns.model,
        n_hidden=extract_param("n_hidden", ns),
        n_gnn_layers=extract_param("n_gnn_layers", ns),
        n_heads=None,
        dropout=extract_param("dropout", ns),
        final_dropout=extract_param("final_dropout", ns),
    )
    sample_args = SimpleNamespace(**vars(ns))
    sample_args.loader_num_workers = 0
    tr_loader, _, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, sample_args, train_shuffle=True
    )
    # get_model subtracts 1 for the synthetic arange id column — sample must still have it.
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, ns)
    model_embedding_dim = int(getattr(model, "embedding_dim", 128))
    model = to_hetero(model, te_data.metadata(), aggr="mean").to(device)
    proj = ContrastiveProjectionHead(
        model_embedding_dim, model_embedding_dim, model_embedding_dim
    ).to(device)

    batch = next(iter(tr_loader))
    seed_ids = get_hetero_seed_edge_ids(batch, tr_loader.data)
    attach_edge_id_from_batch(batch, tr_loader.data)
    batch = batch.to(device)
    seed_ids = seed_ids.to(device)
    edge_dim = int(batch[FORWARD_EDGE_TYPE].edge_attr.shape[1])
    if edge_dim != 8:
        raise SystemExit(f"expected edge_dim=8 after id strip, got {edge_dim}")

    stats: Dict[str, Any] = {}
    view1, view2 = generate_views(
        batch,
        **_contrastive_view_kwargs(ns, stats, seed_edge_ids=seed_ids),
    )
    s1 = stats.get("last_semantic_state_v1", {})
    if s1.get("mask_currency"):
        if not torch.all(view1[FORWARD_EDGE_TYPE].edge_attr[:, 2] == 0):
            raise SystemExit("forward currency mask failed")
        if not torch.all(view1[REVERSE_EDGE_TYPE].edge_attr[:, 2] == 0):
            raise SystemExit("reverse currency mask not synced")

    model.train()
    out1 = model(view1.x_dict, view1.edge_index_dict, view1.edge_attr_dict)
    z1 = out1[FORWARD_EDGE_TYPE]
    if int(z1.shape[-1]) != model_embedding_dim:
        raise SystemExit(f"unexpected z dim {z1.shape}")
    z1p = proj(z1)
    if int(z1p.shape[-1]) != model_embedding_dim:
        raise SystemExit(f"unexpected proj dim {z1p.shape}")
    loss = z1p.float().pow(2).mean()
    loss.backward()
    enc_grad = any(
        p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters()
    )
    if not enc_grad:
        raise SystemExit("encoder gradients are zero")
    if not torch.isfinite(loss).item():
        raise SystemExit("non-finite loss")

    smoke_ckpt = ROOT / "saved-models/checkpoint_schema_mask_scout_smoke.tar"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 0, "loss": float(loss)}, smoke_ckpt)
    blob = torch.load(smoke_ckpt, map_location="cpu")
    model.load_state_dict(blob["model_state_dict"], strict=True)

    report = {
        "passed": True,
        "loss": float(loss.detach().cpu()),
        "z_dim": model_embedding_dim,
        "proj_dim": model_embedding_dim,
        "edge_dim_after_id_strip": edge_dim,
        "encoder_grad_nonzero": enc_grad,
        "semantic_stats": {
            k: (dict(v) if isinstance(v, dict) else v)
            for k, v in stats.items()
            if "semantic" in str(k)
        },
        "checkpoint_roundtrip": str(smoke_ckpt),
        "projected_40ep_hours_from_prior_recipe": 2.7,
        "under_six_hours": True,
        "smoke_wall_sec": time.perf_counter() - t0,
        "predeclared_gate": str(PREDECLARED),
        "code_provenance": code_provenance(),
        "smoke_fix_note": (
            "get_model requires sample batch with arange id column still present "
            "(subtracts 1); attach_edge_id_from_batch runs only on the forward batch."
        ),
    }
    write_json(RESULT_ROOT / "smoke.json", report)
    logging.info("SMOKE PASSED %s", report)


def cmd_train(args: argparse.Namespace) -> None:
    """Delegate to main.py with locked argv (full 40ep)."""
    logger_setup()
    arm = args.arm
    if arm not in ("schema_mask_p025", "schema_mask_p050"):
        raise SystemExit(f"train arm must be masked, got {arm}")
    meta = ARMS[arm]
    ckpt = meta["ckpt"]
    if ckpt.is_file():
        raise SystemExit(f"refusing overwrite {ckpt}")
    last = ROOT / f"saved-models/checkpoint_{meta['unique']}_last.tar"
    if last.is_file():
        raise SystemExit(f"refusing overwrite {last}")
    argv = locked_train_argv(meta["unique"], float(meta["mask_prob"]), n_epochs=40)
    logging.info("Launching main.py train arm=%s argv=%s", arm, " ".join(argv))
    # Import main training entry
    import main as main_mod

    sys.argv = ["main.py"] + argv
    main_mod.main()


def _extract_post128(
    *,
    data: str,
    unique: str,
    emb_subdir: str,
    seed: int,
    train_fit: bool,
    feature_contract: Optional[str],
    random_init: bool,
    batch_size: int,
) -> Path:
    import embedding_extraction as ee

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
        "--representation_source", "post_embedding",
        "--extract_splits", "train,val",
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
    out = ee.run_embedding_extraction(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, ns, data_config
    )
    if (Path(out) / "test.npz").is_file():
        raise SystemExit("test.npz written — forbidden")
    return Path(out)


def _run_amlworld_val(emb_dir: Path, device: torch.device) -> Dict[str, Any]:
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
    y_all = df[dspec.label_col].to_numpy().astype(np.int64)
    x_raw, _, _, _ = mod.build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    tf_feat = np.load(TF_CACHE / "features.npy").astype(np.float32)
    stacks = {}
    for stack_name in ("post128_H_only", "post128_HxXTF"):
        feats = {}
        for sp, expected_ids in (("train", tr_ids), ("val", va_ids)):
            z, y, ids = load_embedding_npz(emb_dir / f"{sp}.npz")
            if not np.array_equal(y, y_all[ids]):
                raise SystemExit(f"AML label mismatch {sp}")
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
            "validation_metrics_at_0.5": metrics_block(feats["val"]["y"], pva, 0.5),
            "validation_metrics_at_val_optimal_f1": {
                **metrics_block(feats["val"]["y"], pva, thr),
                "note": "within-validation diagnostic",
            },
            "ids": {sp: ids_hash(feats[sp]["ids"]) for sp in feats},
            "test_evaluated": False,
        }
    return stacks


def _run_paysim_logistic(emb_dir: Path) -> Dict[str, Any]:
    if (emb_dir / "test.npz").is_file():
        raise SystemExit("test.npz present")
    z_tr, y_tr, ids_tr = load_embedding_npz(emb_dir / "train.npz")
    z_va, y_va, ids_va = load_embedding_npz(emb_dir / "val.npz")
    cw = gin_model_class_weight()
    set_seed(DOWNSTREAM_LOGISTIC_SEED)
    clf = LogisticRegression(
        class_weight=cw, max_iter=1000, random_state=DOWNSTREAM_LOGISTIC_SEED,
        solver="lbfgs", n_jobs=1, C=1.0,
    )
    clf.fit(z_tr, y_tr)
    proba = clf.predict_proba(z_va)[:, 1].astype(np.float64)
    thr = tune_thr_max_f1(y_va, proba)
    return {
        "validation_metrics_at_0.5": metrics_block(y_va, proba, 0.5),
        "validation_metrics_at_val_optimal_f1": {
            **metrics_block(y_va, proba, thr),
            "note": "within-validation diagnostic",
        },
        "ids": {"train": ids_hash(ids_tr), "val": ids_hash(ids_va)},
        "test_evaluated": False,
        "learner": "LogisticRegression",
        "class_weight_mode": "model",
        "C": 1.0,
    }


def cmd_eval_arm(args: argparse.Namespace) -> None:
    logger_setup()
    CELLS.mkdir(parents=True, exist_ok=True)
    EMBED_ROOT.mkdir(parents=True, exist_ok=True)
    if not PREDECLARED.is_file():
        write_predeclared_gate()
    arm = args.arm
    meta = ARMS[arm]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = meta["ckpt"]
    if not ckpt.is_file():
        raise SystemExit(f"missing ckpt {ckpt}")
    sha = sha256_file(ckpt)
    if arm == "control" and sha != CONTROL_SHA:
        raise SystemExit(f"control sha mismatch {sha}")

    # AMLWorld post-128 extract + val MLP
    aml_dir = _extract_post128(
        data="Small-HI",
        unique=meta["unique"],
        emb_subdir=f"{arm}/amlworld_post128",
        seed=ENCODER_SEED,
        train_fit=False,
        feature_contract=None,
        random_init=False,
        batch_size=int(args.batch_size),
    )
    aml = _run_amlworld_val(aml_dir, device)

    paysim = {}
    for contract in (CONTRACT_TYPE_ONLY, CONTRACT_LEGACY):
        role_key = "primary" if contract == CONTRACT_TYPE_ONLY else "sensitivity"
        pre_dir = _extract_post128(
            data="PaySim",
            unique=meta["unique"],
            emb_subdir=f"{arm}/paysim_{contract}_pretrained",
            seed=ENCODER_SEED,
            train_fit=True,
            feature_contract=contract,
            random_init=False,
            batch_size=int(args.batch_size),
        )
        rnd_dir = _extract_post128(
            data="PaySim",
            unique=meta["unique"],
            emb_subdir=f"{arm}/paysim_{contract}_random",
            seed=RANDOM_INIT_SEED,
            train_fit=True,
            feature_contract=contract,
            random_init=True,
            batch_size=int(args.batch_size),
        )
        pre = _run_paysim_logistic(pre_dir)
        rnd = _run_paysim_logistic(rnd_dir)
        paysim[contract] = {
            "role": role_key,
            "pretrained": pre,
            "random": rnd,
            "delta_auprc": pre["validation_metrics_at_0.5"]["auprc"]
            - rnd["validation_metrics_at_0.5"]["auprc"],
            "delta_f1": pre["validation_metrics_at_0.5"]["f1"]
            - rnd["validation_metrics_at_0.5"]["f1"],
            "embeddings_pretrained": str(pre_dir),
            "embeddings_random": str(rnd_dir),
        }

    out = {
        "arm": arm,
        "unique_name": meta["unique"],
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha,
        "mask_prob": meta["mask_prob"],
        "amlworld": aml,
        "paysim": paysim,
        "test_evaluated": False,
        "code_provenance": code_provenance(),
        "predeclared_gate_path": str(PREDECLARED),
    }
    path = CELLS / f"{arm}.json"
    write_json(path, out)
    write_json(RESULT_ROOT / f"eval_{arm}.json", out)
    logging.info("Wrote %s", path)


def cmd_aggregate(_: argparse.Namespace) -> None:
    logger_setup()
    gate = json.loads(PREDECLARED.read_text())
    cells = {}
    for arm in ARMS:
        p = CELLS / f"{arm}.json"
        if not p.is_file():
            raise SystemExit(f"missing {p}")
        cells[arm] = json.loads(p.read_text())

    ctrl = cells["control"]
    ctrl_aml = ctrl["amlworld"]["post128_HxXTF"]["validation_metrics_at_0.5"]["auprc"]
    ctrl_ps = ctrl["paysim"][CONTRACT_TYPE_ONLY]["pretrained"]["validation_metrics_at_0.5"]

    results = []
    for arm in ("schema_mask_p025", "schema_mask_p050"):
        c = cells[arm]
        ps = c["paysim"][CONTRACT_TYPE_ONLY]
        pre = ps["pretrained"]["validation_metrics_at_0.5"]
        rnd = ps["random"]["validation_metrics_at_0.5"]
        aml = c["amlworld"]["post128_HxXTF"]["validation_metrics_at_0.5"]
        d_auprc = pre["auprc"] - ctrl_ps["auprc"]
        d_f1 = pre["f1"] - ctrl_ps["f1"]
        aml_reg = ctrl_aml - aml["auprc"]
        pass_transfer = (d_auprc >= GATE_PAYSIM_AUPRC_DELTA) or (d_f1 >= GATE_PAYSIM_F1_DELTA)
        pass_random = pre["auprc"] > rnd["auprc"]
        pass_aml = aml_reg <= GATE_AML_AUPRC_REGRESSION_MAX
        passed = pass_transfer and pass_random and pass_aml
        results.append(
            {
                "arm": arm,
                "passed": passed,
                "paysim_type_only_val_auprc": pre["auprc"],
                "paysim_type_only_val_f1": pre["f1"],
                "delta_vs_control_auprc": d_auprc,
                "delta_vs_control_f1": d_f1,
                "beats_random": pass_random,
                "aml_hxxtf_val_auprc": aml["auprc"],
                "aml_auprc_regression": aml_reg,
                "checks": {
                    "transfer_improve": pass_transfer,
                    "beat_random": pass_random,
                    "aml_no_big_regress": pass_aml,
                },
            }
        )

    passed_arms = [r for r in results if r["passed"]]
    selected = None
    if passed_arms:
        passed_arms.sort(
            key=lambda r: (r["paysim_type_only_val_auprc"], cells[r["arm"]]["amlworld"]["post128_HxXTF"]["validation_metrics_at_0.5"]["auprc"]),
            reverse=True,
        )
        selected = passed_arms[0]["arm"]

    final = {
        "title": "Schema-level categorical masking scout (seed-2)",
        "scope": "validation_gate_only",
        "not_a_final_transfer_result": True,
        "test_evaluated": False,
        "predeclared_gate": gate,
        "control_baselines": {
            "amlworld_post128_HxXTF_val_auprc": ctrl_aml,
            "amlworld_post128_H_only_val_auprc": ctrl["amlworld"]["post128_H_only"]["validation_metrics_at_0.5"]["auprc"],
            "paysim_type_only_val_auprc": ctrl_ps["auprc"],
            "paysim_type_only_val_f1": ctrl_ps["f1"],
            "paysim_legacy_val_auprc": ctrl["paysim"][CONTRACT_LEGACY]["pretrained"]["validation_metrics_at_0.5"]["auprc"],
        },
        "arms": results,
        "selected_arm": selected,
        "cells": {k: str(CELLS / f"{k}.json") for k in ARMS},
        "code_provenance": code_provenance(),
    }
    write_json(FINAL_JSON, final)
    write_json(RESULT_ROOT / "aggregate.json", final)

    lines = [
        "# Schema-level categorical masking scout (seed-2)",
        "",
        "> Validation-only gate. **Not** a final transfer result. Test never evaluated.",
        "",
        f"- Selected arm: **`{selected}`**" if selected else "- Selected arm: **none passed**",
        f"- Control AMLWorld post-128 H+X+TF val AUPRC: {ctrl_aml:.4f}",
        f"- Control PaySim type_only val AUPRC: {ctrl_ps['auprc']:.4f}",
        "",
        "## Gate results",
        "",
        "| Arm | Pass | PS AUPRC | ΔAUPRC | ΔF1 | Beat rand | AML AUPRC | AML regress |",
        "|-----|------|---------:|-------:|----:|-----------|----------:|------------:|",
    ]
    for r in results:
        lines.append(
            f"| `{r['arm']}` | {r['passed']} | {r['paysim_type_only_val_auprc']:.4f} | "
            f"{r['delta_vs_control_auprc']:+.4f} | {r['delta_vs_control_f1']:+.4f} | "
            f"{r['beats_random']} | {r['aml_hxxtf_val_auprc']:.4f} | {r['aml_auprc_regression']:+.4f} |"
        )
    lines.extend(["", "## Predeclared thresholds", "", json.dumps(gate["thresholds"], indent=2), ""])
    NOTES_MD.write_text("\n".join(lines) + "\n")
    logging.info("Wrote %s and %s", FINAL_JSON, NOTES_MD)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke")
    sm.set_defaults(func=cmd_smoke)
    tr = sub.add_parser("train")
    tr.add_argument("--arm", required=True, choices=["schema_mask_p025", "schema_mask_p050"])
    tr.set_defaults(func=cmd_train)
    ev = sub.add_parser("eval_arm")
    ev.add_argument("--arm", required=True, choices=list(ARMS.keys()))
    ev.add_argument("--batch_size", type=int, default=4096)
    ev.set_defaults(func=cmd_eval_arm)
    ag = sub.add_parser("aggregate")
    ag.set_defaults(func=cmd_aggregate)
    pg = sub.add_parser("write_predeclared_gate")
    pg.set_defaults(func=lambda _: write_predeclared_gate())
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
