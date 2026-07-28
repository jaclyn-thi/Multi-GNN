#!/usr/bin/env python3
"""Minimal joint AMLWorld–PaySim SSL replay scout (exploratory / post-hoc).

Arms (only intended difference = BN running-stat handling):
  shared_bn  — ordinary shared BN buffers across both domains
  domain_bn  — shared learned/affine weights; per-domain BN running_mean/var/num_batches_tracked

Each invocation: preflight → 2-step smoke → 500 joint SSL steps (250/domain, 1:1) →
frozen validation (AML pre-3h H+X+TF MLP; PaySim post-128 H logistic + H+X) → JSON/MD.

No labels in SSL. No test access. table_eligible=false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import to_hetero

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contrastive_loss import edge_identity_infonce_loss  # noqa: E402
from contrastive_projection import project_seed_pair, setup_contrastive_projection  # noqa: E402
from data_loading import get_data  # noqa: E402
from feature_contracts import CONTRACT_LEGACY  # noqa: E402
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
    select_shared_seed_edge_embeddings,
)
from training import _contrastive_view_kwargs, get_model  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

TAG = "joint_replay_scout"
RESULT = ROOT / "results" / "diagnostics" / TAG
CELLS = RESULT / "cells"
EMB_ROOT = ROOT / "embeddings" / TAG
SOURCE_UNIQUE = "gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2"
SOURCE_CKPT = ROOT / f"saved-models/checkpoint_{SOURCE_UNIQUE}.tar"
SOURCE_SHA = "18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6"
CONTRACT = CONTRACT_LEGACY
SEED = 2
TOTAL_STEPS = 500
STEPS_PER_DOMAIN = 250
SMOKE_STEPS = 2
BATCH_SIZE = 8192
NUM_NEIGHS = [100, 100]
TEMP = 0.5
N_NEG = 8192
LOGISTIC_SEED = 1
MLP_SEED = 2
MLP_EPOCHS = 15
MLP_LR = 1e-3
MLP_BS = 8192
TF_CACHE = ROOT / "results/cache/temporal_flow_causal/Small-HI"

REF = {
    "frozen_aml_auprc": 0.021012513105663634,
    "bn_only_auprc": 0.02325522079435137,
    "sequential_aml_init_auprc": 0.04329341608965788,
    "x_only_auprc": 0.004590890212575511,
    "matched_random_auprc": 0.011444637391742028,
    "aml_original_hxxtf_auprc": 0.5337913501667878,
    "sources": {
        "sequential_scout": "results/diagnostics/sequential_aml_to_paysim_ssl_scout.json",
        "x_only": "results/diagnostics/final_corrected_no_preserve_multiseed/cells/control_X_only_paysim_legacy_duplicate_v1.json",
        "random_H": "results/diagnostics/final_corrected_no_preserve_multiseed/cells/control_random_paysim_legacy_duplicate_v1.json",
    },
}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_hash(ids: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(ids, dtype=np.int64).reshape(-1)
    return {
        "n": int(a.shape[0]),
        "n_unique": int(np.unique(a).shape[0]),
        "edge_id_sum": int(a.sum()),
        "sha256_of_ids_bytes": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
    }


def seed_ids_sha(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous().numpy().astype(np.int64)
    return hashlib.sha256(a.tobytes()).hexdigest()


def verify_source() -> Dict[str, Any]:
    if not SOURCE_CKPT.is_file():
        raise SystemExit(f"missing source ckpt {SOURCE_CKPT}")
    sha = sha256_file(SOURCE_CKPT)
    if sha != SOURCE_SHA:
        raise SystemExit(f"source sha mismatch got={sha}")
    return {"path": str(SOURCE_CKPT), "sha256": sha, "unique": SOURCE_UNIQUE}


def is_bn_key(name: str) -> bool:
    return name.endswith(("running_mean", "running_var", "num_batches_tracked"))


def extract_bn(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in sd.items() if is_bn_key(k)}


def apply_bn_(module: nn.Module, bn: Dict[str, torch.Tensor]) -> None:
    sd = module.state_dict()
    for k, v in bn.items():
        if k not in sd:
            raise KeyError(f"BN key missing: {k}")
        sd[k] = v.to(sd[k].device, dtype=sd[k].dtype)
    module.load_state_dict(sd)


def collect_bn_bundle(model: nn.Module, proj: Optional[nn.Module]) -> Dict[str, Dict[str, torch.Tensor]]:
    out = {"model": extract_bn(model.state_dict())}
    if proj is not None:
        out["proj"] = extract_bn(proj.state_dict())
    return out


def apply_bn_bundle_(model: nn.Module, proj: Optional[nn.Module], bundle: Dict[str, Dict[str, torch.Tensor]]) -> None:
    apply_bn_(model, bundle["model"])
    if proj is not None and "proj" in bundle:
        apply_bn_(proj, bundle["proj"])


def clone_bn_bundle(b: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
    return {part: {k: v.clone() for k, v in sd.items()} for part, sd in b.items()}


def domain_schedule(n: int = TOTAL_STEPS) -> List[str]:
    # fixed 1:1: AML, PaySim, AML, PaySim, ...
    out = []
    for i in range(n):
        out.append("aml" if (i % 2 == 0) else "paysim")
    return out


def gin_cw() -> Dict[int, float]:
    args = create_parser().parse_args(["--data", "PaySim", "--model", "gin", "--testing"])
    return {0: float(extract_param("w_ce1", args)), 1: float(extract_param("w_ce2", args))}


def tune_thr(y: np.ndarray, proba: np.ndarray) -> float:
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
        "n": float(y.shape[0]),
        "n_positives": float(int(y.sum())),
        "positive_rate": float(y.mean()) if y.size else 0.0,
    }
    out.update(alert_budget_metrics(y, proba))
    return out


def failure_exit(arm: str, reason: str, detail: Dict[str, Any]) -> int:
    payload = {
        "ok": False,
        "arm": arm,
        "failure": reason,
        "detail": detail,
        "exploratory_posthoc": True,
        "table_eligible": False,
        "test_evaluated": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    write_json(RESULT / f"{arm}_failure.json", payload)
    (ROOT / "notes" / f"joint_replay_scout_{arm}.md").write_text(
        f"# Joint replay scout — {arm}\n\n**FAILED:** `{reason}`\n\n```json\n{json.dumps(detail, indent=2)}\n```\n",
        encoding="utf-8",
    )
    logging.error("FAIL %s: %s", arm, reason)
    return 2


def make_ns(data: str, unique: str) -> argparse.Namespace:
    argv = [
        "--data", data,
        "--model", "gin",
        "--testing",
        "--tqdm",
        "--objective", "contrastive",
        "--unique_name", unique,
        "--seed", str(SEED),
        "--batch_size", str(BATCH_SIZE),
        "--num_neighs", "100", "100",
        "--loader_num_workers", "0",
        "--reverse_mp", "--ego", "--ports", "--emlps", "--tds",
        "--correct_reverse_edge_features",
        "--contrast_projection_head",
        "--contrast_projection_hidden", "128",
        "--contrast_projection_dim", "128",
        "--contrastive_asymmetric",
        "--contrastive_num_neg_samples", str(N_NEG),
        "--contrastive_memory_bank_size", "0",
        "--contrastive_accum_steps", "1",
        "--contrastive_temperature", str(TEMP),
    ]
    if data == "PaySim":
        argv += ["--feature_contract", CONTRACT, "--train_fit_edge_znorm"]
    return create_parser().parse_args(argv)


def build_train_loader(tr_data: HeteroData, transform, *, generator_seed: int) -> LinkNeighborLoader:
    g = torch.Generator()
    g.manual_seed(int(generator_seed))
    edge_label_index = tr_data[FORWARD_EDGE_TYPE].edge_index
    edge_label = tr_data[FORWARD_EDGE_TYPE].y
    return LinkNeighborLoader(
        tr_data,
        num_neighbors=NUM_NEIGHS,
        edge_label_index=((FORWARD_EDGE_TYPE[0], FORWARD_EDGE_TYPE[1], FORWARD_EDGE_TYPE[2]), edge_label_index),
        edge_label=edge_label,
        batch_size=BATCH_SIZE,
        shuffle=True,
        transform=transform,
        num_workers=0,
        generator=g,
    )


def build_model(ns, te_data, sample_batch, device):
    config = SimpleNamespace(
        model="gin",
        n_hidden=extract_param("n_hidden", ns),
        n_gnn_layers=extract_param("n_gnn_layers", ns),
        n_heads=None,
        dropout=extract_param("dropout", ns),
        final_dropout=extract_param("final_dropout", ns),
    )
    model = get_model(sample_batch, config, ns)
    emb_dim = int(getattr(model, "embedding_dim", 128))
    model = to_hetero(model, te_data.metadata(), aggr="mean").to(device)
    return model, emb_dim


def infinite_loader(loader) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch


def contrastive_step(
    *,
    model: nn.Module,
    proj: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch,
    loader_data,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[float, torch.Tensor, str]:
    model.train()
    proj.train()
    seed_edge_ids = get_hetero_seed_edge_ids(batch, loader_data)
    attach_edge_id_from_batch(batch, loader_data)
    sid_hash = seed_ids_sha(seed_edge_ids)
    batch = batch.to(device)
    seed_edge_ids = seed_edge_ids.to(device)
    edge_drop_stats: Dict[str, Any] = {}
    view1, view2 = generate_views(
        batch,
        **_contrastive_view_kwargs(args, edge_drop_stats, seed_edge_ids=seed_edge_ids),
    )
    out1 = model(view1.x_dict, view1.edge_index_dict, view1.edge_attr_dict)
    z1 = out1[FORWARD_EDGE_TYPE]
    with torch.no_grad():
        out2 = model(view2.x_dict, view2.edge_index_dict, view2.edge_attr_dict)
        z2 = out2[FORWARD_EDGE_TYPE]
    z1_seed, seed_id1, z2_seed, seed_id2 = select_shared_seed_edge_embeddings(
        z1, view1[FORWARD_EDGE_TYPE].edge_id, z2, view2[FORWARD_EDGE_TYPE].edge_id, seed_edge_ids
    )
    z2_seed = z2_seed.detach()
    z1_con, z2_con = project_seed_pair(proj, z1_seed, z2_seed)
    loss = edge_identity_infonce_loss(
        z1_con,
        z2_con,
        seed_id1,
        seed_id2,
        temperature=TEMP,
        num_neg_samples=N_NEG,
        symmetric=False,
        memory_queue=None,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # gradient finite check
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grads += [p.grad for p in proj.parameters() if p.grad is not None]
    if not grads:
        raise RuntimeError("no gradients")
    gnorm = torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(proj.parameters()), 1e9)
    if not torch.isfinite(loss) or not torch.isfinite(gnorm):
        raise RuntimeError(f"non-finite loss/grad loss={loss} gnorm={gnorm}")
    optimizer.step()
    return float(loss.detach().cpu()), seed_edge_ids.detach().cpu(), sid_hash


def save_joint_ckpt(
    *,
    unique: str,
    model: nn.Module,
    proj: nn.Module,
    arm: str,
    bn_aml: Optional[Dict],
    bn_ps: Optional[Dict],
    model_dir: Path,
) -> Path:
    if unique == SOURCE_UNIQUE or SOURCE_CKPT.name == f"checkpoint_{unique}.tar":
        raise SystemExit("refusing to overwrite locked source checkpoint")
    blob = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
    blob["model_state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    blob["contrast_projection_state_dict"] = {k: v.detach().cpu() for k, v in proj.state_dict().items()}
    blob.pop("optimizer_state_dict", None)
    blob["joint_replay_meta"] = {
        "arm": arm,
        "domain_bn": arm == "domain_bn",
        "seed": SEED,
        "total_steps": TOTAL_STEPS,
    }
    if arm == "domain_bn":
        blob["bn_aml"] = {p: {k: v.cpu() for k, v in sd.items()} for p, sd in (bn_aml or {}).items()}
        blob["bn_paysim"] = {p: {k: v.cpu() for k, v in sd.items()} for p, sd in (bn_ps or {}).items()}
    dest = ROOT / "saved-models" / f"checkpoint_{unique}.tar"
    torch.save(blob, dest)
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dest, model_dir / dest.name)
    # also write domain BN sidecars
    if arm == "domain_bn":
        torch.save(blob["bn_aml"], model_dir / "bn_aml.pt")
        torch.save(blob["bn_paysim"], model_dir / "bn_paysim.pt")
    return dest


def extract_embeddings(
    *,
    data: str,
    unique: str,
    emb_subdir: str,
    representation_source: str,
    train_fit: bool,
    feature_contract: Optional[str],
) -> Path:
    import embedding_extraction as ee

    argv = [
        "--data", data,
        "--model", "gin",
        "--testing",
        "--tqdm",
        "--unique_name", unique,
        "--embeddings_dir", str(EMB_ROOT),
        "--embeddings_subdir", emb_subdir,
        "--batch_size", "4096",
        "--loader_num_workers", "0",
        "--num_neighs", "100", "100",
        "--representation_source", representation_source,
        "--extract_splits", "train,val",
        "--reverse_mp", "--ego", "--ports", "--tds", "--emlps",
        "--correct_reverse_edge_features",
        "--seed", str(SEED),
    ]
    if train_fit:
        argv.append("--train_fit_edge_znorm")
    if feature_contract:
        argv += ["--feature_contract", feature_contract]
    p = create_parser()
    p.add_argument("--embeddings_dir", type=str, default="embeddings")
    p.add_argument("--random_init", action="store_true")
    p.add_argument("--checkpoint_suffix", type=str, default="")
    p.add_argument("--embeddings_subdir", type=str, default=None)
    p.add_argument("--representation_source", type=str, default="post_embedding")
    p.add_argument("--extract_splits", type=str, default="train,val,test")
    ns = p.parse_args(argv)
    if "test" in ns.extract_splits.split(","):
        raise SystemExit("test extraction forbidden")
    set_seed(SEED)
    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)
    tr, va, te, tr_i, va_i, te_i = get_data(ns, data_config)
    out = Path(ee.run_embedding_extraction(tr, va, te, tr_i, va_i, te_i, ns, data_config))
    if (out / "test.npz").is_file():
        raise SystemExit(f"test.npz written under {out}")
    return out


def fit_logistic_h(z_tr, y_tr, z_va, y_va) -> Dict[str, Any]:
    set_seed(LOGISTIC_SEED)
    clf = LogisticRegression(
        class_weight=gin_cw(), max_iter=1000, random_state=LOGISTIC_SEED, solver="lbfgs", n_jobs=1, C=1.0
    )
    clf.fit(z_tr, y_tr)
    pva = clf.predict_proba(z_va)[:, 1]
    thr = tune_thr(y_va, pva)
    return {
        "learner": "LogisticRegression",
        "validation_metrics_at_0.5": metrics_block(y_va, pva, 0.5),
        "validation_metrics_at_val_optimal_f1": metrics_block(y_va, pva, thr),
        "val_auprc_at_0.5": float(average_precision_score(y_va, pva)),
    }


def fit_logistic_hx(z_tr, x_tr, y_tr, z_va, x_va, y_va) -> Dict[str, Any]:
    scaler = StandardScaler()
    tr = scaler.fit_transform(np.concatenate([z_tr, x_tr], axis=1)).astype(np.float32)
    va = scaler.transform(np.concatenate([z_va, x_va], axis=1)).astype(np.float32)
    set_seed(LOGISTIC_SEED)
    clf = LogisticRegression(
        class_weight=gin_cw(), max_iter=1000, random_state=LOGISTIC_SEED, solver="lbfgs", n_jobs=1, C=1.0
    )
    clf.fit(tr, y_tr)
    pva = clf.predict_proba(va)[:, 1]
    thr = tune_thr(y_va, pva)
    return {
        "learner": "LogisticRegression",
        "stack": "H+X",
        "validation_metrics_at_0.5": metrics_block(y_va, pva, 0.5),
        "validation_metrics_at_val_optimal_f1": metrics_block(y_va, pva, thr),
        "val_auprc_at_0.5": float(average_precision_score(y_va, pva)),
    }


def fit_mlp_hxxtf(mat_tr, y_tr, mat_va, y_va, device) -> Dict[str, Any]:
    scaler = StandardScaler()
    tr = scaler.fit_transform(mat_tr).astype(np.float32)
    va = scaler.transform(mat_va).astype(np.float32)
    torch.manual_seed(MLP_SEED)
    np.random.seed(MLP_SEED)
    model = PaperStyleMLP(tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    x_t = torch.from_numpy(tr)
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    n = tr.shape[0]
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
        pva = _predict_proba(model, va, batch_size=MLP_BS, device=device)
        auprc = float(average_precision_score(y_va, pva))
        if auprc > best_auprc + 1e-12:
            best_auprc = auprc
            best_ep = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(device)
    pva = _predict_proba(model, va, batch_size=MLP_BS, device=device)
    thr = tune_thr(y_va, pva)
    return {
        "learner": "PaperStyleMLP",
        "stack": "pre3h_H+X+TF",
        "best_epoch": best_ep,
        "best_val_auprc": best_auprc,
        "validation_metrics_at_0.5": metrics_block(y_va, pva, 0.5),
        "validation_metrics_at_val_optimal_f1": metrics_block(y_va, pva, thr),
        "val_auprc_at_0.5": float(average_precision_score(y_va, pva)),
        "val_f1_at_0.5": float(metrics_block(y_va, pva, 0.5)["f1"]),
    }


def load_x_and_tf_aml():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "probe_feature_ablation", ROOT / "scripts" / "probe_feature_ablation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["probe_feature_ablation"] = mod
    spec.loader.exec_module(mod)
    df, df_train, _, _, _, dspec = mod.load_dataset_frames("Small-HI", str(ROOT / "data_config.json"))
    y_all = df[dspec.label_col].to_numpy().astype(np.int64)
    x_raw, _, _, _ = mod.build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    tf = np.load(TF_CACHE / "features.npy").astype(np.float32)
    return x_raw.astype(np.float32), tf, y_all


def load_x_paysim():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "probe_feature_ablation", ROOT / "scripts" / "probe_feature_ablation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["probe_feature_ablation"] = mod
    spec.loader.exec_module(mod)
    df, df_train, _, _, _, _ = mod.load_dataset_frames("PaySim", str(ROOT / "data_config.json"))
    x_raw, names, _, _ = mod.build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    return x_raw.astype(np.float32), names


def run_arm(arm: str) -> int:
    assert arm in ("shared_bn", "domain_bn")
    t0 = time.perf_counter()
    RESULT.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(parents=True, exist_ok=True)
    EMB_ROOT.mkdir(parents=True, exist_ok=True)
    unique = f"joint_replay_scout_{arm}_seed2"
    model_dir = ROOT / "saved-models" / unique
    model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("CUDA not available — running on CPU (unexpected for Slurm GPU job)")

    try:
        source = verify_source()
    except SystemExit as e:
        return failure_exit(arm, "source_checkpoint", {"error": str(e)})

    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)

    # Load both domains (train graphs). Labels exist on edges but are unused in SSL.
    set_seed(SEED)
    ns_aml = make_ns("Small-HI", unique)
    ns_ps = make_ns("PaySim", unique)
    logging.info("Loading Small-HI…")
    aml_tr, aml_va, aml_te, aml_tr_i, aml_va_i, aml_te_i = get_data(ns_aml, data_config)
    logging.info("Loading PaySim…")
    ps_tr, ps_va, ps_te, ps_tr_i, ps_va_i, ps_te_i = get_data(ns_ps, data_config)

    transform = AddEgoIds()
    add_arange_ids([aml_tr, aml_va, aml_te])
    add_arange_ids([ps_tr, ps_va, ps_te])

    # Deterministic per-domain generators (same seeds in both arms)
    aml_loader = build_train_loader(aml_tr, transform, generator_seed=SEED * 1000 + 1)
    ps_loader = build_train_loader(ps_tr, transform, generator_seed=SEED * 1000 + 2)
    aml_iter = infinite_loader(aml_loader)
    ps_iter = infinite_loader(ps_loader)

    # Build model from AML sample (dims match corrected ports+tds)
    sample = next(iter(aml_loader))
    model, emb_dim = build_model(ns_aml, aml_te, sample, device)
    proj = setup_contrastive_projection(ns_aml, device, embedding_dim=emb_dim)
    # Strict load source weights
    src_blob = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
    incompat = model.load_state_dict(src_blob["model_state_dict"], strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        return failure_exit(arm, "strict_load_model", {"incompatible": str(incompat)})
    if "contrast_projection_state_dict" in src_blob:
        proj.load_state_dict(src_blob["contrast_projection_state_dict"], strict=True)
    model.to(device)
    proj.to(device)

    # Optimizer reset (fresh Adam on current params)
    lr = float(extract_param("lr", ns_aml))
    optimizer = torch.optim.Adam(list(model.parameters()) + list(proj.parameters()), lr=lr)
    logging.info("Optimizer reset: Adam lr=%s (no optimizer state from source)", lr)

    bn_aml = clone_bn_bundle(collect_bn_bundle(model, proj))
    bn_ps = clone_bn_bundle(bn_aml)

    schedule = domain_schedule(TOTAL_STEPS)
    assert schedule.count("aml") == STEPS_PER_DOMAIN and schedule.count("paysim") == STEPS_PER_DOMAIN

    seed_hash_log = {"aml": [], "paysim": []}
    losses = {"aml": [], "paysim": []}
    args_for_views = ns_aml  # same aug flags

    def maybe_swap(domain: str) -> None:
        if arm != "domain_bn":
            return
        apply_bn_bundle_(model, proj, bn_aml if domain == "aml" else bn_ps)

    def maybe_store(domain: str) -> None:
        if arm != "domain_bn":
            return
        snap = collect_bn_bundle(model, proj)
        if domain == "aml":
            bn_aml.clear()
            bn_aml.update(clone_bn_bundle(snap))
        else:
            bn_ps.clear()
            bn_ps.update(clone_bn_bundle(snap))

    # ---- internal smoke: first 2 optimizer steps ----
    try:
        for si in range(SMOKE_STEPS):
            domain = schedule[si]
            maybe_swap(domain)
            batch = next(aml_iter if domain == "aml" else ps_iter)
            loader_data = aml_tr if domain == "aml" else ps_tr
            loss, _, sid_h = contrastive_step(
                model=model, proj=proj, optimizer=optimizer, batch=batch,
                loader_data=loader_data, args=args_for_views, device=device,
            )
            maybe_store(domain)
            losses[domain].append(loss)
            if len(seed_hash_log[domain]) < 32:
                seed_hash_log[domain].append(sid_h)
        # save/reload smoke
        ckpt_path = save_joint_ckpt(
            unique=unique, model=model, proj=proj, arm=arm,
            bn_aml=bn_aml if arm == "domain_bn" else None,
            bn_ps=bn_ps if arm == "domain_bn" else None,
            model_dir=model_dir,
        )
        reload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(reload["model_state_dict"], strict=True)
        proj.load_state_dict(reload["contrast_projection_state_dict"], strict=True)
        model.to(device)
        proj.to(device)
        if arm == "domain_bn":
            bn_aml = clone_bn_bundle(reload["bn_aml"])
            bn_ps = clone_bn_bundle(reload["bn_paysim"])
        write_json(CELLS / f"{arm}_smoke_ok.json", {
            "ok": True, "steps": SMOKE_STEPS, "losses": losses, "ckpt": str(ckpt_path),
        })
    except Exception as e:
        return failure_exit(arm, "internal_smoke", {"error": str(e)})

    # Re-init optimizer after smoke reload (still "reset"; continue from smoke-adapted weights)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(proj.parameters()), lr=lr)

    # Continue remaining steps (already did 0..SMOKE_STEPS-1)
    logging.info("Continuing joint SSL from step %s → %s", SMOKE_STEPS, TOTAL_STEPS)
    for si in range(SMOKE_STEPS, TOTAL_STEPS):
        domain = schedule[si]
        maybe_swap(domain)
        batch = next(aml_iter if domain == "aml" else ps_iter)
        loader_data = aml_tr if domain == "aml" else ps_tr
        loss, _, sid_h = contrastive_step(
            model=model, proj=proj, optimizer=optimizer, batch=batch,
            loader_data=loader_data, args=args_for_views, device=device,
        )
        maybe_store(domain)
        losses[domain].append(loss)
        if len(seed_hash_log[domain]) < 32:
            seed_hash_log[domain].append(sid_h)
        if (si + 1) % 50 == 0:
            logging.info("step %s/%s domain=%s loss=%.4f", si + 1, TOTAL_STEPS, domain, loss)

    # Final BN: shared uses whatever is in model; domain keeps both
    if arm == "shared_bn":
        bn_aml = collect_bn_bundle(model, proj)
        bn_ps = clone_bn_bundle(bn_aml)
    else:
        # leave model with paysim BN for default save; aml stored separately
        apply_bn_bundle_(model, proj, bn_ps)

    ckpt_path = save_joint_ckpt(
        unique=unique, model=model, proj=proj, arm=arm,
        bn_aml=bn_aml, bn_ps=bn_ps, model_dir=model_dir,
    )
    logging.info("Saved %s", ckpt_path)

    # ---- frozen validation (labels OK now) ----
    # For domain_bn: write temporary AML-BN checkpoint for AML extraction
    eval_device = device
    aml_unique = unique
    if arm == "domain_bn":
        aml_unique = f"{unique}_amlbn"
        # apply AML BN into a copy checkpoint
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        apply_bn_(model, bn_aml["model"])
        if "proj" in bn_aml:
            apply_bn_(proj, bn_aml["proj"])
        blob["model_state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        blob["contrast_projection_state_dict"] = {k: v.detach().cpu() for k, v in proj.state_dict().items()}
        aml_ckpt = ROOT / "saved-models" / f"checkpoint_{aml_unique}.tar"
        torch.save(blob, aml_ckpt)
        shutil.copy2(aml_ckpt, model_dir / aml_ckpt.name)
        # restore paysim BN on live model
        apply_bn_bundle_(model, proj, bn_ps)

    logging.info("Extracting AML pre-3h…")
    emb_aml = extract_embeddings(
        data="Small-HI", unique=aml_unique, emb_subdir=f"{arm}/aml_pre3h",
        representation_source="pre_embedding_3h", train_fit=False, feature_contract=None,
    )
    # nested path
    if (emb_aml / "pre_embedding_3h").is_dir():
        emb_aml_pre = emb_aml / "pre_embedding_3h"
    else:
        emb_aml_pre = emb_aml

    logging.info("Extracting PaySim post-128…")
    emb_ps = extract_embeddings(
        data="PaySim", unique=unique, emb_subdir=f"{arm}/paysim_post128",
        representation_source="post_embedding", train_fit=True, feature_contract=CONTRACT,
    )

    # AML MLP H+X+TF
    z_tr, y_tr, ids_tr = load_embedding_npz(emb_aml_pre / "train.npz")
    z_va, y_va, ids_va = load_embedding_npz(emb_aml_pre / "val.npz")
    x_aml, tf_aml, _y_all = load_x_and_tf_aml()
    # align via edge ids
    mat_tr = np.concatenate([z_tr, x_aml[ids_tr], tf_aml[ids_tr]], axis=1)
    mat_va = np.concatenate([z_va, x_aml[ids_va], tf_aml[ids_va]], axis=1)
    # prefer embedding labels
    aml_mlp = fit_mlp_hxxtf(mat_tr, y_tr, mat_va, y_va, eval_device)
    aml_mlp["ids"] = {"train": ids_hash(ids_tr), "val": ids_hash(ids_va)}
    # optional H-only diagnostic logistic on pre3h
    aml_h = fit_logistic_h(z_tr, y_tr, z_va, y_va)

    # PaySim logistic H + H+X
    zp_tr, yp_tr, idp_tr = load_embedding_npz(emb_ps / "train.npz")
    zp_va, yp_va, idp_va = load_embedding_npz(emb_ps / "val.npz")
    x_ps, _ = load_x_paysim()
    ps_h = fit_logistic_h(zp_tr, yp_tr, zp_va, yp_va)
    ps_h["ids"] = {"train": ids_hash(idp_tr), "val": ids_hash(idp_va)}
    ps_hx = fit_logistic_hx(zp_tr, x_ps[idp_tr], yp_tr, zp_va, x_ps[idp_va], yp_va)

    peak_mem = None
    if device.type == "cuda":
        peak_mem = float(torch.cuda.max_memory_allocated() / (1024 ** 3))

    def _code_prov():
        files = [
            "scripts/joint_replay_scout.py",
            "training.py",
            "train_util.py",
            "graph_augmentations.py",
            "contrastive_losses.py",
        ]
        try:
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
        except Exception:
            head = ""
        return {
            "git_head": head,
            "source_file_sha256": {f: sha256_file(ROOT / f) for f in files if (ROOT / f).is_file()},
        }

    report = {
        "ok": True,
        "arm": arm,
        "unique_name": unique,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "command": f"python scripts/joint_replay_scout.py --arm {arm}",
        "exploratory_posthoc": True,
        "table_eligible": False,
        "test_evaluated": False,
        "test_accessed": False,
        "source_checkpoint": source,
        "optimizer_reset": True,
        "total_optimizer_steps": TOTAL_STEPS,
        "steps_per_domain": {"aml": STEPS_PER_DOMAIN, "paysim": STEPS_PER_DOMAIN},
        "schedule": "fixed_1to1_aml_then_paysim",
        "label_exclusion": {
            "ssl_uses_labels": False,
            "sampling_uses_labels": False,
            "stopping_uses_labels": False,
            "checkpoint_selection_uses_labels": False,
            "labels_only_after_freeze_for_val_probes": True,
        },
        "bn_behavior": {
            "mode": arm,
            "shared_learned_and_affine": True,
            "separate_running_stats": arm == "domain_bn",
            "note": "domain_bn updates only the active domain's BN buffers each step",
        },
        "seed_edge_id_hashes_first_32": seed_hash_log,
        "losses": {
            "aml_mean": float(np.mean(losses["aml"])) if losses["aml"] else None,
            "paysim_mean": float(np.mean(losses["paysim"])) if losses["paysim"] else None,
            "aml_last": losses["aml"][-1] if losses["aml"] else None,
            "paysim_last": losses["paysim"][-1] if losses["paysim"] else None,
            "n_aml": len(losses["aml"]),
            "n_paysim": len(losses["paysim"]),
        },
        "checkpoint": str(ckpt_path),
        "model_dir": str(model_dir),
        "embeddings": {"aml_pre3h": str(emb_aml_pre), "paysim_post128": str(emb_ps)},
        "validation": {
            "amlworld_pre3h_HxXTF_mlp": aml_mlp,
            "amlworld_pre3h_H_logistic_diagnostic": aml_h,
            "paysim_post128_H_logistic_primary": ps_h,
            "paysim_post128_HxX_logistic_secondary": ps_hx,
        },
        "references_not_recomputed": REF,
        "peak_gpu_mem_gb": peak_mem,
        "wall_sec": time.perf_counter() - t0,
        "code_provenance": _code_prov(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_contract_paysim": CONTRACT,
        "paysim_normalization": "train_fit_edge_znorm",
    }
    write_json(RESULT / f"{arm}.json", report)
    write_json(CELLS / f"{arm}_report.json", report)

    lines = [
        f"# Joint replay scout — `{arm}`",
        "",
        "> Exploratory / post-hoc. `table_eligible=false`. Validation only.",
        f"> Twin: `results/diagnostics/joint_replay_scout/{arm}.json`",
        "",
        f"- Job ID: `{report['job_id']}`",
        f"- Source ckpt SHA256: `{SOURCE_SHA}`",
        f"- Steps: {TOTAL_STEPS} (250 AML + 250 PaySim, 1:1)",
        f"- BN mode: **{arm}**",
        f"- Wall sec: {report['wall_sec']:.1f}; peak GPU GB: {peak_mem}",
        "",
        "## PaySim val (post-128 H logistic primary)",
        "",
        f"- H AUPRC@0.5: **{ps_h['val_auprc_at_0.5']:.6f}**",
        f"- H+X AUPRC@0.5: **{ps_hx['val_auprc_at_0.5']:.6f}**",
        f"- Refs: frozen={REF['frozen_aml_auprc']:.4f}, sequential_aml_init={REF['sequential_aml_init_auprc']:.4f}, "
        f"x_only={REF['x_only_auprc']:.4f}, random={REF['matched_random_auprc']:.4f}",
        "",
        "## AMLWorld val (pre-3h H+X+TF MLP)",
        "",
        f"- AUPRC@0.5: **{aml_mlp['val_auprc_at_0.5']:.6f}**  F1@0.5: **{aml_mlp['val_f1_at_0.5']:.6f}**",
        f"- Ref original: {REF['aml_original_hxxtf_auprc']:.4f}",
        "",
        f"- Seed-edge hash count logged: aml={len(seed_hash_log['aml'])} paysim={len(seed_hash_log['paysim'])}",
        "",
    ]
    (ROOT / "notes" / f"joint_replay_scout_{arm}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("DONE arm=%s → %s", arm, RESULT / f"{arm}.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=["shared_bn", "domain_bn"])
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logger_setup()
    args = build_parser().parse_args(argv)
    return run_arm(args.arm)


if __name__ == "__main__":
    raise SystemExit(main())
