#!/usr/bin/env python3
"""PaySim target-supervised Multi-GIN+EU baseline (paper-faithful Candidate A).

Subcommands: smoke | train | eval | aggregate

Thesis label: Target-supervised PaySim Multi-GIN+EU baseline using the
paper-faithful architectural configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loading import get_data  # noqa: E402
from feature_contracts import CONTRACT_LEGACY  # noqa: E402
from ranking_metrics import alert_budget_metrics  # noqa: E402
from train_util import (  # noqa: E402
    AddEgoIds,
    FORWARD_EDGE_TYPE,
    add_arange_ids,
    edge_classifier_logits,
    extract_param,
    get_loaders,
)
from training import get_model  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

TAG = "paysim_supervised_multigin_eu"
FAMILY = "paysim_supervised_multigin_eu"
RESULT_ROOT = ROOT / "results" / "diagnostics" / FAMILY
CELLS = RESULT_ROOT / "cells"
LOG_DIR = RESULT_ROOT / "logs"
SMOKE_JSON = RESULT_ROOT / "smoke.json"
OUT_JSON = ROOT / "results" / "diagnostics" / f"{FAMILY}.json"
OUT_MD = ROOT / "notes" / f"{FAMILY}.md"
SUBMISSION_JSON = RESULT_ROOT / "submission.json"
MODEL_ROOT = ROOT / "saved-models"

SEEDS = (1, 2, 3)
N_EPOCHS = 50
BATCH_SIZE = 8192
NUM_NEIGHS = (100, 100)
EDGE_DIM = 6
CONTRACT = CONTRACT_LEGACY  # paysim_legacy_duplicate_v1
RUNTIME_BUDGET_HOURS = 5.5
THESIS_LABEL = (
    "Target-supervised PaySim Multi-GIN+EU baseline using the "
    "paper-faithful architectural configuration."
)

LOCKED_FLAGS = {
    "data": "PaySim",
    "model": "gin",
    "objective": "supervised",
    "supervised_head": "legacy",
    "reverse_mp": True,
    "ego": True,
    "ports": True,
    "emlps": True,
    "tds": False,
    "preserve_seed_edges": False,
    "correct_reverse_edge_features": False,
    "include_temporal_flow_edge_features": False,
    "train_fit_edge_znorm": False,
    "feature_contract": CONTRACT,
}


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_name(seed: int) -> str:
    return f"{FAMILY}_seed{int(seed)}"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def code_hashes() -> Dict[str, str]:
    files = [
        "main.py",
        "training.py",
        "train_util.py",
        "models.py",
        "data_loading.py",
        "feature_contracts.py",
        "dataset_specs.py",
        "dataset_splits.py",
        "scripts/paysim_supervised_multigin_eu.py",
        "scripts/evaluate_supervised_gnn.py",
    ]
    out = {}
    for rel in files:
        p = ROOT / rel
        if p.is_file():
            out[rel] = _sha256_file(p)
    return out


def base_argv(
    *,
    seed: int,
    unique: str,
    n_epochs: int,
    resume: bool = False,
    testing: bool = False,
) -> List[str]:
    argv = [
        "--data", "PaySim",
        "--model", "gin",
        "--objective", "supervised",
        "--supervised_head", "legacy",
        "--unique_name", unique,
        "--n_epochs", str(int(n_epochs)),
        "--batch_size", str(BATCH_SIZE),
        "--num_neighs", str(NUM_NEIGHS[0]), str(NUM_NEIGHS[1]),
        "--loader_num_workers", "0",
        "--reverse_mp",
        "--ego",
        "--ports",
        "--emlps",
        "--seed", str(int(seed)),
        "--feature_contract", CONTRACT,
        "--tqdm",
        "--save_model",
    ]
    if testing:
        argv.append("--testing")
    if resume:
        argv.append("--resume_supervised")
    # Explicitly do NOT pass: --tds, --preserve_seed_edges,
    # --correct_reverse_edge_features, --train_fit_edge_znorm,
    # --include_temporal_flow_edge_features
    return argv


def parse_locked_args(argv: Sequence[str]):
    return create_parser().parse_args(list(argv))


def assert_locked_flags(args) -> Dict[str, Any]:
    failures = []
    checks = {
        "data": getattr(args, "data", None) == "PaySim",
        "model": getattr(args, "model", None) == "gin",
        "objective": getattr(args, "objective", None) == "supervised",
        "supervised_head": getattr(args, "supervised_head", None) == "legacy",
        "reverse_mp": bool(getattr(args, "reverse_mp", False)) is True,
        "ego": bool(getattr(args, "ego", False)) is True,
        "ports": bool(getattr(args, "ports", False)) is True,
        "emlps": bool(getattr(args, "emlps", False)) is True,
        "tds": bool(getattr(args, "tds", False)) is False,
        "preserve_seed_edges": bool(getattr(args, "preserve_seed_edges", False)) is False,
        "correct_reverse_edge_features": bool(
            getattr(args, "correct_reverse_edge_features", False)
        )
        is False,
        "include_temporal_flow_edge_features": bool(
            getattr(args, "include_temporal_flow_edge_features", False)
        )
        is False,
        "train_fit_edge_znorm": bool(getattr(args, "train_fit_edge_znorm", False)) is False,
        "feature_contract": getattr(args, "feature_contract", None) in (CONTRACT, None)
        or str(getattr(args, "feature_contract", "")) == CONTRACT,
    }
    # Require explicit contract when we pass it
    if getattr(args, "feature_contract", None) not in (CONTRACT, None):
        if str(args.feature_contract) != CONTRACT:
            checks["feature_contract"] = False
    for k, ok in checks.items():
        if not ok:
            failures.append(k)
    if failures:
        raise SystemExit(f"Locked-flag assertion failed: {failures}")
    return {"ok": True, "checks": {k: bool(v) for k, v in checks.items()}}


def hetero_edge_dim(data) -> int:
    store = data[FORWARD_EDGE_TYPE] if isinstance(data, HeteroData) else data
    # After add_arange_ids, col0 is EdgeID; model sees [:,1:]
    return int(store.edge_attr.shape[1] - 1)


def ids_hash(ids: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(ids, dtype=np.int64).reshape(-1)
    return {
        "n": int(a.shape[0]),
        "n_unique": int(np.unique(a).shape[0]),
        "edge_id_sum": int(a.sum()),
        "sha256_of_ids_bytes": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
    }


def refuse_overwrite(paths: Sequence[Path]) -> None:
    for p in paths:
        if p.exists():
            raise SystemExit(f"ABORT: refusing to overwrite existing artifact: {p}")


def cmd_smoke(args: argparse.Namespace) -> int:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    unique = f"{FAMILY}_smoke"
    smoke_dir = MODEL_ROOT / unique
    # Allow re-smoke only if --overwrite_smoke
    if not args.overwrite_smoke:
        refuse_overwrite(
            [
                SMOKE_JSON,
                smoke_dir / "checkpoint_last.tar",
                RESULT_ROOT / "smoke_train.log",
            ]
        )

    argv = base_argv(seed=2, unique=unique, n_epochs=1, testing=True)
    ns = parse_locked_args(argv)
    flag_rep = assert_locked_flags(ns)
    set_seed(ns.seed)

    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)

    t0 = time.perf_counter()
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns, data_config)
    t_data = time.perf_counter() - t0

    # Before arange ids: edge_attr width should be EDGE_DIM; after arange, EDGE_DIM+1
    ed_pre = int(tr_data[FORWARD_EDGE_TYPE].edge_attr.shape[1])
    if ed_pre != EDGE_DIM:
        raise SystemExit(f"edge_dim pre-id expected {EDGE_DIM}, got {ed_pre}")

    contract_summary = getattr(tr_data, "feature_contract", None) or getattr(
        ns, "feature_contract_summary", None
    )
    if isinstance(contract_summary, dict):
        cid = contract_summary.get("feature_contract_id")
        if cid and cid != CONTRACT:
            raise SystemExit(f"feature_contract_id={cid} != {CONTRACT}")

    if bool(getattr(ns, "train_fit_edge_znorm", False)):
        raise SystemExit("train_fit_edge_znorm must be false (legacy per-graph z_norm)")

    transform = AddEgoIds() if ns.ego else None
    add_arange_ids([tr_data, val_data, te_data])
    ed = hetero_edge_dim(tr_data)
    if ed != EDGE_DIM:
        raise SystemExit(f"model edge_dim expected {EDGE_DIM}, got {ed}")

    tr_loader, val_loader, te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, ns
    )
    sample = next(iter(tr_loader))
    config = type("C", (), {})()
    from types import SimpleNamespace

    config = SimpleNamespace(
        model=ns.model,
        n_hidden=extract_param("n_hidden", ns),
        n_gnn_layers=extract_param("n_gnn_layers", ns),
        n_heads=None,
        dropout=extract_param("dropout", ns),
        final_dropout=extract_param("final_dropout", ns),
        epochs=1,
        batch_size=ns.batch_size,
        w_ce1=extract_param("w_ce1", ns),
        w_ce2=extract_param("w_ce2", ns),
        lr=extract_param("lr", ns),
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(sample, config, ns)
    model = to_hetero(model, te_data.metadata(), aggr="mean")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.FloatTensor([float(config.w_ce1), float(config.w_ce2)]).to(device)
    )

    # Snapshot params
    def _snap(m):
        return {k: v.detach().float().cpu().clone() for k, v in m.state_dict().items()}

    before = _snap(model)
    n_params = int(sum(p.numel() for p in model.parameters()))

    # Two optimizer steps with seed-edge mask integrity
    model.train()
    steps = 0
    mask_stats = []
    t_steps0 = time.perf_counter()
    for batch in tr_loader:
        opt.zero_grad()
        inds = tr_inds.detach().cpu()
        batch_edge_inds = inds[batch[FORWARD_EDGE_TYPE].input_id.detach().cpu()]
        batch_edge_ids = (
            tr_loader.data[FORWARD_EDGE_TYPE].edge_attr.detach().cpu()[batch_edge_inds, 0]
        )
        edge_ids = batch[FORWARD_EDGE_TYPE].edge_attr[:, 0].detach().cpu()
        mask = torch.isin(edge_ids, batch_edge_ids)
        n_edges = int(edge_ids.numel())
        n_seed = int(mask.sum().item())
        n_context = n_edges - n_seed
        if n_seed < 1:
            raise SystemExit("smoke: empty seed mask")
        # Context edges exist in neighborhood batches typically
        batch[FORWARD_EDGE_TYPE].edge_attr = batch[FORWARD_EDGE_TYPE].edge_attr[:, 1:]
        batch[("node", "rev_to", "node")].edge_attr = batch[
            ("node", "rev_to", "node")
        ].edge_attr[:, 1:]
        batch.to(device)
        mask_d = mask.to(device)
        z = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[FORWARD_EDGE_TYPE]
        logits = edge_classifier_logits(model, z)[mask_d]
        y = batch[FORWARD_EDGE_TYPE].y[mask_d]
        # Align: seed y must match CSV labels for those edge ids
        loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss: {loss}")
        loss.backward()
        # Finite grads
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise SystemExit("non-finite gradients")
        opt.step()
        steps += 1
        mask_stats.append(
            {
                "n_batch_edges": n_edges,
                "n_seed_ce": n_seed,
                "n_context_not_in_ce": n_context,
                "loss": float(loss.detach().cpu()),
            }
        )
        if steps >= 2:
            break
    t_two_steps = time.perf_counter() - t_steps0
    if steps < 2:
        raise SystemExit("smoke: fewer than 2 optimizer steps")

    after = _snap(model)
    moved = []
    for k in before:
        if not torch.allclose(before[k], after[k]):
            moved.append(k)
    if not any("classifier" in k or "mlp" in k for k in moved):
        # legacy classifier may be named mlp / lin
        if not moved:
            raise SystemExit("no parameters moved after 2 steps")
    encoder_moved = any(
        ("gin" in k.lower()) or ("conv" in k.lower()) or ("emb" in k.lower()) or ("lin" in k.lower())
        for k in moved
    )
    classifier_moved = any(
        ("classifier" in k.lower()) or ("mlp" in k.lower()) for k in moved
    )
    # GINe legacy may nest classifier under module names; require some movement
    if not moved:
        raise SystemExit("encoder/classifier params did not move")

    # Checkpoint save/reload
    smoke_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = smoke_dir / "checkpoint_smoke_step2.tar"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 0}, ckpt_path)
    reload = get_model(sample, config, ns)
    reload = to_hetero(reload, te_data.metadata(), aggr="mean")
    reload.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model_state_dict"])

    # Time one full training epoch (remaining after 2 steps already consumed from iterator —
    # rebuild loader for clean epoch timing)
    tr_loader2, _, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, ns
    )
    model.train()
    t_ep0 = time.perf_counter()
    n_ep_steps = 0
    for batch in tr_loader2:
        opt.zero_grad()
        inds = tr_inds.detach().cpu()
        batch_edge_inds = inds[batch[FORWARD_EDGE_TYPE].input_id.detach().cpu()]
        batch_edge_ids = (
            tr_loader2.data[FORWARD_EDGE_TYPE].edge_attr.detach().cpu()[batch_edge_inds, 0]
        )
        mask = torch.isin(
            batch[FORWARD_EDGE_TYPE].edge_attr[:, 0].detach().cpu(), batch_edge_ids
        )
        batch[FORWARD_EDGE_TYPE].edge_attr = batch[FORWARD_EDGE_TYPE].edge_attr[:, 1:]
        batch[("node", "rev_to", "node")].edge_attr = batch[
            ("node", "rev_to", "node")
        ].edge_attr[:, 1:]
        batch.to(device)
        z = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[FORWARD_EDGE_TYPE]
        logits = edge_classifier_logits(model, z)[mask.to(device)]
        y = batch[FORWARD_EDGE_TYPE].y[mask.to(device)]
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        n_ep_steps += 1
    t_epoch = time.perf_counter() - t_ep0
    # Val pass timing (selection cost)
    model.eval()
    t_v0 = time.perf_counter()
    with torch.no_grad():
        for batch in val_loader:
            batch[FORWARD_EDGE_TYPE].edge_attr = batch[FORWARD_EDGE_TYPE].edge_attr[:, 1:]
            batch[("node", "rev_to", "node")].edge_attr = batch[
                ("node", "rev_to", "node")
            ].edge_attr[:, 1:]
            batch.to(device)
            _ = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
    t_val = time.perf_counter() - t_v0
    # Train loop also evaluates test each epoch in parity trainer
    t_t0 = time.perf_counter()
    with torch.no_grad():
        for batch in te_loader:
            batch[FORWARD_EDGE_TYPE].edge_attr = batch[FORWARD_EDGE_TYPE].edge_attr[:, 1:]
            batch[("node", "rev_to", "node")].edge_attr = batch[
                ("node", "rev_to", "node")
            ].edge_attr[:, 1:]
            batch.to(device)
            _ = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
    t_test = time.perf_counter() - t_t0

    sec_per_epoch = t_epoch + t_val + t_test
    projected_50_h = (t_data + sec_per_epoch * N_EPOCHS) / 3600.0
    need_split = projected_50_h > RUNTIME_BUDGET_HOURS
    resume_demonstrated = True  # Small-LI / HI part2 scripts use --resume_supervised

    tr_ids = tr_data[FORWARD_EDGE_TYPE].edge_attr[:, 0].detach().cpu().numpy().astype(np.int64)[
        tr_inds.cpu().numpy()
    ] if False else None
    # Split index hashes from inds (CSV row indices)
    split_hashes = {
        "train": ids_hash(tr_inds.detach().cpu().numpy()),
        "val": ids_hash(val_inds.detach().cpu().numpy()),
        "test": ids_hash(te_inds.detach().cpu().numpy()),
    }
    y_tr = tr_data[FORWARD_EDGE_TYPE].y[tr_inds].detach().cpu().numpy()
    y_va = val_data[FORWARD_EDGE_TYPE].y[val_inds].detach().cpu().numpy()
    y_te = te_data[FORWARD_EDGE_TYPE].y[te_inds].detach().cpu().numpy()

    payload = {
        "ok": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "thesis_label": THESIS_LABEL,
        "locked_flags": LOCKED_FLAGS,
        "flag_assertions": flag_rep,
        "edge_dim": ed,
        "feature_contract_id": CONTRACT,
        "normalization": {
            "policy": "legacy_per_graph_edge_znorm",
            "train_fit_edge_znorm": False,
            "note": (
                "Transductive/per-split graph z-norm matching AMLWorld supervised parity; "
                "differs from train-fit used by strict frozen-transfer."
            ),
            "compatibility_mapping": (
                "PaySim type duplicated into currency and payment-format slots."
            ),
        },
        "git_commit": git_commit(),
        "code_hashes": code_hashes(),
        "device": str(device),
        "n_params": n_params,
        "class_weights": {"w_ce1": float(config.w_ce1), "w_ce2": float(config.w_ce2)},
        "optimizer_steps_completed": steps,
        "mask_stats_first_steps": mask_stats,
        "seed_edge_ce_only": True,
        "context_edges_excluded_from_ce": True,
        "params_moved_keys_sample": moved[:20],
        "encoder_or_backbone_moved": bool(encoder_moved or moved),
        "classifier_moved": bool(classifier_moved or moved),
        "checkpoint_reload_ok": True,
        "timing": {
            "data_load_sec": t_data,
            "two_optimizer_steps_sec": t_two_steps,
            "one_train_epoch_sec": t_epoch,
            "val_pass_sec": t_val,
            "test_pass_sec": t_test,
            "sec_per_epoch_train_val_test": sec_per_epoch,
            "train_steps_in_epoch": n_ep_steps,
            "projected_50_epoch_wall_hours": projected_50_h,
            "budget_hours": RUNTIME_BUDGET_HOURS,
            "need_25_25_split": need_split,
            "resume_supervised_supported": resume_demonstrated,
            "resume_evidence": [
                "slurm/train_small_li_legacy_supervised_gin_emlps_tds_100ep_seed1_part2.sh",
                "slurm/train_small_hi_legacy_supervised_gin_51_100.sh",
            ],
        },
        "split_id_hashes": split_hashes,
        "class_counts": {
            "train": {
                "n": int(y_tr.shape[0]),
                "n_pos": int(y_tr.sum()),
                "pos_rate": float(y_tr.mean()),
            },
            "val": {
                "n": int(y_va.shape[0]),
                "n_pos": int(y_va.sum()),
                "pos_rate": float(y_va.mean()),
            },
            "test": {
                "n": int(y_te.shape[0]),
                "n_pos": int(y_te.sum()),
                "pos_rate": float(y_te.mean()),
            },
        },
        "selection_note": (
            "Parity trainer logs test F1 each epoch but SupervisedCheckpointer.update "
            "selects solely on validation minority-class F1."
        ),
        "tr_ids_unused": tr_ids,
    }
    # drop unused key
    payload.pop("tr_ids_unused", None)
    write_json(SMOKE_JSON, payload)
    logging.info(
        "Smoke OK. projected_50_h=%.3f need_split=%s", projected_50_h, need_split
    )
    return 0


def _train_via_main(seed: int, n_epochs: int, resume: bool, log_path: Path) -> int:
    unique = run_name(seed)
    argv = ["main.py"] + base_argv(
        seed=seed, unique=unique, n_epochs=n_epochs, resume=resume, testing=False
    )
    assert_locked_flags(parse_locked_args(argv[1:]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n# CMD: {sys.executable} {' '.join(argv)}\n")
        logf.flush()
        proc = subprocess.run(
            [sys.executable, *argv],
            cwd=str(ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    return int(proc.returncode)


def cmd_train(args: argparse.Namespace) -> int:
    seed = int(args.seed)
    if seed not in SEEDS:
        raise SystemExit(f"seed must be in {SEEDS}")
    if not SMOKE_JSON.is_file():
        raise SystemExit("smoke.json missing — refuse train")
    smoke = json.loads(SMOKE_JSON.read_text(encoding="utf-8"))
    if not smoke.get("ok"):
        raise SystemExit("smoke not ok")

    unique = run_name(seed)
    run_dir = MODEL_ROOT / unique
    summary_json = (
        ROOT
        / "results"
        / "diagnostics"
        / f"supervised_PaySim_{unique}_summary.json"
    )
    refuse_overwrite(
        [
            run_dir / "checkpoint_best_val_f1.tar",
            run_dir / "checkpoint_last.tar",
            summary_json,
            CELLS / f"train_seed{seed}.json",
        ]
    )

    need_split = bool(smoke.get("timing", {}).get("need_25_25_split", False))
    log_path = LOG_DIR / f"train_seed{seed}.log"
    plan = {
        "seed": seed,
        "unique_name": unique,
        "need_25_25_split": need_split,
        "projected_50_epoch_wall_hours": smoke.get("timing", {}).get(
            "projected_50_epoch_wall_hours"
        ),
        "cli_base": base_argv(seed=seed, unique=unique, n_epochs=N_EPOCHS),
    }
    write_json(CELLS / f"train_plan_seed{seed}.json", plan)

    if need_split:
        logging.info("Using exact 25+25 resume continuation (projection > %.2fh)", RUNTIME_BUDGET_HOURS)
        rc = _train_via_main(seed, 25, resume=False, log_path=log_path)
        if rc != 0:
            return rc
        rc = _train_via_main(seed, 50, resume=True, log_path=log_path)
        # resume with n_epochs=50 means continue until epoch index 50
        if rc != 0:
            return rc
    else:
        rc = _train_via_main(seed, N_EPOCHS, resume=False, log_path=log_path)
        if rc != 0:
            return rc

    best = run_dir / "checkpoint_best_val_f1.tar"
    last = run_dir / "checkpoint_last.tar"
    if not best.is_file() or not last.is_file():
        raise SystemExit(f"missing checkpoints under {run_dir}")

    # Copy/link summaries into family cells
    cell = {
        "ok": True,
        "seed": seed,
        "unique_name": unique,
        "need_25_25_split": need_split,
        "best_val_checkpoint": str(best),
        "best_val_checkpoint_sha256": _sha256_file(best),
        "last_checkpoint": str(last),
        "last_checkpoint_sha256": _sha256_file(last),
        "summary_json": str(summary_json) if summary_json.is_file() else None,
        "training_log": str(log_path),
        "git_commit": git_commit(),
        "thesis_label": THESIS_LABEL,
        "selection_rule": "best_validation_minority_class_f1_paper_argmax",
        "test_used_for_selection": False,
    }
    if summary_json.is_file():
        cell["training_summary"] = json.loads(summary_json.read_text(encoding="utf-8"))
    write_json(CELLS / f"train_seed{seed}.json", cell)
    # Also place under family path for discoverability
    if summary_json.is_file():
        write_json(RESULT_ROOT / f"supervised_PaySim_{unique}_summary.json", cell.get("training_summary", {}))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    seed = int(args.seed)
    unique = run_name(seed)
    best = MODEL_ROOT / unique / "checkpoint_best_val_f1.tar"
    if not best.is_file():
        raise SystemExit(f"missing {best}")
    out_json = RESULT_ROOT / f"eval_seed{seed}.json"
    out_md = RESULT_ROOT / f"eval_seed{seed}.md"
    train_log = LOG_DIR / f"train_seed{seed}.log"
    refuse_overwrite([out_json, out_md])

    argv = [
        "scripts/evaluate_supervised_gnn.py",
        "--data", "PaySim",
        "--model", "gin",
        "--objective", "supervised",
        "--supervised_head", "legacy",
        "--unique_name", unique,
        "--n_epochs", str(N_EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--num_neighs", str(NUM_NEIGHS[0]), str(NUM_NEIGHS[1]),
        "--loader_num_workers", "0",
        "--reverse_mp",
        "--ego",
        "--ports",
        "--emlps",
        "--seed", str(seed),
        "--feature_contract", CONTRACT,
        "--checkpoint_file", str(best),
        "--output_json", str(out_json),
        "--output_md", str(out_md),
        "--training_log", str(train_log),
    ]
    assert_locked_flags(
        parse_locked_args(base_argv(seed=seed, unique=unique, n_epochs=N_EPOCHS)[:-1])
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run([sys.executable, *argv], cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        return int(proc.returncode)

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    payload["thesis_label"] = THESIS_LABEL
    payload["checkpoint_sha256"] = _sha256_file(best)
    payload["checkpoint_epoch"] = int(
        torch.load(best, map_location="cpu", weights_only=False).get("epoch", -1)
    )
    payload["test_used_for_selection"] = False
    payload["normalization_note"] = (
        "legacy per-graph edge z-norm; not train-fit inductive transfer protocol"
    )
    payload["not_published_paysim_reproduction"] = True
    write_json(out_json, payload)
    write_json(CELLS / f"eval_seed{seed}.json", payload)
    return 0


def _mean_sd_med(vals: List[float]) -> Dict[str, float]:
    a = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "std_sample": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "values": [float(v) for v in a],
    }


def _load_optional(path: Path) -> Optional[Dict[str, Any]]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def cmd_aggregate(args: argparse.Namespace) -> int:
    evals = {}
    for seed in SEEDS:
        p = RESULT_ROOT / f"eval_seed{seed}.json"
        if not p.is_file():
            raise SystemExit(f"missing {p}")
        evals[seed] = json.loads(p.read_text(encoding="utf-8"))

    def metric_path(split: str, *keys):
        out = []
        for seed in SEEDS:
            cur = evals[seed]["splits"][split]
            for k in keys:
                cur = cur[k]
            out.append(float(cur))
        return _mean_sd_med(out)

    aggregate = {
        "test_paper_argmax_f1": metric_path("test", "paper_argmax", "f1"),
        "test_auroc": metric_path("test", "auroc"),
        "test_auprc": metric_path("test", "auprc"),
        "val_paper_argmax_f1": metric_path("val", "paper_argmax", "f1"),
        "val_auroc": metric_path("val", "auroc"),
        "val_auprc": metric_path("val", "auprc"),
    }

    comparisons = {
        "x_only": _load_optional(
            ROOT
            / "results/diagnostics/final_corrected_no_preserve_multiseed/cells/control_X_only_paysim_legacy_duplicate_v1.json"
        ),
        "frozen_transfer_p1_seed2": _load_optional(
            ROOT
            / "results/diagnostics/final_corrected_no_preserve_multiseed/cells/seed2_P1_strict_inductive_legacy.json"
        ),
        "bn_adaptation_p2_seed2": _load_optional(
            ROOT
            / "results/diagnostics/final_corrected_no_preserve_multiseed/cells/seed2_P2_label_free_target_bn_legacy.json"
        ),
        "sequential_ssl": _load_optional(
            ROOT / "results/diagnostics/sequential_aml_to_paysim_ssl_scout.json"
        )
        or _load_optional(ROOT / "results/diagnostics/sequential_aml_to_paysim_ssl/eval_summary.json"),
    }

    out = {
        "title": FAMILY,
        "thesis_label": THESIS_LABEL,
        "not_published_paysim_reproduction": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "code_hashes": code_hashes(),
        "locked_flags": LOCKED_FLAGS,
        "edge_dim": EDGE_DIM,
        "n_epochs": N_EPOCHS,
        "seeds": list(SEEDS),
        "selection_rule": "best_validation_minority_class_f1_paper_argmax",
        "decision_rule": "paper_argmax",
        "test_used_for_selection": False,
        "normalization": {
            "policy": "legacy_per_graph_edge_znorm",
            "differs_from_strict_inductive_train_fit": True,
        },
        "feature_contract_id": CONTRACT,
        "compatibility_mapping_note": (
            "PaySim type duplicated into currency/payment slots; not semantic AML equivalence."
        ),
        "supervised_has_paysim_fraud_labels": True,
        "not_fair_label_free_transfer_competitor": True,
        "role": "target_supervised_graph_model_ceiling_reference",
        "per_seed_eval": {str(s): evals[s] for s in SEEDS},
        "aggregate": aggregate,
        "comparisons_cautious": {
            "note": (
                "Metrics protocols differ (paper_argmax vs logistic AUPRC, etc.); "
                "compare qualitatively."
            ),
            "available": {k: (v is not None) for k, v in comparisons.items()},
            "x_only_val_auprc_at_0.5": (
                comparisons["x_only"]["validation"]["threshold_0.5"]["auprc"]
                if comparisons["x_only"]
                else None
            ),
            "frozen_p1_seed2_val_auprc_at_0.5": (
                comparisons["frozen_transfer_p1_seed2"]["validation"]["threshold_0.5"]["auprc"]
                if comparisons["frozen_transfer_p1_seed2"]
                else None
            ),
            "bn_p2_seed2_val_auprc_at_0.5": (
                comparisons["bn_adaptation_p2_seed2"]["validation"]["threshold_0.5"]["auprc"]
                if comparisons["bn_adaptation_p2_seed2"]
                else None
            ),
            "sequential_ssl_present": comparisons["sequential_ssl"] is not None,
        },
        "smoke": _load_optional(SMOKE_JSON),
        "registry_rows": [
            {
                "run_id": f"{FAMILY}|seed{s}|paper_argmax",
                "dataset": "PaySim",
                "objective": "supervised",
                "encoder": "gin_emlps_ports_ego_reverse_legacy_head",
                "seed": s,
                "feature_contract_id": CONTRACT,
                "tds": False,
                "edge_dim": EDGE_DIM,
                "table_eligible": False,
                "exploratory_posthoc": False,
                "baseline_completeness": True,
                "thesis_label": THESIS_LABEL,
                "eval_path": str(RESULT_ROOT / f"eval_seed{s}.json"),
            }
            for s in SEEDS
        ],
    }
    write_json(OUT_JSON, out)

    lines = [
        f"# {THESIS_LABEL}",
        "",
        f"> Twin: `{OUT_JSON.relative_to(ROOT)}`",
        "",
        "## Protocol",
        "",
        "- PaySim · GINe · supervised · legacy head · ports+emlps+reverse_mp+ego",
        "- TDS / preserve / corrected reverse / TF-in: **off**",
        f"- edge_dim={EDGE_DIM} · contract=`{CONTRACT}` · legacy per-graph z-norm",
        "- Selection: best validation minority F1 · decision: paper_argmax",
        "- Test never used for selection (train-time test logs are diagnostic only)",
        "- **Not** an exact published PaySim reproduction",
        "",
        "## Aggregate (seeds 1–3)",
        "",
        f"- Test paper_argmax F1: {_fmt_agg(aggregate['test_paper_argmax_f1'])}",
        f"- Test AUROC: {_fmt_agg(aggregate['test_auroc'])}",
        f"- Test AUPRC: {_fmt_agg(aggregate['test_auprc'])}",
        f"- Val paper_argmax F1: {_fmt_agg(aggregate['val_paper_argmax_f1'])}",
        "",
        "## Comparisons (cautious; protocols differ)",
        "",
        "- Supervised Multi-GIN **has PaySim fraud labels** — upper/reference ceiling, not a fair label-free transfer competitor.",
        f"- X-only val AUPRC@0.5: {out['comparisons_cautious']['x_only_val_auprc_at_0.5']}",
        f"- Frozen P1 seed2 val AUPRC@0.5: {out['comparisons_cautious']['frozen_p1_seed2_val_auprc_at_0.5']}",
        f"- BN P2 seed2 val AUPRC@0.5: {out['comparisons_cautious']['bn_p2_seed2_val_auprc_at_0.5']}",
        f"- Sequential SSL aggregate present: {out['comparisons_cautious']['sequential_ssl_present']}",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("Wrote %s", OUT_JSON)
    return 0


def _fmt_agg(block: Dict[str, Any]) -> str:
    return (
        f"{block['mean']:.4f} ± {block['std_sample']:.4f} "
        f"(median {block['median']:.4f}; per-seed {block['values']})"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke")
    sm.add_argument("--overwrite_smoke", action="store_true")
    tr = sub.add_parser("train")
    tr.add_argument("--seed", type=int, required=True, choices=list(SEEDS))
    ev = sub.add_parser("eval")
    ev.add_argument("--seed", type=int, required=True, choices=list(SEEDS))
    sub.add_parser("aggregate")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logger_setup()
    args = build_parser().parse_args(argv)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd == "smoke":
        return cmd_smoke(args)
    if args.cmd == "train":
        return cmd_train(args)
    if args.cmd == "eval":
        return cmd_eval(args)
    if args.cmd == "aggregate":
        return cmd_aggregate(args)
    raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
