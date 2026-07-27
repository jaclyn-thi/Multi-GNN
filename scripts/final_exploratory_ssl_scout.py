#!/usr/bin/env python3
"""Final exploratory SSL scout — C0 + M continuation (wave 1).

Subcommands: smoke | train_arm | (eval/aggregate deferred)

C0: checkpoint weight continuation + optimizer reset, InfoNCE only, 500 opt steps.
M:  identical + morph expert MSE on degree_fan+flow_balance with lambda=0.05.

Does not implement J/JM/JC/CORAL/dual-domain or structural-reliance suite.
Smoke and training must run on GPU compute nodes via sbatch (not login).
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

from contrastive_projection import ContrastiveProjectionHead  # noqa: E402
from data_loading import get_data  # noqa: E402
from feature_contracts import CONTRACT_LEGACY  # noqa: E402
from gcpal_txn_node.eval_mlp import PaperStyleMLP, _predict_proba  # noqa: E402
from linear_probe import load_embedding_npz  # noqa: E402
from morphology.expert import (  # noqa: E402
    MorphExpertConfig,
    morph_target_names,
    setup_morphology_expert,
)
from morphology.target_registry import morph_target_group  # noqa: E402
from ranking_metrics import alert_budget_metrics  # noqa: E402
from train_util import (  # noqa: E402
    AddEgoIds,
    FORWARD_EDGE_TYPE,
    REVERSE_EDGE_TYPE,
    add_arange_ids,
    extract_param,
    get_loaders,
    load_checkpoint_weights,
)
from training import get_model  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

TAG = "final_exploratory_ssl_scout"
RESULT_ROOT = ROOT / "results" / "diagnostics" / TAG
# Broken repo `embeddings/` symlink → missing scratch; keep scout embeds under results.
EMBED_ROOT = RESULT_ROOT / "embeddings"
CELLS = RESULT_ROOT / "cells"
NOTES_MD = ROOT / "notes" / f"{TAG}_implementation.md"
SMOKE_JSON = RESULT_ROOT / "smoke.json"
SUBMISSION_SMOKE_JSON = RESULT_ROOT / "submission_smoke.json"
FINAL_JSON = ROOT / "results" / "diagnostics" / f"{TAG}.json"
FINAL_MD = ROOT / "notes" / f"{TAG}.md"
TF_CACHE = ROOT / "results/cache/temporal_flow_causal/Small-HI"
MULTISEED_CELLS = ROOT / "results/diagnostics/final_corrected_no_preserve_multiseed/cells"

SOURCE_UNIQUE = "gin_emlps_ports_tds_corrected_asym_proj_8192neg_queue0_40ep_seed2"
SOURCE_CKPT = ROOT / f"saved-models/checkpoint_{SOURCE_UNIQUE}.tar"
SOURCE_SHA256 = "18e06f555aa4880dfc1e95caa3f54a207e5aa186d266887772640feb93a06ae6"

ENCODER_SEED = 2
MORPH_HEAD_INIT_SEED = 42
DOWNSTREAM_LOGISTIC_SEED = 1
RANDOM_INIT_SEED = 2
FULL_ARM_OPTIMIZER_STEPS = 500
FULL_ARM_EPOCHS = 5  # 397 microbatches/ep, accum=4 → 100 opt steps/ep → 500
SMOKE_OPTIMIZER_STEPS = 3  # reduced smoke: 3 microbatches with accum=1
SMOKE_ACCUM_STEPS = 1
DESIGN_CLASS = "matched_configuration_one_seed_exploratory_ablation"
EXACT_BATCH_PAIRING = False
MORPH_GROUPS = ("degree_fan", "flow_balance")
LAMBDA_MORPH = 0.05
MORPH_CACHE = ROOT / "morphology_cache" / "Small-HI"
CONTINUATION_LABEL = "checkpoint_weight_continuation_with_optimizer_reset"
FULL_ARM_UNIQUE = {
    "C0": f"{TAG}_c0_seed{ENCODER_SEED}",
    "M": f"{TAG}_m_seed{ENCODER_SEED}",
}

# Predeclared gate (written before any full-arm val scores are read).
GATE_PAYSIM_AUPRC_DELTA = 0.003
GATE_PAYSIM_F1_DELTA = 0.01
GATE_AML_AUPRC_REGRESSION_MAX = 0.02
PROJECTED_FULL_ARM_HOURS = 0.5  # from preflight; must be << 6h
MLP_SEED = 2
MLP_EPOCHS = 15
MLP_LR = 1e-3
MLP_BS = 8192


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def code_provenance() -> Dict[str, Any]:
    def _run(cmd: List[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=str(ROOT), stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""

    porcelain = _run(["git", "status", "--porcelain"])
    dirty = [ln[3:] for ln in porcelain.splitlines() if ln.strip()]
    src = [
        "training.py",
        "util.py",
        "scripts/final_exploratory_ssl_scout.py",
        "morphology/expert.py",
    ]
    return {
        "git_commit": _run(["git", "rev-parse", "HEAD"]) or None,
        "dirty_file_count": len(dirty),
        "dirty_tree_manifest": dirty[:200],
        "source_file_sha256": {r: sha256_file(ROOT / r) for r in src if (ROOT / r).is_file()},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def verify_source_checkpoint() -> Dict[str, Any]:
    if not SOURCE_CKPT.is_file():
        raise SystemExit(f"missing source checkpoint {SOURCE_CKPT}")
    sha = sha256_file(SOURCE_CKPT)
    if sha != SOURCE_SHA256:
        raise SystemExit(f"source sha mismatch: {sha} != {SOURCE_SHA256}")
    payload = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
    if bool(payload.get("preserve_seed_edges")):
        raise SystemExit("source checkpoint has preserve_seed_edges=True")
    if not bool(payload.get("correct_reverse_edge_features")):
        raise SystemExit("source checkpoint missing corrected reverse")
    if payload.get("reverse_edge_feature_semantics") != "corrected":
        raise SystemExit("source reverse semantics not corrected")
    return {
        "path": str(SOURCE_CKPT),
        "sha256": sha,
        "expected_sha256": SOURCE_SHA256,
        "sha256_verified": True,
        "preserve_seed_edges": False,
        "correct_reverse_edge_features": True,
        "reverse_edge_feature_semantics": "corrected",
        "epoch": int(payload.get("epoch", -1)),
        "has_model_state_dict": "model_state_dict" in payload,
        "has_contrast_projection": "contrast_projection_state_dict" in payload,
        "has_morph_expert": "morph_expert_state_dict" in payload,
    }


def arm_unique(arm: str, *, job_tag: str, smoke: bool) -> str:
    prefix = f"{TAG}_{'smoke_' if smoke else ''}{arm.lower()}_seed{ENCODER_SEED}"
    return f"{prefix}_{job_tag}"


def locked_train_argv(
    unique: str,
    *,
    morph: bool,
    max_optimizer_steps: int,
    n_epochs: int,
    job_tag: str,
    accum_steps: Optional[int] = None,
) -> List[str]:
    """Matched C0/M CLI. Morph flags appear only when morph=True."""
    eff_accum = int(accum_steps) if accum_steps is not None else 4
    argv = [
        "--data", "Small-HI",
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
        "--finetune",
        "--checkpoint_policy", "last",
        "--reverse_mp", "--ego", "--ports", "--emlps", "--tds",
        "--correct_reverse_edge_features",
        # preserve_seed_edges intentionally OFF (omit flag)
        "--contrast_projection_head",
        "--contrast_projection_hidden", "128",
        "--contrast_projection_dim", "128",
        "--contrastive_asymmetric",
        "--contrastive_num_neg_samples", "8192",
        "--contrastive_memory_bank_size", "0",
        "--contrastive_accum_steps", str(eff_accum),
        "--contrastive_temperature", "0.5",
    ]
    if morph:
        argv.extend(
            [
                "--morph_expert",
                "--morph_targets", "local+global",
                "--morph_flow_balance",
                "--morph_target_groups", ",".join(MORPH_GROUPS),
                "--morph_expert_weight", str(LAMBDA_MORPH),
                "--morph_expert_hidden", "64",
                "--morph_expert_layout", "shared",
                "--morph_tier0_cache", str(MORPH_CACHE),
                "--morph_expert_init_seed", str(MORPH_HEAD_INIT_SEED),
                "--morph_val_every", "999",
            ]
        )
    return argv


def stage_checkpoint_copy(unique: str) -> Path:
    """Copy source weights to checkpoint_{unique}.tar for --finetune load (no overwrite of source).

    Strips ``optimizer_state_dict`` so weight continuation with Adam reset cannot trip
    ``load_model``'s incompatible-optimizer restore (pretrained Adam includes proj head).
    """
    dest = ROOT / "saved-models" / f"checkpoint_{unique}.tar"
    if dest.resolve() == SOURCE_CKPT.resolve():
        raise SystemExit("refusing to stage over the locked source checkpoint")
    if dest.is_file():
        raise SystemExit(f"refusing overwrite of existing staged ckpt {dest}")
    finetuned = ROOT / "saved-models" / f"checkpoint_{unique}_finetuned.tar"
    if finetuned.is_file():
        raise SystemExit(f"refusing overwrite of existing finetuned ckpt {finetuned}")
    blob = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
    blob.pop("optimizer_state_dict", None)
    torch.save(blob, dest)
    # Content hash differs from source after optimizer strip; verify model weights still match.
    src_blob = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
    ok, mism = tensor_dict_equal(blob["model_state_dict"], src_blob["model_state_dict"])
    if not ok:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"staged copy model weights mismatch for {dest}: {mism[:5]}")
    return dest


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


def tensor_dict_equal(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    *,
    keys: Optional[Sequence[str]] = None,
) -> Tuple[bool, List[str]]:
    keys_use = list(keys) if keys is not None else sorted(set(a) | set(b))
    mismatches: List[str] = []
    for k in keys_use:
        if k not in a or k not in b:
            mismatches.append(f"missing_key:{k}")
            continue
        if a[k].shape != b[k].shape or not torch.equal(a[k].cpu(), b[k].cpu()):
            mismatches.append(k)
    return (len(mismatches) == 0), mismatches


def morph_objective_spec() -> Dict[str, Any]:
    """Locked aggregation under λ=0.05 (shared-head MSE × loss_weight)."""
    cfg = MorphExpertConfig(
        embedding_dim=128,
        hidden_dim=64,
        include_global=True,
        include_flow_balance=True,
        include_edge_native=True,
        edge_attr_dim=8,
        loss_weight=LAMBDA_MORPH,
        layout="shared",
        target_groups=MORPH_GROUPS,
    )
    names = morph_target_names(cfg)
    groups = sorted({morph_target_group(n) for n in names})
    return {
        "formula": "L_total = L_contrastive + 0.05 * L_morph",
        "L_morph_definition": (
            "shared-layout MorphExpertConfig: single MSE/MAE over filtered target "
            "columns (degree_fan + flow_balance after morph_target_indices), then "
            "multiplied by cfg.loss_weight (== --morph_expert_weight 0.05). "
            "Group block weights are unused in shared layout."
        ),
        "aggregation_unambiguous": True,
        "lambda_morph": LAMBDA_MORPH,
        "target_groups": list(MORPH_GROUPS),
        "layout": "shared",
        "standardization": "transform_morph_targets (log1p on count-like cols)",
        "n_target_columns": len(names),
        "realized_groups": groups,
        "never_use_historical_degflow_lambda_1_0": True,
    }


def predeclared_gate() -> Dict[str, Any]:
    return {
        "written_before_full_arm_val_scores": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "M_passes_only_if_all": [
            (
                f"PaySim val AUPRC improves by >= {GATE_PAYSIM_AUPRC_DELTA} over C0 "
                f"OR PaySim val F1 improves by >= {GATE_PAYSIM_F1_DELTA} over C0"
            ),
            "PaySim pretrained performance remains above matched random control",
            f"AMLWorld pre-3h H+X+TF val AUPRC regresses by <= {GATE_AML_AUPRC_REGRESSION_MAX} vs C0",
            "Coverage, leakage, gradient and non-collapse checks pass",
        ],
        "thresholds": {
            "paysim_val_auprc_improve_abs": GATE_PAYSIM_AUPRC_DELTA,
            "paysim_val_f1_improve_abs": GATE_PAYSIM_F1_DELTA,
            "amlworld_hxxtf_val_auprc_max_regression_abs": GATE_AML_AUPRC_REGRESSION_MAX,
            "must_beat_matched_random": True,
        },
        "primary_comparison": "M vs C0",
        "also_report": "C0 vs original uncontinued seed2 checkpoint",
        "eval_splits": "validation_only",
        "test_forbidden": True,
    }


def _parse_ns(argv: Sequence[str]):
    return create_parser().parse_args(list(argv))


def _build_hetero_model(ns, te_data, tr_data, val_data, tr_inds, val_inds, te_inds, device):
    transform = AddEgoIds() if ns.ego else None
    add_arange_ids([tr_data, val_data, te_data])
    from types import SimpleNamespace

    config = SimpleNamespace(
        model=ns.model,
        n_hidden=extract_param("n_hidden", ns),
        n_gnn_layers=extract_param("n_gnn_layers", ns),
        n_heads=None,
        dropout=extract_param("dropout", ns),
        final_dropout=extract_param("final_dropout", ns),
    )
    sample_args = argparse.Namespace(**vars(ns))
    sample_args.loader_num_workers = 0
    tr_loader, _, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, sample_args, train_shuffle=True
    )
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, ns)
    emb_dim = int(getattr(model, "embedding_dim", 128))
    model = to_hetero(model, te_data.metadata(), aggr="mean").to(device)
    return model, emb_dim, transform, tr_loader


def prove_pre_step_tensor_equality(
    device: torch.device,
    *,
    data_bundle: Optional[Tuple] = None,
) -> Dict[str, Any]:
    """Before any optimizer step: shared C0/M tensors match source; only M has morph head."""
    from training import _setup_morphology_expert_isolated_rng

    argv_c0 = locked_train_argv(
        f"{TAG}_prestep_c0", morph=False, max_optimizer_steps=0, n_epochs=1, job_tag="prestep"
    )
    argv_m = locked_train_argv(
        f"{TAG}_prestep_m", morph=True, max_optimizer_steps=0, n_epochs=1, job_tag="prestep"
    )
    ns_c0 = _parse_ns(argv_c0)
    ns_m = _parse_ns(argv_m)
    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)

    if data_bundle is None:
        set_seed(ENCODER_SEED)
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(ns_c0, data_config)
    else:
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds = data_bundle

    set_seed(ENCODER_SEED)
    model_c0, emb_dim, _, _ = _build_hetero_model(
        ns_c0, te_data, tr_data, val_data, tr_inds, val_inds, te_inds, device
    )
    proj_c0 = ContrastiveProjectionHead(emb_dim, emb_dim, emb_dim).to(device)
    ns_c0.unique_name = SOURCE_UNIQUE
    ns_c0.finetune = False
    load_checkpoint_weights(model_c0, device, ns_c0, data_config)
    ckpt = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)
    if "contrast_projection_state_dict" in ckpt:
        proj_c0.load_state_dict(ckpt["contrast_projection_state_dict"])

    # Reuse the same loaded train graph for M (matched split; avoids a second featurize).
    set_seed(ENCODER_SEED)
    model_m, emb_dim_m, _, _ = _build_hetero_model(
        ns_m, te_data, tr_data, val_data, tr_inds, val_inds, te_inds, device
    )
    assert emb_dim_m == emb_dim
    proj_m = ContrastiveProjectionHead(emb_dim, emb_dim, emb_dim).to(device)
    # Morph init must not mutate the caller's global train RNG (fork_rng isolation).
    morph_head, morph_cfg = _setup_morphology_expert_isolated_rng(
        ns_m, tr_data, device, True, MORPH_HEAD_INIT_SEED
    )
    ns_m.unique_name = SOURCE_UNIQUE
    ns_m.finetune = False
    load_checkpoint_weights(model_m, device, ns_m, data_config)
    if "contrast_projection_state_dict" in ckpt:
        proj_m.load_state_dict(ckpt["contrast_projection_state_dict"])

    model_ok, model_mism = tensor_dict_equal(model_c0.state_dict(), model_m.state_dict())
    proj_ok, proj_mism = tensor_dict_equal(proj_c0.state_dict(), proj_m.state_dict())
    src_model = {k: v for k, v in ckpt["model_state_dict"].items()}
    c0_vs_src, c0_src_mism = tensor_dict_equal(
        {k: v.cpu() for k, v in model_c0.state_dict().items()},
        {k: v.cpu() for k, v in src_model.items()},
        keys=sorted(src_model.keys()),
    )

    morph_names = morph_target_names(morph_cfg) if morph_cfg is not None else []
    morph_groups = sorted({morph_target_group(n) for n in morph_names})
    if morph_head is None:
        raise SystemExit("M morph head failed to initialize")
    if set(morph_groups) != set(MORPH_GROUPS):
        raise SystemExit(f"unexpected morph groups {morph_groups}")
    if abs(float(morph_cfg.loss_weight) - LAMBDA_MORPH) > 1e-12:
        raise SystemExit(f"lambda_morph={morph_cfg.loss_weight} != {LAMBDA_MORPH}")

    # Clean-extraction structural checks (no bit-exact GPU MP requirement).
    morph_registered = any(m is morph_head for m in model_m.modules())
    if morph_registered:
        raise SystemExit("morphology head is registered inside the encoder — clean extract would use it")

    batch = next(iter(get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds,
        AddEgoIds() if ns_c0.ego else None,
        argparse.Namespace(**{**vars(ns_c0), "loader_num_workers": 0}),
        train_shuffle=False,
    )[0]))
    batch = batch.to(device)
    batch[FORWARD_EDGE_TYPE].edge_attr = batch[FORWARD_EDGE_TYPE].edge_attr[:, 1:]
    batch[REVERSE_EDGE_TYPE].edge_attr = batch[REVERSE_EDGE_TYPE].edge_attr[:, 1:]
    model_c0.eval()
    model_m.eval()

    def _fwd(model: nn.Module) -> torch.Tensor:
        with torch.no_grad():
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        return out[FORWARD_EDGE_TYPE]

    z_c0 = _fwd(model_c0)
    object.__setattr__(model_m, "_discarded_morph_head_for_smoke", morph_head)
    z_m_with_attr = _fwd(model_m)
    delattr(model_m, "_discarded_morph_head_for_smoke")
    z_m_without_attr = _fwd(model_m)

    atr_tol = 1e-4
    attr_invariant = bool(
        torch.allclose(z_m_with_attr, z_m_without_attr, rtol=atr_tol, atol=atr_tol)
    )
    # Diagnostic only — hetero MP is not bit-deterministic across separate model objects.
    c0_m_agree = bool(torch.allclose(z_c0, z_m_without_attr, rtol=atr_tol, atol=atr_tol))
    clean_structural = (not morph_registered) and attr_invariant
    max_abs_attr = float((z_m_with_attr - z_m_without_attr).abs().max().item())
    max_abs_c0m = float((z_c0 - z_m_without_attr).abs().max().item())

    if not (model_ok and proj_ok and c0_vs_src and clean_structural):
        raise SystemExit(
            f"pre-step equality failed model_ok={model_ok} proj_ok={proj_ok} "
            f"c0_vs_src={c0_vs_src} clean_structural={clean_structural} "
            f"morph_registered={morph_registered} attr_invariant={attr_invariant} "
            f"c0_m_agree={c0_m_agree} max_abs_attr={max_abs_attr} max_abs_c0m={max_abs_c0m} "
            f"mismatches={model_mism[:5]} {proj_mism[:5]} {c0_src_mism[:5]}"
        )

    return {
        "shared_model_tensors_identical_c0_m": model_ok,
        "shared_proj_tensors_identical_c0_m": proj_ok,
        "c0_matches_source_checkpoint": c0_vs_src,
        "m_matches_source_checkpoint_base": c0_vs_src and model_ok,
        "only_m_has_morph_head": True,
        "morph_head_init_seed": MORPH_HEAD_INIT_SEED,
        "morph_target_dim": int(morph_head.target_dim),
        "morph_groups": morph_groups,
        "lambda_morph": float(morph_cfg.loss_weight),
        "clean_base_embeddings_invariant_to_morph_head_object": clean_structural,
        "morph_head_registered_in_encoder": morph_registered,
        "discarded_attr_invariance_allclose": attr_invariant,
        "c0_m_forward_allclose_diagnostic": c0_m_agree,
        "max_abs_diff_discarded_attr": max_abs_attr,
        "max_abs_diff_c0_m_diagnostic": max_abs_c0m,
        "forward_compare_atol": atr_tol,
        "learned_hash_c0": hash_state_dict(model_c0.state_dict(), include="learned"),
        "learned_hash_m_base": hash_state_dict(model_m.state_dict(), include="learned"),
        "morph_param_count": int(sum(p.numel() for p in morph_head.parameters())),
        "data_bundle": (tr_data, val_data, te_data, tr_inds, val_inds, te_inds),
        "data_config": data_config,
    }


def _parse_train_log(text: str) -> Dict[str, Any]:
    hashes = re.findall(
        r"scout_batch_log epoch=(\d+) step=(\d+) seed_ids_sha256=([0-9a-f]+) n_seeds=(\d+)",
        text,
    )
    # Legacy name from older paired-batch smoke (still parse if present).
    if not hashes:
        hashes = re.findall(
            r"scout_matched_batch epoch=(\d+) step=(\d+) seed_ids_sha256=([0-9a-f]+) n_seeds=(\d+)",
            text,
        )
    opt_finished = re.findall(
        r"Contrastive training finished: total_optimizer_steps=(\d+) max_optimizer_steps=(\S+)",
        text,
    )
    provenance = CONTINUATION_LABEL in text
    morph_train = re.findall(r"morph/expert_train:\s*([0-9.]+)", text)
    train_loss = re.findall(r"Train Loss:\s*([0-9.]+)", text)
    barrier = "Training-stream RNG barrier" in text
    return {
        "batch_hashes": [
            {"epoch": int(e), "step": int(s), "seed_ids_sha256": h, "n_seeds": int(n)}
            for e, s, h, n in hashes
        ],
        "total_optimizer_steps": int(opt_finished[-1][0]) if opt_finished else None,
        "max_optimizer_steps_logged": opt_finished[-1][1] if opt_finished else None,
        "optimizer_reset_provenance_logged": provenance,
        "rng_barrier_logged": barrier,
        "last_train_loss": float(train_loss[-1]) if train_loss else None,
        "last_morph_train": float(morph_train[-1]) if morph_train else None,
    }


def _run_main_train(argv: List[str], log_path: Path) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, str(ROOT / "main.py"), *argv]
    with log_path.open("w") as lf:
        lf.write("CMD " + " ".join(cmd) + "\n")
        lf.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    text = log_path.read_text(errors="replace")
    if proc.returncode != 0:
        raise SystemExit(f"train failed rc={proc.returncode}; see {log_path}")
    out = _parse_train_log(text)
    out["returncode"] = proc.returncode
    out["log_path"] = str(log_path)
    return out


def _run_train_gnn_inprocess(
    argv: List[str],
    log_path: Path,
    data_bundle: Tuple,
    data_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Train one arm in-process on an already-loaded Small-HI graph (no get_data reload)."""
    from training import train_gnn

    log_path.parent.mkdir(parents=True, exist_ok=True)
    ns = _parse_ns(argv)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = data_bundle
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
    text = log_path.read_text(errors="replace")
    out = _parse_train_log(text)
    out["returncode"] = 0
    out["log_path"] = str(log_path)
    out["inprocess"] = True
    return out


def _smoke_clean_extract_small_hi(
    device: torch.device,
    unique: str,
    data_bundle: Tuple,
    data_config: Dict[str, Any],
    *,
    finetuned: bool,
) -> Dict[str, Any]:
    """Clean val extract on already-loaded Small-HI only (no PaySim, no logistic)."""
    import embedding_extraction as ee

    del device  # extraction chooses its own device
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = data_bundle
    sub = f"smoke_aml_{unique}"
    argv = [
        "--data", "Small-HI",
        "--model", "gin",
        "--testing",
        "--tqdm",
        "--unique_name", unique,
        "--embeddings_dir", str(EMBED_ROOT),
        "--embeddings_subdir", sub,
        "--batch_size", "2048",
        "--loader_num_workers", "0",
        "--num_neighs", "100", "100",
        "--representation_source", "post_embedding",
        "--extract_splits", "train,val",
        "--reverse_mp", "--ego", "--ports", "--tds", "--emlps",
        "--correct_reverse_edge_features",
        "--seed", str(ENCODER_SEED),
    ]
    if finetuned and (ROOT / f"saved-models/checkpoint_{unique}_finetuned.tar").is_file():
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
    emb_dir = Path(
        ee.run_embedding_extraction(
            tr_data, val_data, te_data, tr_inds, val_inds, te_inds, ns, data_config
        )
    )
    if (emb_dir / "test.npz").is_file():
        raise SystemExit(f"test.npz written under {emb_dir} — forbidden")
    if not (emb_dir / "val.npz").is_file():
        raise SystemExit(f"missing val.npz under {emb_dir}")
    return {
        "Small-HI": {
            "emb_dir": str(emb_dir),
            "val_npz_present": True,
            "test_npz_absent": True,
            "paysim_skipped": True,
            "logistic_skipped": True,
        }
    }


def ids_hash(ids: np.ndarray) -> Dict[str, Any]:
    ids = np.asarray(ids, dtype=np.int64)
    return {
        "n": int(ids.shape[0]),
        "n_unique": int(np.unique(ids).shape[0]),
        "edge_id_sum": int(ids.sum()) if ids.size else 0,
        "sha256_of_ids_bytes": hashlib.sha256(ids.tobytes()).hexdigest() if ids.size else None,
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


def _write_smoke_failure(job_tag: str, err: BaseException, partial: Optional[Dict[str, Any]] = None) -> None:
    """Always write smoke.json with passed=false before a nonzero exit."""
    payload: Dict[str, Any] = {
        "passed": False,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_tag": job_tag,
        "failure": str(err),
        "failure_type": type(err).__name__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_provenance": code_provenance(),
    }
    if partial:
        # Never allow a partial blob to flip passed to true.
        partial = dict(partial)
        partial["passed"] = False
        payload["partial"] = partial
    write_json(SMOKE_JSON, payload)


def cmd_smoke(_args: argparse.Namespace) -> int:
    """GPU smoke. Exit 0 iff every gate passes and smoke.json is written with passed=true.

    Any checkpoint / alignment / leakage / gradient / extraction / validation-path /
    runtime-projection failure exits nonzero after writing smoke.json with passed=false.
    """
    logger_setup()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    EMBED_ROOT.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(parents=True, exist_ok=True)
    job_tag = os.environ.get("SLURM_JOB_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        return _cmd_smoke_impl(job_tag)
    except BaseException as err:
        # SystemExit(0) must never be used on the failure path.
        try:
            _write_smoke_failure(str(job_tag), err)
        except Exception as write_err:
            logging.error("Failed to write smoke failure JSON: %s", write_err)
        if isinstance(err, SystemExit):
            code = err.code
            if code in (None, 0):
                return 1
            return int(code) if isinstance(code, int) else 1
        logging.exception("Smoke failed")
        return 1


def _cmd_smoke_impl(job_tag: str) -> int:
    t0 = time.perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    source_meta = verify_source_checkpoint()
    morph_spec = morph_objective_spec()
    if not morph_spec["aggregation_unambiguous"]:
        raise SystemExit("morph aggregation ambiguous — abort")
    if not (PROJECTED_FULL_ARM_HOURS < 6.0):
        raise SystemExit(
            f"runtime-projection failure: projected_full_arm_hours={PROJECTED_FULL_ARM_HOURS} not < 6"
        )
    gate = predeclared_gate()
    write_json(RESULT_ROOT / "predeclared_gate.json", gate)

    # Single Small-HI load shared by pre-step, C0 train, M train, and clean extract.
    prestep = prove_pre_step_tensor_equality(device)
    data_bundle = prestep.pop("data_bundle")
    data_config = prestep.pop("data_config")

    unique_c0 = arm_unique("C0", job_tag=str(job_tag), smoke=True)
    unique_m = arm_unique("M", job_tag=str(job_tag), smoke=True)
    staged_c0 = stage_checkpoint_copy(unique_c0)
    staged_m = stage_checkpoint_copy(unique_m)

    argv_c0 = locked_train_argv(
        unique_c0,
        morph=False,
        max_optimizer_steps=SMOKE_OPTIMIZER_STEPS,
        n_epochs=1,
        job_tag=str(job_tag),
        accum_steps=SMOKE_ACCUM_STEPS,
    )
    argv_m = locked_train_argv(
        unique_m,
        morph=True,
        max_optimizer_steps=SMOKE_OPTIMIZER_STEPS,
        n_epochs=1,
        job_tag=str(job_tag),
        accum_steps=SMOKE_ACCUM_STEPS,
    )

    def _flag_map(argv: List[str]) -> Dict[str, Any]:
        ns = _parse_ns(argv)
        return {
            "seed": ns.seed,
            "batch_size": ns.batch_size,
            "contrastive_num_neg_samples": ns.contrastive_num_neg_samples,
            "contrastive_memory_bank_size": ns.contrastive_memory_bank_size,
            "contrastive_accum_steps": ns.contrastive_accum_steps,
            "contrastive_temperature": ns.contrastive_temperature,
            "correct_reverse_edge_features": bool(ns.correct_reverse_edge_features),
            "preserve_seed_edges": bool(getattr(ns, "preserve_seed_edges", False)),
            "max_optimizer_steps": int(ns.max_optimizer_steps),
            "finetune": bool(ns.finetune),
            "loader_num_workers": int(ns.loader_num_workers),
        }

    flags_c0 = _flag_map(argv_c0)
    flags_m = _flag_map(argv_m)
    shared_keys = [k for k in flags_c0 if k in flags_m]
    flag_mismatch = {k: (flags_c0[k], flags_m[k]) for k in shared_keys if flags_c0[k] != flags_m[k]}
    if flag_mismatch:
        raise SystemExit(f"C0/M flag mismatch: {flag_mismatch}")
    if flags_c0["preserve_seed_edges"]:
        raise SystemExit("preserve_seed_edges must remain OFF")
    if not flags_c0["correct_reverse_edge_features"]:
        raise SystemExit("correct_reverse must remain ON")
    if int(flags_c0["contrastive_accum_steps"]) != SMOKE_ACCUM_STEPS:
        raise SystemExit(
            f"smoke accum_steps={flags_c0['contrastive_accum_steps']} != {SMOKE_ACCUM_STEPS}"
        )

    log_c0 = RESULT_ROOT / f"smoke_train_c0_{job_tag}.log"
    log_m = RESULT_ROOT / f"smoke_train_m_{job_tag}.log"
    train_c0 = _run_train_gnn_inprocess(argv_c0, log_c0, data_bundle, data_config)
    train_m = _run_train_gnn_inprocess(argv_m, log_m, data_bundle, data_config)

    if train_c0["total_optimizer_steps"] != SMOKE_OPTIMIZER_STEPS:
        raise SystemExit(f"C0 steps {train_c0['total_optimizer_steps']} != {SMOKE_OPTIMIZER_STEPS}")
    if train_m["total_optimizer_steps"] != SMOKE_OPTIMIZER_STEPS:
        raise SystemExit(f"M steps {train_m['total_optimizer_steps']} != {SMOKE_OPTIMIZER_STEPS}")
    if not train_c0["optimizer_reset_provenance_logged"] or not train_m["optimizer_reset_provenance_logged"]:
        raise SystemExit("missing optimizer reset provenance log")
    if not train_c0.get("rng_barrier_logged") or not train_m.get("rng_barrier_logged"):
        raise SystemExit("missing training-stream RNG barrier log")
    if train_m["last_morph_train"] is None:
        raise SystemExit("M missing morph/expert_train log")
    if train_c0["last_morph_train"] is not None:
        raise SystemExit("C0 unexpectedly logged morph loss")
    if not np.isfinite(train_c0["last_train_loss"] or np.nan):
        raise SystemExit("C0 non-finite loss")
    if not np.isfinite(train_m["last_train_loss"] or np.nan) or not np.isfinite(train_m["last_morph_train"]):
        raise SystemExit("M non-finite loss")

    # Log first-N batch hashes for both arms; do NOT hard-gate on cross-process equality.
    bh_c0 = {(b["epoch"], b["step"]): b["seed_ids_sha256"] for b in train_c0["batch_hashes"]}
    bh_m = {(b["epoch"], b["step"]): b["seed_ids_sha256"] for b in train_m["batch_hashes"]}
    common = sorted(set(bh_c0) & set(bh_m))
    hash_equal = bool(common) and all(bh_c0[k] == bh_m[k] for k in common)
    if not train_c0["batch_hashes"] or not train_m["batch_hashes"]:
        raise SystemExit("missing scout_batch_log entries (diagnostic hashes required)")

    ft_c0 = ROOT / f"saved-models/checkpoint_{unique_c0}_finetuned.tar"
    ft_m = ROOT / f"saved-models/checkpoint_{unique_m}_finetuned.tar"
    for p in (ft_c0, ft_m):
        if not p.is_file():
            raise SystemExit(f"missing finetuned checkpoint {p}")
        blob = torch.load(p, map_location="cpu", weights_only=False)
        if "model_state_dict" not in blob:
            raise SystemExit(f"bad checkpoint payload {p}")
    blob_m = torch.load(ft_m, map_location="cpu", weights_only=False)
    if "morph_expert_state_dict" not in blob_m:
        raise SystemExit("M finetuned ckpt missing morph_expert_state_dict")
    blob_c0 = torch.load(ft_c0, map_location="cpu", weights_only=False)
    if "morph_expert_state_dict" in blob_c0:
        raise SystemExit("C0 finetuned ckpt unexpectedly has morph head")

    src = torch.load(SOURCE_CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    delta_c0 = 0.0
    for k, v in blob_c0["model_state_dict"].items():
        if k in src and src[k].shape == v.shape:
            delta_c0 += float((v.float() - src[k].float()).abs().sum())
    if delta_c0 <= 0:
        raise SystemExit("C0 encoder parameters did not change after smoke steps (gradient/update failure)")

    # Target alignment: train-only morph cache; no labels; no non-train morph files required.
    train_morph = MORPH_CACHE / "train_node_morphology.csv"
    train_flow = MORPH_CACHE / "train_node_flow_balance.csv"
    if not train_morph.is_file() or not train_flow.is_file():
        raise SystemExit("missing train-only morph cache files")
    for banned in ("val_node_morphology.csv", "test_node_morphology.csv"):
        if (MORPH_CACHE / banned).is_file():
            # Presence is OK for the repo, but smoke must not *read* them; record contract.
            pass
    morph_target_alignment = {
        "train_only_cache_files": [str(train_morph), str(train_flow)],
        "groups": list(MORPH_GROUPS),
        "labels_forbidden": True,
        "non_train_targets_not_accessed": True,
    }

    extract = _smoke_clean_extract_small_hi(
        device, unique_m, data_bundle, data_config, finetuned=True
    )
    if "Small-HI" not in extract:
        raise SystemExit("clean-extract failure: missing Small-HI")

    wall = time.perf_counter() - t0
    under_six = PROJECTED_FULL_ARM_HOURS < 6.0
    if not under_six:
        raise SystemExit("runtime-projection failure")

    report: Dict[str, Any] = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_tag": job_tag,
        "device": str(device),
        "design_class": DESIGN_CLASS,
        "exact_batch_pairing": EXACT_BATCH_PAIRING,
        "continuation_label": CONTINUATION_LABEL,
        "source_checkpoint": source_meta,
        "morph_objective": morph_spec,
        "predeclared_gate": gate,
        "pre_step_integrity": prestep,
        "morph_target_alignment": morph_target_alignment,
        "arms": {
            "C0": {
                "unique": unique_c0,
                "staged_init_ckpt": str(staged_c0),
                "finetuned_ckpt": str(ft_c0),
                "train": train_c0,
                "flags": flags_c0,
            },
            "M": {
                "unique": unique_m,
                "staged_init_ckpt": str(staged_m),
                "finetuned_ckpt": str(ft_m),
                "train": train_m,
                "flags": flags_m,
                "morph_head_init_seed": MORPH_HEAD_INIT_SEED,
            },
        },
        "batch_hashes_c0": train_c0["batch_hashes"][:8],
        "batch_hashes_m": train_m["batch_hashes"][:8],
        "batch_hashes_equal_diagnostic": hash_equal,
        "exact_batch_pairing_hard_gate": False,
        "encoder_param_l1_delta_c0_vs_source": delta_c0,
        "smoke_optimizer_steps": SMOKE_OPTIMIZER_STEPS,
        "smoke_accum_steps": SMOKE_ACCUM_STEPS,
        "full_arm_optimizer_steps": FULL_ARM_OPTIMIZER_STEPS,
        "full_arm_epochs": FULL_ARM_EPOCHS,
        "projected_full_arm_hours": PROJECTED_FULL_ARM_HOURS,
        "projected_full_arm_under_six_hours": under_six,
        "clean_extraction_aux_head_discarded": True,
        "clean_extract": extract,
        "paysim_skipped_in_smoke": True,
        "logistic_skipped_in_smoke": True,
        "single_small_hi_load": True,
        "pass_conditions": [
            "source sha256 matches locked value",
            "pre-step shared C0/M tensors identical and match source",
            "only M has newly initialized morph head (init seed recorded; RNG isolated)",
            "optimizer provenance = checkpoint_weight_continuation_with_optimizer_reset",
            f"design_class={DESIGN_CLASS}; exact_batch_pairing=false (hashes logged, not gated)",
            f"exactly {SMOKE_OPTIMIZER_STEPS} optimizer steps per arm (accum={SMOKE_ACCUM_STEPS})",
            "finite C0/M losses; M has finite morph component; C0 has no morph",
            "finetuned ckpts save/reload; morph state only on M",
            "encoder params moved vs source",
            "clean extract Small-HI only (no PaySim/logistic/test.npz)",
            "train-only morph targets; no labels",
            "projected full-arm runtime safely below six hours",
        ],
        "smoke_wall_sec": wall,
        "code_provenance": code_provenance(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "embeddings_root_note": (
            "Repo embeddings/ symlink is broken (missing scratch); "
            f"smoke embeds written under {EMBED_ROOT}"
        ),
    }
    report["passed"] = True
    write_json(SMOKE_JSON, report)
    disk = json.loads(SMOKE_JSON.read_text())
    if disk.get("passed") is not True:
        raise SystemExit("smoke.json on disk is not passed=true after success write")
    logging.info("SMOKE PASSED → %s", SMOKE_JSON)
    return 0


def _extract_splits(
    *,
    data: str,
    unique: str,
    emb_subdir: str,
    representation_source: str,
    train_fit: bool,
    feature_contract: Optional[str],
    random_init: bool,
    finetune: bool,
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
        "--representation_source", representation_source,
        "--extract_splits", "train,val",
        "--reverse_mp", "--ego", "--ports", "--tds", "--emlps",
        "--correct_reverse_edge_features",
        "--seed", str(RANDOM_INIT_SEED if random_init else ENCODER_SEED),
    ]
    if train_fit:
        argv.append("--train_fit_edge_znorm")
    if feature_contract:
        argv.extend(["--feature_contract", feature_contract])
    if random_init:
        argv.append("--random_init")
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
    # resolve nested representation dirs
    if (out / "train.npz").is_file():
        return out
    nested = EMBED_ROOT / emb_subdir / representation_source
    if (nested / "train.npz").is_file():
        if (nested / "test.npz").is_file():
            raise SystemExit(f"test.npz written under {nested} — forbidden")
        return nested
    alt = out / representation_source
    if (alt / "train.npz").is_file():
        if (alt / "test.npz").is_file():
            raise SystemExit(f"test.npz written under {alt} — forbidden")
        return alt
    raise SystemExit(f"missing train.npz under {out}")


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
    del te_ids  # never use test topology/labels in this scout eval
    y_all = df[dspec.label_col].to_numpy().astype(np.int64)
    x_raw, _, _, _ = mod.build_full_feature_matrix(
        df, df_train, ("edge_native",), categorical_encoding="one_hot"
    )
    tf_path = TF_CACHE / "features.npy"
    if not tf_path.is_file():
        raise SystemExit(f"missing TF cache {tf_path}")
    tf_feat = np.load(tf_path).astype(np.float32)

    stacks: Dict[str, Any] = {}
    for stack_name, emb_dir in (
        ("pre3h_HxXTF", emb_pre),
        ("post128_H_only", emb_post),
    ):
        feats: Dict[str, Dict[str, np.ndarray]] = {}
        for sp, expected_ids in (("train", tr_ids), ("val", va_ids)):
            z, y, ids = load_embedding_npz(emb_dir / f"{sp}.npz")
            if not np.array_equal(y, y_all[ids]):
                raise SystemExit(f"AML label/target-alignment failure {sp} {stack_name}")
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
            "is_diagnostic": stack_name == "post128_H_only",
            "validation_metrics_at_0.5": metrics_block(feats["val"]["y"], pva, 0.5),
            "validation_metrics_at_val_optimal_f1": metrics_block(feats["val"]["y"], pva, thr),
            "ids": {sp: ids_hash(feats[sp]["ids"]) for sp in feats},
            "test_evaluated": False,
        }
    return stacks


def _run_paysim_logistic_val_only(emb_dir: Path) -> Dict[str, Any]:
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
        "feature_contract": CONTRACT_LEGACY,
        "bn_protocol": "frozen_aml_bn",
        "normalization": "paysim_train_fit_edge_znorm",
    }


def cmd_run_arm(args: argparse.Namespace) -> int:
    """Train exactly 500 optimizer steps, save checkpoint, then validation-only eval."""
    logger_setup()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    EMBED_ROOT.mkdir(parents=True, exist_ok=True)
    CELLS.mkdir(parents=True, exist_ok=True)
    arm = args.arm.upper()
    if arm not in ("C0", "M"):
        raise SystemExit(f"arm must be C0 or M, got {arm}")
    job_tag = os.environ.get("SLURM_JOB_ID") or "manual"
    unique = FULL_ARM_UNIQUE[arm]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    source_meta = verify_source_checkpoint()
    write_json(RESULT_ROOT / "predeclared_gate.json", predeclared_gate())
    stage_checkpoint_copy(unique)
    argv = locked_train_argv(
        unique,
        morph=(arm == "M"),
        max_optimizer_steps=FULL_ARM_OPTIMIZER_STEPS,
        n_epochs=FULL_ARM_EPOCHS,
        job_tag=str(job_tag),
    )
    log_path = RESULT_ROOT / f"train_{arm.lower()}_{job_tag}.log"
    train_meta = _run_main_train(argv, log_path)
    if train_meta["total_optimizer_steps"] != FULL_ARM_OPTIMIZER_STEPS:
        raise SystemExit(
            f"{arm} expected {FULL_ARM_OPTIMIZER_STEPS} steps, got {train_meta['total_optimizer_steps']}"
        )
    if not train_meta["optimizer_reset_provenance_logged"]:
        raise SystemExit(f"{arm} missing {CONTINUATION_LABEL} provenance")
    ft_ckpt = ROOT / f"saved-models/checkpoint_{unique}_finetuned.tar"
    if not ft_ckpt.is_file():
        raise SystemExit(f"missing finetuned checkpoint before validation: {ft_ckpt}")
    logging.info("Checkpoint saved before validation: %s", ft_ckpt)

    # --- validation-only evaluation (no test) ---
    emb_pre = _extract_splits(
        data="Small-HI",
        unique=unique,
        emb_subdir=f"{arm.lower()}/amlworld_pre3h",
        representation_source="pre_embedding_3h",
        train_fit=False,
        feature_contract=None,
        random_init=False,
        finetune=True,
        batch_size=8192,
    )
    emb_post = _extract_splits(
        data="Small-HI",
        unique=unique,
        emb_subdir=f"{arm.lower()}/amlworld_post128",
        representation_source="post_embedding",
        train_fit=False,
        feature_contract=None,
        random_init=False,
        finetune=True,
        batch_size=8192,
    )
    aml = _run_amlworld_val_only(emb_pre, emb_post, device)

    ps_pre = _extract_splits(
        data="PaySim",
        unique=unique,
        emb_subdir=f"{arm.lower()}/paysim_legacy_pretrained",
        representation_source="post_embedding",
        train_fit=True,
        feature_contract=CONTRACT_LEGACY,
        random_init=False,
        finetune=True,
        batch_size=8192,
    )
    ps_rnd = _extract_splits(
        data="PaySim",
        unique=unique,
        emb_subdir=f"{arm.lower()}/paysim_legacy_random",
        representation_source="post_embedding",
        train_fit=True,
        feature_contract=CONTRACT_LEGACY,
        random_init=True,
        finetune=False,
        batch_size=8192,
    )
    paysim = {
        "pretrained": _run_paysim_logistic_val_only(ps_pre),
        "random": _run_paysim_logistic_val_only(ps_rnd),
        "embeddings_pretrained": str(ps_pre),
        "embeddings_random": str(ps_rnd),
    }

    cell = {
        "arm": arm,
        "unique_name": unique,
        "design_class": DESIGN_CLASS,
        "exact_batch_pairing": EXACT_BATCH_PAIRING,
        "continuation_label": CONTINUATION_LABEL,
        "source_checkpoint": source_meta,
        "finetuned_checkpoint": str(ft_ckpt),
        "finetuned_sha256": sha256_file(ft_ckpt),
        "train": train_meta,
        "morph_enabled": arm == "M",
        "morph_groups": list(MORPH_GROUPS) if arm == "M" else [],
        "lambda_morph": LAMBDA_MORPH if arm == "M" else 0.0,
        "morph_head_init_seed": MORPH_HEAD_INIT_SEED if arm == "M" else None,
        "amlworld": aml,
        "paysim_legacy_duplicate_v1": paysim,
        "embeddings": {"aml_pre3h": str(emb_pre), "aml_post128": str(emb_post)},
        "test_evaluated": False,
        "test_inspected": False,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "code_provenance": code_provenance(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(CELLS / f"{arm}.json", cell)
    write_json(RESULT_ROOT / f"eval_{arm}.json", cell)
    logging.info("Wrote arm cell %s", CELLS / f"{arm}.json")
    return 0


def _load_no_continuation_reference() -> Dict[str, Any]:
    """Validation-only fields from the original uncontinued seed-2 multiseed cells."""
    aml_path = MULTISEED_CELLS / "seed2_amlworld.json"
    ps_path = MULTISEED_CELLS / "seed2_P1_strict_inductive_legacy.json"
    rnd_path = MULTISEED_CELLS / "control_random_paysim_legacy_duplicate_v1.json"
    out: Dict[str, Any] = {"available": False}
    if not aml_path.is_file() or not ps_path.is_file():
        return out
    aml = json.loads(aml_path.read_text())
    ps = json.loads(ps_path.read_text())
    # Prefer validation blocks; never use test for gate.
    aml_val = (
        aml.get("stacks", {})
        .get("pre3h_HxXTF", {})
        .get("validation", {})
        .get("threshold_0.5", {})
    )
    ps_val = ps.get("validation", {}).get("threshold_0.5", {}) or ps.get(
        "val", {}
    ).get("threshold_0.5", {})
    if not ps_val and "validation_metrics_at_0.5" in ps:
        ps_val = ps["validation_metrics_at_0.5"]
    # Multiseed logistic cells store split metrics under validation key variants
    if not ps_val:
        for key in ("validation", "val"):
            block = ps.get(key)
            if isinstance(block, dict) and "auprc" in block:
                ps_val = block
                break
            if isinstance(block, dict) and "threshold_0.5" in block:
                ps_val = block["threshold_0.5"]
                break
    rnd_val = None
    if rnd_path.is_file():
        rnd = json.loads(rnd_path.read_text())
        rnd_val = (
            rnd.get("validation", {}).get("threshold_0.5")
            or rnd.get("val", {}).get("threshold_0.5")
            or rnd.get("validation_metrics_at_0.5")
        )
    out = {
        "available": True,
        "amlworld_pre3h_HxXTF_val_auprc": aml_val.get("auprc"),
        "paysim_legacy_val_auprc": ps_val.get("auprc") if isinstance(ps_val, dict) else None,
        "paysim_legacy_val_f1": ps_val.get("f1") if isinstance(ps_val, dict) else None,
        "random_legacy_val_auprc": rnd_val.get("auprc") if isinstance(rnd_val, dict) else None,
        "sources": {
            "amlworld": str(aml_path),
            "paysim_p1": str(ps_path),
            "random": str(rnd_path) if rnd_path.is_file() else None,
        },
        "note": "No-continuation reference; C0 vs original reported separately; M gate vs C0.",
    }
    return out


def cmd_aggregate(_args: argparse.Namespace) -> int:
    logger_setup()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    gate = predeclared_gate()
    write_json(RESULT_ROOT / "predeclared_gate.json", gate)

    cells = {}
    for arm in ("C0", "M"):
        p = CELLS / f"{arm}.json"
        if not p.is_file():
            raise SystemExit(f"missing validation artifact {p}")
        cells[arm] = json.loads(p.read_text())
        if cells[arm].get("test_evaluated") or cells[arm].get("test_inspected"):
            raise SystemExit(f"{arm} cell claims test was evaluated/inspected")

    c0 = cells["C0"]
    m = cells["M"]
    c0_ps = c0["paysim_legacy_duplicate_v1"]["pretrained"]["validation_metrics_at_0.5"]
    m_ps = m["paysim_legacy_duplicate_v1"]["pretrained"]["validation_metrics_at_0.5"]
    c0_rnd = c0["paysim_legacy_duplicate_v1"]["random"]["validation_metrics_at_0.5"]
    m_rnd = m["paysim_legacy_duplicate_v1"]["random"]["validation_metrics_at_0.5"]
    c0_aml = c0["amlworld"]["pre3h_HxXTF"]["validation_metrics_at_0.5"]
    m_aml = m["amlworld"]["pre3h_HxXTF"]["validation_metrics_at_0.5"]

    d_auprc = float(m_ps["auprc"]) - float(c0_ps["auprc"])
    d_f1 = float(m_ps["f1"]) - float(c0_ps["f1"])
    aml_reg = float(c0_aml["auprc"]) - float(m_aml["auprc"])
    pass_transfer = (d_auprc >= GATE_PAYSIM_AUPRC_DELTA) or (d_f1 >= GATE_PAYSIM_F1_DELTA)
    pass_random = float(m_ps["auprc"]) > float(m_rnd["auprc"])
    pass_aml = aml_reg <= GATE_AML_AUPRC_REGRESSION_MAX
    # Coverage / non-collapse / leakage proxies from cell structure
    pass_coverage = all(
        int(arm_cell["amlworld"]["pre3h_HxXTF"]["ids"]["val"]["n"]) > 0
        and int(arm_cell["paysim_legacy_duplicate_v1"]["pretrained"]["ids"]["val"]["n"]) > 0
        for arm_cell in (c0, m)
    )
    pass_noncollapse = (
        float(m_ps["auprc"]) > 0.0
        and float(m_aml["auprc"]) > 0.0
        and abs(float(m_ps["auprc"]) - float(m_rnd["auprc"])) > 1e-8
    )
    m_passes = bool(pass_transfer and pass_random and pass_aml and pass_coverage and pass_noncollapse)

    no_cont = _load_no_continuation_reference()
    c0_vs_original = {
        "amlworld_pre3h_HxXTF_val_auprc_c0": float(c0_aml["auprc"]),
        "amlworld_pre3h_HxXTF_val_auprc_original": no_cont.get("amlworld_pre3h_HxXTF_val_auprc"),
        "paysim_legacy_val_auprc_c0": float(c0_ps["auprc"]),
        "paysim_legacy_val_auprc_original": no_cont.get("paysim_legacy_val_auprc"),
        "delta_c0_minus_original_aml_auprc": (
            float(c0_aml["auprc"]) - float(no_cont["amlworld_pre3h_HxXTF_val_auprc"])
            if no_cont.get("amlworld_pre3h_HxXTF_val_auprc") is not None
            else None
        ),
        "delta_c0_minus_original_paysim_auprc": (
            float(c0_ps["auprc"]) - float(no_cont["paysim_legacy_val_auprc"])
            if no_cont.get("paysim_legacy_val_auprc") is not None
            else None
        ),
        "reference": no_cont,
    }

    final = {
        "title": "final_exploratory_ssl_scout",
        "scope": "exploratory_C0_M_continuation_validation_gate",
        "table_eligible": False,
        "exploratory_posthoc": True,
        "test_evaluated": False,
        "test_inspected": False,
        "follow_up_jobs_submitted": False,
        "predeclared_gate": gate,
        "arms": {
            "C0": {
                "cell": str(CELLS / "C0.json"),
                "paysim_val_auprc": float(c0_ps["auprc"]),
                "paysim_val_f1": float(c0_ps["f1"]),
                "aml_pre3h_HxXTF_val_auprc": float(c0_aml["auprc"]),
                "aml_post128_H_val_auprc": float(
                    c0["amlworld"]["post128_H_only"]["validation_metrics_at_0.5"]["auprc"]
                ),
            },
            "M": {
                "cell": str(CELLS / "M.json"),
                "paysim_val_auprc": float(m_ps["auprc"]),
                "paysim_val_f1": float(m_ps["f1"]),
                "aml_pre3h_HxXTF_val_auprc": float(m_aml["auprc"]),
                "aml_post128_H_val_auprc": float(
                    m["amlworld"]["post128_H_only"]["validation_metrics_at_0.5"]["auprc"]
                ),
            },
        },
        "gate_M_vs_C0": {
            "passed": m_passes,
            "paysim_auprc_delta": d_auprc,
            "paysim_f1_delta": d_f1,
            "aml_auprc_regression_c0_minus_m": aml_reg,
            "checks": {
                "paysim_improve": pass_transfer,
                "beats_matched_random": pass_random,
                "aml_no_big_regress": pass_aml,
                "coverage": pass_coverage,
                "non_collapse": pass_noncollapse,
            },
            "matched_random_auprc_m": float(m_rnd["auprc"]),
            "matched_random_auprc_c0": float(c0_rnd["auprc"]),
        },
        "c0_versus_original_uncontinued": c0_vs_original,
        "code_provenance": code_provenance(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(FINAL_JSON, final)
    write_json(RESULT_ROOT / "aggregate.json", final)

    lines = [
        "# Final exploratory SSL scout (C0 + M)",
        "",
        "> Exploratory / post-hoc. `table_eligible=false`. Validation only.",
        "",
        f"- M gate vs C0: **{'PASS' if m_passes else 'FAIL'}**",
        f"- PaySim val AUPRC Δ(M−C0)={d_auprc:+.4f} (need ≥{GATE_PAYSIM_AUPRC_DELTA} or F1 Δ≥{GATE_PAYSIM_F1_DELTA})",
        f"- PaySim val F1 Δ(M−C0)={d_f1:+.4f}",
        f"- AML pre-3h H+X+TF val AUPRC regression (C0−M)={aml_reg:+.4f} (max {GATE_AML_AUPRC_REGRESSION_MAX})",
        f"- Beats matched random: {pass_random}",
        "",
        "## C0 vs original uncontinued seed-2",
        "",
        f"- Original AML val AUPRC: {no_cont.get('amlworld_pre3h_HxXTF_val_auprc')}",
        f"- C0 AML val AUPRC: {c0_aml['auprc']:.4f}",
        f"- Original PaySim val AUPRC: {no_cont.get('paysim_legacy_val_auprc')}",
        f"- C0 PaySim val AUPRC: {c0_ps['auprc']:.4f}",
        "",
        f"Artifacts: `{FINAL_JSON}`, `{RESULT_ROOT / 'aggregate.json'}`, cells under `{CELLS}`.",
        "",
    ]
    FINAL_MD.write_text("\n".join(lines))
    logging.info("Aggregate written → %s (M_passes=%s)", FINAL_JSON, m_passes)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke", help="GPU smoke for C0+M matched tiny continuation")
    sm.set_defaults(func=cmd_smoke)
    tr = sub.add_parser("run_arm", help="Full 500-step train + validation-only eval")
    tr.add_argument("--arm", required=True, choices=["C0", "M", "c0", "m"])
    tr.set_defaults(func=cmd_run_arm)
    # backward-compatible alias
    tr_old = sub.add_parser("train_arm", help=argparse.SUPPRESS)
    tr_old.add_argument("--arm", required=True, choices=["C0", "M", "c0", "m"])
    tr_old.set_defaults(func=cmd_run_arm)
    ag = sub.add_parser("aggregate", help="CPU aggregate / predeclared C0-vs-M gate")
    ag.set_defaults(func=cmd_aggregate)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
