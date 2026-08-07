#!/usr/bin/env python3
"""Phase-4B: MIXED_3DOMAIN (1500) + SMALL_LI_ONLY (1000) shared-core scout.

Training only. No extraction, probes, test evaluation, or DAG jobs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import to_hetero

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contrastive_loss import edge_identity_infonce_loss  # noqa: E402
from data_loading import get_data  # noqa: E402
from direct_r198 import (  # noqa: E402
    LearnedAlphaBeta,
    LossNormState,
    TFMoEBundle,
    combine_direct_h_tfmoe_loss,
    load_tf_moe_context,
    tf_moe_mae_losses,
)
from direct_r198.lr_scheduler import DirectHWarmupLinearScheduler  # noqa: E402
from direct_r198.seed_readout import align_seed_r198_pair, forward_seed_r198_hetero  # noqa: E402
from financial_multidataset_shared_core_contract import FINAL_FEATURE_NAMES  # noqa: E402
from graph_augmentations import generate_views  # noqa: E402
from mixed_ssl_phase2.bn import (  # noqa: E402
    apply_bn_,
    bn_bundle_l1,
    bn_bundles_equal,
    clone_bn_bundle,
    collect_bn_bundle,
)
from mixed_ssl_phase3.hash_util import combined_init_sha  # noqa: E402
from mixed_ssl_phase4a import (  # noqa: E402
    ACCUM_STEPS,
    ALPHABETA_LR,
    BATCH_SIZE,
    CALIB_OBS_PER_DOMAIN,
    CONTRACT_ID,
    ENCODER_LR,
    LOADER_NUM_WORKERS,
    N_NEG,
    NUM_NEIGHS,
    SEED,
    TEMP,
)
from mixed_ssl_phase4a.domain_registry import (  # noqa: E402
    DomainSpec,
    default_smoke_domains,
    domain_order,
    registry_to_json,
)
from mixed_ssl_phase4a.preflight import preflight_phase4a  # noqa: E402
from mixed_ssl_phase4a.schedule import (  # noqa: E402
    loader_generator,
    restore_rng,
    snapshot_rng,
)
from mixed_ssl_phase4b import (  # noqa: E402
    ALL_TRAINABLE_ARMS,
    ARRAY_INDEX_TO_ARM,
    ARMS,
    CANONICAL_DOMAINS,
    MIXED_3DOMAIN_LONG,
    MIXED_STEPS_PER_DOMAIN,
    PHASE3_INIT_SHA_PREFIX,
    PHASE3_SHARED_INIT,
    RESULT_ROOT,
    ROLLING_EVERY,
    arm_alpha_freeze_until,
    arm_checkpoint_steps,
    arm_ckpt_root,
    arm_domains,
    arm_first_ab_update,
    arm_result_root,
    arm_schedule,
    arm_steps_per_domain,
    arm_total_steps,
    arm_unique,
    arm_warmup_decay,
    resolved_recipe,
)
from mixed_ssl_phase4b.matching import (  # noqa: E402
    assert_matching_contract_guaranteed,
    compare_domain_streams,
    extract_domain_hashes,
    init_matching_rng_states,
    load_phase3_mixed_hashes,
    load_phase4b_mixed_hashes,
)
from mixed_ssl_phase4b.plots import plot_training_curves, write_steps_csv  # noqa: E402
from train_util import (  # noqa: E402
    AddEgoIds,
    FORWARD_EDGE_TYPE,
    add_arange_ids,
    attach_edge_id_from_batch,
    extract_param,
    get_hetero_seed_edge_ids,
)
from training import _contrastive_view_kwargs, get_model  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique tmp per process to avoid array-task races on shared result roots.
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def seed_ids_sha(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous().numpy().astype(np.int64)
    return hashlib.sha256(a.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def peak_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def make_ns(data: str, *, unique: str, max_steps: int) -> argparse.Namespace:
    argv = [
        "--data", data,
        "--model", "gin",
        "--objective", "contrastive",
        "--unique_name", unique,
        "--seed", str(SEED),
        "--batch_size", str(BATCH_SIZE),
        "--num_neighs", "100", "100",
        "--loader_num_workers", str(LOADER_NUM_WORKERS),
        "--reverse_mp", "--ego", "--ports", "--emlps", "--tds",
        "--correct_reverse_edge_features",
        "--feature_contract", CONTRACT_ID,
        "--train_fit_edge_znorm",
        "--skip_test_eval",
        "--direct_r198_infonce",
        "--direct_r198_tfmoe",
        "--direct_r198_tfmoe_weight_mode", "adaptive",
        "--contrastive_asymmetric",
        "--contrastive_num_neg_samples", str(N_NEG),
        "--contrastive_memory_bank_size", "0",
        "--contrastive_accum_steps", str(ACCUM_STEPS),
        "--contrastive_temperature", str(TEMP),
        "--max_optimizer_steps", str(max_steps),
    ]
    ns = create_parser().parse_args(argv)
    if bool(ns.preserve_seed_edges):
        raise RuntimeError("preserve_seed_edges must be false")
    if bool(ns.contrast_projection_head):
        raise RuntimeError("contrast_projection_head must be false")
    if bool(getattr(ns, "amp", False)) or bool(getattr(ns, "use_amp", False)):
        raise RuntimeError("AMP must be false")
    return ns


def assert_contract_geometry(ns: argparse.Namespace, tr: HeteroData, name: str) -> None:
    ea = tr[FORWARD_EDGE_TYPE].edge_attr
    if int(ea.shape[1]) != 6:
        raise RuntimeError(f"{name} edge_dim={ea.shape[1]} != 6")
    if str(getattr(ns, "feature_contract", None)) != CONTRACT_ID:
        raise RuntimeError(f"{name} contract mismatch")
    names = list(getattr(ns, "edge_feature_schema_names", []) or [])
    if names and names != list(FINAL_FEATURE_NAMES):
        raise RuntimeError(f"{name} schema {names} != {list(FINAL_FEATURE_NAMES)}")


def specs_for_arm(arm: str) -> Tuple[DomainSpec, ...]:
    all_specs = default_smoke_domains()
    active = set(arm_domains(arm))
    return tuple(s for s in all_specs if s.dataset_id in active)


def loader_offsets() -> Dict[str, int]:
    return {s.dataset_id: s.loader_seed_offset for s in default_smoke_domains()}


def build_train_loader(
    tr_data: HeteroData,
    transform,
    *,
    domain: str,
) -> LinkNeighborLoader:
    offsets = loader_offsets()
    g = loader_generator(
        SEED, domain, domain_order=CANONICAL_DOMAINS, loader_seed_offsets=offsets
    )
    edge_label_index = tr_data[FORWARD_EDGE_TYPE].edge_index
    edge_label = tr_data[FORWARD_EDGE_TYPE].y
    return LinkNeighborLoader(
        tr_data,
        num_neighbors=NUM_NEIGHS,
        edge_label_index=(
            (FORWARD_EDGE_TYPE[0], FORWARD_EDGE_TYPE[1], FORWARD_EDGE_TYPE[2]),
            edge_label_index,
        ),
        edge_label=edge_label,
        batch_size=BATCH_SIZE,
        shuffle=True,
        transform=transform,
        num_workers=LOADER_NUM_WORKERS,
        generator=g,
    )


def infinite_loader(loader) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch


def build_model(ns, metadata_data: HeteroData, sample_batch, device):
    config = SimpleNamespace(
        model="gin",
        n_hidden=extract_param("n_hidden", ns),
        n_gnn_layers=extract_param("n_gnn_layers", ns),
        n_heads=None,
        dropout=extract_param("dropout", ns),
        final_dropout=extract_param("final_dropout", ns),
    )
    ns.direct_r198_infonce = True
    model = get_model(sample_batch, config, ns)
    emb_dim = int(getattr(model, "embedding_dim", 198))
    if emb_dim != 198:
        raise RuntimeError(f"R198 expected embedding_dim=198, got {emb_dim}")
    model = to_hetero(model, metadata_data.metadata(), aggr="mean").to(device)
    return model, emb_dim


def grad_norm(params) -> float:
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum())
    return float(sq ** 0.5)


def effective_rank_diag(z: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        x = z.detach().float()
        norms = x.norm(dim=-1)
        stds = x.std(dim=0)
        try:
            # covariance spectrum proxy
            xc = x - x.mean(dim=0, keepdim=True)
            cov = (xc.T @ xc) / max(1, x.shape[0] - 1)
            evals = torch.linalg.eigvalsh(cov).clamp_min(0)
            p = evals / evals.sum().clamp_min(1e-12)
            ent = -(p * (p + 1e-12).log()).sum()
            erank = float(ent.exp())
        except Exception:
            erank = float("nan")
        return {
            "repr_norm_mean": float(norms.mean()),
            "repr_std_mean": float(stds.mean()),
            "effective_rank": erank,
        }


def verify_phase3_init_compatibility(
    model: nn.Module,
    moe: nn.Module,
    alpha_beta: nn.Module,
    init_path: Path,
) -> Dict[str, Any]:
    if not init_path.is_file():
        raise RuntimeError(f"Phase-3 shared init missing: {init_path}")
    file_sha = file_sha256(init_path)
    blob = torch.load(init_path, map_location="cpu", weights_only=False)
    init_sha = str(blob.get("init_sha256", ""))
    if not init_sha.startswith(PHASE3_INIT_SHA_PREFIX):
        raise RuntimeError(
            f"Phase-3 init SHA prefix mismatch: {init_sha[:16]} != {PHASE3_INIT_SHA_PREFIX}"
        )
    model_sd = blob["model_state_dict"]
    moe_sd = blob["moe_state_dict"]
    ab_sd = blob["alpha_beta_state_dict"]

    cur_m = model.state_dict()
    cur_moe = moe.state_dict()
    cur_ab = alpha_beta.state_dict()
    if set(model_sd.keys()) != set(cur_m.keys()):
        missing = sorted(set(cur_m) - set(model_sd))
        extra = sorted(set(model_sd) - set(cur_m))
        raise RuntimeError(f"model key mismatch missing={missing[:5]} extra={extra[:5]}")
    for k, v in model_sd.items():
        if tuple(v.shape) != tuple(cur_m[k].shape):
            raise RuntimeError(f"model shape mismatch {k}: {tuple(v.shape)} vs {tuple(cur_m[k].shape)}")
    if set(moe_sd.keys()) != set(cur_moe.keys()):
        raise RuntimeError("moe key mismatch vs Phase-4B model")
    for k, v in moe_sd.items():
        if tuple(v.shape) != tuple(cur_moe[k].shape):
            raise RuntimeError(f"moe shape mismatch {k}")
    if set(ab_sd.keys()) != set(cur_ab.keys()):
        raise RuntimeError("alpha/beta key mismatch")
    for k, v in ab_sd.items():
        if tuple(v.shape) != tuple(cur_ab[k].shape):
            raise RuntimeError(f"alpha/beta shape mismatch {k}")

    # Geometry locks from Phase-3 state
    ew = model_sd.get("edge_emb.node__to__node.weight")
    if ew is None or int(ew.shape[-1]) != 6:
        raise RuntimeError("Phase-3 init edge_dim != 6")
    # R198 visible in emlps first linear in
    for k, v in model_sd.items():
        if "emlps.0.0.node__to__node.weight" in k and int(v.shape[-1]) != 198:
            raise RuntimeError(f"R198 dim mismatch at {k}: {tuple(v.shape)}")

    model.load_state_dict(model_sd, strict=True)
    moe.load_state_dict(moe_sd, strict=True)
    alpha_beta.load_state_dict(ab_sd, strict=True)
    alpha_beta.set_frozen(True)
    local = combined_init_sha(model, moe, alpha_beta)
    if local != init_sha:
        raise RuntimeError(f"loaded init sha {local} != Phase-3 {init_sha}")
    return {
        "ok": True,
        "path": str(init_path),
        "file_sha256": file_sha,
        "init_sha256": init_sha,
        "loaded_model_sha_matches": True,
        "edge_dim": 6,
        "r198": 198,
        "n_model_params": len(model_sd),
        "n_moe_params": len(moe_sd),
        "phase3_feature_contract_id_in_blob": blob.get("feature_contract_id"),
        "phase4_feature_contract_id": CONTRACT_ID,
        "geometry_equivalent_despite_contract_id_string": True,
        "seed": blob.get("seed"),
    }


def mixed_step(
    *,
    model,
    moe,
    alpha_beta,
    loss_norm,
    tf_ctx,
    optimizer,
    batch,
    loader_data,
    args,
    device,
) -> Dict[str, Any]:
    model.train()
    moe.train()
    seed_edge_ids = get_hetero_seed_edge_ids(batch, loader_data)
    attach_edge_id_from_batch(batch, loader_data)
    requested = int(seed_edge_ids.numel())
    sid_hash = seed_ids_sha(seed_edge_ids)
    first32 = seed_edge_ids[:32].detach().cpu().tolist()
    batch = batch.to(device)
    seed_edge_ids = seed_edge_ids.to(device)

    view1, view2 = generate_views(
        batch,
        **_contrastive_view_kwargs(args, {}, seed_edge_ids=seed_edge_ids),
    )
    z1_all, id1_all, _ = forward_seed_r198_hetero(model, view1, seed_edge_ids)
    with torch.no_grad():
        z2_all, id2_all, _ = forward_seed_r198_hetero(model, view2, seed_edge_ids)
    z1_seed, seed_id1, z2_seed, seed_id2 = align_seed_r198_pair(
        z1_all, id1_all, z2_all, id2_all
    )
    scored = int(seed_id1.numel())
    z2_seed = z2_seed.detach()
    del z1_all, z2_all, id1_all, id2_all, view1, view2, batch
    if device.type == "cuda":
        torch.cuda.empty_cache()

    contrast_raw = edge_identity_infonce_loss(
        z1_seed,
        z2_seed,
        seed_id1,
        seed_id2,
        temperature=TEMP,
        num_neg_samples=N_NEG,
        symmetric=False,
        memory_queue=None,
    )
    tf_raws, tf_diag = tf_moe_mae_losses(z1_seed, seed_id1, moe, tf_ctx)
    total, stats = combine_direct_h_tfmoe_loss(
        contrast_raw=contrast_raw,
        tf_raws=tf_raws,
        alpha_beta=alpha_beta,
        norm=loss_norm,
        weight_mode="adaptive",
    )
    if not torch.isfinite(total):
        raise RuntimeError(f"non-finite total loss: {stats}")

    optimizer.zero_grad(set_to_none=True)
    total.backward()
    enc_gn = grad_norm(model.parameters())
    moe_gn = grad_norm(moe.parameters())
    ab_gn = grad_norm(alpha_beta.parameters())
    if enc_gn == 0.0:
        raise RuntimeError("encoder gradient norm is zero")
    if moe_gn == 0.0:
        raise RuntimeError("MoE gradient norm is zero")
    for hi, head in enumerate(moe.heads):
        if grad_norm(head.parameters()) == 0.0:
            raise RuntimeError(f"TF expert head {hi} has zero grad")
    # Fail if α/β updated while frozen (grad should be zero / unused)
    if bool(alpha_beta._frozen) and ab_gn > 0.0:
        # frozen params may still have grad buffers; optimizer must not step them.
        # LearnedAlphaBeta.set_frozen should zero requires_grad — verify.
        for p in alpha_beta.parameters():
            if p.requires_grad:
                raise RuntimeError("alpha/beta requires_grad True while frozen")

    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(moe.parameters()) + list(alpha_beta.parameters()),
        1e9,
    )
    optimizer.step()
    repr_diag = effective_rank_diag(z1_seed)
    del z1_seed, z2_seed, seed_id1, seed_id2, contrast_raw, tf_raws, total
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        **stats,
        **repr_diag,
        **{f"k/{k}": v for k, v in tf_diag.items() if isinstance(v, (int, float))},
        "requested_seeds": requested,
        "realized_seeds": scored,
        "scored_seeds": scored,
        "seed_ids_sha256": sid_hash,
        "seed_edge_ids_first32": first32,
        "encoder_grad_norm": enc_gn,
        "moe_grad_norm": moe_gn,
        "alpha_grad_norm": ab_gn,
        "alpha_beta_frozen": bool(alpha_beta._frozen),
        "loss_norm_calibrated": bool(loss_norm.calibrated),
    }


def build_checkpoint(**kwargs) -> Dict[str, Any]:
    model = kwargs["model"]
    moe = kwargs["moe"]
    alpha_beta = kwargs["alpha_beta"]
    return {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "moe_state_dict": {k: v.detach().cpu() for k, v in moe.state_dict().items()},
        "alpha_beta_state_dict": {
            k: v.detach().cpu() for k, v in alpha_beta.state_dict().items()
        },
        "optimizer_state_dict": kwargs["optimizer"].state_dict(),
        "scheduler_state_dict": kwargs["scheduler"].state_dict(),
        "global_optimizer_step": int(kwargs["global_step"]),
        "per_domain_exposure_counts": dict(kwargs["step_counts"]),
        "bn_bundles": {
            d: clone_bn_bundle(kwargs["bn_bundles"][d]) for d in kwargs["bn_bundles"]
        },
        "loss_norm_states": {
            d: {
                "contrast_mean": kwargs["loss_norms"][d].contrast_mean,
                "tf_means": list(kwargs["loss_norms"][d].tf_means),
                "calibrated": kwargs["loss_norms"][d].calibrated,
            }
            for d in kwargs["loss_norms"]
        },
        "edge_scalers": kwargs["edge_scalers"],
        "tf_scalers": {
            d: {
                "mean": kwargs["tf_ctx"][d].scaler_mean.tolist(),
                "scale": kwargs["tf_ctx"][d].scaler_scale.tolist(),
            }
            for d in kwargs["tf_ctx"]
        },
        "domain_registry": kwargs["domain_registry"],
        "scheduler_domain_order": list(kwargs["domains"]),
        "schedule_head": list(kwargs["schedule"][:12]),
        "rng_states": {
            d: {
                "python": kwargs["rng_states"][d]["python"],
                "numpy": kwargs["rng_states"][d]["numpy"],
                "torch": kwargs["rng_states"][d]["torch"],
                **(
                    {"cuda": kwargs["rng_states"][d]["cuda"]}
                    if "cuda" in kwargs["rng_states"][d]
                    else {}
                ),
            }
            for d in kwargs["rng_states"]
        },
        "seed_hash_log": kwargs["seed_hash_log"],
        "init_sha256": kwargs["init_sha"],
        "seed": SEED,
        "arm": kwargs["arm"],
        "unique_name": kwargs["unique"],
        "feature_contract_id": CONTRACT_ID,
        "saml_split_protocol": kwargs["saml_split_protocol"],
        "split_protocol_by_domain": kwargs["split_protocol_by_domain"],
        "resolved": kwargs["resolved"],
        "preflight": kwargs["preflight"],
        "test_evaluated": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def save_ckpt(ckpt: Dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    return file_sha256(path)


def reload_validate_ckpt(path: Path, *, arm: str, expect_step: int, domains: Sequence[str]) -> Dict[str, Any]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    required = [
        "model_state_dict",
        "moe_state_dict",
        "alpha_beta_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "global_optimizer_step",
        "per_domain_exposure_counts",
        "bn_bundles",
        "loss_norm_states",
        "edge_scalers",
        "tf_scalers",
        "domain_registry",
        "feature_contract_id",
        "init_sha256",
        "test_evaluated",
    ]
    missing = [k for k in required if k not in blob]
    ok = not missing
    ok = ok and blob["feature_contract_id"] == CONTRACT_ID
    ok = ok and blob["test_evaluated"] is False
    ok = ok and int(blob["global_optimizer_step"]) == expect_step
    ok = ok and blob.get("arm") == arm
    for d in domains:
        ok = ok and d in blob["bn_bundles"]
        ok = ok and d in blob["loss_norm_states"]
    return {
        "path": str(path),
        "ok": bool(ok),
        "missing_keys": missing,
        "global_optimizer_step": blob.get("global_optimizer_step"),
        "exposures": blob.get("per_domain_exposure_counts"),
        "test_evaluated": blob.get("test_evaluated"),
        "sha256": file_sha256(path),
    }


def run_arm(arm: str, *, dry_match_batches: int = 4) -> Dict[str, Any]:
    if arm not in ALL_TRAINABLE_ARMS:
        raise ValueError(arm)
    is_mixed = arm in ("MIXED_3DOMAIN", MIXED_3DOMAIN_LONG)
    t_wall0 = time.perf_counter()
    unique = arm_unique(arm)
    recipe = resolved_recipe(arm=arm)
    domains = list(arm_domains(arm))
    specs = list(specs_for_arm(arm))
    total_steps = arm_total_steps(arm)
    warmup, decay = arm_warmup_decay(arm)
    freeze_until = arm_alpha_freeze_until(arm)
    first_ab = arm_first_ab_update(arm)
    ckpt_steps = set(arm_checkpoint_steps(arm))
    steps_per_domain = arm_steps_per_domain(arm)
    split_by_dom = {s.dataset_id: s.split_protocol_id for s in default_smoke_domains()}
    reg_json = registry_to_json(default_smoke_domains())

    out_root = ROOT / arm_result_root(arm)
    arm_dir = out_root / "arms" / arm
    logs_dir = arm_dir / "logs"
    fig_dir = out_root / "figures" / arm
    ckpt_dir = ROOT / arm_ckpt_root(arm)
    for p in (arm_dir, logs_dir, fig_dir, ckpt_dir, out_root):
        p.mkdir(parents=True, exist_ok=True)

    # Refuse overwrite of an existing LONG (or any) checkpoint tree.
    existing_ckpts = sorted(ckpt_dir.glob("checkpoint_*.tar")) + sorted(
        ckpt_dir.glob("checkpoint_last*.tar")
    )
    if existing_ckpts:
        raise RuntimeError(
            f"checkpoint path would overwrite existing artifacts under {ckpt_dir}: "
            f"{[str(p.name) for p in existing_ckpts[:8]]}"
        )

    match_contract = assert_matching_contract_guaranteed()
    write_json(arm_dir / "matching_contract.json", match_contract)
    # Shared root copies only from primary mixed task to avoid array-task races.
    if is_mixed:
        write_json(out_root / "matching_contract.json", match_contract)

    # Full registry preflight even for LI-only (locks HI/SAML caches untouched)
    pre = preflight_phase4a(root=ROOT, specs=list(default_smoke_domains()))
    write_json(arm_dir / "preflight.json", pre)
    if is_mixed:
        write_json(out_root / "preflight.json", pre)
    if not pre["ok"]:
        raise RuntimeError("phase4b preflight failed")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)

    mem: Dict[str, Any] = {
        "loader_num_workers": LOADER_NUM_WORKERS,
        "graph_build": {},
        "rss_gib_after_each_domain_load": {},
        "peak_rss_gib_process": None,
        "cuda_peak_alloc_gib": None,
        "time_to_first_step_sec": None,
        "mean_sec_per_step": None,
        "graph_build_total_sec": 0.0,
    }

    ns_by: Dict[str, argparse.Namespace] = {}
    data_by: Dict[str, HeteroData] = {}
    edge_scalers: Dict[str, Any] = {}
    test_access = {
        "test_graph_loaded": False,
        "test_tf_arrays_loaded": False,
        "test_metrics_computed": False,
        "test_labels_used": False,
    }

    for spec in specs:
        d = spec.dataset_id
        logging.info("Loading %s under %s ...", d, CONTRACT_ID)
        t0 = time.perf_counter()
        ns = make_ns(d, unique=unique, max_steps=total_steps)
        ns.direct_r198_tfmoe_cache = str(ROOT / spec.tf_cache_path)
        tr, va, te, tr_i, va_i, te_i = get_data(ns, data_config)
        assert_contract_geometry(ns, tr, d)
        if d == "SAML-D" and int(te_i.numel()) != 0:
            raise RuntimeError("SAML-D te_inds nonempty — refuse test access")
        # Do not retain test graphs
        del va, te, tr_i, va_i
        gc.collect()
        if int(te_i.numel()) > 0 and d != "SAML-D":
            # AMLWorld may expose te_inds; we never use them
            pass
        del te_i
        ns_by[d] = ns
        data_by[d] = tr
        edge_scalers[d] = dict(ns.shared_core_edge_scaler)
        dt = time.perf_counter() - t0
        mem["graph_build"][d] = dt
        mem["graph_build_total_sec"] += dt
        mem["rss_gib_after_each_domain_load"][d] = peak_rss_gib()
        logging.info(
            "Loaded %s in %.1fs RSS=%.2f GiB",
            d,
            dt,
            mem["rss_gib_after_each_domain_load"][d],
        )

    write_json(
        arm_dir / "cache_and_scaler_provenance.json",
        {
            "edge_scalers": edge_scalers,
            "tf_caches": {s.dataset_id: s.tf_cache_path for s in specs},
            "preflight_small_li": pre.get("small_li_locked"),
            "test_access": test_access,
        },
    )

    transform = AddEgoIds()
    for d in domains:
        add_arange_ids([data_by[d]])

    # Throwaway sample — keep training generators virgin
    set_seed(SEED)
    sample_dom = domains[0]
    _sample_loader = build_train_loader(data_by[sample_dom], transform, domain=sample_dom)
    sample = next(iter(_sample_loader))
    del _sample_loader
    model, emb_dim = build_model(ns_by[sample_dom], data_by[sample_dom], sample, device)
    del sample
    moe = TFMoEBundle(in_dim=int(emb_dim), hidden=64, n_targets=3).to(device)
    alpha_beta = LearnedAlphaBeta(n_tf=3, init_alpha=0.6).to(device)
    alpha_beta.set_frozen(True)

    init_path = ROOT / PHASE3_SHARED_INIT
    init_prov = verify_phase3_init_compatibility(model, moe, alpha_beta, init_path)
    write_json(arm_dir / "shared_init_provenance.json", init_prov)
    if arm in ("MIXED_3DOMAIN", MIXED_3DOMAIN_LONG):
        write_json(out_root / "shared_init_provenance.json", init_prov)
    init_sha = init_prov["init_sha256"]
    logging.info("Loaded Phase-3 shared init sha=%s", init_sha)

    # Fresh training loaders
    loaders = {d: build_train_loader(data_by[d], transform, domain=d) for d in domains}
    iters = {d: infinite_loader(loaders[d]) for d in domains}

    # Pre-train stream checks
    stream_pre: Dict[str, Any] = {"dry_match_batches": dry_match_batches}
    if dry_match_batches > 0 and "Small-LI" in domains:

        def _hash_n(domain, n, rng_states_local, it):
            hashes = []
            first32s = []
            for _ in range(n):
                restore_rng(rng_states_local[domain])
                batch = next(it)
                sid = get_hetero_seed_edge_ids(batch, data_by[domain])
                hashes.append(seed_ids_sha(sid))
                first32s.append(sid[:32].detach().cpu().tolist())
                rng_states_local[domain] = snapshot_rng()
            return hashes, first32s

        li_l = build_train_loader(data_by["Small-LI"], transform, domain="Small-LI")
        rng_li = init_matching_rng_states(SEED, active_domains=("Small-LI",))
        li_hashes, li_f32 = _hash_n(
            "Small-LI", dry_match_batches, rng_li, infinite_loader(li_l)
        )
        stream_pre["small_li_first_hashes"] = li_hashes
        stream_pre["small_li_first_first32"] = li_f32
        del li_l, rng_li

    if dry_match_batches > 0 and is_mixed:
        # Adding LI must not change HI stream vs 2-domain offsets
        hi_l2 = build_train_loader(data_by["Small-HI"], transform, domain="Small-HI")
        hi_it = infinite_loader(hi_l2)
        rng_a = init_matching_rng_states(SEED, active_domains=("Small-HI",))
        hi_only = []
        hi_f32 = []
        for _ in range(dry_match_batches):
            restore_rng(rng_a["Small-HI"])
            batch = next(hi_it)
            sid = get_hetero_seed_edge_ids(batch, data_by["Small-HI"])
            hi_only.append(seed_ids_sha(sid))
            hi_f32.append(sid[:32].detach().cpu().tolist())
            rng_a["Small-HI"] = snapshot_rng()

        # mixed RR first HI exposures
        loaders_m = {
            d: build_train_loader(data_by[d], transform, domain=d) for d in domains
        }
        its_m = {d: infinite_loader(loaders_m[d]) for d in domains}
        rng_b = init_matching_rng_states(SEED, active_domains=domains)
        mixed_hi = []
        for i in range(dry_match_batches * len(domains)):
            dom = domains[i % len(domains)]
            restore_rng(rng_b[dom])
            batch = next(its_m[dom])
            if dom == "Small-HI":
                mixed_hi.append(
                    seed_ids_sha(get_hetero_seed_edge_ids(batch, data_by[dom]))
                )
            rng_b[dom] = snapshot_rng()
        if hi_only != mixed_hi:
            raise RuntimeError(f"HI stream changed by LI: {hi_only} vs {mixed_hi}")
        stream_pre["hi_vs_mixed_ok"] = True
        stream_pre["hi_first_hashes"] = hi_only
        stream_pre["hi_first_first32"] = hi_f32

        # Compare to Phase-3 MIXED_1TO1 logs if available
        p3 = load_phase3_mixed_hashes(ROOT, domain="Small-HI", limit=dry_match_batches)
        if p3.get("ok") or p3.get("n_hashes", 0) >= dry_match_batches:
            cmp = compare_domain_streams(
                hi_only,
                p3["hashes"][:dry_match_batches],
                domain="Small-HI",
                n=dry_match_batches,
                label_a="phase4b_mixed",
                label_b="phase3_mixed",
            )
            stream_pre["phase3_hi_match"] = cmp
            if not cmp["ok"]:
                raise RuntimeError(f"Phase-3 HI stream mismatch: {cmp}")
        else:
            stream_pre["phase3_hi_match"] = {
                "ok": False,
                "limitation": "insufficient Phase-3 hashes for dry_match",
                "available": p3,
            }
        p3s = load_phase3_mixed_hashes(ROOT, domain="SAML-D", limit=dry_match_batches)
        # SAML first hashes from dry mixed
        loaders_m2 = {
            d: build_train_loader(data_by[d], transform, domain=d) for d in domains
        }
        its_m2 = {d: infinite_loader(loaders_m2[d]) for d in domains}
        rng_c = init_matching_rng_states(SEED, active_domains=domains)
        mixed_sd = []
        mixed_li = []
        for i in range(dry_match_batches * len(domains)):
            dom = domains[i % len(domains)]
            restore_rng(rng_c[dom])
            batch = next(its_m2[dom])
            sid_h = seed_ids_sha(get_hetero_seed_edge_ids(batch, data_by[dom]))
            if dom == "SAML-D":
                mixed_sd.append(sid_h)
            if dom == "Small-LI":
                mixed_li.append(sid_h)
            rng_c[dom] = snapshot_rng()
        if p3s.get("n_hashes", 0) >= dry_match_batches:
            cmp_s = compare_domain_streams(
                mixed_sd,
                p3s["hashes"][:dry_match_batches],
                domain="SAML-D",
                n=dry_match_batches,
                label_a="phase4b_mixed",
                label_b="phase3_mixed",
            )
            stream_pre["phase3_samld_match"] = cmp_s
            if not cmp_s["ok"]:
                raise RuntimeError(f"Phase-3 SAML stream mismatch: {cmp_s}")
        else:
            stream_pre["phase3_samld_match"] = {
                "ok": False,
                "limitation": "insufficient Phase-3 hashes",
            }
        stream_pre["samld_first_hashes"] = mixed_sd
        stream_pre["li_first_hashes_from_rr"] = mixed_li

        # LONG must match Phase-4B MIXED scout initial stream hashes.
        if arm == MIXED_3DOMAIN_LONG:
            p4_matches = {}
            for dom, local_hashes in (
                ("Small-HI", hi_only),
                ("SAML-D", mixed_sd),
                ("Small-LI", mixed_li),
            ):
                p4 = load_phase4b_mixed_hashes(
                    ROOT, domain=dom, limit=dry_match_batches
                )
                if p4.get("n_hashes", 0) < dry_match_batches:
                    raise RuntimeError(
                        f"Phase-4B MIXED hashes missing for {dom}: {p4}"
                    )
                cmp4 = compare_domain_streams(
                    local_hashes,
                    p4["hashes"][:dry_match_batches],
                    domain=dom,
                    n=dry_match_batches,
                    label_a="MIXED_3DOMAIN_LONG",
                    label_b="MIXED_3DOMAIN_phase4b",
                )
                p4_matches[dom] = cmp4
                if not cmp4["ok"]:
                    raise RuntimeError(
                        f"Phase-4B stream mismatch for {dom}: {cmp4}"
                    )
            stream_pre["phase4b_mixed_stream_match"] = p4_matches

        del hi_l2, loaders_m, its_m, loaders_m2, its_m2, rng_a, rng_b, rng_c
        logging.info("Pre-train stream match OK")

    write_json(arm_dir / "stream_preflight.json", stream_pre)

    tf_ctx = {
        d: load_tf_moe_context(ROOT / next(s for s in specs if s.dataset_id == d).tf_cache_path, device)
        for d in domains
    }
    loss_norms = {d: LossNormState() for d in domains}
    calib = {d: {"contrast": 0.0, "tf": [0.0, 0.0, 0.0], "n": 0} for d in domains}

    optimizer = torch.optim.Adam(
        [
            {"params": list(model.parameters()) + list(moe.parameters()), "lr": ENCODER_LR},
            {"params": list(alpha_beta.parameters()), "lr": ALPHABETA_LR},
        ]
    )
    scheduler = DirectHWarmupLinearScheduler(
        optimizer,
        warmup_steps=warmup,
        linear_steps=decay,
        warmup_start=0.1,
        warmup_end=1.0,
        linear_end=0.1,
        steps_per_epoch=total_steps,
        n_epochs=1,
    )

    bn_init = clone_bn_bundle(collect_bn_bundle(model))
    bn_bundles = {d: clone_bn_bundle(bn_init) for d in domains}
    rng_states = init_matching_rng_states(SEED, active_domains=domains)
    schedule = arm_schedule(arm)
    seed_hash_log = {d: [] for d in domains}
    seed_first32_log = {d: [] for d in domains}
    step_counts = {d: 0 for d in domains}
    enc_grad_by_domain = {d: [] for d in domains}
    moe_grad_by_domain = {d: [] for d in domains}
    alpha_unfrozen_at: Optional[int] = None
    alpha_updated_early = False
    model_init_flat = torch.cat(
        [p.detach().float().reshape(-1).cpu() for p in model.parameters()]
    )
    ab_logit0 = float(alpha_beta.alpha_logit.detach().cpu())

    jsonl_path = logs_dir / "steps.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
    step_times: List[float] = []
    t_first = None
    checkpoints_meta: Dict[str, Any] = {}
    rows_for_plots: List[Dict[str, Any]] = []

    with open(jsonl_path, "w", encoding="utf-8") as jsonl:
        for si in range(total_steps):
            t_step0 = time.perf_counter()
            domain = schedule[si]
            restore_rng(rng_states[domain])
            apply_bn_(model, bn_bundles[domain])
            batch = next(iters[domain])
            was_frozen = bool(alpha_beta._frozen)
            ab_before = float(alpha_beta.alpha_logit.detach().cpu())
            stats = mixed_step(
                model=model,
                moe=moe,
                alpha_beta=alpha_beta,
                loss_norm=loss_norms[domain],
                tf_ctx=tf_ctx[domain],
                optimizer=optimizer,
                batch=batch,
                loader_data=data_by[domain],
                args=ns_by[domain],
                device=device,
            )
            scheduler.step()
            completed = si + 1
            if t_first is None:
                t_first = time.perf_counter() - t_wall0
                mem["time_to_first_step_sec"] = t_first

            ab_after = float(alpha_beta.alpha_logit.detach().cpu())
            if was_frozen and completed <= freeze_until and ab_after != ab_before:
                alpha_updated_early = True
                raise RuntimeError(
                    f"alpha/beta changed while frozen at completed={completed}"
                )

            if calib[domain]["n"] < CALIB_OBS_PER_DOMAIN and not loss_norms[domain].calibrated:
                calib[domain]["contrast"] += float(stats["L_contrast_raw"])
                for m in range(3):
                    calib[domain]["tf"][m] += float(stats[f"L_tf_raw_{m}"])
                calib[domain]["n"] += 1
                if calib[domain]["n"] == CALIB_OBS_PER_DOMAIN:
                    n = float(CALIB_OBS_PER_DOMAIN)
                    loss_norms[domain].contrast_mean = calib[domain]["contrast"] / n
                    loss_norms[domain].tf_means = [
                        calib[domain]["tf"][m] / n for m in range(3)
                    ]
                    loss_norms[domain].calibrated = True
                    logging.info(
                        "CALIBRATION_BOUNDARY domain=%s at global_step=%s",
                        domain,
                        completed,
                    )

            if is_mixed:
                all_calibrated = all(loss_norms[d].calibrated for d in domains)
                if (
                    alpha_beta._frozen
                    and all_calibrated
                    and completed >= freeze_until
                ):
                    alpha_beta.set_frozen(False)
                    alpha_unfrozen_at = completed
                    logging.info(
                        "Unfreezing alpha/beta at completed=%s (first update step=%s)",
                        completed,
                        first_ab,
                    )
            else:
                # Specialist: Phase-3 convention — freeze through step 10
                if alpha_beta._frozen and completed >= freeze_until:
                    alpha_beta.set_frozen(False)
                    alpha_unfrozen_at = completed
                    logging.info(
                        "Unfreezing alpha/beta (specialist) at completed=%s",
                        completed,
                    )

            bn_bundles[domain] = clone_bn_bundle(collect_bn_bundle(model))
            rng_states[domain] = snapshot_rng()
            step_counts[domain] += 1
            if len(seed_hash_log[domain]) < steps_per_domain:
                seed_hash_log[domain].append(stats["seed_ids_sha256"])
                seed_first32_log[domain].append(stats["seed_edge_ids_first32"])
            enc_grad_by_domain[domain].append(stats["encoder_grad_norm"])
            moe_grad_by_domain[domain].append(stats["moe_grad_norm"])

            lrs = scheduler.current_lrs()
            alpha_val = float(torch.sigmoid(alpha_beta.alpha_logit).detach().cpu())
            row = {
                "step": si,
                "global_optimizer_step": completed,
                "domain": domain,
                "domain_exposure_count": step_counts[domain],
                "encoder_lr": lrs[0],
                "alphabeta_lr": lrs[1] if len(lrs) > 1 else lrs[0],
                "schedule_phase": scheduler.phase_at(scheduler.completed_optimizer_steps - 1),
                "calibration_complete_domain": bool(loss_norms[domain].calibrated),
                "all_domains_calibrated": all(loss_norms[d].calibrated for d in domains),
                "alpha_beta_frozen": bool(alpha_beta._frozen),
                "alpha": alpha_val,
                "bn_l1_vs_init": bn_bundle_l1(bn_bundles[domain], bn_init),
                "edge_scaler_sha256": edge_scalers[domain]["scaler_sha256"],
                "tf_scaler_mean": tf_ctx[domain].scaler_mean.tolist(),
                "weight_mode": "adaptive",
                "elapsed_step_sec": time.perf_counter() - t_step0,
                **stats,
            }
            jsonl.write(json.dumps(row) + "\n")
            jsonl.flush()
            rows_for_plots.append(row)
            step_times.append(time.perf_counter() - t_step0)

            if completed % 50 == 0 or completed == 1:
                logging.info(
                    "step %s/%s domain=%s L=%.4f enc_g=%.3f α_frozen=%s RSS=%.2f",
                    completed,
                    total_steps,
                    domain,
                    stats["L_total"],
                    stats["encoder_grad_norm"],
                    alpha_beta._frozen,
                    peak_rss_gib(),
                )

            if completed in ckpt_steps or completed % ROLLING_EVERY == 0:
                ckpt = build_checkpoint(
                    model=model,
                    moe=moe,
                    alpha_beta=alpha_beta,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    bn_bundles=bn_bundles,
                    loss_norms=loss_norms,
                    edge_scalers=edge_scalers,
                    tf_ctx=tf_ctx,
                    step_counts=step_counts,
                    global_step=completed,
                    domain_registry=reg_json,
                    domains=domains,
                    schedule=schedule,
                    rng_states=rng_states,
                    seed_hash_log=seed_hash_log,
                    init_sha=init_sha,
                    saml_split_protocol="samld_calendar_day_rezero_v1",
                    split_protocol_by_domain=split_by_dom,
                    resolved=recipe,
                    preflight=pre,
                    arm=arm,
                    unique=unique,
                )
                if completed in ckpt_steps:
                    p = ckpt_dir / f"checkpoint_step_{completed:04d}.tar"
                    sha = save_ckpt(ckpt, p)
                    checkpoints_meta[f"step_{completed}"] = {"path": str(p), "sha256": sha}
                p_last = ckpt_dir / "checkpoint_last.tar"
                sha_last = save_ckpt(ckpt, p_last)
                checkpoints_meta["last"] = {
                    "path": str(p_last),
                    "sha256": sha_last,
                    "global_step": completed,
                }

            del batch, stats
            if completed % 25 == 0:
                gc.collect()

    mem["mean_sec_per_step"] = float(np.mean(step_times)) if step_times else None
    mem["peak_rss_gib_process"] = peak_rss_gib()
    if device.type == "cuda":
        mem["cuda_peak_alloc_gib"] = float(torch.cuda.max_memory_allocated() / (1024**3))
        mem["cuda_peak_reserved_gib"] = float(torch.cuda.max_memory_reserved() / (1024**3))

    write_steps_csv(rows_for_plots, logs_dir / "steps.csv")
    plot_paths = plot_training_curves(rows_for_plots, fig_dir, arm=arm)

    # Integrity gates
    bn_changed = {d: not bn_bundles_equal(bn_bundles[d], bn_init) for d in domains}
    bn_pairwise_differ = True
    if len(domains) > 1:
        for i, a in enumerate(domains):
            for b in domains[i + 1 :]:
                if bn_bundles_equal(bn_bundles[a], bn_bundles[b]):
                    bn_pairwise_differ = False
        apply_bn_(model, bn_bundles[domains[0]])
        snap_a = clone_bn_bundle(collect_bn_bundle(model))
        apply_bn_(model, bn_bundles[domains[1]])
        snap_b = clone_bn_bundle(collect_bn_bundle(model))
        apply_bn_(model, bn_bundles[domains[0]])
        snap_a2 = clone_bn_bundle(collect_bn_bundle(model))
        bn_swap_ok = bn_bundles_equal(snap_a, snap_a2) and not bn_bundles_equal(snap_a, snap_b)
    else:
        bn_swap_ok = True  # N/A for single-domain

    model_final_flat = torch.cat(
        [p.detach().float().reshape(-1).cpu() for p in model.parameters()]
    )
    encoder_changed = not torch.allclose(model_init_flat, model_final_flat)

    reload_results = {}
    for step in sorted(ckpt_steps):
        key = f"step_{step}"
        if key not in checkpoints_meta:
            reload_results[key] = {"ok": False, "reason": "missing"}
            continue
        reload_results[key] = reload_validate_ckpt(
            Path(checkpoints_meta[key]["path"]),
            arm=arm,
            expect_step=step,
            domains=domains,
        )

    expect_exposures = {d: steps_per_domain for d in domains}
    gates = {
        "exact_global_steps": sum(step_counts.values()) == total_steps,
        "exact_per_domain_exposures": step_counts == expect_exposures,
        "finite_losses": True,
        "nonzero_encoder_grad": all(
            any(g > 0 for g in enc_grad_by_domain[d]) for d in domains
        ),
        "nonzero_moe_grad": all(
            any(g > 0 for g in moe_grad_by_domain[d]) for d in domains
        ),
        "encoder_params_changed": bool(encoder_changed),
        "all_lossnorm_calibrated": all(loss_norms[d].calibrated for d in domains),
        "alpha_unfrozen_at_expected": alpha_unfrozen_at == freeze_until,
        "alpha_no_early_update": not alpha_updated_early,
        "lr_schedule_completed": scheduler.completed_optimizer_steps == total_steps,
        "warmup_decay_resolved": {"warmup": warmup, "decay": decay},
        "required_checkpoints_reload_ok": all(
            reload_results[f"step_{s}"].get("ok") for s in ckpt_steps
        ),
        "no_test_graph_cache_metric": all(v is False for v in test_access.values()),
        "feature_contract_ok": True,
        "edge_dim_6": True,
        "projection_false": True,
        "preserve_seed_edges_false": True,
        "amp_false": True,
        "loader_workers_0": LOADER_NUM_WORKERS == 0,
        "init_sha_matches_phase3": init_sha.startswith(PHASE3_INIT_SHA_PREFIX),
    }
    if is_mixed:
        gates["bn_all_changed"] = all(bn_changed.values())
        gates["bn_all_differ"] = bool(bn_pairwise_differ)
        gates["bn_swap_ok"] = bool(bn_swap_ok)
        gates["batch_stream_per_domain"] = all(
            step_counts[d] == steps_per_domain for d in domains
        )
        if arm == MIXED_3DOMAIN_LONG:
            # Verify first 500/domain against Phase-4B MIXED scout.
            p4_post = {}
            for d in domains:
                p4 = load_phase4b_mixed_hashes(
                    ROOT, domain=d, limit=MIXED_STEPS_PER_DOMAIN
                )
                cmp4 = compare_domain_streams(
                    seed_hash_log[d],
                    p4.get("hashes", []),
                    domain=d,
                    n=MIXED_STEPS_PER_DOMAIN,
                    label_a="MIXED_3DOMAIN_LONG",
                    label_b="MIXED_3DOMAIN_phase4b",
                )
                p4_post[d] = cmp4
            stream_pre["phase4b_mixed_post_train_match_500"] = p4_post
            write_json(arm_dir / "stream_preflight.json", stream_pre)
            gates["phase4b_stream_match_first_500"] = all(
                bool(p4_post[d].get("ok")) for d in domains
            )
    else:
        gates["bn_li_changed"] = bool(bn_changed.get("Small-LI", False))

    # Skip non-bool gate values in all()
    gates["ok"] = all(
        bool(v) for k, v in gates.items() if k not in ("warmup_decay_resolved",)
    )

    summary = {
        "ok": bool(gates["ok"]),
        "arm": arm,
        "unique_name": unique,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "feature_contract_id": CONTRACT_ID,
        "domains": domains,
        "steps": total_steps,
        "exposures": dict(step_counts),
        "schedule_head": schedule[:9],
        "resolved": recipe,
        "calibration": {
            d: {
                "n": calib[d]["n"],
                "calibrated": loss_norms[d].calibrated,
                "contrast_mean": loss_norms[d].contrast_mean,
                "tf_means": list(loss_norms[d].tf_means),
            }
            for d in domains
        },
        "alpha_beta": {
            "frozen_through_completed_step": freeze_until,
            "first_update_step": first_ab,
            "unfrozen_at_completed": alpha_unfrozen_at,
            "init_alpha_logit": ab_logit0,
            "final_alpha": float(torch.sigmoid(alpha_beta.alpha_logit).detach().cpu()),
            "policy_note": recipe["alpha_freeze_policy_note"],
        },
        "gates": gates,
        "memory_runtime": mem,
        "checkpoints": checkpoints_meta,
        "checkpoint_reload": reload_results,
        "init_sha256": init_sha,
        "seed_hash_log_first8": {d: seed_hash_log[d][:8] for d in domains},
        "seed_first32_first": {d: (seed_first32_log[d][0] if seed_first32_log[d] else None) for d in domains},
        "stream_preflight": stream_pre,
        "figures": plot_paths,
        "test_access": test_access,
        "elapsed_sec": time.perf_counter() - t_wall0,
        "test_evaluated": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "representation_quality_not_claimed_from_training_losses": True,
    }
    write_json(arm_dir / "summary.json", summary)
    write_json(arm_dir / "memory_runtime.json", mem)
    write_json(arm_dir / "resolved_run.json", recipe)
    # Persist full seed hash logs for cross-arm matching
    write_json(
        arm_dir / "seed_hash_log.json",
        {
            "hashes": seed_hash_log,
            "first32": {d: seed_first32_log[d][:8] for d in domains},
        },
    )
    return summary


def finalize_phase4b() -> Dict[str, Any]:
    out_root = ROOT / RESULT_ROOT
    arm_summaries = {}
    for arm in ARMS:
        p = out_root / "arms" / arm / "summary.json"
        if not p.is_file():
            raise FileNotFoundError(p)
        arm_summaries[arm] = json.loads(p.read_text(encoding="utf-8"))

    # Cross-arm LI stream match (first 500)
    li_mixed_path = out_root / "arms" / "MIXED_3DOMAIN" / "seed_hash_log.json"
    li_only_path = out_root / "arms" / "SMALL_LI_ONLY" / "seed_hash_log.json"
    li_mixed = json.loads(li_mixed_path.read_text(encoding="utf-8"))["hashes"]["Small-LI"]
    li_only = json.loads(li_only_path.read_text(encoding="utf-8"))["hashes"]["Small-LI"]
    li_cmp = compare_domain_streams(
        li_mixed,
        li_only,
        domain="Small-LI",
        n=MIXED_STEPS_PER_DOMAIN,
        label_a="MIXED_3DOMAIN",
        label_b="SMALL_LI_ONLY",
    )

    # Phase-3 HI/SAML full 500 compare if logs permit
    p3_hi = load_phase3_mixed_hashes(ROOT, domain="Small-HI", limit=MIXED_STEPS_PER_DOMAIN)
    p3_sd = load_phase3_mixed_hashes(ROOT, domain="SAML-D", limit=MIXED_STEPS_PER_DOMAIN)
    mixed_hashes = json.loads(
        (out_root / "arms" / "MIXED_3DOMAIN" / "seed_hash_log.json").read_text()
    )["hashes"]
    hi_cmp = (
        compare_domain_streams(
            mixed_hashes["Small-HI"],
            p3_hi["hashes"],
            domain="Small-HI",
            n=MIXED_STEPS_PER_DOMAIN,
            label_a="phase4b_mixed",
            label_b="phase3_mixed",
        )
        if p3_hi.get("n_hashes", 0) >= MIXED_STEPS_PER_DOMAIN
        else {
            "ok": False,
            "limitation": "Phase-3 HI hashes insufficient",
            "available": p3_hi.get("n_hashes"),
        }
    )
    sd_cmp = (
        compare_domain_streams(
            mixed_hashes["SAML-D"],
            p3_sd["hashes"],
            domain="SAML-D",
            n=MIXED_STEPS_PER_DOMAIN,
            label_a="phase4b_mixed",
            label_b="phase3_mixed",
        )
        if p3_sd.get("n_hashes", 0) >= MIXED_STEPS_PER_DOMAIN
        else {
            "ok": False,
            "limitation": "Phase-3 SAML hashes insufficient",
            "available": p3_sd.get("n_hashes"),
        }
    )

    init_shas = {a: arm_summaries[a].get("init_sha256") for a in ARMS}
    init_eq = len(set(init_shas.values())) == 1

    proposed_frozen = {
        "submitted": False,
        "description": "Proposed validation-only frozen eval (not submitted)",
        "new_cells": [
            {"encoder": "MIXED_3DOMAIN", "eval_domain": "Small-HI"},
            {"encoder": "MIXED_3DOMAIN", "eval_domain": "SAML-D"},
            {"encoder": "MIXED_3DOMAIN", "eval_domain": "Small-LI"},
            {"encoder": "SMALL_LI_ONLY", "eval_domain": "Small-LI"},
        ],
        "reuse_phase3_validation_if_comparable": [
            {
                "encoder": "MIXED_1TO1",
                "eval_domain": "Small-HI",
                "requires": "protocol/feature/scaler/probe comparability documented",
            },
            {
                "encoder": "MIXED_1TO1",
                "eval_domain": "SAML-D",
                "requires": "protocol/feature/scaler/probe comparability documented",
            },
            {
                "encoder": "SMALL_HI_ONLY",
                "eval_domain": "Small-HI",
                "requires": "protocol/feature/scaler/probe comparability documented",
            },
            {
                "encoder": "SAMLD_ONLY",
                "eval_domain": "SAML-D",
                "requires": "protocol/feature/scaler/probe comparability documented",
            },
        ],
        "do_not_silently_merge_incompatible_values": True,
    }
    write_json(out_root / "proposed_frozen_eval.json", proposed_frozen)

    integrity = {
        "ok": all(arm_summaries[a].get("ok") for a in ARMS) and bool(li_cmp.get("ok")),
        "phase": "4b_scout",
        "job_states": {
            a: {
                "job_id": arm_summaries[a].get("job_id"),
                "array_task_id": arm_summaries[a].get("array_task_id"),
                "ok": arm_summaries[a].get("ok"),
                "elapsed_sec": arm_summaries[a].get("elapsed_sec"),
            }
            for a in ARMS
        },
        "resolved_configs": {a: arm_summaries[a].get("resolved") for a in ARMS},
        "checkpoints": {a: arm_summaries[a].get("checkpoints") for a in ARMS},
        "checkpoint_reload": {a: arm_summaries[a].get("checkpoint_reload") for a in ARMS},
        "init_sha256_by_arm": init_shas,
        "init_state_equality": init_eq,
        "update_exposure_counts": {a: arm_summaries[a].get("exposures") for a in ARMS},
        "seed_stream_matching": {
            "small_li_mixed_vs_specialist_first_500": li_cmp,
            "phase3_hi_mixed_first_500": hi_cmp,
            "phase3_samld_mixed_first_500": sd_cmp,
        },
        "alpha_beta_freeze_policy": {
            a: arm_summaries[a].get("alpha_beta") for a in ARMS
        },
        "memory_runtime": {a: arm_summaries[a].get("memory_runtime") for a in ARMS},
        "gates": {a: arm_summaries[a].get("gates") for a in ARMS},
        "test_access": {a: arm_summaries[a].get("test_access") for a in ARMS},
        "proposed_frozen_eval_unsubmitted": True,
        "no_extraction_probe_test_dag_submitted": True,
        "representation_quality_not_claimed_from_training_losses": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # If Phase-3 full match failed only due to limitation flag with ok False but
    # dry-match passed in arm summaries, keep integrity ok when LI match + arms ok
    # and Phase-3 limitation is documented — but require equality when hashes exist.
    if isinstance(hi_cmp, dict) and hi_cmp.get("limitation"):
        pass
    elif isinstance(hi_cmp, dict) and hi_cmp.get("ok") is False:
        integrity["ok"] = False
    if isinstance(sd_cmp, dict) and sd_cmp.get("limitation"):
        pass
    elif isinstance(sd_cmp, dict) and sd_cmp.get("ok") is False:
        integrity["ok"] = False

    write_json(out_root / "training_integrity_summary.json", integrity)
    write_json(ROOT / f"{RESULT_ROOT}.json", integrity)
    return integrity


def run_focused_tests_payload() -> Dict[str, Any]:
    """CPU-only focused checks written before submission."""
    from mixed_ssl_phase4a.preflight import assert_old_contract_unchanged
    from mixed_ssl_phase4a.schedule import first_alpha_beta_update_step

    results = {}
    results["old_contract"] = assert_old_contract_unchanged()
    results["matching_contract"] = assert_matching_contract_guaranteed()
    results["mixed_ab_step"] = first_alpha_beta_update_step(
        n_domains=3, calib_obs_per_domain=CALIB_OBS_PER_DOMAIN
    )
    assert results["mixed_ab_step"] == 16
    assert arm_alpha_freeze_until("MIXED_3DOMAIN") == 15
    assert arm_first_ab_update("MIXED_3DOMAIN") == 16
    assert arm_alpha_freeze_until(MIXED_3DOMAIN_LONG) == 15
    assert arm_first_ab_update(MIXED_3DOMAIN_LONG) == 16
    assert arm_alpha_freeze_until("SMALL_LI_ONLY") == 10
    assert arm_first_ab_update("SMALL_LI_ONLY") == 11
    w_m, d_m = arm_warmup_decay("MIXED_3DOMAIN")
    w_l, d_l = arm_warmup_decay("SMALL_LI_ONLY")
    w_long, d_long = arm_warmup_decay(MIXED_3DOMAIN_LONG)
    results["lr"] = {
        "MIXED_3DOMAIN": {"warmup": w_m, "decay": d_m, "total": w_m + d_m},
        "SMALL_LI_ONLY": {"warmup": w_l, "decay": d_l, "total": w_l + d_l},
        MIXED_3DOMAIN_LONG: {
            "warmup": w_long,
            "decay": d_long,
            "total": w_long + d_long,
        },
        "phase3_fraction": 0.20,
        "not_prompt_10pct_guess": True,
        "long_starts_from_shared_init_not_step1500": True,
    }
    assert w_m == 300 and d_m == 1200
    assert w_l == 200 and d_l == 800
    assert w_long == 600 and d_long == 2400
    assert arm_total_steps(MIXED_3DOMAIN_LONG) == 3000
    assert arm_checkpoint_steps(MIXED_3DOMAIN_LONG) == (750, 1500, 2250, 3000)
    assert arm_schedule(MIXED_3DOMAIN_LONG)[:1500] == arm_schedule("MIXED_3DOMAIN")

    init_path = ROOT / PHASE3_SHARED_INIT
    results["init_file"] = {
        "exists": init_path.is_file(),
        "file_sha256": file_sha256(init_path) if init_path.is_file() else None,
    }
    blob = torch.load(init_path, map_location="cpu", weights_only=False)
    sha = str(blob["init_sha256"])
    results["init_file"]["init_sha256"] = sha
    results["init_file"]["prefix_ok"] = sha.startswith(PHASE3_INIT_SHA_PREFIX)
    results["init_file"]["full_sha_ok"] = (
        sha == "8821c986c7394caf504393830dc33a9c3c97ba4d5fdd3bcbaa19f70421c7aebc"
    )
    ew = blob["model_state_dict"]["edge_emb.node__to__node.weight"]
    results["init_file"]["edge_dim"] = int(ew.shape[-1])
    results["init_file"]["edge_dim_ok"] = int(ew.shape[-1]) == 6
    r198_ok = False
    for k, v in blob["model_state_dict"].items():
        if k.endswith("emlps.0.0.node__to__node.weight"):
            r198_ok = int(v.shape[-1]) == 198
            results["init_file"]["r198_in_dim"] = int(v.shape[-1])
            break
    results["init_file"]["r198_ok"] = r198_ok

    # Phase-3 hash availability
    p3_hi = load_phase3_mixed_hashes(ROOT, domain="Small-HI", limit=MIXED_STEPS_PER_DOMAIN)
    p3_sd = load_phase3_mixed_hashes(ROOT, domain="SAML-D", limit=MIXED_STEPS_PER_DOMAIN)
    results["phase3_stream_logs"] = {
        "hi": {
            "n_hashes": p3_hi.get("n_hashes"),
            "first_hash": p3_hi.get("first_hash"),
            "first32": p3_hi.get("first_first32"),
            "full_500_available": p3_hi.get("n_hashes", 0) >= MIXED_STEPS_PER_DOMAIN,
        },
        "samld": {
            "n_hashes": p3_sd.get("n_hashes"),
            "first_hash": p3_sd.get("first_hash"),
            "full_500_available": p3_sd.get("n_hashes", 0) >= MIXED_STEPS_PER_DOMAIN,
        },
    }

    # Phase-4B MIXED hashes required for LONG stream lock
    p4_refs = {}
    p4_ok = True
    for d in CANONICAL_DOMAINS:
        p4 = load_phase4b_mixed_hashes(ROOT, domain=d, limit=MIXED_STEPS_PER_DOMAIN)
        p4_refs[d] = {
            "n_hashes": p4.get("n_hashes"),
            "first_hash": p4.get("first_hash"),
            "full_500_available": p4.get("n_hashes", 0) >= MIXED_STEPS_PER_DOMAIN,
            "path": p4.get("path"),
        }
        p4_ok = p4_ok and bool(p4_refs[d]["full_500_available"])
    results["phase4b_mixed_stream_logs"] = p4_refs

    long_recipe = resolved_recipe(arm=MIXED_3DOMAIN_LONG)
    recipes = {a: resolved_recipe(arm=a) for a in ARMS}
    recipes[MIXED_3DOMAIN_LONG] = long_recipe
    results["resolved_recipes"] = recipes
    results["long_unique_paths"] = {
        "result_root": long_recipe["result_root"],
        "ckpt_root": long_recipe["ckpt_root"],
        "distinct_from_scout": (
            long_recipe["result_root"] != RESULT_ROOT
            and "mixed_long_3000" in str(long_recipe["result_root"])
            and "mixed_3domain_long" in str(long_recipe["ckpt_root"])
        ),
    }
    ckpt_path = ROOT / str(long_recipe["ckpt_root"])
    existing = list(ckpt_path.glob("checkpoint_*.tar")) if ckpt_path.exists() else []
    results["overwrite_risk"] = {
        "ckpt_root": str(ckpt_path),
        "existing_checkpoints": [p.name for p in existing],
        "ok": len(existing) == 0,
    }
    results["ok"] = (
        results["old_contract"]["ok"]
        and results["matching_contract"]["ok"]
        and results["init_file"]["prefix_ok"]
        and results["init_file"]["full_sha_ok"]
        and results["init_file"]["edge_dim_ok"]
        and results["init_file"]["r198_ok"]
        and recipes["MIXED_3DOMAIN"]["loader_num_workers"] == 0
        and recipes["MIXED_3DOMAIN"]["contrast_projection_head"] is False
        and recipes["MIXED_3DOMAIN"]["preserve_seed_edges"] is False
        and recipes["MIXED_3DOMAIN"]["amp"] is False
        and long_recipe["loader_num_workers"] == 0
        and long_recipe["contrast_projection_head"] is False
        and long_recipe["preserve_seed_edges"] is False
        and long_recipe["amp"] is False
        and long_recipe["max_optimizer_steps"] == 3000
        and long_recipe["warmup_steps"] == 600
        and long_recipe["linear_decay_steps"] == 2400
        and long_recipe["starts_from_shared_init_not_step1500_resume"] is True
        and results["long_unique_paths"]["distinct_from_scout"]
        and results["overwrite_risk"]["ok"]
        and p4_ok
    )
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight_only", action="store_true")
    ap.add_argument("--focused_tests", action="store_true")
    ap.add_argument("--arm", choices=list(ALL_TRAINABLE_ARMS), default=None)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--dry_match_batches", type=int, default=4)
    args = ap.parse_args()
    logger_setup()
    logging.getLogger().setLevel(logging.INFO)

    # Config / focused-test writes go under the arm's result root when arm is set;
    # default scout root for legacy 2-arm gates.
    if args.arm == MIXED_3DOMAIN_LONG:
        out_root = ROOT / arm_result_root(MIXED_3DOMAIN_LONG)
    else:
        out_root = ROOT / RESULT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    if args.focused_tests or args.preflight_only:
        payload = run_focused_tests_payload()
        write_json(out_root / "focused_tests.json", payload)
        if args.preflight_only:
            pre = preflight_phase4a(root=ROOT)
            write_json(out_root / "preflight.json", pre)
            payload["preflight_ok"] = pre["ok"]
            payload["ok"] = bool(payload["ok"] and pre["ok"])
            write_json(out_root / "focused_tests.json", payload)
        print(json.dumps({"ok": payload["ok"]}, indent=2))
        if not payload["ok"]:
            sys.exit(1)
        return

    if args.finalize:
        integrity = finalize_phase4b()
        print(json.dumps({"ok": integrity["ok"]}, indent=2))
        if not integrity["ok"]:
            sys.exit(1)
        return

    arm = args.arm
    if arm is None:
        tid = os.environ.get("SLURM_ARRAY_TASK_ID")
        if tid is None:
            raise SystemExit("Provide --arm or SLURM_ARRAY_TASK_ID")
        tid_i = int(tid)
        if tid_i not in ARRAY_INDEX_TO_ARM:
            raise SystemExit(f"Unknown array task {tid_i}")
        arm = ARRAY_INDEX_TO_ARM[tid_i]

    summary = run_arm(arm, dry_match_batches=args.dry_match_batches)
    print(json.dumps({"ok": summary["ok"], "arm": arm, "job_id": summary.get("job_id")}, indent=2))
    if not summary["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
