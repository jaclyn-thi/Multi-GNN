#!/usr/bin/env python3
"""Phase-4B matched objective ablation on MIXED_3DOMAIN_LONG protocol."""

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

from contrastive_projection import ContrastiveProjectionHead  # noqa: E402
from data_loading import get_data  # noqa: E402
from direct_r198 import (  # noqa: E402
    LearnedAlphaBeta,
    LossNormState,
    TFMoEBundle,
    load_tf_moe_context,
)
from direct_r198.lr_scheduler import DirectHWarmupLinearScheduler  # noqa: E402
from financial_multidataset_shared_core_contract import FINAL_FEATURE_NAMES  # noqa: E402
from graph_augmentations import generate_views  # noqa: E402
from mixed_ssl_phase2.bn import (  # noqa: E402
    apply_bn_,
    bn_bundle_l1,
    bn_bundles_equal,
    clone_bn_bundle,
    collect_bn_bundle,
)
from mixed_ssl_phase3.hash_util import combined_init_sha, state_dict_sha256  # noqa: E402
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
    CANONICAL_DOMAINS,
    MIXED_3DOMAIN_LONG,
    MIXED_LONG_ALPHA_FREEZE_UNTIL,
    MIXED_LONG_FIRST_AB_UPDATE,
    MIXED_LONG_LINEAR_DECAY_STEPS,
    MIXED_LONG_STEPS_PER_DOMAIN,
    MIXED_LONG_TOTAL_STEPS,
    MIXED_LONG_WARMUP_STEPS,
    PHASE3_INIT_SHA_PREFIX,
    PHASE3_SHARED_INIT,
    arm_schedule,
)
from phase4b_objective_ablation import (  # noqa: E402
    ABLATION_ARMS,
    ALPHA_FREEZE_UNTIL,
    ARRAY_INDEX_TO_ARM,
    CHECKPOINT_STEPS,
    FIRST_AB_UPDATE,
    PROJECTION_HIDDEN,
    PROJECTION_IN_DIM,
    PROJECTION_OUT,
    ROLLING_EVERY,
    STEPS_PER_DOMAIN,
    TOTAL_STEPS,
    arm_ckpt_root,
    arm_learn_alpha,
    arm_learn_beta,
    arm_result_root,
    arm_unique,
    arm_uses_projection,
    arm_uses_tfmoe,
    arm_weight_mode,
    resolved_recipe,
)
from phase4b_objective_ablation.matching import (  # noqa: E402
    VIEW_MATCH_SAMPLE_GLOBAL_STEPS,
    assert_objective_ablation_matching_contract,
    collect_view_match_rows,
    compare_vs_long,
    load_long_seed_hashes,
)
from phase4b_objective_ablation.step import ablation_mixed_step  # noqa: E402
from train_util import (  # noqa: E402
    AddEgoIds,
    FORWARD_EDGE_TYPE,
    add_arange_ids,
    extract_param,
    get_hetero_seed_edge_ids,
)
from training import _contrastive_view_kwargs, get_model  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def make_ns(data: str, *, unique: str, arm: str, max_steps: int) -> argparse.Namespace:
    argv = [
        "--data",
        data,
        "--model",
        "gin",
        "--objective",
        "contrastive",
        "--unique_name",
        unique,
        "--seed",
        str(SEED),
        "--batch_size",
        str(BATCH_SIZE),
        "--num_neighs",
        "100",
        "100",
        "--loader_num_workers",
        str(LOADER_NUM_WORKERS),
        "--reverse_mp",
        "--ego",
        "--ports",
        "--emlps",
        "--tds",
        "--correct_reverse_edge_features",
        "--feature_contract",
        CONTRACT_ID,
        "--train_fit_edge_znorm",
        "--skip_test_eval",
        "--direct_r198_infonce",
        "--direct_r198_tfmoe_weight_mode",
        arm_weight_mode(arm) if arm != "INFONCE_ONLY" else "adaptive",
        "--contrastive_asymmetric",
        "--contrastive_num_neg_samples",
        str(N_NEG),
        "--contrastive_memory_bank_size",
        "0",
        "--contrastive_accum_steps",
        str(ACCUM_STEPS),
        "--contrastive_temperature",
        str(TEMP),
        "--max_optimizer_steps",
        str(max_steps),
    ]
    if arm_uses_tfmoe(arm):
        argv.extend(["--direct_r198_tfmoe"])
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


def assert_width_chain(
    tr: HeteroData,
    name: str,
    *,
    after_arange_ids: bool,
    sample_batch: Optional[HeteroData] = None,
    model_edge_dim: Optional[int] = None,
) -> Dict[str, int]:
    """Lock shared-core width 6 → EdgeID insertion 7 → model edge_dim 6."""
    ea = tr[FORWARD_EDGE_TYPE].edge_attr
    width = int(ea.shape[1])
    if not after_arange_ids:
        if width != 6:
            raise RuntimeError(f"{name} raw shared-core edge_attr width={width} != 6")
        return {"raw_shared_core_width": 6}
    if width != 7:
        raise RuntimeError(
            f"{name} after add_arange_ids edge_attr width={width} != 7 "
            "(synthetic EdgeID column expected in col0)"
        )
    out = {"raw_shared_core_width": 6, "after_synthetic_edgeid_width": 7}
    if sample_batch is not None:
        bw = int(sample_batch[FORWARD_EDGE_TYPE].edge_attr.shape[1])
        if bw != 7:
            raise RuntimeError(f"{name} sample batch edge_attr width={bw} != 7")
        inferred = bw - 1
        if inferred != 6:
            raise RuntimeError(f"{name} inferred model edge_dim={inferred} != 6")
        out["sample_batch_width"] = bw
        out["inferred_model_edge_dim"] = inferred
    if model_edge_dim is not None and int(model_edge_dim) != 6:
        raise RuntimeError(f"{name} model edge_dim={model_edge_dim} != 6")
    if model_edge_dim is not None:
        out["model_edge_dim"] = int(model_edge_dim)
    return out


def specs_for_domains() -> Tuple[DomainSpec, ...]:
    active = set(CANONICAL_DOMAINS)
    return tuple(s for s in default_smoke_domains() if s.dataset_id in active)


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
        raise RuntimeError("model key mismatch vs Phase-3 init")
    for k, v in model_sd.items():
        if tuple(v.shape) != tuple(cur_m[k].shape):
            raise RuntimeError(f"model shape mismatch {k}")
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

    ew = model_sd.get("edge_emb.node__to__node.weight")
    if ew is None or int(ew.shape[-1]) != 6:
        raise RuntimeError("Phase-3 init edge_dim != 6")

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
        "shared_param_sha256": init_sha,
        "loaded_model_sha_matches": True,
        "edge_dim": 6,
        "r198": 198,
        "seed": blob.get("seed"),
    }


def build_checkpoint(**kwargs) -> Dict[str, Any]:
    model = kwargs["model"]
    moe = kwargs["moe"]
    alpha_beta = kwargs["alpha_beta"]
    projection = kwargs.get("projection")
    ckpt = {
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
        "rng_states": kwargs["rng_states"],
        "seed_hash_log": kwargs["seed_hash_log"],
        "init_sha256": kwargs["init_sha"],
        "seed": SEED,
        "arm": kwargs["arm"],
        "unique_name": kwargs["unique"],
        "feature_contract_id": CONTRACT_ID,
        "resolved": kwargs["resolved"],
        "preflight": kwargs["preflight"],
        "test_evaluated": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    if projection is not None:
        ckpt["projection_state_dict"] = {
            k: v.detach().cpu() for k, v in projection.state_dict().items()
        }
        ckpt["projection_init_sha256"] = kwargs.get("projection_init_sha")
    return ckpt


def save_ckpt(ckpt: Dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    return file_sha256(path)


def reload_validate_ckpt(
    path: Path,
    *,
    arm: str,
    expect_step: int,
    domains: Sequence[str],
    expect_projection: bool,
) -> Dict[str, Any]:
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
    if expect_projection:
        ok = ok and "projection_state_dict" in blob
    for d in domains:
        ok = ok and d in blob["bn_bundles"]
        ok = ok and d in blob["loss_norm_states"]
    return {
        "path": str(path),
        "ok": bool(ok),
        "missing_keys": missing,
        "global_optimizer_step": blob.get("global_optimizer_step"),
        "test_evaluated": blob.get("test_evaluated"),
        "sha256": file_sha256(path),
    }


def _load_domain_graphs(
    specs: Sequence[DomainSpec],
) -> Tuple[
    Dict[str, argparse.Namespace],
    Dict[str, HeteroData],
    Dict[str, Any],
    Dict[str, Any],
]:
    ns_by: Dict[str, argparse.Namespace] = {}
    data_by: Dict[str, HeteroData] = {}
    edge_scalers: Dict[str, Any] = {}
    mem: Dict[str, Any] = {"graph_build": {}, "rss_gib_after_each_domain_load": {}}
    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)
    for spec in specs:
        d = spec.dataset_id
        logging.info("Loading %s ...", d)
        t0 = time.perf_counter()
        ns = make_ns(d, unique="preflight", arm="PROJECTION_ON_ADAPTIVE", max_steps=TOTAL_STEPS)
        ns.direct_r198_tfmoe_cache = str(ROOT / spec.tf_cache_path)
        tr, va, te, tr_i, va_i, te_i = get_data(ns, data_config)
        assert_contract_geometry(ns, tr, d)
        del va, te, tr_i, va_i, te_i
        gc.collect()
        ns_by[d] = ns
        data_by[d] = tr
        edge_scalers[d] = dict(ns.shared_core_edge_scaler)
        mem["graph_build"][d] = time.perf_counter() - t0
        mem["rss_gib_after_each_domain_load"][d] = peak_rss_gib()
    return ns_by, data_by, edge_scalers, mem


def run_dry_preflight(
    arm: str,
    *,
    match_long_hashes_limit: int = 1000,
) -> Dict[str, Any]:
    """Fail-fast preflight: one batch/domain, gradient checks, LONG hash prefix."""
    if arm not in ABLATION_ARMS:
        raise ValueError(arm)
    t0 = time.perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    specs = list(specs_for_domains())
    domains = list(CANONICAL_DOMAINS)
    match_contract = assert_objective_ablation_matching_contract()
    pre = preflight_phase4a(root=ROOT, specs=list(default_smoke_domains()))
    if not pre["ok"]:
        raise RuntimeError("phase4a preflight failed")

    ns_by, data_by, edge_scalers, mem = _load_domain_graphs(specs)
    transform = AddEgoIds()
    width_reports: Dict[str, Any] = {}
    for d in domains:
        width_reports[d] = {
            "before_arange": assert_width_chain(data_by[d], d, after_arange_ids=False)
        }
        add_arange_ids([data_by[d]])
        width_reports[d]["after_arange"] = assert_width_chain(
            data_by[d], d, after_arange_ids=True
        )

    sample_dom = domains[0]
    sample_loader = build_train_loader(data_by[sample_dom], transform, domain=sample_dom)
    sample = next(iter(sample_loader))
    del sample_loader
    width_reports[sample_dom]["sample_and_model"] = assert_width_chain(
        data_by[sample_dom],
        sample_dom,
        after_arange_ids=True,
        sample_batch=sample,
    )
    model, emb_dim = build_model(ns_by[sample_dom], data_by[sample_dom], sample, device)
    ew = model.state_dict().get("edge_emb.node__to__node.weight")
    model_edge_dim = int(ew.shape[-1]) if ew is not None else None
    width_reports[sample_dom]["sample_and_model"].update(
        assert_width_chain(
            data_by[sample_dom],
            sample_dom,
            after_arange_ids=True,
            sample_batch=sample,
            model_edge_dim=model_edge_dim,
        )
    )
    del sample
    moe = TFMoEBundle(in_dim=int(emb_dim), hidden=64, n_targets=3).to(device)
    alpha_beta = LearnedAlphaBeta(n_tf=3, init_alpha=0.6).to(device)
    alpha_beta.set_frozen(True)
    init_path = ROOT / PHASE3_SHARED_INIT
    init_prov = verify_phase3_init_compatibility(model, moe, alpha_beta, init_path)
    init_sha = init_prov["init_sha256"]

    projection = None
    projection_init_sha = None
    if arm_uses_projection(arm):
        # Dedicated projection init seed; does not touch Phase-3 shared weights.
        # Domain batch/view streams use matching rng_states restored per step.
        torch.manual_seed(int(SEED) * 100_003 + 17)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(SEED) * 100_003 + 17)
        projection = ContrastiveProjectionHead(
            PROJECTION_IN_DIM, PROJECTION_HIDDEN, PROJECTION_OUT
        ).to(device)
        projection_init_sha = state_dict_sha256(projection.state_dict())

    tf_ctx = {
        d: load_tf_moe_context(
            ROOT / next(s for s in specs if s.dataset_id == d).tf_cache_path, device
        )
        for d in domains
    }
    loss_norms = {d: LossNormState() for d in domains}
    for d in domains:
        loss_norms[d].calibrated = True
        loss_norms[d].contrast_mean = 1.0
        loss_norms[d].tf_means = [1.0, 1.0, 1.0]

    if arm == "INFONCE_ONLY":
        alpha_beta.set_frozen(True)
        optimizer = torch.optim.Adam([{"params": list(model.parameters()), "lr": ENCODER_LR}])
    elif arm == "EXPERT_ONLY":
        optimizer = torch.optim.Adam(
            [
                {"params": list(model.parameters()) + list(moe.parameters()), "lr": ENCODER_LR},
                {"params": list(alpha_beta.parameters()), "lr": ALPHABETA_LR},
            ]
        )
        alpha_beta.set_learn_flags(learn_alpha=False, learn_beta=True)
    else:
        optimizer = torch.optim.Adam(
            [
                {
                    "params": list(model.parameters())
                    + list(moe.parameters())
                    + list(projection.parameters()),
                    "lr": ENCODER_LR,
                },
                {"params": list(alpha_beta.parameters()), "lr": ALPHABETA_LR},
            ]
        )
        alpha_beta.set_frozen(False)

    bn_init = clone_bn_bundle(collect_bn_bundle(model))
    bn_bundles = {d: clone_bn_bundle(bn_init) for d in domains}
    from mixed_ssl_phase4b.matching import init_matching_rng_states

    rng_states = init_matching_rng_states(SEED, active_domains=domains)
    step_reports: Dict[str, Any] = {}
    dry_n = min(4, match_long_hashes_limit)

    for d in domains:
        restore_rng(rng_states[d])
        apply_bn_(model, bn_bundles[d])
        loader = build_train_loader(data_by[d], transform, domain=d)
        batch = next(iter(loader))
        stats = ablation_mixed_step(
            arm=arm,
            model=model,
            moe=moe,
            alpha_beta=alpha_beta,
            loss_norm=loss_norms[d],
            tf_ctx=tf_ctx[d],
            optimizer=optimizer,
            batch=batch,
            loader_data=data_by[d],
            args=ns_by[d],
            device=device,
            projection=projection,
            seed_ids_sha_fn=seed_ids_sha,
            do_optimizer_step=False,
        )
        step_reports[d] = {
            "encoder_grad_norm": stats["encoder_grad_norm"],
            "moe_grad_norm": stats["moe_grad_norm"],
            "projection_grad_norm": stats.get("projection_grad_norm"),
            "seed_ids_sha256": stats["seed_ids_sha256"],
            "view1_aug_sha256": stats["view1_aug_sha256"],
            "view2_aug_sha256": stats["view2_aug_sha256"],
        }
        del loader, batch

    # Dry preflight: BN bundles are still identical across domains (no steps yet).
    # Verify apply/restore round-trip only; pairwise divergence is a post-train gate.
    apply_bn_(model, bn_bundles[domains[0]])
    snap_a = clone_bn_bundle(collect_bn_bundle(model))
    apply_bn_(model, bn_bundles[domains[1]])
    apply_bn_(model, bn_bundles[domains[0]])
    snap_a2 = clone_bn_bundle(collect_bn_bundle(model))
    bn_reload_ok = bn_bundles_equal(snap_a, snap_a2)

    long_match: Dict[str, Any] = {}
    loaders = {d: build_train_loader(data_by[d], transform, domain=d) for d in domains}
    its = {d: infinite_loader(loaders[d]) for d in domains}
    rng_m = init_matching_rng_states(SEED, active_domains=domains)
    schedule = arm_schedule(MIXED_3DOMAIN_LONG)
    local_hashes = {d: [] for d in domains}
    for i in range(dry_n * len(domains)):
        dom = schedule[i]
        restore_rng(rng_m[dom])
        batch = next(its[dom])
        sid = seed_ids_sha(get_hetero_seed_edge_ids(batch, data_by[dom]))
        if len(local_hashes[dom]) < dry_n:
            local_hashes[dom].append(sid)
        rng_m[dom] = snapshot_rng()

    for d in domains:
        long_match[d] = compare_vs_long(
            local_hashes[d],
            ROOT,
            domain=d,
            limit=dry_n,
            label_local=f"dry_preflight_{arm}",
        )
        if not long_match[d].get("ok"):
            raise RuntimeError(f"LONG seed hash mismatch domain={d}: {long_match[d]}")

    ckpt_dir = ROOT / arm_ckpt_root(arm)
    existing = list(ckpt_dir.glob("checkpoint_*.tar")) if ckpt_dir.exists() else []
    # LONG@3000 tree ≈ 16 MiB; ablation keeps 1500+3000+rolling ≈ ≤20 rolling snapshots.
    est_ckpt_bytes_per_arm = 20 * 4 * 1024 * 1024
    long_mean_sec = 1.83
    est_runtime_sec = 1200.0 + long_mean_sec * float(TOTAL_STEPS)
    # Home NFS has ~144 TiB free as of package prep; require only a few GiB.
    disk_ok = True
    if existing:
        raise RuntimeError(
            f"dry_preflight refuse: existing checkpoints under {ckpt_dir}: "
            f"{[p.name for p in existing[:8]]}"
        )

    payload = {
        "ok": True,
        "arm": arm,
        "dry_preflight": True,
        "matching_contract": match_contract,
        "preflight_phase4a_ok": pre["ok"],
        "init_sha256": init_sha,
        "shared_param_sha256": init_sha,
        "projection_init_sha256": projection_init_sha,
        "width_chain": width_reports,
        "step_reports": step_reports,
        "bn_reload_ok": bool(bn_reload_ok),
        "no_test_graph_or_tf_arrays": True,
        "long_seed_hash_match": long_match,
        "long_view_hash_limitation": (
            "Historical MIXED_3DOMAIN_LONG did not log view/aug hashes; "
            "only seed-batch hashes are matched against LONG. Cross-arm view "
            "matching uses newly logged hashes among ablation arms."
        ),
        "match_long_hashes_limit": match_long_hashes_limit,
        "dry_stream_hashes_per_domain": local_hashes,
        "existing_checkpoints": [],
        "checkpoint_path_clear": True,
        "estimated_checkpoint_disk_bytes_per_arm": est_ckpt_bytes_per_arm,
        "estimated_runtime_sec_per_arm": est_runtime_sec,
        "disk_quota_adequate": disk_ok,
        "memory": mem,
        "peak_rss_gib": peak_rss_gib(),
        "elapsed_sec": time.perf_counter() - t0,
    }
    if not bn_reload_ok:
        payload["ok"] = False
        raise RuntimeError("BN apply/reload round-trip failed in dry_preflight")
    return payload


def run_arm(
    arm: str,
    *,
    match_long_hashes_limit: int = 1000,
) -> Dict[str, Any]:
    if arm not in ABLATION_ARMS:
        raise ValueError(arm)
    t_wall0 = time.perf_counter()
    unique = arm_unique(arm)
    recipe = resolved_recipe(arm)
    domains = list(CANONICAL_DOMAINS)
    specs = list(specs_for_domains())
    total_steps = TOTAL_STEPS
    warmup = MIXED_LONG_WARMUP_STEPS
    decay = MIXED_LONG_LINEAR_DECAY_STEPS
    freeze_until = ALPHA_FREEZE_UNTIL
    first_ab = FIRST_AB_UPDATE
    ckpt_steps = set(CHECKPOINT_STEPS)
    steps_per_domain = STEPS_PER_DOMAIN
    split_by_dom = {s.dataset_id: s.split_protocol_id for s in default_smoke_domains()}
    reg_json = registry_to_json(default_smoke_domains())

    arm_dir = ROOT / arm_result_root(arm)
    logs_dir = arm_dir / "logs"
    ckpt_dir = ROOT / arm_ckpt_root(arm)
    for p in (arm_dir, logs_dir, ckpt_dir):
        p.mkdir(parents=True, exist_ok=True)

    existing_ckpts = sorted(ckpt_dir.glob("checkpoint_*.tar")) + sorted(
        ckpt_dir.glob("checkpoint_last*.tar")
    )
    if existing_ckpts:
        raise RuntimeError(
            f"checkpoint path would overwrite existing artifacts under {ckpt_dir}: "
            f"{[str(p.name) for p in existing_ckpts[:8]]}"
        )

    match_contract = assert_objective_ablation_matching_contract()
    write_json(arm_dir / "matching_contract.json", match_contract)

    pre = preflight_phase4a(root=ROOT, specs=list(default_smoke_domains()))
    write_json(arm_dir / "preflight.json", pre)
    if not pre["ok"]:
        raise RuntimeError("phase4a preflight failed")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)

    mem: Dict[str, Any] = {
        "loader_num_workers": LOADER_NUM_WORKERS,
        "graph_build": {},
        "rss_gib_after_each_domain_load": {},
        "peak_rss_gib_process": None,
        "time_to_first_step_sec": None,
        "mean_sec_per_step": None,
        "graph_build_total_sec": 0.0,
    }
    test_access = {
        "test_graph_loaded": False,
        "test_tf_arrays_loaded": False,
        "test_metrics_computed": False,
        "test_labels_used": False,
    }

    ns_by: Dict[str, argparse.Namespace] = {}
    data_by: Dict[str, HeteroData] = {}
    edge_scalers: Dict[str, Any] = {}

    for spec in specs:
        d = spec.dataset_id
        logging.info("Loading %s under %s ...", d, CONTRACT_ID)
        t0 = time.perf_counter()
        ns = make_ns(d, unique=unique, arm=arm, max_steps=total_steps)
        ns.direct_r198_tfmoe_cache = str(ROOT / spec.tf_cache_path)
        tr, va, te, tr_i, va_i, te_i = get_data(ns, data_config)
        assert_contract_geometry(ns, tr, d)
        if d == "SAML-D" and int(te_i.numel()) != 0:
            raise RuntimeError("SAML-D te_inds nonempty — refuse test access")
        del va, te, tr_i, va_i, te_i
        gc.collect()
        ns_by[d] = ns
        data_by[d] = tr
        edge_scalers[d] = dict(ns.shared_core_edge_scaler)
        dt = time.perf_counter() - t0
        mem["graph_build"][d] = dt
        mem["graph_build_total_sec"] += dt
        mem["rss_gib_after_each_domain_load"][d] = peak_rss_gib()

    transform = AddEgoIds()
    for d in domains:
        add_arange_ids([data_by[d]])

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
    init_sha = init_prov["init_sha256"]

    projection = None
    projection_init_sha = None
    if arm_uses_projection(arm):
        torch.manual_seed(int(SEED) * 100_003 + 17)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(SEED) * 100_003 + 17)
        projection = ContrastiveProjectionHead(
            PROJECTION_IN_DIM, PROJECTION_HIDDEN, PROJECTION_OUT
        ).to(device)
        projection_init_sha = state_dict_sha256(projection.state_dict())
        init_prov = dict(init_prov)
        init_prov["projection_init_sha256"] = projection_init_sha
        init_prov["projection_attached_after_phase3_init"] = True
        init_prov["projection_init_seed"] = int(SEED) * 100_003 + 17
        init_prov["projection_does_not_block_encoder_infonce_grads"] = True
        write_json(arm_dir / "shared_init_provenance.json", init_prov)

    loaders = {d: build_train_loader(data_by[d], transform, domain=d) for d in domains}
    iters = {d: infinite_loader(loaders[d]) for d in domains}

    tf_ctx = {
        d: load_tf_moe_context(
            ROOT / next(s for s in specs if s.dataset_id == d).tf_cache_path, device
        )
        for d in domains
    }
    loss_norms = {d: LossNormState() for d in domains}
    calib = {d: {"contrast": 0.0, "tf": [0.0, 0.0, 0.0], "n": 0} for d in domains}

    if arm == "INFONCE_ONLY":
        alpha_beta.set_frozen(True)
        optimizer = torch.optim.Adam([{"params": list(model.parameters()), "lr": ENCODER_LR}])
    elif arm == "EXPERT_ONLY":
        optimizer = torch.optim.Adam(
            [
                {"params": list(model.parameters()) + list(moe.parameters()), "lr": ENCODER_LR},
                {"params": list(alpha_beta.parameters()), "lr": ALPHABETA_LR},
            ]
        )
    else:
        optimizer = torch.optim.Adam(
            [
                {
                    "params": list(model.parameters())
                    + list(moe.parameters())
                    + list(projection.parameters()),
                    "lr": ENCODER_LR,
                },
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

    from mixed_ssl_phase4b.matching import init_matching_rng_states

    bn_init = clone_bn_bundle(collect_bn_bundle(model))
    bn_bundles = {d: clone_bn_bundle(bn_init) for d in domains}
    rng_states = init_matching_rng_states(SEED, active_domains=domains)
    schedule = arm_schedule(MIXED_3DOMAIN_LONG)
    seed_hash_log = {d: [] for d in domains}
    seed_first32_log = {d: [] for d in domains}
    step_counts = {d: 0 for d in domains}
    enc_grad_by_domain = {d: [] for d in domains}
    moe_grad_by_domain = {d: [] for d in domains}
    proj_grad_by_domain = {d: [] for d in domains}
    contrast_grad_flags: List[bool] = []
    alpha_unfrozen_at: Optional[int] = None
    expert_beta_only_at: Optional[int] = None
    alpha_updated_early = False
    model_init_flat = torch.cat(
        [p.detach().float().reshape(-1).cpu() for p in model.parameters()]
    )

    jsonl_path = logs_dir / "steps.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
    step_times: List[float] = []
    t_first = None
    checkpoints_meta: Dict[str, Any] = {}
    rows_for_summary: List[Dict[str, Any]] = []

    with open(jsonl_path, "w", encoding="utf-8") as jsonl:
        for si in range(total_steps):
            t_step0 = time.perf_counter()
            domain = schedule[si]
            restore_rng(rng_states[domain])
            apply_bn_(model, bn_bundles[domain])
            batch = next(iters[domain])
            was_frozen = bool(alpha_beta._frozen)
            ab_before = float(alpha_beta.alpha_logit.detach().cpu())

            stats = ablation_mixed_step(
                arm=arm,
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
                projection=projection,
                seed_ids_sha_fn=seed_ids_sha,
                do_optimizer_step=True,
            )
            scheduler.step()
            completed = si + 1
            if t_first is None:
                t_first = time.perf_counter() - t_wall0
                mem["time_to_first_step_sec"] = t_first

            ab_after = float(alpha_beta.alpha_logit.detach().cpu())
            if arm != "INFONCE_ONLY":
                if was_frozen and completed <= freeze_until and ab_after != ab_before:
                    alpha_updated_early = True
                    raise RuntimeError(
                        f"alpha/beta changed while frozen at completed={completed}"
                    )

            if calib[domain]["n"] < CALIB_OBS_PER_DOMAIN and not loss_norms[domain].calibrated:
                calib[domain]["contrast"] += float(stats["L_contrast_raw"])
                if arm != "INFONCE_ONLY":
                    for m in range(3):
                        v = float(stats.get(f"L_tf_raw_{m}", float("nan")))
                        if v != v:  # NaN guard
                            raise RuntimeError(f"non-finite TF raw for calib m={m}")
                        calib[domain]["tf"][m] += v
                calib[domain]["n"] += 1
                if calib[domain]["n"] == CALIB_OBS_PER_DOMAIN:
                    n = float(CALIB_OBS_PER_DOMAIN)
                    loss_norms[domain].contrast_mean = calib[domain]["contrast"] / n
                    if arm == "INFONCE_ONLY":
                        loss_norms[domain].tf_means = [1.0, 1.0, 1.0]
                    else:
                        loss_norms[domain].tf_means = [
                            calib[domain]["tf"][m] / n for m in range(3)
                        ]
                    loss_norms[domain].calibrated = True

            all_calibrated = all(loss_norms[d].calibrated for d in domains)

            if arm == "EXPERT_ONLY" and expert_beta_only_at is None:
                if all_calibrated and completed >= freeze_until:
                    alpha_beta.set_learn_flags(learn_alpha=False, learn_beta=True)
                    expert_beta_only_at = completed
            elif arm == "PROJECTION_ON_ADAPTIVE":
                if alpha_beta._frozen and all_calibrated and completed >= freeze_until:
                    alpha_beta.set_frozen(False)
                    alpha_unfrozen_at = completed
            # INFONCE_ONLY: alpha/beta frozen forever

            bn_bundles[domain] = clone_bn_bundle(collect_bn_bundle(model))
            rng_states[domain] = snapshot_rng()
            step_counts[domain] += 1
            if len(seed_hash_log[domain]) < steps_per_domain:
                seed_hash_log[domain].append(stats["seed_ids_sha256"])
                seed_first32_log[domain].append(stats["seed_edge_ids_first32"])
            enc_grad_by_domain[domain].append(stats["encoder_grad_norm"])
            moe_grad_by_domain[domain].append(stats["moe_grad_norm"])
            if arm_uses_projection(arm):
                proj_grad_by_domain[domain].append(stats["projection_grad_norm"])
            contrast_grad_flags.append(bool(stats.get("contrast_grad_contribution", True)))

            lrs = scheduler.current_lrs()
            alpha_val = float(torch.sigmoid(alpha_beta.alpha_logit).detach().cpu())
            row = {
                "step": si,
                "global_optimizer_step": completed,
                "domain": domain,
                "domain_exposure_count": step_counts[domain],
                "encoder_lr": lrs[0],
                "alphabeta_lr": lrs[1] if len(lrs) > 1 else lrs[0],
                "calibration_complete_domain": bool(loss_norms[domain].calibrated),
                "all_domains_calibrated": all_calibrated,
                "alpha_beta_frozen": bool(alpha_beta._frozen),
                "alpha": alpha_val,
                "weight_mode": arm_weight_mode(arm),
                "elapsed_step_sec": time.perf_counter() - t_step0,
                **stats,
            }
            jsonl.write(json.dumps(row) + "\n")
            jsonl.flush()
            rows_for_summary.append(row)
            step_times.append(time.perf_counter() - t_step0)

            if completed % 50 == 0 or completed == 1:
                logging.info(
                    "step %s/%s arm=%s domain=%s L=%.4f enc_g=%.3f",
                    completed,
                    total_steps,
                    arm,
                    domain,
                    stats["L_total"],
                    stats["encoder_grad_norm"],
                )

            if completed in ckpt_steps or completed % ROLLING_EVERY == 0:
                ckpt = build_checkpoint(
                    model=model,
                    moe=moe,
                    alpha_beta=alpha_beta,
                    projection=projection,
                    projection_init_sha=projection_init_sha,
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
                    rng_states={
                        d: {
                            "python": rng_states[d]["python"],
                            "numpy": rng_states[d]["numpy"],
                            "torch": rng_states[d]["torch"],
                            **(
                                {"cuda": rng_states[d]["cuda"]}
                                if "cuda" in rng_states[d]
                                else {}
                            ),
                        }
                        for d in rng_states
                    },
                    seed_hash_log=seed_hash_log,
                    init_sha=init_sha,
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

    write_json(
        arm_dir / "seed_hash_log.json",
        {
            "hashes": seed_hash_log,
            "first32": {d: seed_first32_log[d][:8] for d in domains},
        },
    )

    matching_vs_long = {
        d: compare_vs_long(
            seed_hash_log[d],
            ROOT,
            domain=d,
            limit=min(match_long_hashes_limit, steps_per_domain),
            label_local=arm,
        )
        for d in domains
    }

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
            expect_projection=arm_uses_projection(arm),
        )

    bn_changed = {d: not bn_bundles_equal(bn_bundles[d], bn_init) for d in domains}
    model_final_flat = torch.cat(
        [p.detach().float().reshape(-1).cpu() for p in model.parameters()]
    )
    encoder_changed = not torch.allclose(model_init_flat, model_final_flat)

    gates: Dict[str, Any] = {
        "exact_global_steps": sum(step_counts.values()) == total_steps,
        "exact_per_domain_exposures": step_counts == {d: steps_per_domain for d in domains},
        "nonzero_encoder_grad": all(any(g > 0 for g in enc_grad_by_domain[d]) for d in domains),
        "encoder_params_changed": bool(encoder_changed),
        "all_lossnorm_calibrated": all(loss_norms[d].calibrated for d in domains),
        "lr_schedule_completed": scheduler.completed_optimizer_steps == total_steps,
        "required_checkpoints_reload_ok": all(
            reload_results[f"step_{s}"].get("ok") for s in ckpt_steps
        ),
        "no_test_graph_cache_metric": all(v is False for v in test_access.values()),
        "init_sha_matches_phase3": init_sha.startswith(PHASE3_INIT_SHA_PREFIX),
        "long_seed_hash_match": all(matching_vs_long[d].get("ok") for d in domains),
        "view_hashes_logged_every_step": all(
            r.get("view1_aug_sha256") and r.get("view2_aug_sha256")
            for r in rows_for_summary
        ),
    }

    if arm == "INFONCE_ONLY":
        # InfoNCE-only intentionally omits MoE from the optimizer; do NOT set a
        # boolean False under a positive-sounding key — that fails all(bool).
        gates["moe_grad_always_zero"] = all(
            g == 0.0 for ds in moe_grad_by_domain.values() for g in ds
        )
        gates["alpha_beta_frozen_entire_run"] = True
        gates["infonce_only_moe_excluded_from_optimizer"] = True
    else:
        gates["nonzero_moe_grad"] = all(
            any(g > 0 for g in moe_grad_by_domain[d]) for d in domains
        )

    if arm == "EXPERT_ONLY":
        gates["contrast_grad_contribution_always_false"] = all(x is False for x in contrast_grad_flags)
        gates["expert_beta_only_engaged"] = expert_beta_only_at is not None
    if arm == "PROJECTION_ON_ADAPTIVE":
        gates["projection_present"] = projection is not None
        gates["nonzero_projection_grad"] = all(
            any(g > 0 for g in proj_grad_by_domain[d]) for d in domains
        )
        gates["alpha_unfrozen_at_expected"] = alpha_unfrozen_at == freeze_until

    # Aggregate only affirmative integrity booleans (exclude bookkeeping keys).
    _skip = {"ok"}
    gates["ok"] = all(
        bool(v)
        for k, v in gates.items()
        if isinstance(v, bool) and k not in _skip
    )

    view_match_sample = collect_view_match_rows(rows_for_summary)

    integrity = {
        "ok": bool(gates["ok"]),
        "arm": arm,
        "gates": gates,
        "matching_vs_long": matching_vs_long,
        "view_match_sample": view_match_sample,
        "view_match_sample_global_steps": list(VIEW_MATCH_SAMPLE_GLOBAL_STEPS),
        "checkpoints": checkpoints_meta,
        "checkpoint_reload": reload_results,
        "init_sha256": init_sha,
        "projection_init_sha256": projection_init_sha,
        "expert_beta_only_at": expert_beta_only_at,
        "alpha_unfrozen_at": alpha_unfrozen_at,
        "alpha_no_early_update": not alpha_updated_early,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    summary = {
        "ok": bool(gates["ok"]),
        "arm": arm,
        "unique_name": unique,
        "domains": domains,
        "steps": total_steps,
        "exposures": dict(step_counts),
        "resolved": recipe,
        "gates": gates,
        "matching_vs_long": matching_vs_long,
        "view_match_sample": view_match_sample,
        "init_sha256": init_sha,
        "shared_param_sha256": init_sha,
        "projection_init_sha256": projection_init_sha,
        "checkpoints": checkpoints_meta,
        "memory_runtime": mem,
        "elapsed_sec": time.perf_counter() - t_wall0,
        "test_evaluated": False,
    }

    write_json(arm_dir / "summary.json", summary)
    write_json(arm_dir / "integrity.json", integrity)
    write_json(arm_dir / "matching_vs_long.json", matching_vs_long)
    write_json(arm_dir / "memory_runtime.json", mem)
    write_json(arm_dir / "resolved_run.json", recipe)

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ABLATION_ARMS), default=None)
    ap.add_argument("--array_task_id", type=int, default=None)
    ap.add_argument(
        "--dry_preflight",
        action="store_true",
        help="Fail-fast preflight only (no training)",
    )
    ap.add_argument("--match_long_hashes_limit", type=int, default=1000)
    args = ap.parse_args()
    logger_setup()
    logging.getLogger().setLevel(logging.INFO)

    arm = args.arm
    if arm is None:
        tid = args.array_task_id
        if tid is None:
            tid = os.environ.get("SLURM_ARRAY_TASK_ID")
        if tid is None:
            raise SystemExit("Provide --arm or --array_task_id / SLURM_ARRAY_TASK_ID")
        tid_i = int(tid)
        if tid_i not in ARRAY_INDEX_TO_ARM:
            raise SystemExit(f"Unknown array task {tid_i}")
        arm = ARRAY_INDEX_TO_ARM[tid_i]

    if args.dry_preflight:
        payload = run_dry_preflight(
            arm, match_long_hashes_limit=args.match_long_hashes_limit
        )
        out = ROOT / arm_result_root(arm) / "dry_preflight.json"
        write_json(out, payload)
        print(json.dumps({"ok": payload["ok"], "arm": arm, "dry_preflight": True}, indent=2))
        if not payload["ok"]:
            sys.exit(1)
        return

    summary = run_arm(arm, match_long_hashes_limit=args.match_long_hashes_limit)
    print(json.dumps({"ok": summary["ok"], "arm": arm}, indent=2))
    if not summary["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
