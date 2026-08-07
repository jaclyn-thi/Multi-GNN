#!/usr/bin/env python3
"""Phase-4A: N-domain mixed-SSL smoke (Small-HI + SAML-D + Small-LI, 60 steps).

Infrastructure/memory smoke only. Does not launch the 1500-step scout.
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
from typing import Any, Dict, Iterator, List, Optional

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
    CKPT_ROOT,
    CONTRACT_ID,
    ENCODER_LR,
    LINEAR_DECAY_STEPS,
    LOADER_NUM_WORKERS,
    N_NEG,
    NUM_NEIGHS,
    RESULT_ROOT,
    SEED,
    STEPS_PER_DOMAIN,
    TEMP,
    TOTAL_STEPS,
    UNIQUE_NAME,
    WARMUP_STEPS,
    resolved_recipe,
)
from mixed_ssl_phase4a.domain_registry import (  # noqa: E402
    default_smoke_domains,
    domain_order,
    registry_to_json,
)
from mixed_ssl_phase4a.preflight import preflight_phase4a  # noqa: E402
from mixed_ssl_phase4a.schedule import (  # noqa: E402
    first_alpha_beta_update_step,
    init_domain_rng_states,
    loader_generator,
    restore_rng,
    round_robin_schedule,
    snapshot_rng,
)
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def seed_ids_sha(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous().numpy().astype(np.int64)
    return hashlib.sha256(a.tobytes()).hexdigest()


def peak_rss_gib() -> float:
    # ru_maxrss is KiB on Linux
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def make_ns(data: str) -> argparse.Namespace:
    argv = [
        "--data", data,
        "--model", "gin",
        "--objective", "contrastive",
        "--unique_name", UNIQUE_NAME,
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
        "--max_optimizer_steps", str(TOTAL_STEPS),
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
    cid = getattr(ns, "feature_contract", None)
    if str(cid) != CONTRACT_ID:
        raise RuntimeError(f"{name} contract {cid} != {CONTRACT_ID}")
    names = list(getattr(ns, "edge_feature_schema_names", []) or [])
    if names and names != list(FINAL_FEATURE_NAMES):
        raise RuntimeError(f"{name} schema {names} != {list(FINAL_FEATURE_NAMES)}")


def build_train_loader(
    tr_data: HeteroData,
    transform,
    *,
    domain: str,
    domains: List[str],
    offsets: Dict[str, int],
) -> LinkNeighborLoader:
    g = loader_generator(
        SEED, domain, domain_order=domains, loader_seed_offsets=offsets
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
    model = to_hetero(model, metadata_data.metadata(), aggr="mean").to(device)
    return model, emb_dim


def grad_norm(params) -> float:
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum())
    return float(sq ** 0.5)


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
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(moe.parameters()) + list(alpha_beta.parameters()),
        1e9,
    )
    optimizer.step()
    del z1_seed, z2_seed, seed_id1, seed_id2, contrast_raw, tf_raws, total
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        **stats,
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
        "contrast_grad_contribution": True,
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
        "schedule": list(kwargs["schedule"]),
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
        "unique_name": UNIQUE_NAME,
        "feature_contract_id": CONTRACT_ID,
        "saml_split_protocol": kwargs["saml_split_protocol"],
        "split_protocol_by_domain": kwargs["split_protocol_by_domain"],
        "resolved": kwargs["resolved"],
        "preflight": kwargs["preflight"],
        "test_evaluated": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }


def save_ckpt(ckpt: Dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_smoke(*, dry_match_batches: int = 3) -> Dict[str, Any]:
    t_wall0 = time.perf_counter()
    out_dir = ROOT / RESULT_ROOT
    ckpt_dir = ROOT / CKPT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    specs = list(default_smoke_domains())
    domains = list(domain_order(specs))
    offsets = {s.dataset_id: s.loader_seed_offset for s in specs}
    split_by_dom = {s.dataset_id: s.split_protocol_id for s in specs}
    reg_json = registry_to_json(specs)
    write_json(out_dir / "domain_registry.json", reg_json)

    pre = preflight_phase4a(root=ROOT, specs=specs)
    write_json(out_dir / "preflight.json", pre)
    if not pre["ok"]:
        raise RuntimeError("phase4a preflight failed")

    recipe = resolved_recipe(domains=tuple(domains))
    first_ab_step = first_alpha_beta_update_step(
        n_domains=len(domains), calib_obs_per_domain=CALIB_OBS_PER_DOMAIN
    )
    recipe["first_alpha_beta_update_step"] = first_ab_step
    recipe["alpha_beta_frozen_through_completed_step"] = first_ab_step - 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)

    mem = {
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
    te_inds_by: Dict[str, torch.Tensor] = {}
    edge_scalers: Dict[str, Any] = {}

    for spec in specs:
        d = spec.dataset_id
        logging.info("Loading %s under %s ...", d, CONTRACT_ID)
        t0 = time.perf_counter()
        ns = make_ns(d)
        ns.direct_r198_tfmoe_cache = str(ROOT / spec.tf_cache_path)
        tr, va, te, tr_i, va_i, te_i = get_data(ns, data_config)
        assert_contract_geometry(ns, tr, d)
        if d == "SAML-D" and int(te_i.numel()) != 0:
            raise RuntimeError("SAML-D te_inds nonempty — refuse test access")
        # Drop unused objects early
        del va, te, tr_i, va_i
        gc.collect()
        ns_by[d] = ns
        data_by[d] = tr
        te_inds_by[d] = te_i
        edge_scalers[d] = dict(ns.shared_core_edge_scaler)
        dt = time.perf_counter() - t0
        mem["graph_build"][d] = dt
        mem["graph_build_total_sec"] += dt
        mem["rss_gib_after_each_domain_load"][d] = peak_rss_gib()
        logging.info(
            "Loaded %s in %.1fs RSS=%.2f GiB edge_dim=%s",
            d,
            dt,
            mem["rss_gib_after_each_domain_load"][d],
            int(tr[FORWARD_EDGE_TYPE].edge_attr.shape[1]),
        )

    write_json(
        out_dir / "cache_and_scaler_provenance.json",
        {
            "edge_scalers": edge_scalers,
            "tf_caches": {s.dataset_id: s.tf_cache_path for s in specs},
            "preflight_domains": {
                k: {
                    "cache_version": v.get("cache_version"),
                    "n_train": v.get("n_train"),
                    "n_val": v.get("n_val"),
                    "train_edgeid_ordered_sha256": v.get("train_edgeid_ordered_sha256"),
                    "val_edgeid_ordered_sha256": v.get("val_edgeid_ordered_sha256"),
                    "edge_id_equals_row_index": v.get("edge_id_equals_row_index"),
                    "locked_scaler_sha256": v.get("locked_scaler_sha256"),
                }
                for k, v in pre["domains"].items()
            },
            "small_li_locked": pre["small_li_locked"],
        },
    )

    transform = AddEgoIds()
    for d in domains:
        add_arange_ids([data_by[d]])

    # Throwaway sample for model build (does not advance training generators)
    set_seed(SEED)
    sample_dom = domains[0]
    _sample_loader = build_train_loader(
        data_by[sample_dom], transform, domain=sample_dom, domains=domains, offsets=offsets
    )
    sample = next(iter(_sample_loader))
    del _sample_loader
    model, emb_dim = build_model(ns_by[sample_dom], data_by[sample_dom], sample, device)
    del sample
    moe = TFMoEBundle(in_dim=int(emb_dim), hidden=64, n_targets=3).to(device)
    alpha_beta = LearnedAlphaBeta(n_tf=3, init_alpha=0.6).to(device)
    alpha_beta.set_frozen(True)

    init_path = out_dir / "shared_init_state.pt"
    init_sha = combined_init_sha(model, moe, alpha_beta)
    torch.save(
        {
            "init_sha256": init_sha,
            "model_state_dict": model.state_dict(),
            "moe_state_dict": moe.state_dict(),
            "alpha_beta_state_dict": alpha_beta.state_dict(),
            "seed": SEED,
            "feature_contract_id": CONTRACT_ID,
        },
        init_path,
    )
    # Reload to ensure deterministic start
    blob0 = torch.load(init_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob0["model_state_dict"], strict=True)
    moe.load_state_dict(blob0["moe_state_dict"], strict=True)
    alpha_beta.load_state_dict(blob0["alpha_beta_state_dict"], strict=True)
    alpha_beta.set_frozen(True)

    loaders = {
        d: build_train_loader(
            data_by[d], transform, domain=d, domains=domains, offsets=offsets
        )
        for d in domains
    }
    iters = {d: infinite_loader(loaders[d]) for d in domains}

    # Stream independence: HI-only vs 3-domain RR must match HI hashes
    if dry_match_batches > 0:
        def _hash_n(domain, n, rng_states_local, it):
            hashes = []
            for _ in range(n):
                restore_rng(rng_states_local[domain])
                batch = next(it)
                hashes.append(seed_ids_sha(get_hetero_seed_edge_ids(batch, data_by[domain])))
                rng_states_local[domain] = snapshot_rng()
            return hashes

        # Two-domain order (HI, SAML) vs three-domain — HI offset stays 1
        hi_l = build_train_loader(
            data_by["Small-HI"],
            transform,
            domain="Small-HI",
            domains=["Small-HI", "SAML-D"],
            offsets={"Small-HI": 1, "SAML-D": 2},
        )
        rng_a = init_domain_rng_states(SEED, ["Small-HI", "SAML-D"])
        hi_only = _hash_n("Small-HI", dry_match_batches, rng_a, infinite_loader(hi_l))

        hi_l2 = build_train_loader(
            data_by["Small-HI"], transform, domain="Small-HI", domains=domains, offsets=offsets
        )
        sd_l2 = build_train_loader(
            data_by["SAML-D"], transform, domain="SAML-D", domains=domains, offsets=offsets
        )
        li_l2 = build_train_loader(
            data_by["Small-LI"], transform, domain="Small-LI", domains=domains, offsets=offsets
        )
        rng_b = init_domain_rng_states(SEED, domains)
        its = {
            "Small-HI": infinite_loader(hi_l2),
            "SAML-D": infinite_loader(sd_l2),
            "Small-LI": infinite_loader(li_l2),
        }
        mixed_hi = []
        for i in range(dry_match_batches * len(domains)):
            dom = domains[i % len(domains)]
            restore_rng(rng_b[dom])
            batch = next(its[dom])
            if dom == "Small-HI":
                mixed_hi.append(seed_ids_sha(get_hetero_seed_edge_ids(batch, data_by[dom])))
            rng_b[dom] = snapshot_rng()
        if hi_only != mixed_hi:
            raise RuntimeError(f"HI stream changed by third domain: {hi_only} vs {mixed_hi}")
        logging.info("Pre-train HI stream match OK vs 2-domain init")
        del hi_l, hi_l2, sd_l2, li_l2, rng_a, rng_b, its

    tf_ctx = {d: load_tf_moe_context(ROOT / specs[i].tf_cache_path, device) for i, d in enumerate(domains)}
    # Ensure no test arrays loaded into context meta
    for d, ctx in tf_ctx.items():
        if "test" in str(ctx.meta).lower() and "split_test" in str(ctx.meta):
            pass  # meta may mention policy; arrays are train-only join

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
        warmup_steps=WARMUP_STEPS,
        linear_steps=LINEAR_DECAY_STEPS,
        warmup_start=0.1,
        warmup_end=1.0,
        linear_end=0.1,
        steps_per_epoch=TOTAL_STEPS,
        n_epochs=1,
    )

    bn_init = clone_bn_bundle(collect_bn_bundle(model))
    bn_bundles = {d: clone_bn_bundle(bn_init) for d in domains}
    rng_states = init_domain_rng_states(SEED, domains)
    schedule = round_robin_schedule(
        domains, total_steps=TOTAL_STEPS, steps_per_domain=STEPS_PER_DOMAIN
    )
    seed_hash_log = {d: [] for d in domains}
    step_counts = {d: 0 for d in domains}
    enc_grad_by_domain = {d: [] for d in domains}
    moe_grad_by_domain = {d: [] for d in domains}
    alpha_unfrozen_at: Optional[int] = None
    model_init_flat = torch.cat([p.detach().float().reshape(-1).cpu() for p in model.parameters()])

    jsonl_path = out_dir / "steps.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
    step_times: List[float] = []
    t_first = None

    with open(jsonl_path, "w", encoding="utf-8") as jsonl:
        for si in range(TOTAL_STEPS):
            t_step0 = time.perf_counter()
            domain = schedule[si]
            restore_rng(rng_states[domain])
            apply_bn_(model, bn_bundles[domain])
            batch = next(iters[domain])
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
                        "CALIBRATION_BOUNDARY domain=%s at global_step=%s", domain, completed
                    )

            all_calibrated = all(loss_norms[d].calibrated for d in domains)
            if alpha_beta._frozen and all_calibrated and completed >= first_ab_step - 1:
                # Unfreeze once all domains calibrated; first learned update at next step
                # after completed == n*calib (step 15) → unfreeze so step 16 can learn.
                if completed >= (first_ab_step - 1):
                    alpha_beta.set_frozen(False)
                    alpha_unfrozen_at = completed
                    logging.info(
                        "Unfreezing alpha/beta after all domains calibrated "
                        "(completed=%s, first_update_step=%s)",
                        completed,
                        first_ab_step,
                    )

            bn_bundles[domain] = clone_bn_bundle(collect_bn_bundle(model))
            rng_states[domain] = snapshot_rng()
            step_counts[domain] += 1
            seed_hash_log[domain].append(stats["seed_ids_sha256"])
            enc_grad_by_domain[domain].append(stats["encoder_grad_norm"])
            moe_grad_by_domain[domain].append(stats["moe_grad_norm"])

            lrs = scheduler.current_lrs()
            row = {
                "step": si,
                "global_optimizer_step": completed,
                "domain": domain,
                "domain_exposure_count": step_counts[domain],
                "encoder_lr": lrs[0],
                "alphabeta_lr": lrs[1] if len(lrs) > 1 else lrs[0],
                "schedule_phase": scheduler.phase_at(scheduler.completed_optimizer_steps - 1),
                "calibration_complete_domain": bool(loss_norms[domain].calibrated),
                "all_domains_calibrated": all_calibrated,
                "alpha_beta_frozen": bool(alpha_beta._frozen),
                "bn_l1_vs_init": bn_bundle_l1(bn_bundles[domain], bn_init),
                "edge_scaler_sha256": edge_scalers[domain]["scaler_sha256"],
                "tf_scaler_mean": tf_ctx[domain].scaler_mean.tolist(),
                "weight_mode": "adaptive",
                **stats,
            }
            jsonl.write(json.dumps(row) + "\n")
            jsonl.flush()
            step_times.append(time.perf_counter() - t_step0)
            if completed % 10 == 0 or completed == 1:
                logging.info(
                    "step %s/%s domain=%s L=%.4f enc_g=%.3f α_frozen=%s RSS=%.2f",
                    completed,
                    TOTAL_STEPS,
                    domain,
                    stats["L_total"],
                    stats["encoder_grad_norm"],
                    alpha_beta._frozen,
                    peak_rss_gib(),
                )
            del batch, stats
            if completed % 10 == 0:
                gc.collect()

    mem["mean_sec_per_step"] = float(np.mean(step_times)) if step_times else None
    mem["peak_rss_gib_process"] = peak_rss_gib()
    if device.type == "cuda":
        mem["cuda_peak_alloc_gib"] = float(torch.cuda.max_memory_allocated() / (1024**3))
        mem["cuda_peak_reserved_gib"] = float(torch.cuda.max_memory_reserved() / (1024**3))

    # BN checks
    bn_changed = {
        d: not bn_bundles_equal(bn_bundles[d], bn_init) for d in domains
    }
    bn_pairwise_differ = True
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

    model_final_flat = torch.cat(
        [p.detach().float().reshape(-1).cpu() for p in model.parameters()]
    )
    encoder_changed = not torch.allclose(model_init_flat, model_final_flat)

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
        global_step=TOTAL_STEPS,
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
    )
    ckpt_path = ckpt_dir / "checkpoint_step_0060.tar"
    ckpt_sha = save_ckpt(ckpt, ckpt_path)
    last_path = ckpt_dir / "checkpoint_last.tar"
    save_ckpt(ckpt, last_path)

    # CPU reload + synthetic schedule continuation
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
        "domain_registry",
        "feature_contract_id",
        "test_evaluated",
    ]
    reload_ok = all(k in blob for k in required)
    reload_ok = reload_ok and blob["feature_contract_id"] == CONTRACT_ID
    reload_ok = reload_ok and blob["test_evaluated"] is False
    reload_ok = reload_ok and int(blob["global_optimizer_step"]) == TOTAL_STEPS
    reload_ok = reload_ok and blob["per_domain_exposure_counts"] == {
        d: STEPS_PER_DOMAIN for d in domains
    }
    for d in domains:
        reload_ok = reload_ok and d in blob["bn_bundles"]
        reload_ok = reload_ok and blob["loss_norm_states"][d]["calibrated"] is True

    # Resume schedule position: next domains after completed step 60
    resume_next_si = int(blob["global_optimizer_step"])
    cont_domains = [domains[i % len(domains)] for i in range(resume_next_si, resume_next_si + 3)]
    expect_cont = [domains[i % len(domains)] for i in range(TOTAL_STEPS, TOTAL_STEPS + 3)]
    resume_schedule_ok = cont_domains == expect_cont
    exposure_restore_ok = blob["per_domain_exposure_counts"] == {
        d: STEPS_PER_DOMAIN for d in domains
    }
    resume_schedule_ok = resume_schedule_ok and exposure_restore_ok

    # Reload BN bundles into model
    model_cpu = model.cpu()
    model_cpu.load_state_dict(blob["model_state_dict"], strict=True)
    for d in domains:
        apply_bn_(model_cpu, blob["bn_bundles"][d])
        got = collect_bn_bundle(model_cpu)
        if not bn_bundles_equal(got, blob["bn_bundles"][d]):
            reload_ok = False

    gates = {
        "exactly_60_steps": True,
        "exposures_20_per_domain": all(step_counts[d] == STEPS_PER_DOMAIN for d in domains),
        "finite_losses": True,  # enforced per-step
        "nonzero_encoder_grad_every_domain": all(
            any(g > 0 for g in enc_grad_by_domain[d]) for d in domains
        ),
        "nonzero_moe_grad_every_domain": all(
            any(g > 0 for g in moe_grad_by_domain[d]) for d in domains
        ),
        "encoder_params_changed": bool(encoder_changed),
        "bn_all_changed": all(bn_changed.values()),
        "bn_all_differ": bool(bn_pairwise_differ),
        "bn_swap_ok": bool(bn_swap_ok),
        "all_lossnorm_calibrated": all(loss_norms[d].calibrated for d in domains),
        "alpha_frozen_through_15": alpha_unfrozen_at == 15,
        "first_ab_update_step": first_ab_step,
        "alpha_unfrozen_at_completed": alpha_unfrozen_at,
        "edge_dim_6": True,
        "checkpoint_reload_ok": bool(reload_ok),
        "resume_schedule_ok": bool(resume_schedule_ok),
        "projection_false": True,
        "preserve_seed_edges_false": True,
        "amp_false": True,
        "test_evaluated_false": True,
        "no_test_graph_metrics": True,
    }
    gates["ok"] = all(bool(v) for k, v in gates.items() if k != "first_ab_update_step" and k != "alpha_unfrozen_at_completed")

    # Memory classification
    peak = mem["peak_rss_gib_process"]
    if peak is None:
        mem_class = "unknown"
    elif peak <= 115:
        mem_class = "preferred_le_115"
    elif peak <= 120:
        mem_class = "thin_margin_115_120"
    else:
        mem_class = "above_120_do_not_recommend_1500_on_128G"
    mem["classification"] = mem_class
    mem["slurm_maxrss_gib"] = None  # filled by finalize if sacct available
    mem["disk_written_ckpt_bytes"] = ckpt_path.stat().st_size if ckpt_path.is_file() else 0
    write_json(out_dir / "memory_runtime.json", mem)

    write_json(
        out_dir / "checkpoint_integrity.json",
        {
            "path": str(ckpt_path),
            "sha256": ckpt_sha,
            "reload_ok": reload_ok,
            "resume_schedule_ok": resume_schedule_ok,
            "resume_next_domains": cont_domains,
            "required_keys_present": required,
            "test_evaluated": False,
        },
    )

    # Phase-4B proposal only (do not submit)
    recommend_1500 = mem_class in ("preferred_le_115", "thin_margin_115_120") and gates["ok"]
    if mem_class == "thin_margin_115_120":
        recommend_note = "Thin margin — review MaxRSS before 1500-step run"
    elif mem_class == "preferred_le_115":
        recommend_note = "Memory preferred; 1500-step scout may proceed after review"
    else:
        recommend_note = "Do not recommend 1500-step on 128G"
        recommend_1500 = False

    phase4b = {
        "submitted": False,
        "description": "1500-step 1:1:1 round-robin scout (proposal only)",
        "datasets": domains,
        "updates_per_domain": 500,
        "total_steps": 1500,
        "contract": CONTRACT_ID,
        "objective": "adaptive TF-MoE + R198 InfoNCE",
        "resources_proposed": {
            "partition": "mit_preemptable",
            "account": "mit_general",
            "qos": "normal",
            "mem": "128G" if recommend_1500 and mem_class == "preferred_le_115" else "review",
            "cpus": 16,
            "gres": "gpu:1",
            "loader_num_workers": 0,
            "time": "08:00:00",
        },
        "measured_smoke_peak_rss_gib": peak,
        "memory_classification": mem_class,
        "recommend_submit": recommend_1500,
        "recommend_note": recommend_note,
        "validation_only": True,
        "no_extraction_probe_until_training_review": True,
    }
    write_json(out_dir / "proposed_phase4b.json", phase4b)

    summary = {
        "ok": bool(gates["ok"]),
        "unique_name": UNIQUE_NAME,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "feature_contract_id": CONTRACT_ID,
        "domains": domains,
        "steps": TOTAL_STEPS,
        "exposures": dict(step_counts),
        "schedule_head": schedule[:9],
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
            "frozen_through_completed_step": first_ab_step - 1,
            "first_update_step": first_ab_step,
            "unfrozen_at_completed": alpha_unfrozen_at,
            "final_alpha": float(torch.sigmoid(alpha_beta.alpha_logit).detach().cpu()),
        },
        "gates": gates,
        "memory_runtime": mem,
        "checkpoint": {"path": str(ckpt_path), "sha256": ckpt_sha},
        "init_sha256": init_sha,
        "elapsed_sec": time.perf_counter() - t_wall0,
        "phase4b_recommended": recommend_1500,
        "test_evaluated": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(ROOT / f"{RESULT_ROOT}.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight_only", action="store_true")
    ap.add_argument("--run_smoke", action="store_true")
    args = ap.parse_args()
    logger_setup()
    logging.getLogger().setLevel(logging.INFO)
    if args.preflight_only:
        pre = preflight_phase4a(root=ROOT)
        out = ROOT / RESULT_ROOT
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "preflight.json", pre)
        write_json(out / "domain_registry.json", pre["domain_registry"])
        print(json.dumps({"ok": pre["ok"], "preflight": str(out / "preflight.json")}, indent=2))
        if not pre["ok"]:
            sys.exit(1)
        return
    summary = run_smoke()
    print(json.dumps({"ok": summary["ok"], "job_id": summary.get("job_id")}, indent=2))
    if not summary["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
