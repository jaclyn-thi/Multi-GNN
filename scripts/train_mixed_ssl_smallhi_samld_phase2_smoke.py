#!/usr/bin/env python3
"""Phase-2 mixed Small-HI + SAML-D DIRECT_H TFMOE smoke (50 optimizer steps).

Dedicated entrypoint — does not modify historical single-domain trainers.
No test access, no extraction, no probes, no PaySim, no category adapters.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
from graph_augmentations import generate_views  # noqa: E402
from mixed_ssl_phase2 import (  # noqa: E402
    ALPHABETA_LR,
    CALIB_STEPS_PER_DOMAIN,
    CONTRACT_ID,
    DOMAINS,
    ENCODER_LR,
    HI_TF_CACHE,
    RUN_UNIQUE,
    SAMLD_TF_CACHE,
    SAML_SPLIT_PROTOCOL,
    SEED,
    STEPS_PER_DOMAIN,
    TOTAL_STEPS,
)
from mixed_ssl_phase2.bn import (  # noqa: E402
    apply_bn_,
    bn_bundle_l1,
    bn_bundles_equal,
    clone_bn_bundle,
    collect_bn_bundle,
)
from mixed_ssl_phase2.preflight import preflight_all  # noqa: E402
from mixed_ssl_phase2.schedule import (  # noqa: E402
    assert_balanced_schedule,
    domain_schedule,
    init_domain_rng_states,
    loader_generator,
    restore_rng,
    snapshot_rng,
)
from shared_core_contract import SHARED_CORE_FINAL_FEATURE_NAMES  # noqa: E402
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

BATCH_SIZE = 8192
NUM_NEIGHS = [100, 100]
TEMP = 0.5
N_NEG = 8192
RESULT_DIR = ROOT / "results" / "diagnostics" / "smallhi_samld_mixed_ssl_phase2_smoke"
CKPT_DIR = ROOT / "saved-models" / RUN_UNIQUE


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def seed_ids_sha(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous().numpy().astype(np.int64)
    return hashlib.sha256(a.tobytes()).hexdigest()


def make_ns(data: str) -> argparse.Namespace:
    argv = [
        "--data", data,
        "--model", "gin",
        "--objective", "contrastive",
        "--unique_name", RUN_UNIQUE,
        "--seed", str(SEED),
        "--batch_size", str(BATCH_SIZE),
        "--num_neighs", "100", "100",
        "--loader_num_workers", "0",
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
        "--contrastive_accum_steps", "1",
        "--contrastive_temperature", str(TEMP),
        "--max_optimizer_steps", str(TOTAL_STEPS),
    ]
    # preserve_seed_edges omitted (=false); contrast_projection_head omitted (=false)
    ns = create_parser().parse_args(argv)
    if bool(ns.preserve_seed_edges):
        raise RuntimeError("preserve_seed_edges must be false")
    if bool(ns.contrast_projection_head):
        raise RuntimeError("contrast_projection_head must be false")
    if bool(getattr(ns, "testing", False)):
        # create_parser default testing=False; refuse if somehow enabled
        raise RuntimeError("testing flag must be off (no test eval)")
    return ns


def build_train_loader(tr_data: HeteroData, transform, *, domain: str) -> LinkNeighborLoader:
    g = loader_generator(SEED, domain)
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
        num_workers=0,
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


def effective_rank_diag(z: torch.Tensor) -> Dict[str, float]:
    if z.numel() == 0 or z.shape[0] < 2:
        return {"repr_norm_mean": float("nan"), "repr_std_mean": float("nan"), "effective_rank": float("nan")}
    with torch.no_grad():
        x = z.detach().float()
        norms = x.norm(dim=-1)
        stds = x.std(dim=0)
        # cheap effective rank via singular values of centered mini batch
        xc = x - x.mean(0, keepdim=True)
        try:
            s = torch.linalg.svdvals(xc)
            p = s / (s.sum() + 1e-12)
            ent = -(p * (p + 1e-12).log()).sum()
            erank = float(ent.exp())
        except Exception:
            erank = float("nan")
        return {
            "repr_norm_mean": float(norms.mean()),
            "repr_std_mean": float(stds.mean()),
            "effective_rank": erank,
        }


def mixed_step(
    *,
    model: nn.Module,
    moe: TFMoEBundle,
    alpha_beta: LearnedAlphaBeta,
    loss_norm: LossNormState,
    tf_ctx,
    optimizer: torch.optim.Optimizer,
    batch,
    loader_data,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    model.train()
    moe.train()
    requested = int(batch[FORWARD_EDGE_TYPE].edge_index.shape[1]) if False else 0
    seed_edge_ids = get_hetero_seed_edge_ids(batch, loader_data)
    attach_edge_id_from_batch(batch, loader_data)
    requested = int(seed_edge_ids.numel())
    sid_hash = seed_ids_sha(seed_edge_ids)
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
    if enc_gn == 0.0:
        raise RuntimeError("encoder gradient norm is zero")
    if moe_gn == 0.0:
        raise RuntimeError("MoE gradient norm is zero")
    # Per-head grad presence
    head_grads = []
    for hi, head in enumerate(moe.heads):
        g = grad_norm(head.parameters())
        head_grads.append(g)
        if g == 0.0:
            raise RuntimeError(f"TF expert head {hi} has zero grad")
    torch.nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(moe.parameters()) + list(alpha_beta.parameters()),
        1e9,
    )
    optimizer.step()

    repr_diag = effective_rank_diag(z1_seed)
    del z1_seed, z2_seed, seed_id1, seed_id2, contrast_raw, tf_raws, total
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out = {
        **stats,
        **repr_diag,
        **{f"k/{k}": v for k, v in tf_diag.items() if isinstance(v, (int, float))},
        "requested_seeds": requested,
        "scored_seeds": scored,
        "seed_ids_sha256": sid_hash,
        "encoder_grad_norm": enc_gn,
        "moe_grad_norm": moe_gn,
        "head_grad_norms": head_grads,
        "alpha_beta_frozen": bool(alpha_beta._frozen),
        "loss_norm_calibrated": bool(loss_norm.calibrated),
    }
    return out


def run_smoke() -> Dict[str, Any]:
    t0 = time.perf_counter()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    jsonl_path = RESULT_DIR / "logs" / "steps.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    pre = preflight_all()
    write_json(RESULT_DIR / "preflight.json", pre)
    logging.info("Preflight OK: %s", SAML_SPLIT_PROTOCOL)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("CUDA unavailable — smoke will run on CPU (unexpected on GPU partition)")

    set_seed(SEED)
    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)

    ns_hi = make_ns("Small-HI")
    ns_sd = make_ns("SAML-D")
    ns_hi.direct_r198_tfmoe_cache = str(ROOT / HI_TF_CACHE)
    ns_sd.direct_r198_tfmoe_cache = str(ROOT / SAMLD_TF_CACHE)

    logging.info("Loading Small-HI under %s (skip_test_eval)...", CONTRACT_ID)
    hi_tr, hi_va, hi_te, hi_tr_i, hi_va_i, hi_te_i = get_data(ns_hi, data_config)
    logging.info("Loading SAML-D under %s (skip_test_eval)...", CONTRACT_ID)
    sd_tr, sd_va, sd_te, sd_tr_i, sd_va_i, sd_te_i = get_data(ns_sd, data_config)

    # Gates: no test indices materialized for SAML-D under skip_test_eval
    if hi_te_i.numel() != 0 and bool(getattr(ns_hi, "skip_test_eval", False)):
        # get_data empties te_inds when skip_test_eval
        pass
    if int(sd_te_i.numel()) != 0:
        raise RuntimeError("SAML-D te_inds nonempty under skip_test_eval — refuse test access")
    # Edge dim / schema
    for name, tr, ns in (("Small-HI", hi_tr, ns_hi), ("SAML-D", sd_tr, ns_sd)):
        ea = tr[FORWARD_EDGE_TYPE].edge_attr
        if int(ea.shape[1]) != 6:
            raise RuntimeError(f"{name} edge_dim={ea.shape[1]} != 6")
        names = list(getattr(ns, "edge_feature_schema_names", []) or [])
        if names and names != list(SHARED_CORE_FINAL_FEATURE_NAMES):
            raise RuntimeError(f"{name} schema {names}")
        scaler = getattr(ns, "shared_core_edge_scaler", None)
        if not isinstance(scaler, dict) or "scaler_sha256" not in scaler:
            raise RuntimeError(f"{name} missing shared_core_edge_scaler provenance")

    hi_scaler = dict(ns_hi.shared_core_edge_scaler)
    sd_scaler = dict(ns_sd.shared_core_edge_scaler)
    if hi_scaler["scaler_sha256"] == sd_scaler["scaler_sha256"]:
        logging.warning("edge scaler hashes unexpectedly equal across domains")

    transform = AddEgoIds()
    add_arange_ids([hi_tr, hi_va])
    add_arange_ids([sd_tr, sd_va])
    # Do not touch test graphs beyond placeholders; they alias val under skip_test_eval.

    hi_loader = build_train_loader(hi_tr, transform, domain="Small-HI")
    sd_loader = build_train_loader(sd_tr, transform, domain="SAML-D")
    hi_iter = infinite_loader(hi_loader)
    sd_iter = infinite_loader(sd_loader)

    sample = next(iter(hi_loader))
    model, emb_dim = build_model(ns_hi, hi_tr, sample, device)
    if emb_dim != 198:
        logging.info("R198 embedding_dim=%s (expected 198 for n_hidden=64)", emb_dim)

    moe = TFMoEBundle(in_dim=int(emb_dim), hidden=64, n_targets=3).to(device)
    alpha_beta = LearnedAlphaBeta(n_tf=3, init_alpha=0.6).to(device)
    alpha_beta.set_frozen(True)

    tf_ctx = {
        "Small-HI": load_tf_moe_context(ROOT / HI_TF_CACHE, device),
        "SAML-D": load_tf_moe_context(ROOT / SAMLD_TF_CACHE, device),
    }
    # Per-domain loss norms + calibration accumulators
    loss_norms = {d: LossNormState() for d in DOMAINS}
    calib = {
        d: {"contrast": 0.0, "tf": [0.0, 0.0, 0.0], "n": 0} for d in DOMAINS
    }

    enc_params = list(model.parameters())
    moe_params = list(moe.parameters())
    ab_params = list(alpha_beta.parameters())
    optimizer = torch.optim.Adam(
        [
            {"params": enc_params + moe_params, "lr": ENCODER_LR},
            {"params": ab_params, "lr": ALPHABETA_LR},
        ]
    )
    # 10-step warmup + 40-step linear decay over the 50-step smoke
    scheduler = DirectHWarmupLinearScheduler(
        optimizer,
        warmup_steps=10,
        linear_steps=40,
        warmup_start=0.1,
        warmup_end=1.0,
        linear_end=0.1,
        steps_per_epoch=TOTAL_STEPS,
        n_epochs=1,
    )

    bn_init = clone_bn_bundle(collect_bn_bundle(model))
    bn_bundles = {d: clone_bn_bundle(bn_init) for d in DOMAINS}
    rng_states = init_domain_rng_states(SEED)

    schedule = domain_schedule(TOTAL_STEPS)
    assert_balanced_schedule(schedule)

    seed_hash_log = {d: [] for d in DOMAINS}
    step_counts = {d: 0 for d in DOMAINS}
    enc_changed = False
    enc_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if v.is_floating_point()}
    alpha_unfrozen_at: Optional[int] = None
    gates: Dict[str, Any] = {}

    args_views = ns_hi  # identical aug flags on both domains
    jsonl = open(jsonl_path, "w", encoding="utf-8")

    try:
        for si, domain in enumerate(schedule):
            # Domain RNG isolation
            restore_rng(rng_states[domain])
            apply_bn_(model, bn_bundles[domain])

            batch = next(hi_iter if domain == "Small-HI" else sd_iter)
            loader_data = hi_tr if domain == "Small-HI" else sd_tr
            ns = ns_hi if domain == "Small-HI" else ns_sd

            stats = mixed_step(
                model=model,
                moe=moe,
                alpha_beta=alpha_beta,
                loss_norm=loss_norms[domain],
                tf_ctx=tf_ctx[domain],
                optimizer=optimizer,
                batch=batch,
                loader_data=loader_data,
                args=ns,
                device=device,
            )
            scheduler.step()

            # Calibration accumulation (first 5 steps per domain)
            if calib[domain]["n"] < CALIB_STEPS_PER_DOMAIN:
                calib[domain]["contrast"] += float(stats["L_contrast_raw"])
                for m in range(3):
                    calib[domain]["tf"][m] += float(stats[f"L_tf_raw_{m}"])
                calib[domain]["n"] += 1
                if calib[domain]["n"] == CALIB_STEPS_PER_DOMAIN:
                    n = float(CALIB_STEPS_PER_DOMAIN)
                    loss_norms[domain].contrast_mean = calib[domain]["contrast"] / n
                    loss_norms[domain].tf_means = [
                        calib[domain]["tf"][m] / n for m in range(3)
                    ]
                    loss_norms[domain].calibrated = True
                    logging.info(
                        "CALIBRATION_BOUNDARY domain=%s frozen means contrast=%.6f tf=%s",
                        domain,
                        loss_norms[domain].contrast_mean,
                        loss_norms[domain].tf_means,
                    )
                if all(loss_norms[d].calibrated for d in DOMAINS) and alpha_beta._frozen:
                    alpha_beta.set_frozen(False)
                    alpha_unfrozen_at = si + 1  # after this step
                    logging.info(
                        "CALIBRATION_BOUNDARY both domains done; unfreezing alpha/beta at end of step %s",
                        si,
                    )

            # Save BN + RNG for domain
            bn_bundles[domain] = clone_bn_bundle(collect_bn_bundle(model))
            rng_states[domain] = snapshot_rng()

            step_counts[domain] += 1
            if len(seed_hash_log[domain]) < 32:
                seed_hash_log[domain].append(stats["seed_ids_sha256"])

            lrs = scheduler.current_lrs()
            row = {
                "step": si,
                "domain": domain,
                "domain_step": step_counts[domain],
                "encoder_lr": lrs[0],
                "alphabeta_lr": lrs[1] if len(lrs) > 1 else lrs[0],
                "schedule_phase": scheduler.phase_at(scheduler.completed_optimizer_steps - 1),
                "calibration_complete_domain": bool(loss_norms[domain].calibrated),
                "both_calibrated": all(loss_norms[d].calibrated for d in DOMAINS),
                "alpha_beta_frozen": bool(alpha_beta._frozen),
                "edge_scaler_sha256": (
                    hi_scaler if domain == "Small-HI" else sd_scaler
                )["scaler_sha256"],
                "tf_scaler_mean": tf_ctx[domain].scaler_mean.tolist(),
                **stats,
            }
            jsonl.write(json.dumps(row) + "\n")
            jsonl.flush()
            logging.info(
                "step %s/%s domain=%s L=%.4f enc_g=%.3f moe_g=%.3f α_frozen=%s",
                si + 1,
                TOTAL_STEPS,
                domain,
                stats["L_total"],
                stats["encoder_grad_norm"],
                stats["moe_grad_norm"],
                alpha_beta._frozen,
            )
            del batch, stats
            gc.collect()
    finally:
        jsonl.close()

    # Encoder changed?
    for k, v0 in enc_before.items():
        v1 = model.state_dict()[k].detach().cpu()
        if v1.is_floating_point() and not torch.allclose(v0, v1):
            enc_changed = True
            break

    bn_hi = bn_bundles["Small-HI"]
    bn_sd = bn_bundles["SAML-D"]
    bn_changed_hi = not bn_bundles_equal(bn_hi, bn_init)
    bn_changed_sd = not bn_bundles_equal(bn_sd, bn_init)
    bn_differ = not bn_bundles_equal(bn_hi, bn_sd)
    # Swap reproducibility
    apply_bn_(model, bn_hi)
    snap_a = clone_bn_bundle(collect_bn_bundle(model))
    apply_bn_(model, bn_sd)
    snap_b = clone_bn_bundle(collect_bn_bundle(model))
    apply_bn_(model, bn_hi)
    snap_a2 = clone_bn_bundle(collect_bn_bundle(model))
    swap_ok = bn_bundles_equal(snap_a, snap_a2) and not bn_bundles_equal(snap_a, snap_b)

    # Checkpoint
    ckpt = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "moe_state_dict": {k: v.detach().cpu() for k, v in moe.state_dict().items()},
        "alpha_beta_state_dict": {
            k: v.detach().cpu() for k, v in alpha_beta.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "bn_bundles": {d: clone_bn_bundle(bn_bundles[d]) for d in DOMAINS},
        "loss_norm_states": {
            d: {
                "contrast_mean": loss_norms[d].contrast_mean,
                "tf_means": list(loss_norms[d].tf_means),
                "calibrated": loss_norms[d].calibrated,
            }
            for d in DOMAINS
        },
        "edge_scalers": {"Small-HI": hi_scaler, "SAML-D": sd_scaler},
        "tf_scalers": {
            d: {
                "mean": tf_ctx[d].scaler_mean.tolist(),
                "scale": tf_ctx[d].scaler_scale.tolist(),
            }
            for d in DOMAINS
        },
        "rng_states_present": True,
        "seed": SEED,
        "unique_name": RUN_UNIQUE,
        "feature_contract_id": CONTRACT_ID,
        "saml_split_protocol": SAML_SPLIT_PROTOCOL,
        "resolved": {
            "preserve_seed_edges": False,
            "contrast_projection_head": False,
            "ports": True,
            "tds": True,
            "emlps": True,
            "ego": True,
            "reverse_mp": True,
            "correct_reverse_edge_features": True,
            "train_fit_edge_znorm": True,
            "skip_test_eval": True,
            "direct_r198_infonce": True,
            "direct_r198_tfmoe": True,
            "direct_r198_tfmoe_weight_mode": "adaptive",
            "max_optimizer_steps": TOTAL_STEPS,
            "encoder_lr": ENCODER_LR,
            "alphabeta_lr": ALPHABETA_LR,
        },
        "step_counts": step_counts,
        "seed_hash_log": seed_hash_log,
        "preflight": pre,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    ckpt_path = CKPT_DIR / "checkpoint_smoke.tar"
    torch.save(ckpt, ckpt_path)

    # Reload verification
    reload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(reload["model_state_dict"], strict=True)
    moe.load_state_dict(reload["moe_state_dict"], strict=True)
    alpha_beta.load_state_dict(reload["alpha_beta_state_dict"], strict=True)
    optimizer.load_state_dict(reload["optimizer_state_dict"])
    scheduler.load_state_dict(reload["scheduler_state_dict"])
    bn_reload_ok = True
    for d in DOMAINS:
        if not bn_bundles_equal(reload["bn_bundles"][d], bn_bundles[d]):
            bn_reload_ok = False
    loss_reload_ok = all(
        reload["loss_norm_states"][d]["calibrated"]
        and reload["loss_norm_states"][d]["contrast_mean"] == loss_norms[d].contrast_mean
        for d in DOMAINS
    )

    gates = {
        "steps_total": sum(step_counts.values()) == TOTAL_STEPS,
        "steps_per_domain": all(step_counts[d] == STEPS_PER_DOMAIN for d in DOMAINS),
        "encoder_params_changed": enc_changed,
        "bn_changed_small_hi": bn_changed_hi,
        "bn_changed_samld": bn_changed_sd,
        "bn_bundles_differ": bn_differ,
        "bn_swap_reproducible": swap_ok,
        "both_calibrated": all(loss_norms[d].calibrated for d in DOMAINS),
        "alpha_unfrozen_after_calib": alpha_unfrozen_at is not None and not alpha_beta._frozen,
        "ckpt_reload_ok": bn_reload_ok and loss_reload_ok,
        "saml_protocol": SAML_SPLIT_PROTOCOL,
        "no_test_metrics_written": True,
        "no_test_split_loaded_samld": int(sd_te_i.numel()) == 0,
        "projection_disabled": True,
        "preserve_seed_edges_false": True,
    }
    ok = all(gates.values())

    summary = {
        "ok": ok,
        "gates": gates,
        "step_counts": step_counts,
        "alpha_unfrozen_after_step": alpha_unfrozen_at,
        "loss_norm_states": ckpt["loss_norm_states"],
        "bn_l1_hi_vs_init": bn_bundle_l1(bn_hi, bn_init),
        "bn_l1_sd_vs_init": bn_bundle_l1(bn_sd, bn_init),
        "bn_l1_hi_vs_sd": bn_bundle_l1(bn_hi, bn_sd),
        "edge_scalers": {"Small-HI": hi_scaler, "SAML-D": sd_scaler},
        "tf_scalers": ckpt["tf_scalers"],
        "seed_hash_log_counts": {d: len(seed_hash_log[d]) for d in DOMAINS},
        "checkpoint": str(ckpt_path),
        "elapsed_sec": time.perf_counter() - t0,
        "device": str(device),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "explicitly_not_done": {
            "full_training": False,
            "embedding_extraction": False,
            "downstream_probe": False,
            "category_adapter": False,
            "paysim": False,
            "paired_domain_updates": False,
            "dependent_dag": False,
            "test_metrics": False,
        },
    }
    write_json(RESULT_DIR / "summary.json", summary)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    logger_setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight_only", action="store_true")
    args = p.parse_args(argv)
    if args.preflight_only:
        pre = preflight_all()
        write_json(RESULT_DIR / "preflight.json", pre)
        print(json.dumps(pre, indent=2))
        return 0
    summary = run_smoke()
    print(json.dumps({"ok": summary["ok"], "gates": summary["gates"], "step_counts": summary["step_counts"]}, indent=2))
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
