#!/usr/bin/env python3
"""Phase-3 three-arm Small-HI + SAML-D DIRECT_H TFMOE scout (1000 optimizer steps).

Dedicated entrypoint — does not modify historical single-domain trainers.
No test access, no extraction, no probes, no PaySim, no category adapters.
"""

from __future__ import annotations

import argparse
import fcntl
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
from mixed_ssl_phase2.bn import (  # noqa: E402
    apply_bn_,
    bn_bundle_l1,
    bn_bundles_equal,
    clone_bn_bundle,
    collect_bn_bundle,
)
from mixed_ssl_phase2.schedule import (  # noqa: E402
    init_domain_rng_states,
    loader_generator,
    restore_rng,
    snapshot_rng,
)
from mixed_ssl_phase3 import (  # noqa: E402
    ACCUM_STEPS,
    ALPHA_FREEZE_UNTIL_STEP,
    ALPHABETA_LR,
    ARMS,
    ARRAY_INDEX_TO_ARM,
    BATCH_SIZE,
    CALIB_OBS_PER_DOMAIN,
    CHECKPOINT_STEPS,
    CKPT_ROOT,
    CONTRACT_ID,
    DOMAINS,
    ENCODER_LR,
    HI_TF_CACHE,
    LINEAR_DECAY_STEPS,
    MIXED_STEPS_PER_DOMAIN,
    N_NEG,
    NUM_NEIGHS,
    RESULT_ROOT,
    ROLLING_EVERY,
    SAMLD_TF_CACHE,
    SAML_SPLIT_PROTOCOL,
    SECONDARY_COMPARISON_CAVEAT,
    SEED,
    TEMP,
    TOTAL_STEPS,
    WARMUP_STEPS,
    arm_schedule,
    arm_unique,
    resolved_recipe,
)
from mixed_ssl_phase3.amount_diag import amount_received_diagnostic  # noqa: E402
from mixed_ssl_phase3.hash_util import combined_init_sha, state_dict_sha256  # noqa: E402
from mixed_ssl_phase3.matching import assert_matching_contract_guaranteed  # noqa: E402
from mixed_ssl_phase3.plots import plot_training_curves, write_steps_csv  # noqa: E402
from mixed_ssl_phase3.preflight import preflight_phase3  # noqa: E402
from shared_core_contract import (  # noqa: E402
    HISTORICAL_SUPERVISED_PORTS_ONLY_NOTE,
    SHARED_CORE_FINAL_FEATURE_NAMES,
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


def make_ns(data: str, unique: str) -> argparse.Namespace:
    argv = [
        "--data", data,
        "--model", "gin",
        "--objective", "contrastive",
        "--unique_name", unique,
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
        "--contrastive_accum_steps", str(ACCUM_STEPS),
        "--contrastive_temperature", str(TEMP),
        "--max_optimizer_steps", str(TOTAL_STEPS),
    ]
    # preserve_seed_edges omitted (=false); contrast_projection_head omitted (=false)
    ns = create_parser().parse_args(argv)
    if bool(ns.preserve_seed_edges):
        raise RuntimeError("preserve_seed_edges must be false")
    if bool(ns.contrast_projection_head):
        raise RuntimeError("contrast_projection_head must be false")
    if bool(getattr(ns, "amp", False)) or bool(getattr(ns, "use_amp", False)):
        raise RuntimeError("AMP must be false")
    return ns


def assert_shared_core_not_historical(ns: argparse.Namespace, tr: HeteroData, name: str) -> None:
    ea = tr[FORWARD_EDGE_TYPE].edge_attr
    if int(ea.shape[1]) != 6:
        raise RuntimeError(f"{name} edge_dim={ea.shape[1]} != 6")
    cid = getattr(ns, "feature_contract", None) or getattr(ns, "feature_contract_id", None)
    if str(cid) != CONTRACT_ID:
        raise RuntimeError(f"{name} contract {cid} != {CONTRACT_ID}")
    names = list(getattr(ns, "edge_feature_schema_names", []) or [])
    if names and names != list(SHARED_CORE_FINAL_FEATURE_NAMES):
        raise RuntimeError(
            f"{name} schema {names} — refused historical ports-only dim-6. "
            f"{HISTORICAL_SUPERVISED_PORTS_ONLY_NOTE}"
        )
    # Explicit geometry gate: shared-core requires ports+tds (base2+ports2+tds2).
    if not (bool(ns.ports) and bool(ns.tds) and bool(ns.emlps)):
        raise RuntimeError(f"{name}: ports/tds/emlps must all be true for shared-core")
    summary = getattr(ns, "feature_contract_summary", None)
    if isinstance(summary, dict) and not summary.get(
        "not_historical_supervised_ports_only_dim6", True
    ):
        raise RuntimeError(f"{name}: historical dim-6 protocol refused")


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
        return {
            "repr_norm_mean": float("nan"),
            "repr_std_mean": float("nan"),
            "effective_rank": float("nan"),
        }
    with torch.no_grad():
        x = z.detach().float()
        norms = x.norm(dim=-1)
        stds = x.std(dim=0)
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


def ensure_shared_init(
    *,
    init_path: Path,
    model: nn.Module,
    moe: nn.Module,
    alpha_beta: nn.Module,
) -> str:
    """Create or load shared init so all arms start from identical weights."""
    init_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = init_path.with_suffix(".lock")
    with open(lock_path, "w", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        if init_path.is_file():
            blob = torch.load(init_path, map_location="cpu", weights_only=False)
            model.load_state_dict(blob["model_state_dict"], strict=True)
            moe.load_state_dict(blob["moe_state_dict"], strict=True)
            alpha_beta.load_state_dict(blob["alpha_beta_state_dict"], strict=True)
            sha = str(blob["init_sha256"])
            local = combined_init_sha(model, moe, alpha_beta)
            if local != sha:
                raise RuntimeError(f"loaded init sha mismatch local={local} file={sha}")
            return sha
        sha = combined_init_sha(model, moe, alpha_beta)
        blob = {
            "init_sha256": sha,
            "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "moe_state_dict": {k: v.detach().cpu().clone() for k, v in moe.state_dict().items()},
            "alpha_beta_state_dict": {
                k: v.detach().cpu().clone() for k, v in alpha_beta.state_dict().items()
            },
            "seed": SEED,
            "feature_contract_id": CONTRACT_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        tmp = init_path.with_suffix(f".tmp.{os.getpid()}")
        torch.save(blob, tmp)
        os.replace(tmp, init_path)
        return sha


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


def build_checkpoint(
    *,
    model,
    moe,
    alpha_beta,
    optimizer,
    scheduler,
    bn_bundles,
    loss_norms,
    edge_scalers,
    tf_ctx,
    step_counts,
    global_step,
    arm,
    unique,
    init_sha,
    rng_states,
    seed_hash_log,
    amount_diag,
    preflight,
) -> Dict[str, Any]:
    return {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "moe_state_dict": {k: v.detach().cpu() for k, v in moe.state_dict().items()},
        "alpha_beta_state_dict": {
            k: v.detach().cpu() for k, v in alpha_beta.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_optimizer_step": int(global_step),
        "per_domain_exposure_counts": dict(step_counts),
        "bn_bundles": {d: clone_bn_bundle(bn_bundles[d]) for d in bn_bundles},
        "loss_norm_states": {
            d: {
                "contrast_mean": loss_norms[d].contrast_mean,
                "tf_means": list(loss_norms[d].tf_means),
                "calibrated": loss_norms[d].calibrated,
            }
            for d in loss_norms
        },
        "edge_scalers": edge_scalers,
        "tf_scalers": {
            d: {
                "mean": tf_ctx[d].scaler_mean.tolist(),
                "scale": tf_ctx[d].scaler_scale.tolist(),
            }
            for d in tf_ctx
        },
        "alpha_beta_logits": {
            k: v.detach().cpu() for k, v in alpha_beta.state_dict().items()
        },
        "rng_states": {
            d: {
                "python": rng_states[d]["python"],
                "numpy": rng_states[d]["numpy"],
                "torch": rng_states[d]["torch"],
                # cuda RNG may be large; store if present
                **(
                    {"cuda": rng_states[d]["cuda"]}
                    if "cuda" in rng_states[d]
                    else {}
                ),
            }
            for d in rng_states
        },
        "seed_hash_log": seed_hash_log,
        "init_sha256": init_sha,
        "seed": SEED,
        "arm": arm,
        "unique_name": unique,
        "feature_contract_id": CONTRACT_ID,
        "saml_split_protocol": SAML_SPLIT_PROTOCOL,
        "resolved": resolved_recipe(arm=arm),
        "amount_scale_diagnostic": amount_diag,
        "preflight": preflight,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
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


def manual_resume_command(arm: str, ckpt_path: Path) -> str:
    return (
        f"sbatch --export=ALL,PHASE3_ARM={arm},PHASE3_RESUME={ckpt_path} "
        f"slurm/run_mixed_ssl_phase3_scout.sh\n"
        f"# or interactive:\n"
        f"python scripts/train_mixed_ssl_phase3_scout.py --arm {arm} "
        f"--resume {ckpt_path}"
    )


def maybe_write_aggregate(result_root: Path) -> Optional[Path]:
    arm_summaries = {}
    for arm in ARMS:
        p = result_root / "arms" / arm / "summary.json"
        if not p.is_file():
            return None
        arm_summaries[arm] = json.loads(p.read_text(encoding="utf-8"))
    if not all(arm_summaries[a].get("ok") for a in ARMS):
        # still write integrity with ok=false
        pass

    # Seed-stream matching across arms
    from mixed_ssl_phase3.matching import compare_domain_streams, extract_domain_hashes

    def _load_rows(arm: str) -> List[dict]:
        path = result_root / "arms" / arm / "logs" / "steps.jsonl"
        rows = []
        if path.is_file():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
        return rows

    hi_rows = _load_rows("SMALL_HI_ONLY")
    sd_rows = _load_rows("SAMLD_ONLY")
    mx_rows = _load_rows("MIXED_1TO1")
    match_hi = compare_domain_streams(
        extract_domain_hashes(hi_rows, "Small-HI"),
        extract_domain_hashes(mx_rows, "Small-HI"),
        domain="Small-HI",
        n=MIXED_STEPS_PER_DOMAIN,
    )
    match_sd = compare_domain_streams(
        extract_domain_hashes(sd_rows, "SAML-D"),
        extract_domain_hashes(mx_rows, "SAML-D"),
        domain="SAML-D",
        n=MIXED_STEPS_PER_DOMAIN,
    )

    init_shas = {a: arm_summaries[a].get("init_sha256") for a in ARMS}
    init_equal = len(set(init_shas.values())) == 1 and None not in init_shas.values()

    integrity = {
        "ok": all(arm_summaries[a].get("ok") for a in ARMS)
        and bool(match_hi.get("ok"))
        and bool(match_sd.get("ok"))
        and init_equal,
        "phase": 3,
        "primary_comparison": "equal_optimizer_budget_1000_steps_all_arms",
        "secondary_comparison": {
            "description": "mixed@500/domain vs single@500 checkpoint",
            "caveat": SECONDARY_COMPARISON_CAVEAT,
            "not_perfectly_lr_phase_matched": True,
        },
        "job_states": {
            a: {
                "job_id": arm_summaries[a].get("job_id"),
                "elapsed_sec": arm_summaries[a].get("elapsed_sec"),
                "ok": arm_summaries[a].get("ok"),
                "preemption_resume_events": arm_summaries[a].get(
                    "preemption_resume_events", []
                ),
            }
            for a in ARMS
        },
        "resolved_configs": {a: resolved_recipe(arm=a) for a in ARMS},
        "checkpoints": {a: arm_summaries[a].get("checkpoints") for a in ARMS},
        "init_sha256_by_arm": init_shas,
        "init_state_equality": init_equal,
        "update_exposure_counts": {
            a: arm_summaries[a].get("step_counts") for a in ARMS
        },
        "seed_stream_matching": {"Small-HI": match_hi, "SAML-D": match_sd},
        "amount_scale_diagnostic": arm_summaries.get("SMALL_HI_ONLY", {}).get(
            "amount_scale_diagnostic"
        )
        or arm_summaries.get("MIXED_1TO1", {}).get("amount_scale_diagnostic"),
        "alpha_beta_freeze_policy": (
            f"frozen through global step {ALPHA_FREEZE_UNTIL_STEP - 1}; "
            f"unfrozen when global_step>={ALPHA_FREEZE_UNTIL_STEP} (matched all arms)"
        ),
        "representation_quality_not_claimed_from_training_losses": True,
        "proposed_next_eval_matrix_unsubmitted": {
            "encoders": [
                "SMALL_HI_ONLY@step1000",
                "SAMLD_ONLY@step1000",
                "MIXED_1TO1@step1000",
            ],
            "targets": ["Small-HI", "SAML-D"],
            "cells": 6,
            "protocol": "frozen full-subgraph R198; same downstream probe; validation-only",
            "no_test_access": True,
            "submitted": False,
        },
        "no_extraction_probe_test_dag_submitted": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out = result_root / "training_integrity_summary.json"
    write_json(out, integrity)

    notes = ROOT / "notes" / "smallhi_samld_mixed_ssl_phase3_scout.md"
    lines = [
        "# Small-HI + SAML-D mixed SSL Phase-3 scout",
        "",
        f"> Twin: `results/diagnostics/smallhi_samld_mixed_ssl_phase3_scout.json`",
        f"> Integrity: `{out}`",
        "",
        f"**ok={integrity['ok']}** — validation-free training-integrity only.",
        "",
        "## Seed-stream matching",
        f"- Small-HI: `{match_hi}`",
        f"- SAML-D: `{match_sd}`",
        "",
        "## Init SHA equality",
        f"- equal={init_equal}: `{init_shas}`",
        "",
        "## Secondary comparison caveat",
        SECONDARY_COMPARISON_CAVEAT,
        "",
        "## Proposed next eval (NOT submitted)",
        json.dumps(integrity["proposed_next_eval_matrix_unsubmitted"], indent=2),
        "",
        "Stop for human review before extraction.",
        "",
    ]
    notes.parent.mkdir(parents=True, exist_ok=True)
    # Append integrity block without wiping earlier preflight content if present
    existing = notes.read_text(encoding="utf-8") if notes.is_file() else ""
    marker = "\n## Post-training integrity\n"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"
    notes.write_text(existing + marker + "\n".join(lines[5:]) + "\n", encoding="utf-8")
    return out


def run_arm(
    arm: str,
    *,
    resume_path: Optional[Path] = None,
    dry_match_batches: int = 4,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if arm not in ARMS:
        raise ValueError(arm)
    unique = arm_unique(arm)
    result_dir = ROOT / RESULT_ROOT / "arms" / arm
    ckpt_dir = ROOT / CKPT_ROOT / arm
    result_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "logs").mkdir(parents=True, exist_ok=True)
    (result_dir / "figures").mkdir(parents=True, exist_ok=True)
    jsonl_path = result_dir / "logs" / "steps.jsonl"

    pre = preflight_phase3(root=ROOT)
    write_json(result_dir / "preflight.json", pre)
    matching = assert_matching_contract_guaranteed()
    write_json(ROOT / RESULT_ROOT / "matching_contract.json", matching)
    if not pre["ok"]:
        raise RuntimeError("phase3 preflight failed")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("CUDA unavailable — unexpected on GPU partition")

    set_seed(SEED)
    with open(ROOT / "data_config.json") as f:
        data_config = json.load(f)

    need_hi = arm in ("SMALL_HI_ONLY", "MIXED_1TO1")
    need_sd = arm in ("SAMLD_ONLY", "MIXED_1TO1")

    ns_hi = make_ns("Small-HI", unique) if need_hi else None
    ns_sd = make_ns("SAML-D", unique) if need_sd else None
    if ns_hi is not None:
        ns_hi.direct_r198_tfmoe_cache = str(ROOT / HI_TF_CACHE)
    if ns_sd is not None:
        ns_sd.direct_r198_tfmoe_cache = str(ROOT / SAMLD_TF_CACHE)

    hi_tr = hi_va = hi_te = None
    hi_tr_i = hi_va_i = hi_te_i = None
    sd_tr = sd_va = sd_te = None
    sd_tr_i = sd_va_i = sd_te_i = None

    if need_hi:
        logging.info("Loading Small-HI under %s...", CONTRACT_ID)
        hi_tr, hi_va, hi_te, hi_tr_i, hi_va_i, hi_te_i = get_data(ns_hi, data_config)
        assert_shared_core_not_historical(ns_hi, hi_tr, "Small-HI")
    if need_sd:
        logging.info("Loading SAML-D under %s...", CONTRACT_ID)
        sd_tr, sd_va, sd_te, sd_tr_i, sd_va_i, sd_te_i = get_data(ns_sd, data_config)
        assert_shared_core_not_historical(ns_sd, sd_tr, "SAML-D")
        if int(sd_te_i.numel()) != 0:
            raise RuntimeError("SAML-D te_inds nonempty — refuse test access")

    hi_scaler = dict(ns_hi.shared_core_edge_scaler) if ns_hi is not None else None
    sd_scaler = dict(ns_sd.shared_core_edge_scaler) if ns_sd is not None else None
    edge_scalers = {}
    if hi_scaler is not None:
        edge_scalers["Small-HI"] = hi_scaler
    if sd_scaler is not None:
        edge_scalers["SAML-D"] = sd_scaler

    amount_diag = None
    if need_hi:
        amount_diag = amount_received_diagnostic(
            train_edge_attr=hi_tr[FORWARD_EDGE_TYPE].edge_attr,
            scaler=hi_scaler,
        )
        write_json(result_dir / "amount_scale_diagnostic.json", amount_diag)
        logging.info(
            "Amount Received std=%.6e max_abs_z=%.4f (documentation only)",
            amount_diag["standard_deviation"],
            amount_diag["max_abs_normalized"],
        )

    transform = AddEgoIds()
    if need_hi:
        add_arange_ids([hi_tr, hi_va])
    if need_sd:
        add_arange_ids([sd_tr, sd_va])

    # Throwaway sample loader so training LinkNeighborLoader generators stay virgin
    # (required for cross-arm seed-stream matching).
    set_seed(SEED)
    meta_data = hi_tr if need_hi else sd_tr
    sample_ns = ns_hi if need_hi else ns_sd
    sample_domain = "Small-HI" if need_hi else "SAML-D"
    sample_tr = hi_tr if need_hi else sd_tr
    _sample_loader = build_train_loader(sample_tr, transform, domain=sample_domain)
    sample = next(iter(_sample_loader))
    del _sample_loader
    model, emb_dim = build_model(sample_ns, meta_data, sample, device)
    moe = TFMoEBundle(in_dim=int(emb_dim), hidden=64, n_targets=3).to(device)
    alpha_beta = LearnedAlphaBeta(n_tf=3, init_alpha=0.6).to(device)
    alpha_beta.set_frozen(True)
    init_path = ROOT / RESULT_ROOT / "shared_init_state.pt"
    init_sha = ensure_shared_init(
        init_path=init_path, model=model, moe=moe, alpha_beta=alpha_beta
    )
    write_json(result_dir / "init_sha256.json", {"init_sha256": init_sha, "arm": arm})
    logging.info("init_sha256=%s", init_sha)

    # Training loaders AFTER init sample — fresh generators for matching contract
    hi_loader = build_train_loader(hi_tr, transform, domain="Small-HI") if need_hi else None
    sd_loader = build_train_loader(sd_tr, transform, domain="SAML-D") if need_sd else None
    hi_iter = infinite_loader(hi_loader) if hi_loader is not None else None
    sd_iter = infinite_loader(sd_loader) if sd_loader is not None else None

    # Fail-before-training: in-process stream matching when both domains available
    if arm == "MIXED_1TO1" and dry_match_batches > 0:
        from train_util import get_hetero_seed_edge_ids as _gid

        def _hash_n(loader, tr, domain, n, rng_states_local):
            it = infinite_loader(loader)
            hashes = []
            for _i in range(n):
                restore_rng(rng_states_local[domain])
                batch = next(it)
                sid = _gid(batch, tr)
                hashes.append(seed_ids_sha(sid))
                rng_states_local[domain] = snapshot_rng()
            return hashes

        hi_l2 = build_train_loader(hi_tr, transform, domain="Small-HI")
        rng_a = init_domain_rng_states(SEED)
        hi_only_h = _hash_n(hi_l2, hi_tr, "Small-HI", dry_match_batches, rng_a)
        hi_l3 = build_train_loader(hi_tr, transform, domain="Small-HI")
        sd_l3 = build_train_loader(sd_tr, transform, domain="SAML-D")
        rng_b = init_domain_rng_states(SEED)
        mixed_hi_h = []
        hi_it = infinite_loader(hi_l3)
        sd_it = infinite_loader(sd_l3)
        for i in range(dry_match_batches * 2):
            domain = "Small-HI" if i % 2 == 0 else "SAML-D"
            restore_rng(rng_b[domain])
            batch = next(hi_it if domain == "Small-HI" else sd_it)
            if domain == "Small-HI":
                mixed_hi_h.append(seed_ids_sha(_gid(batch, hi_tr)))
            rng_b[domain] = snapshot_rng()
        if hi_only_h != mixed_hi_h:
            raise RuntimeError(
                f"seed-stream matching failed before training: {hi_only_h} vs {mixed_hi_h}"
            )
        # Also check SAML-only vs mixed SAML exposures
        sd_l2 = build_train_loader(sd_tr, transform, domain="SAML-D")
        rng_c = init_domain_rng_states(SEED)
        sd_only_h = _hash_n(sd_l2, sd_tr, "SAML-D", dry_match_batches, rng_c)
        sd_l4 = build_train_loader(sd_tr, transform, domain="SAML-D")
        hi_l4 = build_train_loader(hi_tr, transform, domain="Small-HI")
        rng_d = init_domain_rng_states(SEED)
        mixed_sd_h = []
        hi_it2 = infinite_loader(hi_l4)
        sd_it2 = infinite_loader(sd_l4)
        for i in range(dry_match_batches * 2):
            domain = "Small-HI" if i % 2 == 0 else "SAML-D"
            restore_rng(rng_d[domain])
            batch = next(hi_it2 if domain == "Small-HI" else sd_it2)
            if domain == "SAML-D":
                mixed_sd_h.append(seed_ids_sha(_gid(batch, sd_tr)))
            rng_d[domain] = snapshot_rng()
        if sd_only_h != mixed_sd_h:
            raise RuntimeError(
                f"SAML seed-stream matching failed before training: {sd_only_h} vs {mixed_sd_h}"
            )
        logging.info(
            "Pre-train seed-stream match OK for first %s HI and SAML batches",
            dry_match_batches,
        )
        del hi_l2, hi_l3, sd_l3, sd_l2, sd_l4, hi_l4, rng_a, rng_b, rng_c, rng_d

    tf_ctx = {}
    if need_hi:
        tf_ctx["Small-HI"] = load_tf_moe_context(ROOT / HI_TF_CACHE, device)
    if need_sd:
        tf_ctx["SAML-D"] = load_tf_moe_context(ROOT / SAMLD_TF_CACHE, device)

    # Per-domain loss norms for active domains; keep schema keys for checkpoint
    active_domains = []
    if need_hi:
        active_domains.append("Small-HI")
    if need_sd:
        active_domains.append("SAML-D")
    loss_norms = {d: LossNormState() for d in active_domains}
    calib = {d: {"contrast": 0.0, "tf": [0.0, 0.0, 0.0], "n": 0} for d in active_domains}

    enc_params = list(model.parameters())
    moe_params = list(moe.parameters())
    ab_params = list(alpha_beta.parameters())
    optimizer = torch.optim.Adam(
        [
            {"params": enc_params + moe_params, "lr": ENCODER_LR},
            {"params": ab_params, "lr": ALPHABETA_LR},
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
    bn_bundles = {d: clone_bn_bundle(bn_init) for d in active_domains}
    rng_states = init_domain_rng_states(SEED)
    # Only keep RNG for active domains (still init both for matching parity)
    rng_states = {d: rng_states[d] for d in active_domains}

    schedule = arm_schedule(arm)
    if arm == "MIXED_1TO1":
        if schedule.count("Small-HI") != MIXED_STEPS_PER_DOMAIN:
            raise RuntimeError("mixed schedule not 500/500")

    seed_hash_log = {d: [] for d in active_domains}
    step_counts = {d: 0 for d in active_domains}
    checkpoints_meta: Dict[str, Any] = {}
    alpha_unfrozen_at: Optional[int] = None
    preemption_resume_events: List[Dict[str, Any]] = []
    start_step = 0

    if resume_path is not None:
        resume_path = Path(resume_path)
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model_state_dict"], strict=True)
        moe.load_state_dict(blob["moe_state_dict"], strict=True)
        alpha_beta.load_state_dict(blob["alpha_beta_state_dict"], strict=True)
        optimizer.load_state_dict(blob["optimizer_state_dict"])
        scheduler.load_state_dict(blob["scheduler_state_dict"])
        for d in blob["bn_bundles"]:
            if d in bn_bundles:
                bn_bundles[d] = clone_bn_bundle(blob["bn_bundles"][d])
        for d, st in blob["loss_norm_states"].items():
            if d not in loss_norms:
                continue
            loss_norms[d].contrast_mean = st["contrast_mean"]
            loss_norms[d].tf_means = list(st["tf_means"])
            loss_norms[d].calibrated = bool(st["calibrated"])
        step_counts = dict(blob["per_domain_exposure_counts"])
        start_step = int(blob["global_optimizer_step"])
        seed_hash_log = blob.get("seed_hash_log", seed_hash_log)
        if "rng_states" in blob:
            for d, st in blob["rng_states"].items():
                if d in rng_states:
                    rng_states[d] = st
        alpha_beta.set_frozen(start_step < ALPHA_FREEZE_UNTIL_STEP)
        preemption_resume_events.append(
            {
                "resumed_from": str(resume_path),
                "start_step": start_step,
                "at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        logging.info("Resumed from %s at global_step=%s", resume_path, start_step)
        # NOTE: loader iterators are not fully restored; manual resume may desync
        # seed streams after mid-run resume. Prefer restart from scratch if matching
        # integrity is required. Documented in summary.
        if jsonl_path.exists() and start_step > 0:
            # keep existing jsonl; append
            pass
        else:
            jsonl_path.write_text("", encoding="utf-8")
    else:
        if jsonl_path.exists():
            jsonl_path.unlink()

    args_views = ns_hi if ns_hi is not None else ns_sd
    jsonl = open(jsonl_path, "a" if resume_path else "w", encoding="utf-8")

    try:
        for si in range(start_step, TOTAL_STEPS):
            domain = schedule[si]
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
            completed = si + 1  # global optimizer steps completed

            # Calibrate LossNorm from first 5 observations of this domain
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
                        si,
                    )

            # Matched α/β freeze through global step 10 in ALL arms
            if alpha_beta._frozen and completed >= ALPHA_FREEZE_UNTIL_STEP:
                alpha_beta.set_frozen(False)
                alpha_unfrozen_at = completed
                logging.info("Unfreezing alpha/beta after global step %s", completed)

            bn_bundles[domain] = clone_bn_bundle(collect_bn_bundle(model))
            rng_states[domain] = snapshot_rng()
            step_counts[domain] += 1
            if len(seed_hash_log[domain]) < MIXED_STEPS_PER_DOMAIN:
                seed_hash_log[domain].append(stats["seed_ids_sha256"])

            bn_l1_init = bn_bundle_l1(bn_bundles[domain], bn_init)
            bn_cross = None
            if "Small-HI" in bn_bundles and "SAML-D" in bn_bundles:
                bn_cross = bn_bundle_l1(bn_bundles["Small-HI"], bn_bundles["SAML-D"])

            lrs = scheduler.current_lrs()
            row = {
                "step": si,
                "global_optimizer_step": completed,
                "domain": domain,
                "domain_step": step_counts[domain],
                "domain_exposure_count": step_counts[domain],
                "encoder_lr": lrs[0],
                "alphabeta_lr": lrs[1] if len(lrs) > 1 else lrs[0],
                "schedule_phase": scheduler.phase_at(scheduler.completed_optimizer_steps - 1),
                "calibration_complete_domain": bool(loss_norms[domain].calibrated),
                "alpha_beta_frozen": bool(alpha_beta._frozen),
                "bn_l1_vs_init": bn_l1_init,
                "bn_l1_hi_vs_sd": bn_cross,
                "edge_scaler_sha256": edge_scalers[domain]["scaler_sha256"],
                "tf_scaler_mean": tf_ctx[domain].scaler_mean.tolist(),
                **stats,
            }
            # Flatten useful weight keys if present under alternate names
            for src, dst in (
                ("w_c", "w_contrast"),
                ("w_contrast", "w_contrast"),
            ):
                if src in stats and dst not in row:
                    row[dst] = stats[src]
            jsonl.write(json.dumps(row) + "\n")
            jsonl.flush()

            if (si + 1) % 50 == 0 or si == 0:
                logging.info(
                    "step %s/%s domain=%s L=%.4f enc_g=%.3f α_frozen=%s",
                    completed,
                    TOTAL_STEPS,
                    domain,
                    stats["L_total"],
                    stats["encoder_grad_norm"],
                    alpha_beta._frozen,
                )

            # Checkpoints
            ckpt = None
            if completed in CHECKPOINT_STEPS or completed % ROLLING_EVERY == 0:
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
                    arm=arm,
                    unique=unique,
                    init_sha=init_sha,
                    rng_states=rng_states,
                    seed_hash_log=seed_hash_log,
                    amount_diag=amount_diag,
                    preflight=pre,
                )
            if completed in CHECKPOINT_STEPS:
                p = ckpt_dir / f"checkpoint_step_{completed:04d}.tar"
                sha = save_ckpt(ckpt, p)
                checkpoints_meta[f"step_{completed}"] = {"path": str(p), "sha256": sha}
                logging.info("Saved %s sha=%s", p, sha)
            if completed % ROLLING_EVERY == 0:
                p = ckpt_dir / "checkpoint_last.tar"
                sha = save_ckpt(ckpt, p)
                checkpoints_meta["last"] = {
                    "path": str(p),
                    "sha256": sha,
                    "global_step": completed,
                }

            del batch, stats
            if completed % 20 == 0:
                gc.collect()
    finally:
        jsonl.close()

    # BN swap check when both domains present
    swap_ok = True
    bn_differ = False
    if "Small-HI" in bn_bundles and "SAML-D" in bn_bundles:
        bn_differ = not bn_bundles_equal(bn_bundles["Small-HI"], bn_bundles["SAML-D"])
        apply_bn_(model, bn_bundles["Small-HI"])
        snap_a = clone_bn_bundle(collect_bn_bundle(model))
        apply_bn_(model, bn_bundles["SAML-D"])
        snap_b = clone_bn_bundle(collect_bn_bundle(model))
        apply_bn_(model, bn_bundles["Small-HI"])
        snap_a2 = clone_bn_bundle(collect_bn_bundle(model))
        swap_ok = bn_bundles_equal(snap_a, snap_a2) and not bn_bundles_equal(snap_a, snap_b)

    # Reload verification for required checkpoints
    reload_ok = True
    for step in CHECKPOINT_STEPS:
        key = f"step_{step}"
        if key not in checkpoints_meta:
            reload_ok = False
            continue
        p = Path(checkpoints_meta[key]["path"])
        blob = torch.load(p, map_location="cpu", weights_only=False)
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
            "feature_contract_id",
            "saml_split_protocol",
            "resolved",
        ]
        for k in required:
            if k not in blob:
                reload_ok = False
                logging.error("checkpoint %s missing %s", p, k)
        if int(blob.get("global_optimizer_step", -1)) != step:
            reload_ok = False

    # Plots + CSV
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    write_steps_csv(rows, result_dir / "logs" / "steps.csv")
    plot_paths = plot_training_curves(rows, result_dir / "figures", arm=arm)

    expected_counts = {
        "SMALL_HI_ONLY": {"Small-HI": TOTAL_STEPS},
        "SAMLD_ONLY": {"SAML-D": TOTAL_STEPS},
        "MIXED_1TO1": {
            "Small-HI": MIXED_STEPS_PER_DOMAIN,
            "SAML-D": MIXED_STEPS_PER_DOMAIN,
        },
    }[arm]
    counts_ok = all(step_counts.get(d, 0) == n for d, n in expected_counts.items())

    gates = {
        "steps_total": sum(step_counts.values()) == TOTAL_STEPS,
        "exposure_counts_ok": counts_ok,
        "init_sha_recorded": bool(init_sha),
        "alpha_unfrozen_at_step_ge_10": alpha_unfrozen_at is not None
        and alpha_unfrozen_at >= ALPHA_FREEZE_UNTIL_STEP,
        "ckpt_reload_ok": reload_ok,
        "projection_disabled": True,
        "preserve_seed_edges_false": True,
        "amp_false": True,
        "saml_protocol": SAML_SPLIT_PROTOCOL if need_sd else True,
        "no_test_split_loaded_samld": (
            True if not need_sd else int(sd_te_i.numel()) == 0
        ),
        "shared_core_contract": CONTRACT_ID,
        "bn_swap_ok": swap_ok,
    }
    if arm == "MIXED_1TO1":
        gates["bn_bundles_differ"] = bn_differ
        gates["both_domains_calibrated"] = all(
            loss_norms[d].calibrated for d in active_domains
        )

    ok = all(bool(v) if not isinstance(v, str) else True for v in gates.values())

    summary = {
        "ok": ok,
        "gates": gates,
        "arm": arm,
        "unique_name": unique,
        "step_counts": step_counts,
        "init_sha256": init_sha,
        "alpha_unfrozen_after_step": alpha_unfrozen_at,
        "alpha_freeze_policy": (
            f"freeze through step {ALPHA_FREEZE_UNTIL_STEP - 1}; "
            f"unfreeze at completed>={ALPHA_FREEZE_UNTIL_STEP} (all arms matched)"
        ),
        "loss_norm_states": {
            d: {
                "contrast_mean": loss_norms[d].contrast_mean,
                "tf_means": list(loss_norms[d].tf_means),
                "calibrated": loss_norms[d].calibrated,
            }
            for d in loss_norms
        },
        "edge_scalers": edge_scalers,
        "amount_scale_diagnostic": amount_diag,
        "checkpoints": checkpoints_meta,
        "figures": plot_paths,
        "elapsed_sec": time.perf_counter() - t0,
        "device": str(device),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "preemption_resume_events": preemption_resume_events,
        "manual_resume_command": manual_resume_command(
            arm, ckpt_dir / "checkpoint_last.tar"
        ),
        "secondary_comparison_caveat": SECONDARY_COMPARISON_CAVEAT,
        "resolved": resolved_recipe(arm=arm),
        "explicitly_not_done": {
            "extraction": False,
            "probe": False,
            "test_metrics": False,
            "category_adapter": False,
            "paysim": False,
            "paired_domain_updates": False,
            "dependent_dag": False,
            "representation_quality_claim": False,
        },
    }
    write_json(result_dir / "summary.json", summary)
    maybe_write_aggregate(ROOT / RESULT_ROOT)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    logger_setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight_only", action="store_true")
    p.add_argument("--arm", choices=list(ARMS), default=None)
    p.add_argument(
        "--array_task_id",
        type=int,
        default=None,
        help="Override SLURM_ARRAY_TASK_ID mapping 0=SMALL_HI_ONLY,1=SAMLD_ONLY,2=MIXED_1TO1",
    )
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--dry_match_batches", type=int, default=4)
    args = p.parse_args(argv)

    if args.preflight_only:
        pre = preflight_phase3(root=ROOT)
        out = ROOT / RESULT_ROOT / "preflight.json"
        write_json(out, pre)
        # also mirror top-level twin JSON stub
        write_json(
            ROOT / "results" / "diagnostics" / "smallhi_samld_mixed_ssl_phase3_scout.json",
            {
                "phase": 3,
                "status": "preflight",
                "preflight_ok": pre["ok"],
                "runtime_projection": pre["runtime_projection"],
                "storage_projection": pre["storage_projection"],
                "arms": {a: pre["arms"][a]["resolved"] for a in ARMS},
                "slurm": pre["slurm"],
            },
        )
        print(json.dumps(pre, indent=2, default=str))
        return 0 if pre["ok"] else 2

    arm = args.arm
    if arm is None:
        tid = args.array_task_id
        if tid is None:
            env = os.environ.get("SLURM_ARRAY_TASK_ID")
            tid = int(env) if env is not None else None
        if tid is None and os.environ.get("PHASE3_ARM"):
            arm = os.environ["PHASE3_ARM"]
        elif tid is not None:
            if tid not in ARRAY_INDEX_TO_ARM:
                raise SystemExit(f"unknown array task id {tid}")
            arm = ARRAY_INDEX_TO_ARM[tid]
        else:
            raise SystemExit("must pass --arm or SLURM_ARRAY_TASK_ID")

    resume = args.resume or os.environ.get("PHASE3_RESUME") or None
    summary = run_arm(
        arm,
        resume_path=Path(resume) if resume else None,
        dry_match_batches=int(args.dry_match_batches),
    )
    print(
        json.dumps(
            {
                "ok": summary["ok"],
                "arm": arm,
                "gates": summary["gates"],
                "step_counts": summary["step_counts"],
                "init_sha256": summary["init_sha256"],
            },
            indent=2,
        )
    )
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
