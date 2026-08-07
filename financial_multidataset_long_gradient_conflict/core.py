"""Core no-update gradient extraction matching Phase-4B mixed_step."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from contrastive_loss import edge_identity_infonce_loss
from direct_r198 import LearnedAlphaBeta, LossNormState, combine_direct_h_tfmoe_loss, tf_moe_mae_losses
from direct_r198.seed_readout import align_seed_r198_pair, forward_seed_r198_hetero
from financial_multidataset_long_gradient_conflict import N_NEG, RECON_ATOL, RECON_RTOL, TEMP
from financial_multidataset_long_gradient_conflict.grad_math import (
    accumulate_dot,
    accumulate_grad_stats,
    add_grads,
    bn_bundle_sha256,
    cosine_from_norms_dot,
    encoder_parameters,
    reconstruction_ok,
    scale_grads,
    state_tensor_sha256,
    tensor_sha256,
)
from graph_augmentations import generate_views
from mixed_ssl_phase2.bn import apply_bn_, bn_bundles_equal, clone_bn_bundle, collect_bn_bundle
from train_util import attach_edge_id_from_batch, get_hetero_seed_edge_ids
from training import _contrastive_view_kwargs

FWD = ("node", "to", "node")


def _autograd_enc(
    loss: torch.Tensor,
    enc_params: Sequence[nn.Parameter],
    *,
    retain_graph: bool,
) -> List[Optional[torch.Tensor]]:
    grads = torch.autograd.grad(
        loss,
        enc_params,
        retain_graph=retain_graph,
        allow_unused=True,
        create_graph=False,
    )
    return list(grads)


def hash_view_forward(view) -> str:
    store = view[FWD]
    parts = [
        tensor_sha256(store.edge_index.cpu()),
        tensor_sha256(store.edge_attr.cpu()),
    ]
    if hasattr(store, "edge_id") and store.edge_id is not None:
        parts.append(tensor_sha256(store.edge_id.cpu()))
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()


def capture_rng_state(device: torch.device) -> Dict[str, Any]:
    st: Dict[str, Any] = {"torch": torch.get_rng_state()}
    if device.type == "cuda" and torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    else:
        st["cuda"] = None
    return st


def restore_rng_state(st: Dict[str, Any]) -> None:
    torch.set_rng_state(st["torch"])
    if st.get("cuda") is not None:
        torch.cuda.set_rng_state_all(st["cuda"])


@torch.no_grad()
def seed_ids_sha(ids: torch.Tensor) -> str:
    return tensor_sha256(ids.detach().cpu().long().contiguous())


def rng_state_sha256(rng_state: Dict[str, Any]) -> str:
    raw = bytes(rng_state["torch"].cpu().numpy().tobytes())
    return hashlib.sha256(raw).hexdigest()


def compute_component_grads_for_batch(
    *,
    model: nn.Module,
    moe: nn.Module,
    alpha_beta: LearnedAlphaBeta,
    loss_norm: LossNormState,
    tf_ctx,
    batch,
    loader_data,
    args,
    device: torch.device,
    bn_locked: Dict[str, torch.Tensor],
    rng_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train-mode forward + component encoder grads; restore BN; never step."""
    model.train()
    moe.train()
    alpha_beta.set_frozen(True)

    apply_bn_(model, bn_locked)
    model_sha_before = state_tensor_sha256(model)
    moe_sha_before = state_tensor_sha256(moe)
    ab_sha_before = state_tensor_sha256(alpha_beta)

    seed_edge_ids = get_hetero_seed_edge_ids(batch, loader_data)
    attach_edge_id_from_batch(batch, loader_data)
    requested = int(seed_edge_ids.numel())
    sid_hash = seed_ids_sha(seed_edge_ids)

    batch = batch.to(device)
    seed_edge_ids = seed_edge_ids.to(device)

    if rng_state is not None:
        restore_rng_state(rng_state)
    else:
        rng_state = capture_rng_state(device)

    view1, view2 = generate_views(
        batch,
        **_contrastive_view_kwargs(args, {}, seed_edge_ids=seed_edge_ids),
    )
    v1_hash = hash_view_forward(view1)
    v2_hash = hash_view_forward(view2)

    z1_all, id1_all, _ = forward_seed_r198_hetero(model, view1, seed_edge_ids)
    with torch.no_grad():
        z2_all, id2_all, _ = forward_seed_r198_hetero(model, view2, seed_edge_ids)
    z1_seed, seed_id1, z2_seed, seed_id2 = align_seed_r198_pair(
        z1_all, id1_all, z2_all, id2_all
    )
    aligned = int(seed_id1.numel())
    z2_seed = z2_seed.detach()

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
    tf_raws, _tf_diag = tf_moe_mae_losses(z1_seed, seed_id1, moe, tf_ctx)

    with torch.no_grad():
        alpha_t, beta_t = alpha_beta()
        alpha = float(alpha_t.detach().cpu())
        betas = [float(beta_t[i].detach().cpu()) for i in range(3)]
    alpha_c = contrast_raw.new_tensor(alpha)
    beta_c = [contrast_raw.new_tensor(betas[i]) for i in range(3)]

    c_n = loss_norm.normalize_contrast(contrast_raw)
    tf_ns = [loss_norm.normalize_tf(tf_raws[m], m) for m in range(3)]
    tf_agg = sum(beta_c[m] * tf_ns[m] for m in range(3))
    total_direct = alpha_c * c_n + (1.0 - alpha_c) * tf_agg

    total_official, stats = combine_direct_h_tfmoe_loss(
        contrast_raw=contrast_raw,
        tf_raws=tf_raws,
        alpha_beta=alpha_beta,
        norm=loss_norm,
        weight_mode="adaptive",
    )
    if abs(float(total_direct.detach()) - float(total_official.detach())) > 1e-5:
        raise RuntimeError(
            f"total mismatch direct={float(total_direct)} official={float(total_official)}"
        )

    enc_params = encoder_parameters(model)

    g_contrast_raw = _autograd_enc(contrast_raw, enc_params, retain_graph=True)
    g_contrast_norm = _autograd_enc(c_n, enc_params, retain_graph=True)
    g_tf0 = _autograd_enc(tf_ns[0], enc_params, retain_graph=True)
    g_tf1 = _autograd_enc(tf_ns[1], enc_params, retain_graph=True)
    g_tf2 = _autograd_enc(tf_ns[2], enc_params, retain_graph=True)
    g_tf_aggregate = _autograd_enc(tf_agg, enc_params, retain_graph=True)
    g_total_direct = _autograd_enc(total_direct, enc_params, retain_graph=False)

    g_contrast_weighted = scale_grads(g_contrast_norm, alpha)
    g_tf_weighted = scale_grads(g_tf_aggregate, 1.0 - alpha)
    g_total_recon = add_grads(g_contrast_weighted, g_tf_weighted)
    recon = reconstruction_ok(
        g_total_recon, g_total_direct, rtol=RECON_RTOL, atol=RECON_ATOL
    )
    if not recon["ok"]:
        raise RuntimeError(f"gradient reconstruction failed: {recon}")

    def _n(gs):
        return accumulate_grad_stats(gs)[0]

    def _cos(a, b):
        na, _ = accumulate_grad_stats(a)
        nb, _ = accumulate_grad_stats(b)
        return cosine_from_norms_dot(accumulate_dot(a, b), na, nb)

    apply_bn_(model, bn_locked)
    bn_after = collect_bn_bundle(model)
    moe_sha_after = state_tensor_sha256(moe)
    ab_sha_after = state_tensor_sha256(alpha_beta)
    model_sha_final = state_tensor_sha256(model)
    bn_restored = bn_bundles_equal(bn_after, bn_locked)

    row = {
        "requested_seeds": requested,
        "aligned_seeds": aligned,
        "seed_edge_ids_sha256": sid_hash,
        "view1_aug_sha256": v1_hash,
        "view2_aug_sha256": v2_hash,
        "rng_state_sha256": rng_state_sha256(rng_state),
        "L_contrast_raw": float(contrast_raw.detach().cpu()),
        "L_contrast_norm": float(c_n.detach().cpu()),
        "L_tf0_norm": float(tf_ns[0].detach().cpu()),
        "L_tf1_norm": float(tf_ns[1].detach().cpu()),
        "L_tf2_norm": float(tf_ns[2].detach().cpu()),
        "L_tf_aggregate": float(tf_agg.detach().cpu()),
        "L_total": float(total_direct.detach().cpu()),
        "alpha": alpha,
        "beta0": betas[0],
        "beta1": betas[1],
        "beta2": betas[2],
        "w_contrast": alpha,
        "w_tf0": (1.0 - alpha) * betas[0],
        "w_tf1": (1.0 - alpha) * betas[1],
        "w_tf2": (1.0 - alpha) * betas[2],
        "weighted_contrast": float((alpha_c * c_n).detach().cpu()),
        "weighted_tf_agg": float(((1.0 - alpha_c) * tf_agg).detach().cpu()),
        "norm_g_contrast_raw": _n(g_contrast_raw),
        "norm_g_contrast_norm": _n(g_contrast_norm),
        "norm_g_tf0_norm": _n(g_tf0),
        "norm_g_tf1_norm": _n(g_tf1),
        "norm_g_tf2_norm": _n(g_tf2),
        "norm_g_tf_aggregate": _n(g_tf_aggregate),
        "norm_g_contrast_weighted": _n(g_contrast_weighted),
        "norm_g_tf_weighted": _n(g_tf_weighted),
        "norm_g_total_direct": _n(g_total_direct),
        "norm_g_total_recon": _n(g_total_recon),
        "cos_contrast_tf0": _cos(g_contrast_norm, g_tf0),
        "cos_contrast_tf1": _cos(g_contrast_norm, g_tf1),
        "cos_contrast_tf2": _cos(g_contrast_norm, g_tf2),
        "cos_contrast_tf_agg": _cos(g_contrast_norm, g_tf_aggregate),
        "cos_weighted_contrast_tf": _cos(g_contrast_weighted, g_tf_weighted),
        "share_contrast_weighted": _n(g_contrast_weighted) / max(_n(g_total_direct), 1e-12),
        "share_tf_weighted": _n(g_tf_weighted) / max(_n(g_total_direct), 1e-12),
        "recon_ok": bool(recon["ok"]),
        "recon_diff_l2": float(recon["diff_l2"]),
        "recon_rel_error": float(recon["rel_error"]),
        "bn_restored": bool(bn_restored),
        "model_sha_restored": model_sha_final == model_sha_before,
        "moe_sha_unchanged": moe_sha_after == moe_sha_before,
        "alpha_beta_sha_unchanged": ab_sha_after == ab_sha_before,
        "bn_locked_sha256": bn_bundle_sha256(bn_locked),
        "official_stats_alpha": float(stats["alpha"]),
    }
    del (
        z1_seed,
        z2_seed,
        contrast_raw,
        tf_raws,
        c_n,
        tf_ns,
        tf_agg,
        total_direct,
        total_official,
        view1,
        view2,
        batch,
        g_contrast_raw,
        g_contrast_norm,
        g_tf0,
        g_tf1,
        g_tf2,
        g_tf_aggregate,
        g_contrast_weighted,
        g_tf_weighted,
        g_total_direct,
        g_total_recon,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"metrics": row, "rng_state": rng_state}
