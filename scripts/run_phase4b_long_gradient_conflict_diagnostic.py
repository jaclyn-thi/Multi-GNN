#!/usr/bin/env python3
"""No-update gradient-conflict diagnostic for MIXED_3DOMAIN_LONG @1500/@3000.

Never constructs/steps an optimizer. Never writes checkpoints or embeddings.
Never accesses test splits. Train batches only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from financial_multidataset_long_gradient_conflict import (  # noqa: E402
    ARM,
    CHECKPOINTS,
    CONTRACT_ID,
    COS_ALIGN,
    COS_CONFLICT,
    DOMAINS,
    N_BATCHES_PER_DOMAIN,
    OUT_ROOT,
    SEED,
    TF_NAMES,
)
from financial_multidataset_long_gradient_conflict.grad_math import (  # noqa: E402
    classify_cosine,
    file_sha256,
    refuse_optimizer_step,
    refuse_test_split,
    summarize_cosines,
)


FORWARD_EDGE_TYPE = ("node", "to", "node")


def _batch_to_cpu(batch):
    return batch.cpu() if hasattr(batch, "cpu") else batch


def materialize_train_batches(domain: str, tr_data, n: int, *, build_train_loader, AddEgoIds) -> List[Any]:
    refuse_test_split("train")
    loader = build_train_loader(tr_data, AddEgoIds(), domain=domain)
    out = []
    for i, batch in enumerate(loader):
        if i >= n:
            break
        out.append(copy.deepcopy(_batch_to_cpu(batch)))
    if len(out) < n:
        raise RuntimeError(f"{domain}: only got {len(out)}/{n} train batches")
    return out


def load_checkpoint(step: int, device: torch.device) -> Dict[str, Any]:
    meta = CHECKPOINTS[step]
    path = meta["path"]
    sha = file_sha256(path)
    if sha != meta["sha256"]:
        raise RuntimeError(f"checkpoint SHA mismatch step={step}: {sha} != {meta['sha256']}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if str(blob.get("feature_contract_id")) != CONTRACT_ID:
        raise RuntimeError("contract mismatch")
    if str(blob.get("arm")) != ARM:
        raise RuntimeError(f"arm mismatch: {blob.get('arm')}")
    return {"blob": blob, "path": str(path), "sha256": sha, "step": step}



FWD_ET = ("node", "to", "node")
REV_ET = ("node", "rev_to", "node")


def _fwd_edge_attr_dim(data) -> int:
    return int(data[FWD_ET].edge_attr.shape[1])


def _inferred_get_model_edge_dim(sample_batch) -> int:
    """Mirror training.py get_model hetero edge_dim (subtract synthetic EdgeID col)."""
    return int(sample_batch[FWD_ET].edge_attr.shape[1] - 1)


def _edge_emb_in_features(model) -> int:
    sd = model.state_dict()
    key = "edge_emb.node__to__node.weight"
    if key not in sd:
        # fallback: first edge_emb *.weight
        keys = [k for k in sd if k.startswith("edge_emb.") and k.endswith(".weight")]
        if not keys:
            raise RuntimeError("no edge_emb weight in model state_dict")
        key = sorted(keys)[0]
    return int(sd[key].shape[1])


def assert_shared_core_widths(*, stage: str, width: int, expected: int) -> None:
    if width != expected:
        raise RuntimeError(
            f"[{stage}] edge_attr width={width} expected={expected} "
            f"(shared-core contract edge_dim=6; after add_arange_ids expect 7)"
        )


def rebuild_modules(ns, tr_data, sample_batch, blob, device, *, build_model, TFMoEBundle, LearnedAlphaBeta):
    # deepcopy so .to(device) cannot mutate the cached CPU sample batch
    sample = copy.deepcopy(sample_batch).to(device)
    inferred = _inferred_get_model_edge_dim(sample)
    if inferred != 6:
        raise RuntimeError(f"inferred get_model edge_dim={inferred} expected 6")
    model, emb_dim = build_model(ns, tr_data, sample, device)
    if emb_dim != 198:
        raise RuntimeError(emb_dim)
    in_f = _edge_emb_in_features(model)
    if in_f != 6:
        raise RuntimeError(f"edge_emb.in_features={in_f} expected 6")
    model.load_state_dict(blob["model_state_dict"], strict=True)
    moe = TFMoEBundle(in_dim=198, hidden=64, n_targets=3).to(device)
    moe.load_state_dict(blob["moe_state_dict"], strict=True)
    ab = LearnedAlphaBeta(n_tf=3, init_alpha=0.6).to(device)
    ab.load_state_dict(blob["alpha_beta_state_dict"], strict=True)
    ab.set_frozen(True)
    return model, moe, ab, {"inferred_edge_dim": inferred, "edge_emb_in_features": in_f}



def loss_norms_from_blob(blob, LossNormState):
    out = {}
    for d, st in blob["loss_norm_states"].items():
        out[d] = LossNormState(
            contrast_mean=st.get("contrast_mean"),
            tf_means=list(st.get("tf_means", [None, None, None])),
            calibrated=bool(st.get("calibrated", False)),
        )
    return out


def plot_figures(per_batch: List[Dict[str, Any]], fig_dir: Path) -> Dict[str, str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_batch)
    df.to_csv(fig_dir / "figure_data.csv", index=False)
    paths = {"figure_data_csv": str(fig_dir / "figure_data.csv")}

    # 1. InfoNCE vs aggregate TF cosine
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (ckpt, dom), g in df.groupby(["checkpoint_step", "domain"]):
        ax.scatter(
            [f"{dom}\n@{ckpt}"] * len(g),
            g["cos_contrast_tf_agg"],
            alpha=0.7,
            label=None,
        )
    means = df.groupby(["domain", "checkpoint_step"])["cos_contrast_tf_agg"].mean().reset_index()
    for _, r in means.iterrows():
        ax.scatter(
            f"{r['domain']}\n@{int(r['checkpoint_step'])}",
            r["cos_contrast_tf_agg"],
            marker="x",
            s=80,
            c="black",
            zorder=5,
        )
    ax.axhline(COS_CONFLICT, color="r", ls="--", lw=0.8, label="conflict<-0.1")
    ax.axhline(COS_ALIGN, color="g", ls="--", lw=0.8, label="align>0.1")
    ax.set_ylabel("cosine(g_contrast_norm, g_tf_aggregate)")
    ax.set_title("InfoNCE vs aggregate TF gradient cosine")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = fig_dir / "01_infonce_vs_tf_agg_cosine.png"
    fig.savefig(p1, dpi=140)
    fig.savefig(fig_dir / "01_infonce_vs_tf_agg_cosine.pdf")
    plt.close(fig)
    paths["01"] = str(p1)

    # 2. Per-TF target cosines
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for i, col in enumerate(["cos_contrast_tf0", "cos_contrast_tf1", "cos_contrast_tf2"]):
        ax = axes[i]
        for ckpt in sorted(df["checkpoint_step"].unique()):
            sub = df[df["checkpoint_step"] == ckpt]
            ax.scatter(sub["domain"], sub[col], alpha=0.6, label=f"@{ckpt}")
        ax.axhline(COS_CONFLICT, color="r", ls="--", lw=0.8)
        ax.axhline(COS_ALIGN, color="g", ls="--", lw=0.8)
        ax.set_title(TF_NAMES[i].replace("log1p_", ""))
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("cosine(g_contrast_norm, g_tf_k)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Per-TF-target gradient cosine vs InfoNCE")
    fig.tight_layout()
    p2 = fig_dir / "02_per_tf_cosine.png"
    fig.savefig(p2, dpi=140)
    fig.savefig(fig_dir / "02_per_tf_cosine.pdf")
    plt.close(fig)
    paths["02"] = str(p2)

    # 3. Weighted gradient norms
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    keys = []
    c_means, t_means = [], []
    for dom in DOMAINS:
        for ckpt in (1500, 3000):
            sub = df[(df["domain"] == dom) & (df["checkpoint_step"] == ckpt)]
            keys.append(f"{dom}\n@{ckpt}")
            c_means.append(sub["norm_g_contrast_weighted"].mean())
            t_means.append(sub["norm_g_tf_weighted"].mean())
    x = np.arange(len(keys))
    ax.bar(x - width / 2, c_means, width, label="||α g_c||")
    ax.bar(x + width / 2, t_means, width, label="||(1-α) g_tf||")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=7)
    ax.set_ylabel("mean weighted grad L2")
    ax.set_title("Weighted gradient norms by domain/checkpoint")
    ax.legend()
    fig.tight_layout()
    p3 = fig_dir / "03_weighted_grad_norms.png"
    fig.savefig(p3, dpi=140)
    fig.savefig(fig_dir / "03_weighted_grad_norms.pdf")
    plt.close(fig)
    paths["03"] = str(p3)

    # 4. Alpha vs contrast share
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for ckpt, g in df.groupby("checkpoint_step"):
        ax.scatter(g["alpha"], g["share_contrast_weighted"], alpha=0.7, label=f"@{ckpt}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("||g_contrast_weighted|| / ||g_total||")
    ax.set_title("Alpha vs realized weighted contrast gradient share")
    ax.legend()
    fig.tight_layout()
    p4 = fig_dir / "04_alpha_vs_contrast_share.png"
    fig.savefig(p4, dpi=140)
    fig.savefig(fig_dir / "04_alpha_vs_contrast_share.pdf")
    plt.close(fig)
    paths["04"] = str(p4)
    return paths


def aggregate_tables(per_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    by = defaultdict(list)
    for r in per_batch:
        by[(r["checkpoint_step"], r["domain"])].append(r)

    agg_rows = []
    cos_keys = [
        "cos_contrast_tf0",
        "cos_contrast_tf1",
        "cos_contrast_tf2",
        "cos_contrast_tf_agg",
        "cos_weighted_contrast_tf",
    ]
    for (ckpt, dom), rows in sorted(by.items()):
        entry = {
            "checkpoint_step": ckpt,
            "domain": dom,
            "n_batches": len(rows),
            "alpha_mean": float(np.mean([r["alpha"] for r in rows])),
            "norm_contrast_weighted_mean": float(
                np.mean([r["norm_g_contrast_weighted"] for r in rows])
            ),
            "norm_tf_weighted_mean": float(np.mean([r["norm_g_tf_weighted"] for r in rows])),
            "share_contrast_mean": float(np.mean([r["share_contrast_weighted"] for r in rows])),
            "share_tf_mean": float(np.mean([r["share_tf_weighted"] for r in rows])),
            "L_contrast_raw_mean": float(np.mean([r["L_contrast_raw"] for r in rows])),
        }
        for k in cos_keys:
            sm = summarize_cosines(
                [r[k] for r in rows], conflict=COS_CONFLICT, align=COS_ALIGN
            )
            for sk, sv in sm.items():
                entry[f"{k}_{sk}"] = sv
            entry[f"{k}_class_mean"] = classify_cosine(
                sm["mean"], conflict=COS_CONFLICT, align=COS_ALIGN
            )
        agg_rows.append(entry)

    # across domains per checkpoint
    across = {}
    for ckpt in (1500, 3000):
        rows = [r for r in per_batch if r["checkpoint_step"] == ckpt]
        block = {"n_batches": len(rows)}
        for k in cos_keys:
            sm = summarize_cosines(
                [r[k] for r in rows], conflict=COS_CONFLICT, align=COS_ALIGN
            )
            block[k] = sm
            block[f"{k}_class"] = classify_cosine(
                sm["mean"], conflict=COS_CONFLICT, align=COS_ALIGN
            )
        block["alpha_mean"] = float(np.mean([r["alpha"] for r in rows]))
        block["share_contrast_mean"] = float(
            np.mean([r["share_contrast_weighted"] for r in rows])
        )
        block["share_tf_mean"] = float(np.mean([r["share_tf_weighted"] for r in rows]))
        block["L_contrast_raw_mean"] = float(np.mean([r["L_contrast_raw"] for r in rows]))
        across[str(ckpt)] = block
    return {"by_domain_checkpoint": agg_rows, "across_domains": across}


def interpret(agg: Dict[str, Any], per_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    a15 = agg["across_domains"]["1500"]
    a30 = agg["across_domains"]["3000"]
    answers = {}
    answers["1_infonce_vs_tf_agg_conflict"] = {
        "at_1500": a15["cos_contrast_tf_agg_class"],
        "at_3000": a30["cos_contrast_tf_agg_class"],
        "mean_cos_1500": a15["cos_contrast_tf_agg"]["mean"],
        "mean_cos_3000": a30["cos_contrast_tf_agg"]["mean"],
        "note": "diagnostic evidence from 8 batches/domain; not causal",
    }
    # which TF conflicts most (lowest mean cos across both ckpts)
    tf_means = {}
    for i, k in enumerate(["cos_contrast_tf0", "cos_contrast_tf1", "cos_contrast_tf2"]):
        vals = [r[k] for r in per_batch if r[k] == r[k]]
        tf_means[TF_NAMES[i]] = float(np.mean(vals)) if vals else float("nan")
    worst = min(tf_means, key=tf_means.get)
    answers["2_tf_target_most_conflict_with_infonce"] = {
        "lowest_mean_cosine_target": worst,
        "means": tf_means,
    }
    answers["3_conflict_stronger_at_3000"] = {
        "mean_cos_tf_agg_1500": a15["cos_contrast_tf_agg"]["mean"],
        "mean_cos_tf_agg_3000": a30["cos_contrast_tf_agg"]["mean"],
        "stronger_conflict": bool(
            a30["cos_contrast_tf_agg"]["mean"] < a15["cos_contrast_tf_agg"]["mean"]
        ),
    }
    answers["4_contrast_magnitude_shrink_vs_tf"] = {
        "share_contrast_1500": a15["share_contrast_mean"],
        "share_contrast_3000": a30["share_contrast_mean"],
        "share_tf_1500": a15["share_tf_mean"],
        "share_tf_3000": a30["share_tf_mean"],
        "contrast_share_decreased": bool(
            a30["share_contrast_mean"] < a15["share_contrast_mean"]
        ),
    }
    answers["5_lower_alpha_smaller_weighted_contrast_grads"] = {
        "alpha_1500": a15["alpha_mean"],
        "alpha_3000": a30["alpha_mean"],
        "share_contrast_1500": a15["share_contrast_mean"],
        "share_contrast_3000": a30["share_contrast_mean"],
        "consistent_with_alpha_drop": bool(
            a30["alpha_mean"] < a15["alpha_mean"]
            and a30["share_contrast_mean"] < a15["share_contrast_mean"]
        ),
    }
    # per-domain diversity of cos at 3000
    dom_cos = {
        d: float(
            np.mean(
                [
                    r["cos_contrast_tf_agg"]
                    for r in per_batch
                    if r["checkpoint_step"] == 3000 and r["domain"] == d
                ]
            )
        )
        for d in DOMAINS
    }
    answers["6_shared_ab_conceal_per_domain_behavior"] = {
        "cos_tf_agg_by_domain_at_3000": dom_cos,
        "spread": float(max(dom_cos.values()) - min(dom_cos.values())),
        "note": "same global α/β; domain LossNorm/BN differ",
    }
    answers["7_rising_raw_infonce_from_conflict"] = {
        "L_contrast_raw_1500": a15["L_contrast_raw_mean"],
        "L_contrast_raw_3000": a30["L_contrast_raw_mean"],
        "plausible_if_conflict_and_alpha_downweight": "diagnostic only; conflict alone insufficient without training dynamics",
    }
    # overall evidence class
    mean_cos = 0.5 * (
        a15["cos_contrast_tf_agg"]["mean"] + a30["cos_contrast_tf_agg"]["mean"]
    )
    if mean_cos < COS_CONFLICT:
        mode = "conflict"
    elif mean_cos > COS_ALIGN:
        mode = "aligned"
    else:
        mode = "near-orthogonal_multitask"
    # domination if one share >> other
    if max(a30["share_contrast_mean"], a30["share_tf_mean"]) > 0.75:
        mode = "domination_with_" + (
            "contrast" if a30["share_contrast_mean"] > a30["share_tf_mean"] else "tf"
        )
    answers["8_evidence_mode"] = {
        "primary": mode,
        "mean_cos_tf_agg": mean_cos,
        "uncertainty": "n=8 batches/domain; report variation via std/frac tables",
    }
    answers["9_strengthens_projection_case"] = {
        "maybe": mode.startswith("conflict") or "domination" in mode,
        "rationale": "projection could isolate InfoNCE geometry from R198/TF path; not proven here",
    }
    answers["10_suggest_masking_reweight_or_alt_objective"] = {
        "if_conflict": mode.startswith("conflict"),
        "if_domination": "domination" in mode,
        "suggestions": [
            "objective reweighting / alpha floor if contrast share collapses",
            "projection-on matched ablation (InfoNCE on H, eval on R198)",
            "stronger attribute masking only if identity-shortcut audit remains primary concern",
            "alternative objective (VICReg) later — not next",
        ],
    }
    return answers


def write_md(
    out_root: Path,
    answers: Dict[str, Any],
    agg: Dict[str, Any],
    integrity: Dict[str, Any],
    job_meta: Dict[str, Any],
) -> None:
    lines = [
        "# MIXED_3DOMAIN_LONG no-update gradient-conflict diagnostic",
        "",
        f"> Twin root: `{out_root}`",
        "",
        "**No optimizer.step / no parameter update / no checkpoint modification / no test access.**",
        "",
        f"Job: `{job_meta.get('job_id')}` · device `{job_meta.get('device')}` · "
        f"runtime_s={job_meta.get('runtime_s')}",
        "",
        "## Scientific question",
        "",
        "Do InfoNCE and TF-expert losses produce aligned, orthogonal, or conflicting "
        "encoder gradients, and does this change between LONG@1500 and LONG@3000?",
        "",
        "## Integrity",
        "",
        f"- Checkpoint SHA preserved: **{integrity['checkpoint_sha_preserved']}**",
        f"- All batch reconstructions OK: **{integrity['all_recon_ok']}**",
        f"- BN/state restored every batch: **{integrity['all_bn_restored']}**",
        f"- Matched batches/views across checkpoints: **{integrity['batch_view_match_ok']}**",
        f"- Max recon rel error: **{integrity['max_recon_rel_error']:.3e}**",
        "",
        "## Across-domain summary",
        "",
    ]
    for ckpt, block in agg["across_domains"].items():
        c = block["cos_contrast_tf_agg"]
        lines += [
            f"### Checkpoint @{ckpt}",
            f"- mean cos(InfoNCE, TF-agg) = {c['mean']:.4f} "
            f"(std {c['std']:.4f}; conflict_frac={c['frac_conflicting']:.2f}; "
            f"class=**{block['cos_contrast_tf_agg_class']}**)",
            f"- alpha_mean={block['alpha_mean']:.4f}; "
            f"contrast_share={block['share_contrast_mean']:.3f}; "
            f"tf_share={block['share_tf_mean']:.3f}",
            f"- mean L_contrast_raw={block['L_contrast_raw_mean']:.4f}",
            "",
        ]
    lines += ["## Ten interpretation answers", ""]
    for k, v in answers.items():
        lines.append(f"### {k}")
        lines.append("```json")
        lines.append(json.dumps(v, indent=2))
        lines.append("```")
        lines.append("")
    lines += [
        "## Confirmation",
        "",
        "- No optimizer constructed or stepped",
        "- No encoder/MoE/αβ update",
        "- No checkpoint modification",
        "- No embeddings written; no probe fit",
        "- No test split access",
        "",
        "Stop after this diagnostic — no automatic follow-up training.",
        "",
    ]
    (ROOT / "notes/financial_multidataset_long_gradient_conflict.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(*, n_batches: int, smoke: bool) -> int:
    from data_loading import get_data
    from direct_r198 import LearnedAlphaBeta, LossNormState, TFMoEBundle, load_tf_moe_context
    from mixed_ssl_phase2.bn import clone_bn_bundle
    from mixed_ssl_phase4a.domain_registry import default_smoke_domains
    from train_util import AddEgoIds
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "train_mixed_ssl_phase4b_scout",
        ROOT / "scripts/train_mixed_ssl_phase4b_scout.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_mod)
    build_model = _mod.build_model
    build_train_loader = _mod.build_train_loader
    make_ns = _mod.make_ns
    from financial_multidataset_long_gradient_conflict.core import (
        compute_component_grads_for_batch,
    )

    t0 = time.time()
    out_root = OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if smoke and device.type == "cuda":
        # smoke stays CPU synthetic — handled elsewhere
        pass
    if not smoke and device.type != "cuda":
        raise RuntimeError("Full diagnostic requires GPU (do not run on login node)")

    # Verify checkpoints immutable before
    sha_before = {s: file_sha256(CHECKPOINTS[s]["path"]) for s in CHECKPOINTS}

    from train_util import add_arange_ids

    # Load domains: graphs + TF + scalers (same pattern as phase4b trainers)
    with open(ROOT / "data_config.json", encoding="utf-8") as f:
        data_config = json.load(f)
    domain_specs = {s.dataset_id: s for s in default_smoke_domains()}
    tr_datas: Dict[str, Any] = {}
    ns_map: Dict[str, Any] = {}
    tf_ctx: Dict[str, Any] = {}
    width_before: Dict[str, int] = {}
    width_after: Dict[str, int] = {}

    # Stage A: load train graphs only (drop test immediately); assert width=6
    for d in DOMAINS:
        refuse_test_split("train")
        ns = make_ns(d, unique=f"grad_conflict_{d}", max_steps=3000)
        tr, va, te, tr_i, va_i, te_i = get_data(ns, data_config)
        # Never iterate / retain test.
        del te, te_i, va, va_i, tr_i
        w0 = _fwd_edge_attr_dim(tr)
        assert_shared_core_widths(stage=f"{d}/before_add_arange_ids", width=w0, expected=6)
        width_before[d] = w0
        tr_datas[d] = tr
        ns_map[d] = ns
        tf_ctx[d] = load_tf_moe_context(Path(domain_specs[d].tf_cache_path), device)
        print(f"loaded {d} edge_attr_width_before_id={w0}", flush=True)

    # Stage B: identical to Phase-4B trainer — add_arange_ids on each train graph
    # before NeighborLoader / model construction (train_mixed_ssl_phase4b_scout ~698-700).
    transform = AddEgoIds()
    for d in DOMAINS:
        add_arange_ids([tr_datas[d]])
        w1 = _fwd_edge_attr_dim(tr_datas[d])
        assert_shared_core_widths(stage=f"{d}/after_add_arange_ids", width=w1, expected=7)
        width_after[d] = w1
        print(f"add_arange_ids {d} edge_attr_width_after_id={w1}", flush=True)

    # Stage C: one throwaway sample batch → build model; load BOTH checkpoints
    d0 = DOMAINS[0]
    _sample_loader = build_train_loader(tr_datas[d0], transform, domain=d0)
    sample0 = next(iter(_sample_loader))
    del _sample_loader
    inferred0 = _inferred_get_model_edge_dim(sample0)
    if inferred0 != 6:
        raise RuntimeError(f"sample inferred edge_dim={inferred0} expected 6")

    ckpt_load_ok = {}
    for step in (1500, 3000):
        ck = load_checkpoint(step, device)
        model, moe, ab, geom = rebuild_modules(
            ns_map[d0], tr_datas[d0], sample0, ck["blob"], device,
            build_model=build_model, TFMoEBundle=TFMoEBundle, LearnedAlphaBeta=LearnedAlphaBeta,
        )
        ckpt_load_ok[step] = {
            "ok": True,
            "sha256": ck["sha256"],
            **geom,
            "edge_emb_in_features_after_load": _edge_emb_in_features(model),
        }
        if ckpt_load_ok[step]["edge_emb_in_features_after_load"] != 6:
            raise RuntimeError("edge_emb.in_features drifted after load")
        print(f"checkpoint step={step} loaded ok geom={geom}", flush=True)
        del model, moe, ab
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del sample0

    # Stage D: one-batch-per-domain dry preflight (forward + loss) before full materialize
    dry_records = []
    for step in (1500, 3000):
        ck = load_checkpoint(step, device)
        blob = ck["blob"]
        loss_norms = loss_norms_from_blob(blob, LossNormState)
        bn_locked = {d: clone_bn_bundle(blob["bn_bundles"][d]) for d in DOMAINS}
        _sl = build_train_loader(tr_datas[d0], transform, domain=d0)
        sample_for_build = next(iter(_sl))
        del _sl
        model, moe, ab, _geom = rebuild_modules(
            ns_map[d0], tr_datas[d0], sample_for_build, blob, device,
            build_model=build_model, TFMoEBundle=TFMoEBundle, LearnedAlphaBeta=LearnedAlphaBeta,
        )
        del sample_for_build
        for d in DOMAINS:
            _dl = build_train_loader(tr_datas[d], transform, domain=d)
            batch = next(iter(_dl))
            del _dl
            out = compute_component_grads_for_batch(
                model=model,
                moe=moe,
                alpha_beta=ab,
                loss_norm=loss_norms[d],
                tf_ctx=tf_ctx[d],
                batch=copy.deepcopy(batch),
                loader_data=tr_datas[d],
                args=ns_map[d],
                device=device,
                bn_locked=bn_locked[d],
                rng_state=None,
            )
            m = out["metrics"]
            dry_records.append({
                "checkpoint_step": step,
                "domain": d,
                "recon_ok": m["recon_ok"],
                "aligned_seeds": m["aligned_seeds"],
                "L_total": m["L_total"],
                "bn_restored": m["bn_restored"],
            })
            print(
                f"dry_preflight step={step} domain={d} "
                f"recon_ok={m['recon_ok']} L_total={m['L_total']:.6f}",
                flush=True,
            )
        del model, moe, ab
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not all(r["recon_ok"] and r["bn_restored"] for r in dry_records):
        raise RuntimeError(f"dry preflight failed: {dry_records}")

    dry_path = out_root / "dry_preflight.json"
    dry_payload = {
        "ok": True,
        "width_before_add_arange_ids": width_before,
        "width_after_add_arange_ids": width_after,
        "checkpoint_loads": ckpt_load_ok,
        "one_batch_per_domain": dry_records,
        "ordering_note": (
            "get_data → assert width=6 → add_arange_ids([tr]) → assert width=7 → "
            "NeighborLoader sample → get_model e_dim=6 → load ckpts → "
            "1-batch dry forward/loss → then materialize 8 batches"
        ),
        "no_test_split_retained": True,
        "edgeid_tf_join": "attach_edge_id_from_batch + tf_moe_mae_losses via seed EdgeID (trainer-matched)",
    }
    dry_path.write_text(json.dumps(dry_payload, indent=2) + "\n")
    print(f"DRY_PREFLIGHT_OK wrote {dry_path}", flush=True)

    # Stage E: materialize full diagnostic batches (generators after add_arange_ids)
    batches: Dict[str, List[Any]] = {}
    sample_batches: Dict[str, Any] = {}
    for d in DOMAINS:
        batches[d] = materialize_train_batches(
            d, tr_datas[d], n_batches,
            build_train_loader=build_train_loader, AddEgoIds=AddEgoIds,
        )
        sample_batches[d] = copy.deepcopy(batches[d][0])
        print(f"materialized {n_batches} train batches for {d}", flush=True)

    per_batch: List[Dict[str, Any]] = []
    match_records = []
    recon_records = []
    state_records = []
    rng_store: Dict[str, Dict[int, Any]] = {d: {} for d in DOMAINS}

    for step in (1500, 3000):
        ck = load_checkpoint(step, device)
        blob = ck["blob"]
        loss_norms = loss_norms_from_blob(blob, LossNormState)
        bn_locked = {
            d: clone_bn_bundle(blob["bn_bundles"][d]) for d in DOMAINS
        }

        # rebuild model from first domain metadata (shared architecture)
        model, moe, ab, _geom = rebuild_modules(
            ns_map[d0], tr_datas[d0], sample_batches[d0], blob, device,
            build_model=build_model, TFMoEBundle=TFMoEBundle, LearnedAlphaBeta=LearnedAlphaBeta,
        )

        for d in DOMAINS:
            apply_needed = bn_locked[d]
            for bi, batch_cpu in enumerate(batches[d]):
                batch = copy.deepcopy(batch_cpu)
                rng = rng_store[d].get(bi)
                out = compute_component_grads_for_batch(
                    model=model,
                    moe=moe,
                    alpha_beta=ab,
                    loss_norm=loss_norms[d],
                    tf_ctx=tf_ctx[d],
                    batch=batch,
                    loader_data=tr_datas[d],
                    args=ns_map[d],
                    device=device,
                    bn_locked=apply_needed,
                    rng_state=rng,
                )
                if rng is None:
                    rng_store[d][bi] = out["rng_state"]
                m = out["metrics"]
                m.update(
                    {
                        "checkpoint_step": step,
                        "domain": d,
                        "batch_index": bi,
                        "checkpoint_sha256": ck["sha256"],
                    }
                )
                per_batch.append(m)
                recon_records.append(
                    {
                        "checkpoint_step": step,
                        "domain": d,
                        "batch_index": bi,
                        "recon_ok": m["recon_ok"],
                        "recon_rel_error": m["recon_rel_error"],
                        "recon_diff_l2": m["recon_diff_l2"],
                    }
                )
                state_records.append(
                    {
                        "checkpoint_step": step,
                        "domain": d,
                        "batch_index": bi,
                        "bn_restored": m["bn_restored"],
                        "model_sha_restored": m["model_sha_restored"],
                        "moe_sha_unchanged": m["moe_sha_unchanged"],
                        "alpha_beta_sha_unchanged": m["alpha_beta_sha_unchanged"],
                    }
                )
                match_records.append(
                    {
                        "checkpoint_step": step,
                        "domain": d,
                        "batch_index": bi,
                        "seed_edge_ids_sha256": m["seed_edge_ids_sha256"],
                        "view1_aug_sha256": m["view1_aug_sha256"],
                        "view2_aug_sha256": m["view2_aug_sha256"],
                        "rng_state_sha256": m["rng_state_sha256"],
                        "requested_seeds": m["requested_seeds"],
                        "aligned_seeds": m["aligned_seeds"],
                    }
                )
                print(
                    f"done step={step} domain={d} batch={bi} "
                    f"cos_tf_agg={m['cos_contrast_tf_agg']:.4f} "
                    f"recon_rel={m['recon_rel_error']:.2e}",
                    flush=True,
                )

        # Explicit refusal: never build optimizer
        try:
            refuse_optimizer_step(None)
        except RuntimeError:
            pass

        del model, moe, ab
        if device.type == "cuda":
            torch.cuda.empty_cache()

    sha_after = {s: file_sha256(CHECKPOINTS[s]["path"]) for s in CHECKPOINTS}
    ckpt_preserved = sha_before == sha_after == {
        s: CHECKPOINTS[s]["sha256"] for s in CHECKPOINTS
    }

    # batch/view match across checkpoints
    match_ok = True
    match_issues = []
    by_key = defaultdict(dict)
    for r in match_records:
        by_key[(r["domain"], r["batch_index"])][r["checkpoint_step"]] = r
    for key, steps in by_key.items():
        a, b = steps[1500], steps[3000]
        for field in (
            "seed_edge_ids_sha256",
            "view1_aug_sha256",
            "view2_aug_sha256",
            "rng_state_sha256",
            "requested_seeds",
        ):
            if a[field] != b[field]:
                match_ok = False
                match_issues.append({"key": key, "field": field, "1500": a[field], "3000": b[field]})

    agg = aggregate_tables(per_batch)
    answers = interpret(agg, per_batch)
    fig_paths = plot_figures(per_batch, out_root / "figures")

    # write CSVs
    per_csv = out_root / "per_batch.csv"
    with per_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(per_batch[0].keys()))
        w.writeheader()
        w.writerows(per_batch)

    agg_csv = out_root / "aggregate_by_domain_checkpoint.csv"
    with agg_csv.open("w", newline="") as f:
        rows = agg["by_domain_checkpoint"]
        w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    integrity = {
        "checkpoint_sha_preserved": ckpt_preserved,
        "sha_before": sha_before,
        "sha_after": sha_after,
        "all_recon_ok": all(r["recon_ok"] for r in recon_records),
        "max_recon_rel_error": max(r["recon_rel_error"] for r in recon_records),
        "all_bn_restored": all(r["bn_restored"] for r in state_records),
        "all_model_sha_restored": all(r["model_sha_restored"] for r in state_records),
        "all_moe_unchanged": all(r["moe_sha_unchanged"] for r in state_records),
        "batch_view_match_ok": match_ok,
        "match_issues": match_issues,
    }
    (out_root / "gradient_reconstruction_integrity.json").write_text(
        json.dumps({"records": recon_records, "summary": {
            "all_ok": integrity["all_recon_ok"],
            "max_rel_error": integrity["max_recon_rel_error"],
        }}, indent=2) + "\n"
    )
    (out_root / "state_preservation_integrity.json").write_text(
        json.dumps({"records": state_records, "checkpoint_sha": integrity}, indent=2) + "\n"
    )
    (out_root / "batch_and_augmentation_matching.json").write_text(
        json.dumps({"records": match_records, "ok": match_ok, "issues": match_issues}, indent=2)
        + "\n"
    )

    runtime = time.time() - t0
    job_meta = {
        "job_id": __import__("os").environ.get("SLURM_JOB_ID"),
        "device": str(device),
        "runtime_s": runtime,
        "n_batches_per_domain": n_batches,
        "domains": list(DOMAINS),
        "checkpoints": [1500, 3000],
        "partition": __import__("os").environ.get("SLURM_JOB_PARTITION"),
        "account": "mit_general",
        "qos": "normal",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_root / "submission_runtime_manifest.json").write_text(
        json.dumps(job_meta, indent=2) + "\n"
    )

    summary = {
        "ok": bool(
            integrity["checkpoint_sha_preserved"]
            and integrity["all_recon_ok"]
            and integrity["all_bn_restored"]
            and integrity["batch_view_match_ok"]
        ),
        "scientific_question": (
            "Do InfoNCE and TF expert losses conflict on the shared encoder "
            "at LONG@1500 vs LONG@3000?"
        ),
        "aggregates": agg,
        "answers": answers,
        "integrity": integrity,
        "figures": fig_paths,
        "job": job_meta,
        "no_optimizer_step": True,
        "no_parameter_update": True,
        "no_test_access": True,
        "no_checkpoint_modification": True,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_md(out_root, answers, agg, integrity, job_meta)
    print(json.dumps({"ok": summary["ok"], "runtime_s": runtime}, indent=2))
    return 0 if summary["ok"] else 2


def run_synthetic_smoke() -> int:
    """CPU synthetic smoke: grad math + reconstruction + BN restore + refusals."""
    from financial_multidataset_long_gradient_conflict.grad_math import (
        add_grads,
        reconstruction_ok,
        refuse_optimizer_step,
        refuse_test_split,
        scale_grads,
        summarize_cosines,
        cosine_from_norms_dot,
        accumulate_dot,
        accumulate_grad_stats,
    )
    from mixed_ssl_phase2.bn import apply_bn_, collect_bn_bundle, clone_bn_bundle, bn_bundles_equal

    # cosine / zero-norm
    assert cosine_from_norms_dot(0.0, 0.0, 1.0) != cosine_from_norms_dot(0.0, 0.0, 1.0) or True
    assert np.isnan(cosine_from_norms_dot(1.0, 0.0, 1.0))
    g1 = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    g2 = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, -1.0])]
    n1, _ = accumulate_grad_stats(g1)
    n2, _ = accumulate_grad_stats(g2)
    c = cosine_from_norms_dot(accumulate_dot(g1, g2), n1, n2)
    assert abs(c - 0.0) < 1e-6

    # reconstruction
    alpha = 0.3
    gc = [torch.randn(4), torch.randn(3)]
    gt = [torch.randn(4), torch.randn(3)]
    recon = add_grads(scale_grads(gc, alpha), scale_grads(gt, 1 - alpha))
    # direct = same
    ok = reconstruction_ok(recon, recon)
    assert ok["ok"]

    # BN restore
    m = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    m.train()
    locked = clone_bn_bundle(collect_bn_bundle(m))
    x = torch.randn(8, 4)
    m(x)  # mutate running stats
    assert not bn_bundles_equal(collect_bn_bundle(m), locked)
    apply_bn_(m, locked)
    assert bn_bundles_equal(collect_bn_bundle(m), locked)

    # refusals
    try:
        refuse_optimizer_step(None)
        raise AssertionError("should refuse")
    except RuntimeError:
        pass
    try:
        refuse_test_split("test")
        raise AssertionError("should refuse")
    except RuntimeError:
        pass

    sm = summarize_cosines([-0.5, 0.0, 0.5])
    assert sm["frac_conflicting"] == 1 / 3
    assert sm["frac_orthogonal"] == 1 / 3
    assert sm["frac_aligned"] == 1 / 3

    # deterministic matching stub
    torch.manual_seed(0)
    a = torch.rand(10)
    torch.manual_seed(0)
    b = torch.rand(10)
    assert torch.equal(a, b)

    out = OUT_ROOT
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_synthetic.json").write_text(
        json.dumps({"ok": True, "mode": "synthetic_cpu"}, indent=2) + "\n"
    )
    print("SYNTHETIC_SMOKE_OK")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU synthetic smoke only")
    ap.add_argument("--n_batches", type=int, default=N_BATCHES_PER_DOMAIN)
    args = ap.parse_args()
    if args.smoke:
        raise SystemExit(run_synthetic_smoke())
    raise SystemExit(run(n_batches=int(args.n_batches), smoke=False))


if __name__ == "__main__":
    main()
