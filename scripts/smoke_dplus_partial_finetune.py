#!/usr/bin/env python3
"""Bounded GPU smoke for D+ partial fine-tune (no full job submission)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dplus_partial_finetune import (
    CLF_LR,
    DPLUS_CKPT,
    DPLUS_CKPT_SHA256,
    DPLUS_PRE3H_DIR,
    ENC_LR,
    FinetuneState,
    MAX_TOTAL_EPOCHS,
    STACK_DIM,
    WARMUP_EPOCHS_DEFAULT,
    apply_trainability,
    assert_dplus_checkpoint,
    build_graph_args,
    build_optimizer,
    equivalence_check_pre3h,
    eval_clf,
    GradPreEmbeddingCapture,
    load_dplus_hetero_encoder,
    load_finetune_checkpoint,
    load_hxxtf_matrices,
    optimizer_group_report,
    param_deltas,
    run_partial_epoch_online,
    save_finetune_checkpoint,
    set_encoder_modes,
    snapshot_params,
    tf_leakage_audit,
    train_mlp_epoch,
)
from data_loading import get_data
from gcpal_txn_node.eval_mlp import PaperStyleMLP
from train_util import AddEgoIds, get_loaders
from util import logger_setup, set_seed

# ROOT already set above for sys.path


def _gpu_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return float(torch.cuda.max_memory_allocated() / (1024**3))


def main() -> None:
    logger_setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--partial_steps", type=int, default=2)
    p.add_argument("--equiv_batches", type=int, default=2)
    p.add_argument("--loader_num_workers", type=int, default=0)
    p.add_argument(
        "--output_json",
        default="results/diagnostics/dplus_partial_finetune_smoke.json",
    )
    p.add_argument(
        "--output_md",
        default="notes/dplus_partial_finetune_smoke.md",
    )
    p.add_argument(
        "--ckpt_dir",
        default="saved-models/dplus_partial_finetune_smoke_seed2",
    )
    args_ns = p.parse_args()
    set_seed(args_ns.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    report: dict = {
        "title": "dplus_partial_finetune_smoke",
        "pass": False,
        "device": str(device),
        "source_checkpoint": str(DPLUS_CKPT),
        "source_sha256": DPLUS_CKPT_SHA256,
        "mlp_init_path": None,
        "full_job_command": None,
        "projected_full_runtime_h": None,
        "six_hour_safe": None,
    }

    failures = []

    # --- checkpoint lock ---
    try:
        ckpt_meta = assert_dplus_checkpoint()
        report["checkpoint_meta"] = ckpt_meta
    except Exception as e:
        failures.append(f"checkpoint: {e}")
        _write(report, failures, args_ns, t0)
        raise

    leak = tf_leakage_audit()
    report["tf_leakage_audit"] = leak
    if not leak["ok"]:
        failures.append("TF leakage audit failed")

    # --- features / frozen reference path ---
    feats = load_hxxtf_matrices(str(ROOT / "data_config.json"))
    report["mlp_init_path"] = feats["mlp_init_path"]
    report["winning_mlp_weights_found"] = feats["winning_mlp_weights_found"]
    report["winning_mlp_note"] = feats["winning_mlp_note"]
    report["stack_dim"] = int(feats["x_tr"].shape[1])
    if feats["x_tr"].shape[1] != STACK_DIM:
        failures.append(f"stack dim {feats['x_tr'].shape[1]} != {STACK_DIM}")

    with open(ROOT / "data_config.json", encoding="utf-8") as f:
        data_config = json.load(f)
    graph_args = build_graph_args(seed=args_ns.seed, loader_num_workers=args_ns.loader_num_workers)
    logging.info("Loading graph…")
    t_data = time.perf_counter()
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(graph_args, data_config)
    report["data_load_s"] = time.perf_counter() - t_data

    encoder, head_spec, pre_dim, ckpt_epoch, _cfg = load_dplus_hetero_encoder(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, graph_args, data_config, device
    )
    report["loaded_ckpt_epoch"] = ckpt_epoch
    report["pre_dim"] = pre_dim

    # --- equivalence: live pre-3h vs cached ---
    equiv = equivalence_check_pre3h(
        encoder,
        head_spec,
        pre_dim,
        tr_data,
        tr_inds,
        graph_args,
        device,
        feats["splits"]["train"]["Z"],
        feats["ids_tr"],
        max_batches=args_ns.equiv_batches,
    )
    report["equivalence_pre3h"] = equiv
    if not equiv["ok"]:
        failures.append(
            f"pre-3h equivalence failed matched={equiv['matched_ids']} max_abs={equiv['max_abs_diff']}"
        )

    # --- Stage 1 warmup (classifier only) ---
    trainability_w = apply_trainability(encoder, "warmup")
    set_encoder_modes(encoder, "warmup")
    enc_before = snapshot_params(encoder)
    clf = PaperStyleMLP(STACK_DIM, hidden=128, dropout=0.1).to(device)
    for p in clf.parameters():
        p.requires_grad = True
    opt = build_optimizer("warmup", encoder, clf)
    report["stage1_trainable_encoder"] = trainability_w["trainable"]
    report["stage1_frozen_encoder_n"] = len(trainability_w["frozen"])
    report["stage1_optimizer_groups"] = optimizer_group_report(opt)

    # Reproduce frozen-reference behavior (short smoke: 1 epoch; full uses 15)
    warmup_losses = []
    frozen_ref = None
    for ep in range(args_ns.warmup_epochs):
        loss = train_mlp_epoch(
            clf,
            feats["x_tr"],
            feats["y_tr"],
            optimizer=opt,
            device=device,
            seed=args_ns.seed,
            epoch=ep,
        )
        warmup_losses.append(loss)
        if not np.isfinite(loss):
            failures.append(f"non-finite warmup loss ep{ep}")
    enc_after = snapshot_params(encoder)
    deltas_warm = param_deltas(enc_before, enc_after)
    report["stage1_encoder_param_deltas"] = deltas_warm
    if deltas_warm:
        failures.append(f"encoder changed during warmup: {list(deltas_warm)[:5]}")
    frozen_ref = eval_clf(clf, feats["x_va"], feats["y_va"], device=device, split_name="val")
    report["stage1_val_after_warmup"] = frozen_ref
    report["stage1_losses"] = warmup_losses
    # Guard: must not touch test during selection
    try:
        eval_clf(clf, feats["x_te"], feats["y_te"], device=device, split_name="test", allow_test=False)
        failures.append("test eval was allowed during selection")
    except RuntimeError:
        report["test_blocked_during_selection"] = True

    # --- save/resume across stage transition ---
    ckpt_dir = ROOT / args_ns.ckpt_dir
    ckpt_path = ckpt_dir / "checkpoint_last.tar"
    state = FinetuneState(stage="warmup", global_epoch=args_ns.warmup_epochs, warmup_epochs_done=args_ns.warmup_epochs)
    save_finetune_checkpoint(
        ckpt_path,
        encoder=encoder,
        clf=clf,
        optimizer=opt,
        state=state,
        scaler=feats["scaler"],
        meta={"smoke": True},
    )
    # reload into fresh optimizer after stage switch
    trainability_p = apply_trainability(encoder, "partial")
    set_encoder_modes(encoder, "partial")
    for p in clf.parameters():
        p.requires_grad = True
    opt2 = build_optimizer("partial", encoder, clf)
    state2 = load_finetune_checkpoint(ckpt_path, encoder, clf, optimizer=None)
    report["resume_stage_after_load"] = state2.stage
    report["stage2_trainable_encoder"] = trainability_p["trainable"]
    report["stage2_trainable_encoder_n_params"] = int(
        sum(p.numel() for n, p in encoder.named_parameters() if n in set(trainability_p["trainable"]))
    )
    report["stage2_optimizer_groups"] = optimizer_group_report(opt2)
    rev = [n for n in trainability_p["trainable"] if "rev_to" in n]
    report["stage2_reverse_trainable"] = rev
    if not rev:
        failures.append("no reverse-relation params trainable in stage 2")
    if any(n.startswith("embedding_head.") for n in trainability_p["trainable"]):
        failures.append("embedding_head unexpectedly trainable")

    # --- Stage 2: one partial update with param delta check ---
    before_partial = snapshot_params(encoder, trainability_p["trainable"])
    before_frozen0 = snapshot_params(
        encoder, [n for n in trainability_p["frozen"] if n.startswith("convs.0.")]
    )
    transform = AddEgoIds() if graph_args.ego else None
    tr_loader, _, _ = get_loaders(
        tr_data,
        val_data,
        te_data,
        tr_inds,
        val_inds,
        te_inds,
        transform,
        graph_args,
        train_shuffle=True,
    )
    capture = GradPreEmbeddingCapture(encoder, pre_dim=pre_dim, emb_dim=128, head_spec=head_spec)
    t_batch0 = time.perf_counter()
    partial_stats = run_partial_epoch_online(
        encoder,
        clf,
        opt2,
        tr_loader,
        tr_inds,
        capture,
        feats["x_raw"],
        feats["tf_feat"],
        feats["scaler"],
        device,
        max_batches=args_ns.partial_steps,
    )
    batch_wall = time.perf_counter() - t_batch0
    report["stage2_partial_stats"] = partial_stats
    report["stage2_batch_runtime_s"] = batch_wall
    report["stage2_sec_per_batch"] = batch_wall / max(args_ns.partial_steps, 1)
    after_partial = snapshot_params(encoder, trainability_p["trainable"])
    after_frozen0 = snapshot_params(
        encoder, [n for n in trainability_p["frozen"] if n.startswith("convs.0.")]
    )
    d_train = param_deltas(before_partial, after_partial)
    d_frozen = param_deltas(before_frozen0, after_frozen0)
    report["stage2_trainable_deltas_n"] = len(d_train)
    report["stage2_trainable_delta_examples"] = dict(list(d_train.items())[:8])
    report["stage2_frozen_convs0_deltas"] = d_frozen
    if not d_train:
        failures.append("no trainable encoder params changed after partial step")
    if d_frozen:
        failures.append(f"frozen convs.0 changed: {list(d_frozen)[:3]}")
    if not np.isfinite(partial_stats["loss"]):
        failures.append("non-finite partial loss")

    # validation pass (selection metric only)
    # For smoke: use cached val features with updated clf (encoder change not fully reflected
    # in cached H). Still exercises val AUPRC path without test.
    val_metrics = eval_clf(clf, feats["x_va"], feats["y_va"], device=device, split_name="val")
    report["smoke_val_metrics"] = val_metrics
    state.early.step(1, val_metrics["auprc"], val_metrics["f1_at_selected"])
    report["early_stop_state"] = {
        "best_auprc": state.early.best_auprc,
        "best_f1": state.early.best_f1,
        "uses_test": False,
    }

    # save after stage2 step
    state.stage = "partial"
    state.partial_epochs_done = 1
    state.global_epoch = args_ns.warmup_epochs + 1
    save_finetune_checkpoint(
        ckpt_dir / "checkpoint_after_partial.tar",
        encoder=encoder,
        clf=clf,
        optimizer=opt2,
        state=state,
        scaler=feats["scaler"],
        meta={"smoke": True, "partial_steps": args_ns.partial_steps},
    )
    # reload check
    _ = load_finetune_checkpoint(ckpt_dir / "checkpoint_after_partial.tar", encoder, clf, optimizer=None)
    report["checkpoint_save_reload_ok"] = True

    # runtime projection: ~397 train batches/epoch * remaining epochs after warmup
    sec_per = report["stage2_sec_per_batch"]
    n_train_batches = int(np.ceil(int(tr_inds.numel()) / int(graph_args.batch_size)))
    warmup_full_s = 15 * 60.0  # conservative wall for full 15-ep MLP on cached H
    # remaining partial epochs after default warmup 5 → 15 epochs online
    remaining = MAX_TOTAL_EPOCHS - WARMUP_EPOCHS_DEFAULT
    partial_s = remaining * n_train_batches * sec_per
    # val pass each epoch ~ val batches * sec_per (forward only ≈ same order)
    n_val_batches = int(np.ceil(int(val_inds.numel()) / int(graph_args.batch_size)))
    val_s = remaining * n_val_batches * sec_per * 0.7
    data_s = report["data_load_s"]
    projected_h = (warmup_full_s + partial_s + val_s + data_s) / 3600.0
    report["n_train_batches"] = n_train_batches
    report["n_val_batches"] = n_val_batches
    report["projected_full_runtime_h"] = projected_h
    report["six_hour_safe"] = bool(projected_h < 5.0)
    report["peak_gpu_mem_gb"] = _gpu_mem_gb()
    report["resume_strategy"] = (
        "single_6h_job"
        if report["six_hour_safe"]
        else "two_jobs_with_checkpoint_resume (warmup then partial)"
    )

    report["full_job_command"] = (
        "python scripts/run_dplus_partial_finetune.py "
        "--seed 2 --warmup_epochs 5 --max_epochs 20 "
        "--early_stop_patience 5 --loader_num_workers 0 "
        "--unique_name dplus_partial_finetune_hxxtf_seed2 "
        f"--init_checkpoint {DPLUS_CKPT} "
        "--output_dir saved-models/dplus_partial_finetune_hxxtf_seed2"
    )
    # Do not submit; command only.

    report["pass"] = len(failures) == 0
    report["failures"] = failures
    report["wall_s"] = time.perf_counter() - t0
    _write(report, failures, args_ns, t0)
    if failures:
        raise SystemExit(f"SMOKE FAIL: {failures}")
    logging.info("SMOKE PASS")


def _write(report, failures, args_ns, t0):
    report["failures"] = failures
    report["pass"] = len(failures) == 0
    report.setdefault("wall_s", time.perf_counter() - t0)
    out_json = Path(args_ns.output_json)
    out_md = Path(args_ns.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    status = "PASS" if report["pass"] else "FAIL"
    lines = [
        "# D+ partial fine-tune smoke",
        "",
        f"**{status}**",
        "",
        "Bounded GPU smoke for the locked partial fine-tune protocol "
        "([`final_dplus_experiment_preflight.md`](final_dplus_experiment_preflight.md)). "
        "Full fine-tune job was **not** submitted.",
        "",
        "## Locked inputs",
        "",
        f"- Device: `{report.get('device')}`",
        f"- Source ckpt: `{report.get('source_checkpoint')}`",
        f"- Source sha256: `{report.get('source_sha256')}`",
        f"- Loaded epoch: {report.get('loaded_ckpt_epoch')}",
        f"- MLP init path: **{report.get('mlp_init_path')}** "
        f"(18678029 weights found: {report.get('winning_mlp_weights_found')})",
        f"- Note: {report.get('winning_mlp_note')}",
        f"- Stack dim: {report.get('stack_dim')} (H+X+TF)",
        f"- TF leakage audit ok: {report.get('tf_leakage_audit', {}).get('ok')}",
        f"- Pre-3h equivalence ok: {report.get('equivalence_pre3h', {}).get('ok')} "
        f"(matched={report.get('equivalence_pre3h', {}).get('matched_ids')}, "
        f"max_abs={report.get('equivalence_pre3h', {}).get('max_abs_diff')})",
        "",
        "## Stage 1 (frozen encoder / classifier warmup)",
        "",
        f"- Trainable encoder params: {report.get('stage1_trainable_encoder')}",
        f"- Frozen encoder tensors: {report.get('stage1_frozen_encoder_n')}",
        f"- Optimizer groups: `{report.get('stage1_optimizer_groups')}`",
        f"- Warmup losses: {report.get('stage1_losses')}",
        f"- Encoder param deltas during warmup (must be empty): "
        f"{report.get('stage1_encoder_param_deltas')}",
        f"- Val after warmup: `{report.get('stage1_val_after_warmup')}`",
        f"- Test blocked during selection: {report.get('test_blocked_during_selection')}",
        "",
        "## Stage 2 (partial unfreeze)",
        "",
        f"- Trainable encoder n_params: {report.get('stage2_trainable_encoder_n_params')}",
        f"- Reverse trainable count: {len(report.get('stage2_reverse_trainable') or [])}",
        f"- Optimizer groups: `{report.get('stage2_optimizer_groups')}`",
        f"- Partial stats (loss / grad_norm / steps): `{report.get('stage2_partial_stats')}`",
        f"- Trainable deltas: {report.get('stage2_trainable_deltas_n')} "
        f"(examples: `{report.get('stage2_trainable_delta_examples')}`)",
        f"- Frozen convs.0 deltas (must be empty): `{report.get('stage2_frozen_convs0_deltas')}`",
        f"- Smoke val metrics: `{report.get('smoke_val_metrics')}`",
        f"- Checkpoint save/reload ok: {report.get('checkpoint_save_reload_ok')}",
        "",
        "## Resources / projection",
        "",
        f"- Peak GPU GiB: {report.get('peak_gpu_mem_gb')}",
        f"- Sec/batch (partial): {report.get('stage2_sec_per_batch')}",
        f"- Train/val batches: {report.get('n_train_batches')} / {report.get('n_val_batches')}",
        f"- Projected full runtime h: {report.get('projected_full_runtime_h')}",
        f"- 6h safe: {report.get('six_hour_safe')} → {report.get('resume_strategy')}",
        f"- Wall s: {report.get('wall_s')}",
        "",
        "## Full-job command (NOT submitted)",
        "",
        "```bash",
        str(report.get("full_job_command")),
        "```",
        "",
    ]
    if failures:
        lines += ["## Failures", ""] + [f"- {f}" for f in failures] + [""]
    out_md.write_text("\n".join(lines) + "\n")
    logging.info("Wrote %s and %s", out_json, out_md)


if __name__ == "__main__":
    main()
