#!/usr/bin/env python3
"""Full D+ partial fine-tune runner (do not auto-submit; invoke explicitly)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loading import get_data
from dplus_partial_finetune import (
    CLF_LR,
    DPLUS_CKPT,
    EarlyStopState,
    ENC_LR,
    FinetuneState,
    GradPreEmbeddingCapture,
    MAX_TOTAL_EPOCHS,
    STACK_DIM,
    WARMUP_EPOCHS_DEFAULT,
    apply_trainability,
    assert_dplus_checkpoint,
    build_graph_args,
    build_optimizer,
    equivalence_check_pre3h,
    eval_clf,
    load_dplus_hetero_encoder,
    load_finetune_checkpoint,
    load_hxxtf_matrices,
    optimizer_group_report,
    run_partial_epoch_online,
    save_finetune_checkpoint,
    set_encoder_modes,
    tf_leakage_audit,
    train_mlp_epoch,
)
from gcpal_txn_node.eval_mlp import PaperStyleMLP
from train_util import AddEgoIds, get_loaders
from util import logger_setup, set_seed


def main() -> None:
    logger_setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS_DEFAULT)
    p.add_argument("--max_epochs", type=int, default=MAX_TOTAL_EPOCHS)
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--loader_num_workers", type=int, default=0)
    p.add_argument("--unique_name", default="dplus_partial_finetune_hxxtf_seed2")
    p.add_argument("--init_checkpoint", default=str(DPLUS_CKPT))
    p.add_argument("--output_dir", default="saved-models/dplus_partial_finetune_hxxtf_seed2")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_equivalence", action="store_true")
    p.add_argument("--eval_test_after_lock", action="store_true",
                   help="Only after training ends; never used for selection.")
    args = p.parse_args()
    if args.max_epochs > MAX_TOTAL_EPOCHS:
        raise SystemExit(f"--max_epochs capped at {MAX_TOTAL_EPOCHS}")
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_last = out_dir / "checkpoint_last.tar"
    ckpt_best = out_dir / "checkpoint_best_val_auprc.tar"

    assert_dplus_checkpoint(Path(args.init_checkpoint))
    leak = tf_leakage_audit()
    if not leak["ok"]:
        raise SystemExit(f"TF leakage audit failed: {leak}")

    feats = load_hxxtf_matrices(str(ROOT / "data_config.json"))
    with open(ROOT / "data_config.json", encoding="utf-8") as f:
        data_config = json.load(f)
    graph_args = build_graph_args(seed=args.seed, loader_num_workers=args.loader_num_workers)
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(graph_args, data_config)
    encoder, head_spec, pre_dim, _, _ = load_dplus_hetero_encoder(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, graph_args, data_config, device
    )
    if not args.skip_equivalence:
        eq = equivalence_check_pre3h(
            encoder, head_spec, pre_dim, tr_data, tr_inds, graph_args, device,
            feats["splits"]["train"]["Z"], feats["ids_tr"], max_batches=2,
        )
        logging.info("equivalence %s", eq)
        if not eq["ok"]:
            raise SystemExit(f"pre-3h equivalence failed: {eq}")

    clf = PaperStyleMLP(STACK_DIM, hidden=128, dropout=0.1).to(device)
    state = FinetuneState()
    state.early = EarlyStopState(
        patience=args.early_stop_patience,
        patience_left=args.early_stop_patience,
    )

    if args.resume and ckpt_last.is_file():
        state = load_finetune_checkpoint(ckpt_last, encoder, clf, optimizer=None)
        logging.info("Resumed stage=%s global_epoch=%s", state.stage, state.global_epoch)

    transform = AddEgoIds() if graph_args.ego else None
    tr_loader, val_loader, _ = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, graph_args, train_shuffle=True
    )
    capture = GradPreEmbeddingCapture(encoder, pre_dim=pre_dim, emb_dim=128, head_spec=head_spec)

    # ---- Stage 1 ----
    while state.warmup_epochs_done < args.warmup_epochs and state.global_epoch < args.max_epochs:
        apply_trainability(encoder, "warmup")
        set_encoder_modes(encoder, "warmup")
        for p in clf.parameters():
            p.requires_grad = True
        opt = build_optimizer("warmup", encoder, clf)
        loss = train_mlp_epoch(
            clf, feats["x_tr"], feats["y_tr"], optimizer=opt, device=device,
            seed=args.seed, epoch=state.warmup_epochs_done,
        )
        val = eval_clf(clf, feats["x_va"], feats["y_va"], device=device, split_name="val")
        state.warmup_epochs_done += 1
        state.global_epoch += 1
        state.stage = "warmup"
        improved = state.early.step(state.global_epoch, val["auprc"], val["f1_at_selected"])
        row = {"stage": "warmup", "epoch": state.global_epoch, "loss": loss, "val": val}
        state.history.append(row)
        logging.info("warmup ep=%s loss=%.4f valA=%.4f", state.global_epoch, loss, val["auprc"])
        save_finetune_checkpoint(ckpt_last, encoder=encoder, clf=clf, optimizer=opt, state=state,
                                 scaler=feats["scaler"], meta={"unique_name": args.unique_name})
        if improved:
            save_finetune_checkpoint(ckpt_best, encoder=encoder, clf=clf, optimizer=opt, state=state,
                                     scaler=feats["scaler"], meta={"unique_name": args.unique_name, "best": True})

    # ---- Stage 2 ----
    # Warmup must finish before unfreeze. When first entering Stage 2, reset
    # patience so Stage-1 non-improvements cannot terminate Stage 2 early;
    # keep global best_* across both stages. Skip reset on mid-partial resume.
    if state.partial_epochs_done == 0:
        state.early.patience_left = int(state.early.patience)
        state.early.stopped = False
        logging.info(
            "stage transition warmup→partial: reset patience=%s keep best_val_auprc=%.6f @ ep %s",
            state.early.patience_left,
            state.early.best_auprc,
            state.early.best_epoch,
        )
    apply_trainability(encoder, "partial")
    set_encoder_modes(encoder, "partial")
    for p in clf.parameters():
        p.requires_grad = True
    opt = build_optimizer("partial", encoder, clf)
    logging.info("partial optimizer groups %s", optimizer_group_report(opt))
    state.stage = "partial"
    if state.partial_epochs_done == 0:
        save_finetune_checkpoint(
            ckpt_last, encoder=encoder, clf=clf, optimizer=opt, state=state,
            scaler=feats["scaler"],
            meta={"unique_name": args.unique_name, "stage_transition": "warmup_to_partial"},
        )

    while state.global_epoch < args.max_epochs and not state.early.stopped:
        stats = run_partial_epoch_online(
            encoder, clf, opt, tr_loader, tr_inds, capture,
            feats["x_raw"], feats["tf_feat"], feats["scaler"], device,
        )
        # Val: re-extract is expensive; for selection use online val loader forward
        # Collect val predictions
        encoder.eval()
        # keep final block BN in eval for stable val? Spec: batch_norms.1 may train in train;
        # for val use eval on whole model except we already set modes — force eval for metrics
        set_encoder_modes(encoder, "warmup")  # all eval for val pass
        clf.eval()
        # Use cached H for val ranking during smoke-scale; for full fidelity rebuild via online.
        # Protocol: val AUPRC on current clf with features from current encoder.
        from dplus_partial_finetune import forward_pre3h_from_loader_batch, pack_online_features
        probs = []
        ys = []
        with torch.no_grad():
            for batch in val_loader:
                h, yb, eids = forward_pre3h_from_loader_batch(
                    encoder, batch, val_inds, val_loader.data, capture, device
                )
                feats_b = pack_online_features(h, eids, feats["x_raw"], feats["tf_feat"], feats["scaler"], device)
                logits = clf(feats_b)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                ys.append(yb.cpu().numpy())
        proba = np.concatenate(probs)
        yv = np.concatenate(ys).astype(np.int64)
        from dplus_partial_finetune import ranking_metrics, f1_at_threshold
        from gcpal_txn_node.eval_mlp import _select_threshold_f1
        thr = _select_threshold_f1(yv, proba)
        val = ranking_metrics(yv, proba)
        val["f1_at_selected"] = f1_at_threshold(yv, proba, thr)
        val["threshold"] = float(thr)

        state.partial_epochs_done += 1
        state.global_epoch += 1
        improved = state.early.step(state.global_epoch, val["auprc"], val["f1_at_selected"])
        state.history.append({"stage": "partial", "epoch": state.global_epoch, "train": stats, "val": val})
        logging.info(
            "partial ep=%s loss=%.4f valA=%.4f bestA=%.4f patience=%s",
            state.global_epoch, stats["loss"], val["auprc"], state.early.best_auprc, state.early.patience_left,
        )
        save_finetune_checkpoint(ckpt_last, encoder=encoder, clf=clf, optimizer=opt, state=state,
                                 scaler=feats["scaler"], meta={"unique_name": args.unique_name})
        if improved:
            save_finetune_checkpoint(ckpt_best, encoder=encoder, clf=clf, optimizer=opt, state=state,
                                     scaler=feats["scaler"], meta={"unique_name": args.unique_name, "best": True})
        # restore train modes for next epoch
        apply_trainability(encoder, "partial")
        set_encoder_modes(encoder, "partial")

    summary = {
        "unique_name": args.unique_name,
        "best_epoch": state.early.best_epoch,
        "best_val_auprc": state.early.best_auprc,
        "best_val_f1": state.early.best_f1,
        "stopped": state.early.stopped,
        "history": state.history,
        "mlp_init_path": feats["mlp_init_path"],
        "clf_lr": CLF_LR,
        "enc_lr": ENC_LR,
        "test_used_for_selection": False,
    }
    if args.eval_test_after_lock:
        # Load best, report test once — never for selection
        load_finetune_checkpoint(ckpt_best if ckpt_best.is_file() else ckpt_last, encoder, clf)
        test = eval_clf(clf, feats["x_te"], feats["y_te"], device=device, split_name="test", allow_test=True)
        summary["test_after_lock_cached_H_warning"] = (
            "Cached-H test only if encoder not re-extracted; prefer online extract for final report."
        )
        summary["test_cached_stack"] = test
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logging.info("Done. best val AUPRC=%.4f @ ep %s", state.early.best_auprc, state.early.best_epoch)


if __name__ == "__main__":
    main()
