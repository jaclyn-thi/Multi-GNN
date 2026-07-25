#!/usr/bin/env python3
"""Resume positive-complete txn-node scouts from epoch 5 → 20 with checkpoints.

The original 5-epoch scouts did not save model/optimizer checkpoints. This script:

1. Verifies KNN/coverage diagnostics from the completed 5ep JSON (preflight).
2. Deterministically replays epochs 1–5 with the same seeds/sampler, verifies
   per-epoch mean loss against the saved history, and writes epoch-5 checkpoint.
3. Continues training through epoch 20, saving checkpoints at 10, 15, and 20.
4. Evaluates checkpoints 5/10/15/20 with the deterministic downstream protocol.
5. Writes NEW 20ep artifacts; never overwrites existing *_5ep_seed2 files.

Not an exact GCPAL reproduction.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from gcpal_txn_node.adjacency import (
    adjacency_list_from_edge_index,
    build_directed_flow_adjacency,
    induce_edge_index,
)
from gcpal_txn_node.checkpointing import (
    load_training_checkpoint,
    restore_rng_from_checkpoint,
    save_training_checkpoint,
)
from gcpal_txn_node.data import load_small_hi_frame
from gcpal_txn_node.eval_mlp import train_eval_mlp_suite
from gcpal_txn_node.features import fit_feature_preprocessor
from gcpal_txn_node.knn_adapter import load_train_knn_cache
from gcpal_txn_node.model import SharedTxnNodeEncoder
from gcpal_txn_node.spec import DEFAULT_KNN_CACHE, LAMBDA_MIX, NOT_EXACT_REPRODUCTION, TEMPERATURE
from gcpal_txn_node.train_step import StepConfig, run_positive_complete_step

EVAL_EPOCHS = (5, 10, 15, 20)
LOSS_ATOL = 2e-2
LOSS_RTOL = 1e-2
# Predetermined validation selection metric (never uses test).
VAL_SELECT_PROTOCOL = "temporal"
VAL_SELECT_REP = "HxX"
VAL_SELECT_KEY = "auprc"


def _configure_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


@torch.no_grad()
def encode_nodes_induced(
    encoder: SharedTxnNodeEncoder,
    x_all: np.ndarray,
    flow_ei: np.ndarray,
    node_ids: np.ndarray,
    device: torch.device,
    chunk: int = 4096,
) -> np.ndarray:
    encoder.eval()
    out = np.zeros((node_ids.shape[0], encoder.gin.out_dim), dtype=np.float32)
    for start in range(0, node_ids.shape[0], chunk):
        ids = node_ids[start : start + chunk]
        x = torch.from_numpy(x_all[ids]).to(device)
        ei = torch.from_numpy(induce_edge_index(flow_ei, ids)).to(device)
        h, _ = encoder(x, ei)
        out[start : start + chunk] = h.detach().cpu().numpy()
    return out


def verify_five_ep_diagnostics(ref: dict) -> Dict[str, Any]:
    """Recover coverage stats from completed 5ep JSON; raise if incomplete/bad."""
    bd = (ref.get("batch_diagnostics_head") or [None])[0]
    if not bd:
        raise SystemExit("5ep JSON missing batch_diagnostics_head; refusing resume")
    g = bd["growth_stats"]
    pos = bd["positive_stats"]
    mode = ref["mode"]
    report = {
        "mode": mode,
        "frac_anchors_knn_ge1": float(g["frac_anchors_knn_ge1"]),
        "frac_anchors_knn_all_available": float(g["frac_anchors_knn_all_available"]),
        "frac_anchors_struct_ge1": float(g["frac_anchors_struct_ge1"]),
        "mean_knn_pos_growth": float(g["mean_knn_pos"]),
        "mean_structural_pos_growth": float(g["mean_structural_pos"]),
        "mean_identity_pos_growth": float(g.get("mean_identity_pos", 1.0)),
        "loss_mask_mean_knn_pos": float(pos["mean_knn_pos"]),
        "loss_mask_mean_structural_pos": float(pos["mean_structural_pos"]),
        "loss_mask_mean_total_pos": float(pos["mean_total_pos"]),
        "rejected_not_allowed": float(g.get("rejected_not_allowed", 0.0)),
        "realized_n_anchors": int(bd["realized_n_anchors"]),
        "n_nodes": int(bd["n_nodes"]),
    }
    if report["frac_anchors_knn_ge1"] < 0.95:
        raise SystemExit(f"preflight fail: knn_ge1={report['frac_anchors_knn_ge1']}")
    if report["frac_anchors_knn_all_available"] < 0.90:
        raise SystemExit(f"preflight fail: knn_all={report['frac_anchors_knn_all_available']}")
    if report["rejected_not_allowed"] != 0.0:
        raise SystemExit("preflight fail: split leakage indicated in growth stats")
    if mode == "B_gcpal":
        if report["loss_mask_mean_knn_pos"] < 14.0:
            raise SystemExit("preflight fail: B expected ~15 KNN positives in loss mask")
    if mode == "A_identity":
        if report["loss_mask_mean_knn_pos"] != 0.0:
            raise SystemExit("preflight fail: A should have identity-only loss mask")
    return report


def losses_match(replayed: float, reference: float) -> bool:
    return abs(replayed - reference) <= max(LOSS_ATOL, LOSS_RTOL * abs(reference))


def run_epochs(
    *,
    encoder: SharedTxnNodeEncoder,
    opt: torch.optim.Optimizer,
    rng: np.random.RandomState,
    start_epoch: int,
    end_epoch: int,
    steps_per_epoch: int,
    n: int,
    x_train: np.ndarray,
    flow_ei: np.ndarray,
    flow_adj,
    knn_adj,
    knn,
    cfg: StepConfig,
    pos_flow,
    pos_knn,
    allowed,
    labels,
    device: torch.device,
    seed: int,
    history: List[dict],
    total_opt_steps: int,
    total_anchor_exposures: int,
    batch_diags: List[dict],
    ckpt_dir: Path,
    meta: Dict[str, Any],
    reference_history: Optional[List[dict]] = None,
    verify_replay: bool = False,
) -> Tuple[List[dict], int, int, np.random.RandomState]:
    for epoch in range(start_epoch, end_epoch + 1):
        perm = rng.permutation(n).astype(np.int64)
        cursor = 0
        losses: List[float] = []
        for step_i in range(steps_per_epoch):
            if cursor >= n:
                perm = rng.permutation(n).astype(np.int64)
                cursor = 0
            out = run_positive_complete_step(
                encoder=encoder,
                x_all=x_train,
                flow_edge_index=flow_ei,
                flow_adj=flow_adj,
                knn_adj=knn_adj,
                knn_graph_edge_fn=knn.edge_index_for_nodes,
                candidate_stream=perm,
                stream_start=cursor,
                device=device,
                cfg=cfg,
                seed=seed * 10007 + (epoch - 1) * 97 + step_i,
                pos_flow_adj=pos_flow,
                pos_knn_adj=pos_knn,
                allowed_ids=allowed,
                labels=labels,
            )
            loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_opt_steps += 1
            total_anchor_exposures += int(out["n_anchors"])
            losses.append(float(loss.detach().cpu()))
            if len(batch_diags) < 8:
                batch_diags.append(
                    {
                        "epoch": epoch,
                        "realized_n_anchors": int(out["n_anchors"]),
                        "n_nodes": int(out["n_sampled_nodes"]),
                        "growth_stats": out["growth_stats"],
                        "positive_stats": out["positive_stats"],
                        "step_time_seconds": out["step_time_seconds"],
                        "peak_cuda_mib": out["peak_cuda_mib"],
                    }
                )
            cursor = int(out["stream_next"])
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss_mean": mean_loss, "n_steps": len(losses)})
        logging.info("epoch %s loss=%.6f steps=%d", epoch, mean_loss, len(losses))

        if verify_replay and reference_history is not None:
            ref = next(h for h in reference_history if int(h["epoch"]) == epoch)
            if not losses_match(mean_loss, float(ref["loss_mean"])):
                raise SystemExit(
                    f"Replay loss mismatch at epoch {epoch}: "
                    f"replayed={mean_loss:.8f} reference={float(ref['loss_mean']):.8f} "
                    f"(atol={LOSS_ATOL}, rtol={LOSS_RTOL}). Refusing to continue."
                )
            logging.info("Replay loss verified for epoch %s", epoch)

        if epoch in EVAL_EPOCHS:
            ckpt_path = ckpt_dir / f"epoch_{epoch:02d}.pt"
            save_training_checkpoint(
                ckpt_path,
                epoch=epoch,
                encoder=encoder,
                optimizer=opt,
                rng=rng,
                history=history,
                total_opt_steps=total_opt_steps,
                total_anchor_exposures=total_anchor_exposures,
                meta=meta,
            )
            logging.info("Wrote checkpoint %s", ckpt_path)

    return history, total_opt_steps, total_anchor_exposures, rng


def evaluate_checkpoint(
    *,
    ckpt_path: Path,
    encoder: SharedTxnNodeEncoder,
    opt: torch.optim.Optimizer,
    x_train: np.ndarray,
    x_full: np.ndarray,
    flow_ei_train: np.ndarray,
    flow_full: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    y_all: np.ndarray,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    load_training_checkpoint(ckpt_path, encoder=encoder, optimizer=opt, map_location=str(device))
    encoder.eval()
    # Deterministic full encode for all splits (comparable across epochs).
    h_full = np.zeros((x_full.shape[0], encoder.gin.out_dim), dtype=np.float32)
    h_full[tr] = encode_nodes_induced(encoder, x_full, flow_full, tr.astype(np.int64), device)
    h_full[va] = encode_nodes_induced(encoder, x_full, flow_full, va.astype(np.int64), device)
    h_full[te] = encode_nodes_induced(encoder, x_full, flow_full, te.astype(np.int64), device)

    temporal = train_eval_mlp_suite(
        h_full[tr],
        x_full[tr],
        y_all[tr],
        h_full[te],
        x_full[te],
        y_all[te],
        h_val=h_full[va],
        x_val=x_full[va],
        y_val=y_all[va],
        seed=seed,
        device=device,
    )
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.4, random_state=seed)
    tr_r, te_r = next(sss.split(np.arange(len(y_all)), y_all))
    sss_inner = StratifiedShuffleSplit(n_splits=1, train_size=0.75, random_state=seed + 1)
    tr_r2, va_r = next(sss_inner.split(tr_r, y_all[tr_r]))
    tr_idx = tr_r[tr_r2]
    va_idx = tr_r[va_r]
    random40 = train_eval_mlp_suite(
        h_full[tr_idx],
        x_full[tr_idx],
        y_all[tr_idx],
        h_full[te_r],
        x_full[te_r],
        y_all[te_r],
        h_val=h_full[va_idx],
        x_val=x_full[va_idx],
        y_val=y_all[va_idx],
        seed=seed,
        device=device,
    )
    return {
        "checkpoint": str(ckpt_path),
        "temporal_primary": temporal,
        "random40_diagnostic": random40,
        "encode_note": "All splits encoded via induced full-graph chunks from the checkpoint weights.",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["A_identity", "B_gcpal"], required=True)
    p.add_argument("--data_config", default="data_config.json")
    p.add_argument("--knn_cache", default=DEFAULT_KNN_CACHE)
    p.add_argument("--max_total_nodes", type=int, default=2048)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--reference_5ep_json",
        default=None,
        help="Completed 5ep scout JSON (default: results/diagnostics/..._5ep_seed2.json)",
    )
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_md", required=True)
    p.add_argument(
        "--existing_epoch5_ckpt",
        default=None,
        help="If set, load this epoch-5 checkpoint instead of replaying 1–5.",
    )
    args = p.parse_args()
    assert NOT_EXACT_REPRODUCTION
    if Path(args.output_json).exists():
        raise SystemExit(f"Refusing overwrite existing {args.output_json}")

    ref_path = Path(
        args.reference_5ep_json
        or f"results/diagnostics/gcpal_txn_node_poscomplete_scout_{args.mode}_5ep_seed2.json"
    )
    if not ref_path.is_file():
        raise SystemExit(f"Missing reference 5ep JSON: {ref_path}")
    # Never overwrite original 5ep artifacts.
    if "5ep" in Path(args.output_json).name and Path(args.output_json).resolve() == ref_path.resolve():
        raise SystemExit("Output path collides with reference 5ep artifact")

    ref = json.loads(ref_path.read_text())
    preflight = verify_five_ep_diagnostics(ref)
    logging.info("Preflight diagnostics OK: %s", json.dumps(preflight))

    ckpt_dir = Path(
        args.ckpt_dir
        or f"checkpoints/gcpal_txn_node_poscomplete_{args.mode}_20ep_seed{args.seed}"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _configure_determinism(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df, tr, va, te, meta = load_small_hi_frame(args.data_config)
    df_train = df.iloc[tr].reset_index(drop=True).copy()
    n = len(df_train)
    allowed = set(range(n))
    labels = df_train[meta["label_col"]].to_numpy().astype(np.int64)
    prep = fit_feature_preprocessor(df_train, amount_col=meta["amount_col"])
    x_train = prep.transform(df_train)

    flow_ei, flow_stats = build_directed_flow_adjacency(
        df_train["from_id"].to_numpy(),
        df_train["to_id"].to_numpy(),
        df_train["Timestamp"].astype(float).to_numpy(),
        policy="immediate_next",
    )
    flow_adj = adjacency_list_from_edge_index(flow_ei, n)
    knn = load_train_knn_cache(args.knn_cache, expected_k=15)
    if knn.node_ids.shape[0] != n:
        raise SystemExit("KNN/train size mismatch")
    knn_adj = knn.adjacency_lists()
    empty_adj = [np.zeros(0, dtype=np.int64) for _ in range(n)]

    if args.mode == "A_identity":
        cfg = StepConfig(
            edge_drop=0.1,
            feature_drop=0.1,
            lambda_mix=1.0,
            temperature=TEMPERATURE,
            include_identity=True,
            use_knn_contrast=False,
            max_total_nodes=int(args.max_total_nodes),
            batching_mode="positive_complete",
            knn_k=15,
        )
        pos_flow, pos_knn = empty_adj, empty_adj
    else:
        cfg = StepConfig(
            edge_drop=0.1,
            feature_drop=0.1,
            lambda_mix=LAMBDA_MIX,
            temperature=TEMPERATURE,
            include_identity=True,
            use_knn_contrast=True,
            max_total_nodes=int(args.max_total_nodes),
            batching_mode="positive_complete",
            knn_k=15,
        )
        pos_flow, pos_knn = flow_adj, knn_adj

    encoder = SharedTxnNodeEncoder(in_dim=x_train.shape[1], emb_dim=128).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    steps_per_epoch = int(np.ceil(n / 2048.0))
    assert steps_per_epoch == int(ref.get("steps_per_epoch", steps_per_epoch))

    ckpt_meta = {
        "mode": args.mode,
        "seed": int(args.seed),
        "max_total_nodes": int(args.max_total_nodes),
        "steps_per_epoch": steps_per_epoch,
        "lambda_mix": float(cfg.lambda_mix),
        "temperature": float(cfg.temperature),
        "use_knn_contrast": bool(cfg.use_knn_contrast),
        "reference_5ep_json": str(ref_path),
        "not_exact_reproduction": True,
    }

    history: List[dict] = []
    batch_diags: List[dict] = []
    total_opt_steps = 0
    total_anchor_exposures = 0
    t0 = time.perf_counter()
    resume_note: Dict[str, Any]

    existing = Path(args.existing_epoch5_ckpt) if args.existing_epoch5_ckpt else None
    ep5_path = ckpt_dir / "epoch_05.pt"

    if existing is not None and existing.is_file():
        logging.info("Loading existing epoch-5 checkpoint %s", existing)
        ckpt = load_training_checkpoint(existing, encoder=encoder, optimizer=opt, map_location=str(device))
        if int(ckpt["epoch"]) != 5:
            raise SystemExit(f"Expected epoch=5 checkpoint, got {ckpt['epoch']}")
        rng = restore_rng_from_checkpoint(ckpt)
        history = list(ckpt["history"])
        total_opt_steps = int(ckpt["total_opt_steps"])
        total_anchor_exposures = int(ckpt["total_anchor_exposures"])
        if not ep5_path.exists():
            save_training_checkpoint(
                ep5_path,
                epoch=5,
                encoder=encoder,
                optimizer=opt,
                rng=rng,
                history=history,
                total_opt_steps=total_opt_steps,
                total_anchor_exposures=total_anchor_exposures,
                meta=ckpt_meta,
            )
        resume_note = {
            "method": "loaded_existing_epoch5_checkpoint",
            "path": str(existing),
            "replay_verified": False,
        }
        start_continue = 6
    elif ep5_path.is_file():
        logging.info("Reusing previously written epoch-5 checkpoint %s", ep5_path)
        ckpt = load_training_checkpoint(ep5_path, encoder=encoder, optimizer=opt, map_location=str(device))
        rng = restore_rng_from_checkpoint(ckpt)
        history = list(ckpt["history"])
        total_opt_steps = int(ckpt["total_opt_steps"])
        total_anchor_exposures = int(ckpt["total_anchor_exposures"])
        resume_note = {
            "method": "loaded_ckpt_dir_epoch5",
            "path": str(ep5_path),
            "replay_verified": bool((ckpt.get("meta") or {}).get("replay_verified", False)),
        }
        start_continue = 6
    else:
        logging.info(
            "No epoch-5 checkpoint found; deterministically replaying epochs 1–5 "
            "and verifying loss against %s",
            ref_path,
        )
        resume_note = {
            "method": "deterministic_replay_epochs_1_to_5",
            "reason": "original_5ep_scouts_did_not_save_checkpoints",
            "reference_json": str(ref_path),
            "loss_atol": LOSS_ATOL,
            "loss_rtol": LOSS_RTOL,
        }
        rng = np.random.RandomState(args.seed)
        ckpt_meta_replay = dict(ckpt_meta)
        ckpt_meta_replay["replay_verified"] = True
        history, total_opt_steps, total_anchor_exposures, rng = run_epochs(
            encoder=encoder,
            opt=opt,
            rng=rng,
            start_epoch=1,
            end_epoch=5,
            steps_per_epoch=steps_per_epoch,
            n=n,
            x_train=x_train,
            flow_ei=flow_ei,
            flow_adj=flow_adj,
            knn_adj=knn_adj,
            knn=knn,
            cfg=cfg,
            pos_flow=pos_flow,
            pos_knn=pos_knn,
            allowed=allowed,
            labels=labels,
            device=device,
            seed=int(args.seed),
            history=history,
            total_opt_steps=total_opt_steps,
            total_anchor_exposures=total_anchor_exposures,
            batch_diags=batch_diags,
            ckpt_dir=ckpt_dir,
            meta=ckpt_meta_replay,
            reference_history=ref["history"],
            verify_replay=True,
        )
        resume_note["replay_verified"] = True
        resume_note["replayed_history"] = history[:5]
        start_continue = 6

    # Continue 6 → 20
    history, total_opt_steps, total_anchor_exposures, rng = run_epochs(
        encoder=encoder,
        opt=opt,
        rng=rng,
        start_epoch=start_continue,
        end_epoch=20,
        steps_per_epoch=steps_per_epoch,
        n=n,
        x_train=x_train,
        flow_ei=flow_ei,
        flow_adj=flow_adj,
        knn_adj=knn_adj,
        knn=knn,
        cfg=cfg,
        pos_flow=pos_flow,
        pos_knn=pos_knn,
        allowed=allowed,
        labels=labels,
        device=device,
        seed=int(args.seed),
        history=history,
        total_opt_steps=total_opt_steps,
        total_anchor_exposures=total_anchor_exposures,
        batch_diags=batch_diags,
        ckpt_dir=ckpt_dir,
        meta=ckpt_meta,
        verify_replay=False,
    )
    train_s = time.perf_counter() - t0

    # Build full features once for eval
    flow_full, _ = build_directed_flow_adjacency(
        df["from_id"].to_numpy(),
        df["to_id"].to_numpy(),
        df["Timestamp"].astype(float).to_numpy(),
        policy="immediate_next",
    )
    x_full = np.zeros((len(df), x_train.shape[1]), dtype=np.float32)
    x_full[tr] = x_train
    x_full[va] = prep.transform(df.iloc[va])
    x_full[te] = prep.transform(df.iloc[te])
    y_all = df[meta["label_col"]].to_numpy().astype(np.int64)

    learning_curve: Dict[str, Any] = {}
    for ep in EVAL_EPOCHS:
        path = ckpt_dir / f"epoch_{ep:02d}.pt"
        if not path.is_file():
            raise SystemExit(f"Missing checkpoint for eval: {path}")
        logging.info("Evaluating checkpoint epoch %s", ep)
        # Fresh opt shell for load API (state restored from ckpt).
        opt_eval = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        learning_curve[str(ep)] = evaluate_checkpoint(
            ckpt_path=path,
            encoder=encoder,
            opt=opt_eval,
            x_train=x_train,
            x_full=x_full,
            flow_ei_train=flow_ei,
            flow_full=flow_full,
            tr=tr,
            va=va,
            te=te,
            y_all=y_all,
            device=device,
            seed=int(args.seed),
        )

    # Predetermined validation selection (temporal HxX val AUPRC); never uses test.
    selected_epoch = None
    selected_val = float("-inf")
    for ep in EVAL_EPOCHS:
        block = learning_curve[str(ep)]["temporal_primary"][VAL_SELECT_REP]["val_ranking"]
        score = float(block.get(VAL_SELECT_KEY, float("nan")))
        if score > selected_val:
            selected_val = score
            selected_epoch = int(ep)
    selection = {
        "protocol": VAL_SELECT_PROTOCOL,
        "representation": VAL_SELECT_REP,
        "metric": VAL_SELECT_KEY,
        "split": "validation",
        "selected_epoch": selected_epoch,
        "selected_value": selected_val,
        "note": "Checkpoint chosen by validation metric only; test metrics reported but not used for selection.",
    }

    payload = {
        "not_exact_reproduction": True,
        "mode": args.mode,
        "batching_mode": "positive_complete",
        "max_total_nodes": int(args.max_total_nodes),
        "seed": int(args.seed),
        "target_epochs": 20,
        "eval_epochs": list(EVAL_EPOCHS),
        "steps_per_epoch": steps_per_epoch,
        "history": history,
        "train_seconds": train_s,
        "total_opt_steps": total_opt_steps,
        "total_anchor_exposures": total_anchor_exposures,
        "checkpoint_dir": str(ckpt_dir),
        "resume": resume_note,
        "preflight_diagnostics": preflight,
        "reference_5ep_json": str(ref_path),
        "batch_diagnostics_head": batch_diags,
        "flow_stats": {
            "n_nodes": flow_stats.n_nodes,
            "n_edges": flow_stats.n_edges,
            "policy": flow_stats.policy,
            "note": flow_stats.note,
        },
        "knn_deviations": knn.deviation_notes,
        "learning_curve": learning_curve,
        "validation_checkpoint_selection": selection,
        "eval_notes": [
            "Temporal protocol is primary; stratified random-40 is diagnostic only.",
            "Reports X / H / H||X with AUROC/AUPRC, F1/P/R at 0.5 and val-selected thr, "
            "positive prediction rate, and confusion counts.",
            "Checkpoint selection uses validation temporal HxX AUPRC only (never test).",
            "Original 5ep artifacts were not overwritten.",
            "Do not extend automatically beyond epoch 20.",
        ],
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    def _fmt_block(ep: int, proto: str, block: dict) -> List[str]:
        lines = [f"### Epoch {ep} — {proto}", ""]
        for rep in ("X", "H", "HxX"):
            f = block[rep]["threshold_0.5"]
            v = block[rep]["threshold_val_selected"]
            vr = block[rep].get("val_ranking", {})
            lines.append(
                f"- **{rep}** val AUPRC={vr.get('auprc')} | "
                f"@0.5 auprc={f['auprc']:.4f} f1={f['f1']:.4f} "
                f"ppr={f.get('positive_prediction_rate')} "
                f"tp/fp/tn/fn={int(f.get('tp',0))}/{int(f.get('fp',0))}/{int(f.get('tn',0))}/{int(f.get('fn',0))}"
            )
            lines.append(
                f"- **{rep}** @val-thr={v.get('validation_selected_threshold', v.get('threshold'))}: "
                f"auprc={v['auprc']:.4f} f1={v['f1']:.4f} "
                f"p={v['precision']:.4f} r={v['recall']:.4f} "
                f"ppr={v.get('positive_prediction_rate')} "
                f"tp/fp/tn/fn={int(v.get('tp',0))}/{int(v.get('fp',0))}/{int(v.get('tn',0))}/{int(v.get('fn',0))}"
            )
        lines.append("")
        return lines

    md: List[str] = [
        f"# Positive-complete txn-node resume to 20ep (`{args.mode}`)",
        "",
        "**Not an exact GCPAL reproduction.**",
        "",
        f"- Resume method: `{resume_note.get('method')}`",
        f"- seed={args.seed} cap={args.max_total_nodes} steps/epoch={steps_per_epoch}",
        f"- train_seconds={train_s:.1f} opt_steps={total_opt_steps} anchor_exposures={total_anchor_exposures}",
        f"- checkpoints: `{ckpt_dir}`",
        f"- val-selected checkpoint: epoch **{selected_epoch}** "
        f"(temporal HxX val AUPRC={selected_val:.6f})",
        "",
        "## Preflight (from 5ep JSON)",
        "",
        f"```json\n{json.dumps(preflight, indent=2)}\n```",
        "",
        "## Loss curve",
        "",
    ]
    for h in history:
        md.append(f"- ep {h['epoch']}: loss={h['loss_mean']:.6f}")
    md.extend(["", "## Learning curve evaluations", ""])
    for ep in EVAL_EPOCHS:
        md.extend(_fmt_block(ep, "temporal primary", learning_curve[str(ep)]["temporal_primary"]))
        md.extend(
            _fmt_block(ep, "random-40 diagnostic", learning_curve[str(ep)]["random40_diagnostic"])
        )
    out_md.write_text("\n".join(md) + "\n")
    logging.info("Wrote %s", out_json)
    logging.info("Wrote %s", out_md)
    print(out_json)


if __name__ == "__main__":
    main()
