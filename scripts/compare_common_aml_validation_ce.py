#!/usr/bin/env python3
"""Post-hoc common AMLWorld validation CE comparison (probes vs supervised Multi-GIN).

Standalone: does NOT edit TFMOE ablation / reeval / packaging training paths.

- Prefer reusing saved predictions.
- If supervised val logits missing: one validation-only inference pass (both last + best-val).
- Probe per-example logits were never saved; native final_probe_val_bce is retained as
  common unweighted CE when cohort size matches; weighted CE marked unavailable pending
  a separate deterministic re-probe follow-up (not auto-launched here).

Never loads/evaluates the test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loading import get_data  # noqa: E402
from train_util import (  # noqa: E402
    AddEgoIds,
    FORWARD_EDGE_TYPE,
    add_arange_ids,
    edge_classifier_logits,
    extract_param,
    get_loaders,
)
from training import get_model  # noqa: E402
from util import create_parser, logger_setup, set_seed  # noqa: E402

OUT_DIR = ROOT / "results/diagnostics/common_aml_validation_ce_comparison"
PRED_DIR = OUT_DIR / "predictions"
NOTE_PATH = ROOT / "notes/common_aml_validation_ce_comparison.md"
JSON_PATH = ROOT / "results/diagnostics/common_aml_validation_ce_comparison.json"
CSV_PATH = ROOT / "results/diagnostics/common_aml_validation_ce_comparison.csv"

REEVAL = ROOT / "results/diagnostics/direct_r198_40ep_linear_lr_full_extract_reeval"
R198_CELLS = REEVAL / "r198_only_lr_analysis/cells"
SRC_CELLS = REEVAL / "cells"
EMB_ROOT = ROOT / "embeddings/direct_r198_40ep_linear_lr_full_extract"

SUP_RUN = "small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2"
SUP_DIR = ROOT / "saved-models" / SUP_RUN
SUP_SUMMARY = (
    ROOT
    / "results/diagnostics/supervised_Small-HI_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2_summary.json"
)
SUP_HISTORY = (
    ROOT
    / "results/diagnostics/supervised_Small-HI_small_hi_legacy_supervised_gin_emlps_ports_50ep_seed2_epoch_history.json"
)
CE_AUDIT = (
    ROOT / "results/diagnostics/direct_r198_tfmoe_40ep_linear_lr_sweep/ce_audit.json"
)

EPS_CLIP = 1e-12


# ---------------------------------------------------------------------------
# CE metrics
# ---------------------------------------------------------------------------


def binary_nll_unweighted(y: np.ndarray, p: np.ndarray, eps: float = EPS_CLIP) -> float:
    y = y.astype(np.float64).reshape(-1)
    p = np.clip(p.astype(np.float64).reshape(-1), eps, 1.0 - eps)
    return float(np.mean(-y * np.log(p) - (1.0 - y) * np.log(1.0 - p)))


def binary_nll_from_logit_unweighted(y: np.ndarray, logit: np.ndarray) -> float:
    """Stable BCE-with-logits mean (matches F.binary_cross_entropy_with_logits reduction=mean)."""
    y_t = torch.as_tensor(y, dtype=torch.float64).reshape(-1)
    z = torch.as_tensor(logit, dtype=torch.float64).reshape(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y_t, reduction="mean")
    return float(loss.item())


def supervised_weighted_ce(
    y: np.ndarray,
    logits_2: np.ndarray,
    weight: Sequence[float],
) -> float:
    """Match CrossEntropyLoss(weight=..., reduction='mean') on a full cohort.

    L = sum_i w[y_i] * (-log p_i[y_i]) / sum_i w[y_i]
    """
    y = y.astype(np.int64).reshape(-1)
    z = torch.as_tensor(logits_2, dtype=torch.float64)
    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError(f"expected [N,2] logits, got {tuple(z.shape)}")
    w = torch.as_tensor(list(weight), dtype=torch.float64)
    log_p = torch.log_softmax(z, dim=-1)
    nll = -log_p[torch.arange(z.shape[0]), torch.as_tensor(y)]
    wy = w[torch.as_tensor(y)]
    return float((wy * nll).sum().item() / wy.sum().item())


def p_pos_from_two_logit(logits_2: np.ndarray) -> np.ndarray:
    z = torch.as_tensor(logits_2, dtype=torch.float64)
    return torch.softmax(z, dim=-1)[:, 1].numpy()


def test_weighted_ce_matches_pytorch() -> None:
    torch.manual_seed(0)
    n = 128
    logits = torch.randn(n, 2, dtype=torch.float64)
    y = torch.randint(0, 2, (n,))
    w = torch.tensor([1.0000182882773443, 6.275014431494497], dtype=torch.float64)
    ref = torch.nn.functional.cross_entropy(logits, y, weight=w, reduction="mean")
    got = supervised_weighted_ce(y.numpy(), logits.numpy(), w.tolist())
    assert abs(float(ref) - got) < 1e-10, (float(ref), got)


# ---------------------------------------------------------------------------
# Probe rows (scalars only)
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _probe_block(cell: Dict[str, Any], protocol: str) -> Dict[str, Any]:
    if protocol == "R198_only":
        # packaged r198-only cells put R198-only under primary; source cells under diagnostic
        if int((cell.get("primary") or {}).get("input_dim", -1)) == 198:
            return cell["primary"]
        return cell["diagnostic"]
    # R198+X+TF
    p = cell["primary"]
    if int(p.get("input_dim", -1)) != 227:
        raise ValueError(f"expected primary input_dim=227 for XTF, got {p.get('input_dim')}")
    return p


def _history_bce(block: Dict[str, Any], epoch: int) -> Optional[float]:
    for row in block.get("epoch_history") or []:
        if int(row.get("epoch", -1)) == int(epoch):
            return float(row["val_bce"])
    return None


def emb_val_ids(run: str, ssl_epoch: int) -> Tuple[np.ndarray, np.ndarray]:
    emb = EMB_ROOT / f"{run}_epoch{ssl_epoch:02d}" / "pre_embedding_3h"
    va = np.load(emb / "val.npz")
    eid = va["edge_id"].astype(np.int64).reshape(-1)
    y = va["y"].astype(np.int64).reshape(-1)
    if eid.size != np.unique(eid).size:
        raise RuntimeError(f"duplicate edge_id in {emb}")
    if (emb / "test.npz").is_file():
        raise RuntimeError(f"test.npz present (forbidden): {emb}")
    return eid, y


def sha_sorted_ids(ids: np.ndarray) -> str:
    ordered = np.sort(ids.astype(np.int64))
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def probe_rows() -> List[Dict[str, Any]]:
    """Build probe table rows from trusted cell JSONs (no re-probe)."""
    specs = [
        # best R198-only
        {
            "method": "DIRECT_R198",
            "feature_protocol": "R198_only",
            "run": "direct_r198_infonce_40ep_seed2_linear_lr1e-3",
            "ssl_epoch": 3,
            "cell": R198_CELLS
            / "direct_r198_infonce_40ep_seed2_linear_lr1e-3"
            / "epoch_03.json",
            "role": "best_ssl_by_r198_only_auprc",
        },
        {
            "method": "DIRECT_H_TFMOE",
            "feature_protocol": "R198_only",
            "run": "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3",
            "ssl_epoch": 10,
            "cell": R198_CELLS
            / "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3"
            / "epoch_10.json",
            "role": "best_ssl_by_r198_only_auprc",
        },
        # best R198+X+TF (collaborator source cells)
        {
            "method": "DIRECT_R198",
            "feature_protocol": "R198_X_TF",
            "run": "direct_r198_infonce_40ep_seed2_linear_lr6p2e-3",
            "ssl_epoch": 10,
            "cell": SRC_CELLS
            / "direct_r198_infonce_40ep_seed2_linear_lr6p2e-3"
            / "epoch_10.json",
            "role": "best_ssl_by_r198_x_tf_auprc",
        },
        {
            "method": "DIRECT_H_TFMOE",
            "feature_protocol": "R198_X_TF",
            "run": "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3",
            "ssl_epoch": 10,
            "cell": SRC_CELLS
            / "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3"
            / "epoch_10.json",
            "role": "best_ssl_by_r198_x_tf_auprc",
        },
        # final SSL epoch 40
        {
            "method": "DIRECT_R198",
            "feature_protocol": "R198_only",
            "run": "direct_r198_infonce_40ep_seed2_linear_lr1e-3",
            "ssl_epoch": 40,
            "cell": R198_CELLS
            / "direct_r198_infonce_40ep_seed2_linear_lr1e-3"
            / "epoch_40.json",
            "role": "final_ssl_epoch_40",
        },
        {
            "method": "DIRECT_H_TFMOE",
            "feature_protocol": "R198_only",
            "run": "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3",
            "ssl_epoch": 40,
            "cell": R198_CELLS
            / "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3"
            / "epoch_40.json",
            "role": "final_ssl_epoch_40",
        },
        {
            "method": "DIRECT_R198",
            "feature_protocol": "R198_X_TF",
            "run": "direct_r198_infonce_40ep_seed2_linear_lr6p2e-3",
            "ssl_epoch": 40,
            "cell": SRC_CELLS
            / "direct_r198_infonce_40ep_seed2_linear_lr6p2e-3"
            / "epoch_40.json",
            "role": "final_ssl_epoch_40",
        },
        {
            "method": "DIRECT_H_TFMOE",
            "feature_protocol": "R198_X_TF",
            "run": "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3",
            "ssl_epoch": 40,
            "cell": SRC_CELLS
            / "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3"
            / "epoch_40.json",
            "role": "final_ssl_epoch_40",
        },
    ]
    rows: List[Dict[str, Any]] = []
    for s in specs:
        cell = _load_json(s["cell"])
        verify = cell.get("verify") or {}
        if not verify.get("ok"):
            raise RuntimeError(f"verify.ok false: {s['cell']}")
        if int(verify.get("train_val_intersect", -1)) != 0:
            raise RuntimeError(f"train∩val != 0: {s['cell']}")
        n_val = int(verify.get("n_val") or 0)
        if n_val < 500000:  # refuse seed-only ~138k
            raise RuntimeError(f"n_val too small (seed-only?): {n_val} in {s['cell']}")
        block = _probe_block(cell, s["feature_protocol"])
        eid, y = emb_val_ids(s["run"], s["ssl_epoch"])
        if eid.shape[0] != n_val and abs(eid.shape[0] - n_val) > 2:
            # some cells report n_val from verify; embedding is authoritative
            logging.warning(
                "n_val mismatch cell=%s verify=%s emb=%s", s["cell"], n_val, eid.shape[0]
            )
        best_ep = int(block.get("best_probe_epoch", -1))
        final_bce = float(block["final_probe_val_bce"])
        selected_bce = _history_bce(block, best_ep)

        # Final classifier epoch (= MLP epoch 20) row
        rows.append(
            {
                "method": s["method"],
                "feature_protocol": s["feature_protocol"],
                "model_checkpoint": f"{s['run']} SSL ep{s['ssl_epoch']}",
                "ssl_epoch": s["ssl_epoch"],
                "ssl_role": s["role"],
                "classifier_checkpoint_meaning": "final_probe_epoch_20",
                "is_final_classifier": True,
                "is_validation_selected_classifier": False,
                "is_final_ssl": s["ssl_epoch"] == 40,
                "is_validation_selected_ssl": s["role"].startswith("best_ssl"),
                "native_loss_definition": (
                    "unweighted binary CE via BCEWithLogits (one logit); "
                    "class_weights=None; pos_weight=None; reduction=mean; "
                    "final_probe_val_bce = last PaperStyleMLP epoch (20)"
                ),
                "native_final_val_loss": final_bce,
                "common_unweighted_val_ce": final_bce,
                "common_unweighted_note": (
                    "Equals native final_probe_val_bce (same formula/cohort size). "
                    "Per-example logits not saved; EdgeID-aligned recompute not possible "
                    "without deterministic re-probe."
                ),
                "common_supervised_weighted_val_ce": None,
                "common_weighted_note": (
                    "UNAVAILABLE: probe per-example logits/probs were not persisted. "
                    "Requires separate deterministic PaperStyleMLP re-probe on existing "
                    "full-subgraph embeddings (not auto-launched)."
                ),
                "n": int(eid.shape[0]),
                "positives": int((y == 1).sum()),
                "prevalence": float(y.mean()),
                "id_hash": sha_sorted_ids(eid),
                "predictions_exist": False,
                "source_cell": str(s["cell"]),
                "best_probe_epoch": best_ep,
                "validation_auprc_selected": float(block.get("validation_auprc", float("nan"))),
            }
        )
        # Selected-classifier epoch row (never labeled final)
        if selected_bce is not None:
            rows.append(
                {
                    "method": s["method"],
                    "feature_protocol": s["feature_protocol"],
                    "model_checkpoint": f"{s['run']} SSL ep{s['ssl_epoch']}",
                    "ssl_epoch": s["ssl_epoch"],
                    "ssl_role": s["role"],
                    "classifier_checkpoint_meaning": f"validation_selected_probe_epoch_{best_ep}",
                    "is_final_classifier": False,
                    "is_validation_selected_classifier": True,
                    "is_final_ssl": s["ssl_epoch"] == 40,
                    "is_validation_selected_ssl": s["role"].startswith("best_ssl"),
                    "native_loss_definition": (
                        "unweighted binary CE at best-val-AUPRC PaperStyleMLP epoch "
                        "(from epoch_history.val_bce); NOT the final probe epoch"
                    ),
                    "native_final_val_loss": None,
                    "native_selected_probe_val_bce": selected_bce,
                    "common_unweighted_val_ce": selected_bce,
                    "common_unweighted_note": "From epoch_history at best_probe_epoch; not final.",
                    "common_supervised_weighted_val_ce": None,
                    "common_weighted_note": "UNAVAILABLE (no per-example preds)",
                    "n": int(eid.shape[0]),
                    "positives": int((y == 1).sum()),
                    "prevalence": float(y.mean()),
                    "id_hash": sha_sorted_ids(eid),
                    "predictions_exist": False,
                    "source_cell": str(s["cell"]),
                    "best_probe_epoch": best_ep,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Supervised inference
# ---------------------------------------------------------------------------


def _build_model_config(args) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        model=args.model,
        n_hidden=extract_param("n_hidden", args),
        n_gnn_layers=extract_param("n_gnn_layers", args),
        n_heads=extract_param("n_heads", args) if args.model == "gat" else None,
        dropout=extract_param("dropout", args),
        final_dropout=extract_param("final_dropout", args),
    )


@torch.no_grad()
def collect_val_logits_edgeids(
    loader, split_inds, model, data, device, args
) -> Dict[str, np.ndarray]:
    """Validation-only: two-logit logits, labels, global EdgeIDs. No test access."""
    y_chunks: List[torch.Tensor] = []
    logit_chunks: List[torch.Tensor] = []
    eid_chunks: List[torch.Tensor] = []
    hetero = isinstance(data, HeteroData)
    split_inds_cpu = split_inds.detach().cpu()
    store = FORWARD_EDGE_TYPE

    for batch in loader:
        if hetero:
            batch_edge_inds = split_inds_cpu[batch[store].input_id.detach().cpu()]
            batch_edge_ids = loader.data[store].edge_attr.detach().cpu()[batch_edge_inds, 0]
            edge_ids = batch[store].edge_attr[:, 0].detach().cpu()
            mask = torch.isin(edge_ids, batch_edge_ids)
            # Preserve global ids for scored rows before dropping id column
            scored_eids = edge_ids[mask].long()
            batch[store].edge_attr = batch[store].edge_attr[:, 1:]
            batch[("node", "rev_to", "node")].edge_attr = batch[
                ("node", "rev_to", "node")
            ].edge_attr[:, 1:]
            batch.to(device)
            z = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[store]
            logits = edge_classifier_logits(model, z)[mask.to(device)]
            y = batch[store].y[mask.to(device)]
        else:
            batch_edge_inds = split_inds_cpu[batch.input_id.detach().cpu()]
            batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            edge_ids = batch.edge_attr[:, 0].detach().cpu()
            mask = torch.isin(edge_ids, batch_edge_ids)
            scored_eids = edge_ids[mask].long()
            batch.edge_attr = batch.edge_attr[:, 1:]
            batch.to(device)
            z = model(batch.x, batch.edge_index, batch.edge_attr)
            logits = model.classifier(z)[mask.to(device)]
            y = batch.y[mask.to(device)]
        y_chunks.append(y.detach().cpu().long())
        logit_chunks.append(logits.detach().cpu().to(torch.float32))
        eid_chunks.append(scored_eids)

    y_np = torch.cat(y_chunks).numpy().astype(np.int64)
    z_np = torch.cat(logit_chunks).numpy().astype(np.float32)
    eid_np = torch.cat(eid_chunks).numpy().astype(np.int64)
    if eid_np.size != np.unique(eid_np).size:
        raise RuntimeError("duplicate EdgeIDs in supervised val collection")
    return {"y": y_np, "logits": z_np, "edge_id": eid_np}


def class_weights_from_checkpoint(ckpt: Dict[str, Any]) -> Tuple[float, float, str]:
    cfg = ckpt.get("config") or {}
    if "w_ce1" in cfg and "w_ce2" in cfg:
        return float(cfg["w_ce1"]), float(cfg["w_ce2"]), "checkpoint['config']"
    raise KeyError("class weights not found in checkpoint config")


def infer_supervised_both(device: torch.device) -> Dict[str, Any]:
    """Score checkpoint_last and checkpoint_best_val_f1 on validation only."""
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    last_path = SUP_DIR / "checkpoint_last.tar"
    best_path = SUP_DIR / "checkpoint_best_val_f1.tar"
    ckpt_last = torch.load(last_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt_last["args"]
    # create_parser() requires --data/--model; supply placeholders then override from ckpt.
    parser = create_parser()
    args, _unknown = parser.parse_known_args(
        [
            "--data",
            str(ckpt_args["data"]),
            "--model",
            str(ckpt_args["model"]),
            "--unique_name",
            str(ckpt_args["unique_name"]),
            "--seed",
            str(int(ckpt_args["seed"])),
            "--batch_size",
            str(int(ckpt_args["batch_size"])),
            "--n_epochs",
            str(int(ckpt_args["n_epochs"])),
            "--num_neighs",
            *[str(x) for x in list(ckpt_args["num_neighs"])],
            "--supervised_head",
            str(ckpt_args["supervised_head"]),
            "--objective",
            "supervised",
            "--skip_test_eval",
            "--tqdm",
            "--testing",
        ]
        + (["--reverse_mp"] if ckpt_args["reverse_mp"] else [])
        + (["--emlps"] if ckpt_args["emlps"] else [])
        + (["--ports"] if ckpt_args["ports"] else [])
        + (["--tds"] if ckpt_args["tds"] else [])
        + (["--ego"] if ckpt_args["ego"] else [])
    )
    set_seed(args.seed)

    with open("data_config.json", "r", encoding="utf-8") as f:
        data_config = json.load(f)

    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    if te_inds is not None and hasattr(te_inds, "numel") and int(te_inds.numel()) != 0:
        # Hard lock: empty test seed set under skip_test_eval
        logging.warning("Emptying te_inds for skip_test_eval safety")
        te_inds = te_inds[:0]
    transform = AddEgoIds() if args.ego else None
    add_arange_ids([tr_data, val_data, te_data])
    # Loaders: validation only used for scoring; train loader needed for model shape sample
    tr_loader, val_loader, te_loader = get_loaders(
        tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args, train_shuffle=False
    )
    # Do not iterate te_loader
    del te_loader

    sample_batch = next(iter(tr_loader))
    config = _build_model_config(args)
    w0, w1, w_src = class_weights_from_checkpoint(ckpt_last)
    meta = {
        "class_weights": {"0": w0, "1": w1},
        "class_weights_source": w_src,
        "model_mode": "train_mode_bn_match_training_val_pass",
        "skip_test_eval": True,
        "test_accessed": False,
    }

    out_meta = {"checkpoints": {}, **meta}
    for label, path, ckpt in (
        ("final_epoch_50", last_path, ckpt_last),
        (
            "best_validation_f1",
            best_path,
            torch.load(best_path, map_location="cpu", weights_only=False),
        ),
    ):
        model = get_model(sample_batch, config, args)
        if args.reverse_mp:
            model = to_hetero(model, val_data.metadata(), aggr="mean")
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        # Match training validation pass: do NOT call model.eval()
        model.train()
        pack = collect_val_logits_edgeids(val_loader, val_inds, model, val_data, device, args)
        pred_path = PRED_DIR / f"supervised_{label}_val.npz"
        np.savez_compressed(
            pred_path,
            y=pack["y"],
            logits=pack["logits"],
            edge_id=pack["edge_id"],
        )
        p = p_pos_from_two_logit(pack["logits"])
        lu = binary_nll_unweighted(pack["y"], p)
        lw = supervised_weighted_ce(pack["y"], pack["logits"], [w0, w1])
        out_meta["checkpoints"][label] = {
            "checkpoint_path": str(path),
            "checkpoint_epoch": int(ckpt.get("epoch", -1)),
            "selected_epoch": ckpt.get("selected_epoch"),
            "best_validation_f1": ckpt.get("best_validation_f1"),
            "pred_path": str(pred_path),
            "n": int(pack["y"].shape[0]),
            "positives": int((pack["y"] == 1).sum()),
            "prevalence": float(pack["y"].mean()),
            "id_hash": sha_sorted_ids(pack["edge_id"]),
            "common_unweighted_val_ce": lu,
            "common_supervised_weighted_val_ce": lw,
        }
        logging.info(
            "supervised %s ep=%s n=%s L_unw=%.6f L_w=%.6f",
            label,
            ckpt.get("epoch"),
            pack["y"].shape[0],
            lu,
            lw,
        )
    (PRED_DIR / "supervised_infer_meta.json").write_text(json.dumps(out_meta, indent=2) + "\n")
    return out_meta


def _index_by_edge_id(edge_id: np.ndarray, y: np.ndarray, logits: np.ndarray) -> Dict[int, int]:
    if edge_id.size != np.unique(edge_id).size:
        raise RuntimeError("duplicate EdgeIDs in prediction artifact")
    return {int(e): i for i, e in enumerate(edge_id.tolist())}


def align_supervised_preds_to_common_ids(
    infer_meta: Dict[str, Any],
    *,
    common_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Recompute supervised CE on one shared EdgeID ordering (intersection).

    NeighborLoader seed coverage can differ by a few edges across checkpoint passes;
    refuse silent row-position comparison and align by global EdgeID instead.
    """
    w0 = float(infer_meta["class_weights"]["0"])
    w1 = float(infer_meta["class_weights"]["1"])
    packs = {}
    for key, meta in infer_meta["checkpoints"].items():
        d = np.load(meta["pred_path"])
        packs[key] = {
            "edge_id": d["edge_id"].astype(np.int64).reshape(-1),
            "y": d["y"].astype(np.int64).reshape(-1),
            "logits": d["logits"].astype(np.float64),
        }
    id_sets = [set(p["edge_id"].tolist()) for p in packs.values()]
    inter = set.intersection(*id_sets) if id_sets else set()
    if common_ids is not None:
        inter &= set(common_ids.astype(np.int64).tolist())
    ordered = np.array(sorted(inter), dtype=np.int64)
    if ordered.size == 0:
        raise RuntimeError("empty supervised EdgeID intersection")

    aligned_meta = dict(infer_meta)
    aligned_meta["alignment"] = {
        "policy": "intersection_of_supervised_pred_edge_ids_sorted",
        "n_common": int(ordered.size),
        "per_checkpoint_n_raw": {k: int(p["edge_id"].size) for k, p in packs.items()},
        "coverage_vs_raw": {
            k: float(ordered.size / max(int(p["edge_id"].size), 1)) for k, p in packs.items()
        },
        "id_hash_common": sha_sorted_ids(ordered),
    }
    aligned_meta["checkpoints"] = {}
    for key, p in packs.items():
        idx_map = _index_by_edge_id(p["edge_id"], p["y"], p["logits"])
        ii = np.array([idx_map[int(e)] for e in ordered], dtype=np.int64)
        y = p["y"][ii]
        logits = p["logits"][ii]
        # Label consistency across checkpoints on shared IDs
        if key != next(iter(packs)):
            y0 = aligned_meta["checkpoints"][next(iter(packs))]["_y_aligned"]
            if not np.array_equal(y0, y):
                raise RuntimeError(f"label disagreement on common EdgeIDs for {key}")
        p_pos = p_pos_from_two_logit(logits)
        lu = binary_nll_unweighted(y, p_pos)
        lw = supervised_weighted_ce(y, logits, [w0, w1])
        raw = infer_meta["checkpoints"][key]
        aligned_meta["checkpoints"][key] = {
            **{k: v for k, v in raw.items() if k not in ("n", "positives", "prevalence", "id_hash",
                                                         "common_unweighted_val_ce",
                                                         "common_supervised_weighted_val_ce")},
            "n": int(y.size),
            "positives": int((y == 1).sum()),
            "prevalence": float(y.mean()),
            "id_hash": sha_sorted_ids(ordered),
            "common_unweighted_val_ce": lu,
            "common_supervised_weighted_val_ce": lw,
            "common_unweighted_val_ce_raw_coverage": raw.get("common_unweighted_val_ce"),
            "common_supervised_weighted_val_ce_raw_coverage": raw.get(
                "common_supervised_weighted_val_ce"
            ),
            "_y_aligned": y,  # transient for label check; stripped before write
        }
    # strip transient arrays before JSON serialization
    for key in aligned_meta["checkpoints"]:
        aligned_meta["checkpoints"][key].pop("_y_aligned", None)
    return aligned_meta


def supervised_rows(infer_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    hist = _load_json(SUP_HISTORY)
    epochs_list = hist["epochs"] if isinstance(hist, dict) and "epochs" in hist else hist
    by_ep = {int(r["epoch"]): r for r in epochs_list}
    summary = _load_json(SUP_SUMMARY)
    rows = []
    mapping = [
        (
            "final_epoch_50",
            "final_supervised_epoch_50",
            True,
            False,
            50,
            "native logged value is TRAIN weighted CE (not val)",
        ),
        (
            "best_validation_f1",
            "best_validation_supervised_checkpoint",
            False,
            True,
            int(summary["best_validation_epoch"]),
            "native logged value is TRAIN weighted CE at best-val epoch (not val)",
        ),
    ]
    for key, meaning, is_final, is_sel, ep, native_note in mapping:
        m = infer_meta["checkpoints"][key]
        native_train = None
        if ep in by_ep:
            native_train = float(by_ep[ep].get("train_loss", float("nan")))
        rows.append(
            {
                "method": "supervised_MultiGIN",
                "feature_protocol": "supervised_raw_edge_features",
                "model_checkpoint": f"{SUP_RUN} ep{m['checkpoint_epoch']}",
                "ssl_epoch": None,
                "ssl_role": None,
                "classifier_checkpoint_meaning": meaning,
                "is_final_classifier": is_final,
                "is_validation_selected_classifier": is_sel,
                "is_final_ssl": None,
                "is_validation_selected_ssl": None,
                "native_loss_definition": (
                    "2-logit CrossEntropyLoss(weight=[w0,w1], reduction=mean); "
                    f"weights from {infer_meta['class_weights_source']}: "
                    f"{infer_meta['class_weights']}; "
                    "logged epoch loss = example-mean TRAIN weighted CE "
                    "(sum batch_mean*n / N_train). Validation CE was never logged."
                ),
                "native_final_val_loss": None,
                "native_train_weighted_ce": native_train,
                "native_train_ce_note": native_note,
                "common_unweighted_val_ce": m["common_unweighted_val_ce"],
                "common_supervised_weighted_val_ce": m["common_supervised_weighted_val_ce"],
                "common_unweighted_note": (
                    "Recomputed from saved val logits on EdgeID-aligned common cohort."
                ),
                "common_weighted_note": (
                    "Recomputed on EdgeID-aligned common cohort; matches "
                    "PyTorch CE(weight, reduction=mean)."
                ),
                "n": m["n"],
                "positives": m["positives"],
                "prevalence": m["prevalence"],
                "id_hash": m["id_hash"],
                "predictions_exist": True,
                "source_pred": m["pred_path"],
                "class_weights": infer_meta["class_weights"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Audit + report
# ---------------------------------------------------------------------------


def build_audit(preds_needed: bool) -> Dict[str, Any]:
    return {
        "probe_predictions": {
            "per_example_logits_or_probs": False,
            "available": "aggregate final_probe_val_bce + epoch_history only",
            "cells_checked": [
                str(R198_CELLS / "direct_r198_infonce_40ep_seed2_linear_lr1e-3/epoch_03.json"),
                str(R198_CELLS / "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3/epoch_10.json"),
                str(SRC_CELLS / "direct_r198_infonce_40ep_seed2_linear_lr6p2e-3/epoch_10.json"),
                str(SRC_CELLS / "direct_r198_tfmoe_40ep_seed2_linear_lr2e-3/epoch_10.json"),
            ],
            "classifier_final_vs_selected": (
                "final_probe_val_bce = PaperStyleMLP epoch 20; "
                "ranking metrics / best_probe_epoch = best val AUPRC epoch"
            ),
            "seed_only_excluded": True,
        },
        "supervised_predictions": {
            "per_example_before_this_script": False,
            "checkpoints": {
                "final": str(SUP_DIR / "checkpoint_last.tar"),
                "best_val": str(SUP_DIR / "checkpoint_best_val_f1.tar"),
            },
            "native_ce_0_011657_source": (
                f"{SUP_HISTORY} epoch 50 train_loss "
                "(also ce_audit.json supervised_native.final_epoch.train_loss_weighted_ce)"
            ),
            "native_ce_0_011319_source": (
                f"{SUP_HISTORY} epoch 43 train_loss "
                "(best-validation epoch; TRAIN weighted CE, not validation)"
            ),
            "inference_necessary": preds_needed,
        },
        "loss_definitions": {
            "probe": {
                "loss": "binary_cross_entropy_with_logits",
                "logits": "one_logit",
                "class_weights": None,
                "pos_weight": None,
                "reduction": "mean",
                "mlp_epochs": 20,
                "final_vs_selected": (
                    "final = last probe epoch BCE; selected = best_val_auprc epoch "
                    "(ranking); both recorded separately"
                ),
            },
            "supervised": {
                "loss": "CrossEntropyLoss",
                "logits": "two_logit",
                "class_weights_source": "checkpoint config w_ce1/w_ce2",
                "reduction": "mean",
                "train_loss_aggregation": "example-mean of batch means (sum loss*n / N)",
                "validation_ce_logged": False,
            },
        },
        "tfmoe_ablation_untouched": True,
        "test_accessed": False,
    }


def write_outputs(audit: Dict[str, Any], rows: List[Dict[str, Any]], infer_meta: Optional[Dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Align ID hash check across probe emb refs and supervised
    hashes = {r["id_hash"] for r in rows if r.get("id_hash")}
    cohort_note = {
        "unique_id_hashes": sorted(hashes),
        "n_distinct_cohorts": len(hashes),
        "policy": (
            "Probe rows use full-subgraph embedding val EdgeIDs; supervised uses "
            "inferred val EdgeIDs. If hashes match, cohorts are identical."
        ),
    }
    # Main-table subset: final classifier rows + supervised both
    main = [
        r
        for r in rows
        if r.get("is_final_classifier")
        or (
            r.get("method") == "supervised_MultiGIN"
            and r.get("classifier_checkpoint_meaning")
            in ("final_supervised_epoch_50", "best_validation_supervised_checkpoint")
        )
    ]
    # Prefer best-SSL rows + final SSL + supervised for main table readability
    main_preferred = []
    for r in main:
        if r["method"] == "supervised_MultiGIN":
            main_preferred.append(r)
        elif r.get("is_final_classifier") and (
            r.get("is_validation_selected_ssl") or r.get("is_final_ssl")
        ):
            main_preferred.append(r)

    payload = {
        "audit": audit,
        "cohort": cohort_note,
        "infer_meta": infer_meta,
        "rows": rows,
        "main_table_rows": main_preferred,
        "answers": _answers(audit, main_preferred, infer_meta, hashes),
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    # CSV
    import csv

    fields = [
        "method",
        "feature_protocol",
        "model_checkpoint",
        "classifier_checkpoint_meaning",
        "native_loss_definition",
        "native_final_val_loss",
        "native_train_weighted_ce",
        "common_unweighted_val_ce",
        "common_supervised_weighted_val_ce",
        "n",
        "positives",
        "id_hash",
        "is_final_classifier",
        "is_validation_selected_classifier",
        "is_final_ssl",
        "is_validation_selected_ssl",
    ]
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in main_preferred:
            w.writerow(r)

    NOTE_PATH.write_text(_markdown(audit, main_preferred, rows, payload["answers"], cohort_note))
    logging.info("Wrote %s %s %s", JSON_PATH, CSV_PATH, NOTE_PATH)


def _answers(audit, main_rows, infer_meta, hashes) -> Dict[str, Any]:
    unw = {
        f"{r['method']}|{r['feature_protocol']}|{r['classifier_checkpoint_meaning']}|ssl{r.get('ssl_epoch')}": r.get(
            "common_unweighted_val_ce"
        )
        for r in main_rows
    }
    wtd = {
        f"{r['method']}|{r['feature_protocol']}|{r['classifier_checkpoint_meaning']}|ssl{r.get('ssl_epoch')}": r.get(
            "common_supervised_weighted_val_ce"
        )
        for r in main_rows
    }
    return {
        "1_existing_predictions_sufficient": False,
        "1_detail": (
            "Probe aggregates existed; supervised per-example val preds were missing "
            "(required one inference job)."
        ),
        "2_slurm_job_submitted": bool(infer_meta),
        "2_detail": (
            json.dumps(_load_json(OUT_DIR / "submission.json"))
            if (OUT_DIR / "submission.json").is_file()
            else "infer_meta present; see submission.json if written"
        ),
        "3_native_losses_directly_comparable": False,
        "3_detail": (
            "Probe native loss = unweighted val BCE; supervised logged loss = TRAIN "
            "weighted CE. Different objective, split, and weighting."
        ),
        "4_common_unweighted_ce": unw,
        "5_common_supervised_weighted_ce": wtd,
        "6_recomputed_match_logged": {
            "probe_unweighted_equals_native_final_bce": True,
            "supervised_weighted_val_vs_logged_train_ce": (
                "Not expected to match: logged values are TRAIN CE; recomputed are VAL CE."
            ),
        },
        "7_final_vs_selected": {
            "final": [
                r
                for r in main_rows
                if r.get("is_final_classifier") or r.get("classifier_checkpoint_meaning") == "final_supervised_epoch_50"
            ],
            "validation_selected": [
                r
                for r in main_rows
                if r.get("is_validation_selected_classifier")
                or r.get("classifier_checkpoint_meaning") == "best_validation_supervised_checkpoint"
            ],
        },
        "8_test_data_accessed": False,
        "9_tfmoe_jobs_or_code_modified": False,
        "cohort_id_hashes": sorted(hashes),
    }


def _markdown(audit, main_rows, all_rows, answers, cohort) -> str:
    lines = [
        "# Common AMLWorld validation CE comparison",
        "",
        "## 1. Artifact audit",
        "",
        "```json",
        json.dumps(audit, indent=2),
        "```",
        "",
        "## 2. Loss definitions",
        "",
        "- **Probe:** one-logit `binary_cross_entropy_with_logits`, no class/pos weights, "
        "`reduction=mean`, 20 PaperStyleMLP epochs. "
        "`final_probe_val_bce` = epoch 20; ranking uses `best_probe_epoch` by val AUPRC.",
        "- **Supervised:** two-logit `CrossEntropyLoss(weight=[w_ce1,w_ce2], reduction=mean)`. "
        "Weights read from checkpoint `config` (not hardcoded). "
        "Logged epoch loss is **train** example-mean weighted CE; validation CE was not logged.",
        "",
        "## 3. Cohort",
        "",
        f"- Distinct EdgeID hashes: `{cohort['unique_id_hashes']}`",
        f"- {cohort['policy']}",
        "",
        "## 4. Main table (final classifier / supervised checkpoints)",
        "",
        "| Method | Feature protocol | Model checkpoint | Classifier meaning | Native final val loss | Common unweighted val CE | Common supervised-weighted val CE | n | positives | ID hash |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in main_rows:
        lines.append(
            "| {method} | {fp} | {mc} | {cm} | {nv} | {cu} | {cw} | {n} | {pos} | `{h}` |".format(
                method=r["method"],
                fp=r["feature_protocol"],
                mc=r["model_checkpoint"],
                cm=r["classifier_checkpoint_meaning"],
                nv=("" if r.get("native_final_val_loss") is None else f"{r['native_final_val_loss']:.6f}"),
                cu=("" if r.get("common_unweighted_val_ce") is None else f"{r['common_unweighted_val_ce']:.6f}"),
                cw=(
                    "N/A"
                    if r.get("common_supervised_weighted_val_ce") is None
                    else f"{r['common_supervised_weighted_val_ce']:.6f}"
                ),
                n=r["n"],
                pos=r["positives"],
                h=r["id_hash"][:16],
            )
        )
    lines += [
        "",
        "Selected-classifier probe rows (not final) are in the JSON under `rows` "
        "with `is_validation_selected_classifier=true`.",
        "",
        "## 5. Direct answers",
        "",
        f"1. Existing predictions sufficient? **{answers['1_existing_predictions_sufficient']}** — {answers['1_detail']}",
        f"2. Slurm job submitted? **{answers['2_slurm_job_submitted']}** — {answers['2_detail']}",
        f"3. Native losses directly comparable? **{answers['3_native_losses_directly_comparable']}** — {answers['3_detail']}",
        "4. Common unweighted CE values:",
        "```json",
        json.dumps(answers["4_common_unweighted_ce"], indent=2),
        "```",
        "5. Common supervised-weighted CE values:",
        "```json",
        json.dumps(answers["5_common_supervised_weighted_ce"], indent=2),
        "```",
        f"6. Recomputed match logged? {json.dumps(answers['6_recomputed_match_logged'])}",
        "7. Final vs validation-selected: see JSON `answers.7_final_vs_selected` "
        "(probe final = MLP ep20; supervised final = ep50; selected SSL/probe/supervised flagged separately).",
        "8. Test data accessed? **no**",
        "9. Active TFMOE jobs or their code paths modified? **no**",
        "",
        "## 6. Proposed follow-up (NOT launched)",
        "",
        "Deterministic PaperStyleMLP re-probe on existing full-subgraph embeddings for the "
        "cells listed in the audit, saving `edge_id`, `y`, and one-logit `logits` for "
        "final (ep20) and optionally selected probe epochs, to enable EdgeID-aligned "
        "common weighted CE for probes. Single dedicated job; do not touch TFMOE ablation DAG.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true", help="Run weighted-CE unit test and exit")
    ap.add_argument(
        "--infer-supervised",
        action="store_true",
        help="Run validation-only supervised inference for last + best-val checkpoints",
    )
    ap.add_argument(
        "--write-report",
        action="store_true",
        help="Assemble audit/table/note from cells + any saved supervised preds",
    )
    args = ap.parse_args()

    if args.self_test:
        test_weighted_ce_matches_pytorch()
        print("self_test_ok")
        return 0

    pred_last = PRED_DIR / "supervised_final_epoch_50_val.npz"
    pred_best = PRED_DIR / "supervised_best_validation_f1_val.npz"
    have_preds = pred_last.is_file() and pred_best.is_file()
    audit = build_audit(preds_needed=not have_preds)

    infer_meta = None
    if args.infer_supervised or (args.write_report and not have_preds):
        if have_preds and not args.infer_supervised:
            infer_meta = _load_json(PRED_DIR / "supervised_infer_meta.json")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logging.info("Running supervised val-only inference on %s", device)
            infer_meta = infer_supervised_both(device)
    elif have_preds:
        infer_meta = _load_json(PRED_DIR / "supervised_infer_meta.json")

    if args.write_report or args.infer_supervised:
        if infer_meta is None:
            raise SystemExit("Supervised predictions missing; re-run with --infer-supervised")
        # Align supervised preds onto common EdgeIDs (handles tiny NeighborLoader coverage drift).
        probe_rows_raw = probe_rows()
        # Prefer also intersecting with a representative probe embedding cohort when unique.
        ref_probe_ids = None
        probe_hashes = {r["id_hash"] for r in probe_rows_raw}
        if len(probe_hashes) == 1:
            eid, _y = emb_val_ids("direct_r198_tfmoe_40ep_seed2_linear_lr2e-3", 10)
            ref_probe_ids = eid
        infer_meta = align_supervised_preds_to_common_ids(infer_meta, common_ids=ref_probe_ids)
        (PRED_DIR / "supervised_infer_meta_aligned.json").write_text(
            json.dumps(infer_meta, indent=2) + "\n"
        )
        rows = probe_rows_raw + supervised_rows(infer_meta)
        sup_hash = infer_meta["alignment"]["id_hash_common"]
        logging.info(
            "Aligned supervised cohort n=%s hash=%s coverage=%s",
            infer_meta["alignment"]["n_common"],
            sup_hash[:16],
            infer_meta["alignment"]["coverage_vs_raw"],
        )
        if probe_hashes and sup_hash not in probe_hashes:
            logging.warning(
                "Common supervised hash %s differs from probe embedding hashes %s "
                "(probe weighted CE still unavailable; unweighted probe BCE is native-cohort).",
                sup_hash[:16],
                {h[:16] for h in probe_hashes},
            )
        write_outputs(audit, rows, infer_meta)
        print(json.dumps({"status": "ok", "json": str(JSON_PATH), "note": str(NOTE_PATH)}, indent=2))
        return 0

    # Audit-only
    print(json.dumps(audit, indent=2))
    print(
        "Predictions sufficient for supervised weighted CE?",
        have_preds,
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
